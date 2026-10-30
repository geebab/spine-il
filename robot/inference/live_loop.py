#!/usr/bin/env python3
"""Closed-loop policy on the two arms: what the policy sees is what the robot did.

rollout.py runs the policy against the viewer alone — the next state is the pose the policy just commanded ("the robot
followed perfectly") — and the robot plays the finished trajectory back afterwards. Here the arms move while the policy
runs, one action chunk at a time:

    measured tool poses (tracker) ─┬→ observation.state
                                   └→ viewer step session (render API /v1/step) → endoscope image, bone cut where the
                                      robot tools really were
    policy (chunk-relative ACT) → the next N poses from the measured pose → safety gate → the arms (endo_follow stream)

Per chunk: predict → gate every target → send the N targets (one per 1/(fps·speed) s) → the arms move (nothing is
rendered meanwhile: the recorder page shares the GPU and a concurrent render stalls its drill loop) → the arms hold,
trigger off → the measured pose at each target time is stepped into the viewer session → one more step at the settled
pose → that state and image are the next observation. The holds exist only on the wall clock: in policy time (the
viewer session, the phase rules, the trigger the policy sees) the frames follow each other at 1/fps.
Scripted entry (ready pose → working pose) and exit as rollout.py --cycle, also measured.

--obs-url (10-06, user: "정책이 기록 페이지의 화면을 직접 보게 해"): the step session is a second simulation — it cuts where the
measured poses are fed in afterwards, the recorder page cuts as the robot moves, and the two differ (165–200 mm³ against
265–327 in the same runs), so the policy looked at bone the run had already removed. With --obs-url there is no step
session: after a chunk, while the arms hold, the image and the bridge metrics are read from the recorder page itself
(robot_origin_rec.py --obs-port, page_obs.py) — the bone the policy sees is the bone the run is judged by. Nothing is fed:
the page saw the tools through the tracker. What the page removed during a chunk is known as a total, once per chunk.

    # 1. plan: the ready pose and the robot start joints (no robot, no policy) → a seed trajectory for the run script
    python robot/inference/live_loop.py --prepare-out seed.json --dataset DS --episode 6 --start-frame 0 --gate-anchor-s 17 \
        --gate-pair 3868 --cycle [--start-state start_k.json]
    # 2. run: started by robot/eval/robot_origin_run.sh (RO_LIVE=<args file>) once the recorder page is up
    python robot/inference/live_loop.py --live-run RUN --policy-path CKPT --phase-rules … -o live.json

Output JSON: frames = the measured trajectory (what the viewer session was fed), frames_cmd = the commanded one, meta as
rollout.py's plus timing (inference, render step), tracking deviation per chunk and aborts.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import page_obs as PG          # noqa: E402
import remote_policy as RP     # noqa: E402
import rollout as RO            # noqa: E402
from rollout import AC          # noqa: E402
from safety import limit_step, step_size, _slerp   # noqa: E402
import stream_plan as SP       # noqa: E402


def log(*a) -> None:
    print("[live]", time.strftime("%T"), *a, flush=True)


def render_busy(api: str) -> list:
    """Ids of the render API's queued / running jobs. A step session queued behind a job is closed by the server's
    keepalive after ~40 s, so the session is opened only when the renderer is free."""
    import urllib.request
    base = api.replace("ws://", "http://").replace("wss://", "https://").split("/v1/")[0]
    tok = RO.token()
    rq = urllib.request.Request(base + "/v1/jobs", headers={"Authorization": f"Bearer {tok}"} if tok else {})
    jobs = json.loads(urllib.request.urlopen(rq, timeout=30).read())
    return [j["id"] for j in jobs if j.get("status") in ("queued", "running", "pending", "started")]


async def run(a) -> int:
    if not (a.cycle and a.gate_pair):
        raise SystemExit("--cycle and --gate-pair are required (the robot starts at the gate-checked ready pose)")
    P = RO.prepare(a)
    ds, meta, states, hdr, flags, dt_ms, k0 = (P[k] for k in ("ds", "meta", "states", "hdr", "flags", "dt_ms", "k0"))
    portals, shafts, gate, q_ready, cyc = (P[k] for k in ("portals", "shafts", "gate", "q_ready", "cyc"))
    ready = cyc["ready"]
    if a.prepare_out:   # the seed: 1 s at the ready pose (policy2rec → the recording the arms and the recorder page open on)
        Path(a.prepare_out).write_text(json.dumps({
            "fps": a.fps, "frames": [ready.tolist()] * 15, "ref_recording": meta["recording"],
            "meta": {"policy": "live-seed", "dataset": str(ds), "episode": a.episode, "start_frame": k0, "start_state": a.start_state,
                     "registration": P["registration"],
                     "cycle": {"segments": [["ready", 0]], "ready_out_mm": a.ready_out_mm, "robot_start": q_ready}}}))
        print(json.dumps({"seed": a.prepare_out, "robot_start": q_ready}))
        return 0

    import torch
    from live_robot import RobotLink
    torch.set_num_threads(a.torch_threads)
    # the link first: its trigger heartbeat (off) must reach the relay before the recorder page starts (btnSrc robot)
    link = RobotLink(P["hdr_robot"], ready, a.live_run, tap_port=a.tap_port, scene_map=a.scene_map, latency_dir=a.latency_dir)
    ws = None
    try:   # anything that fails before the run starts must still tell the arms and the relay (link.close)
        pol = pre = post = remote = None; n_state = 15
        if not a.demo_replay and a.policy_url:
            # the policy is on another machine (remote_policy.py, user 10-09: "Nebius 서버로의 연결 해"): the image, the two
            # tool poses and the phase text go out, absolute action rows come back and are turned into the chunk-relative
            # chunk the rest of this loop was written for. Its model card (/info) is checked before the arms are released.
            remote = RP.RemotePolicy(a.policy_url, RP.read_token(a.policy_token_file), a.policy_timeout_s, a.policy_retries, run=a.live_run or "",
                                     ca_file=a.policy_ca_file, image=a.policy_image)
            rinfo = remote.info()
            want_opt, got_opt = (P.get("optics") or "air"), (rinfo.get("image") or {}).get("optics")
            if (rinfo.get("image") or {}).get("size") != a.size or got_opt != want_opt:
                raise SystemExit(f"the policy server wants {rinfo.get('image')}, this run observes {a.size} px in {want_opt} (--size / --optics)")
            end_kind = remote.end_kind; end_level = None if end_kind is None else float(a.end_level or RO.END_LEVEL[end_kind])
            done_out = end_kind is not None; step_norm = None
            # the goal in words (user, 10-13): only a model trained with an instruction set that names it (lang1) takes one
            remote.goal = RP.INS.parse_goal(a.goal)
            if a.goal_text_file:     # the drilling sentence from a file, as written — one the model may never have met (10-16)
                own = Path(a.goal_text_file).read_text().strip()
                if not own or "\n" in own:
                    raise SystemExit(f"--goal-text-file {a.goal_text_file}: one sentence on one line")
                remote.goal = {**(remote.goal or {}), "drill_text": own}
            try:
                said = {p: remote.task(p) for p in (3, 4)}
            except ValueError as e:
                raise SystemExit(f"--goal {a.goal}: {e}")
            known = rinfo.get("tasks") or []
            if a.goal_text_file and said[4] not in known:
                log(f"the drilling sentence is not one the model was trained with (a test of the words' meaning): {said[4]!r}")
            if known and not {v for p, v in said.items() if not (a.goal_text_file and p == 4)} <= set(known):
                raise SystemExit(f"the loop would say {sorted(set(said.values()) - set(known))}, the server lists {known[:3]} … as its model's instructions")
            log(f"instructions {(rinfo.get('instructions') or {}).get('name') or 'phase1'}: approach {said[3]!r}, drilling {said[4]!r}")
            log(f"remote policy {rinfo.get('model')} at {a.policy_url}: {rinfo['action_horizon']} rows per answer, end output {end_kind}"
                + (f" (the drilling ends above {end_level:g})" if done_out else " (the harness's bounds end the drilling)"))
        elif not a.demo_replay:
            pol, pre, post = RO.load_policy(a.policy_path)
            rp = Path(a.policy_path) / "action_rep.json"
            if not (rp.exists() and json.loads(rp.read_text()).get("rep") == "chunk_relative"):
                raise SystemExit("live_loop runs chunk-relative policies (train/lerobot-train-rel.py: action_rep.json)")
            n_state = int(pol.config.input_features["observation.state"].shape[0])
            step_norm = RO.load_step_norm(a.policy_path)
            end_kind, end_level = RO.end_output(json.loads(rp.read_text()))   # the policy ends the drilling itself (14th value)
            if end_kind and a.end_level:
                end_level = float(a.end_level)
            done_out = end_kind is not None
        n_exec = int(a.n_action_steps or (pol.config.n_action_steps if pol else min(16, rinfo["action_horizon"]) if remote else 20))
        if remote is not None and n_exec > rinfo["action_horizon"]:
            raise SystemExit(f"--n-action-steps {n_exec} but the policy server answers with {rinfo['action_horizon']} rows")
        if pol is None and remote is None:
            done_out, end_kind, end_level = False, None, None
        dt_wall_ms = 1000.0 / (a.fps * a.speed)
        dt_feed = dt_wall_ms if a.session_dt == "wall" else dt_ms   # the viewer session's clock per fed frame
        page = bool(a.obs_url)   # the policy looks at the recorder page itself: no viewer session of its own
        img, m = None, {}
        if not page:
            # the viewer session opens before the arms are released: a busy renderer stops the run here, not mid-air
            t_w = time.time()
            while (busy := render_busy(a.api)):
                if time.time() - t_w > a.render_wait_s:
                    raise SystemExit(f"the renderer is still busy after {a.render_wait_s:.0f} s (jobs {busy})")
                await asyncio.sleep(5.0)
            ws, r = await RO.session(a.api, hdr, ready, a.size, flags, allow_outside=True, optics=P.get("optics"))
            img, m = await RO.step(ws, ready, dt_ms)
        if a.ready_flag:
            Path(a.ready_flag).write_text("ready")
        log(f"waiting for the arms of {a.live_run} (policy loaded, " + (f"observations from the recorder page {a.obs_url}" if page else "viewer session open")
            + f", {n_exec} steps per chunk, {dt_wall_ms:.0f} ms per step)")
        t_w = t_k = time.time()
        while not link.engaged():
            if time.time() - t_w > a.engage_timeout_s:
                raise SystemExit(f"the arms did not engage in {a.engage_timeout_s:.0f} s")
            if ws is not None and time.time() - t_k > 20.0:   # the session closes after 60 s without a message
                img, m = await RO.step(ws, ready, 1.0); t_k = time.time()
            await asyncio.sleep(0.25)
        await asyncio.sleep(1.5)
        vm, age = link.state_at(window=(time.time() * 1000.0 - 1000.0, time.time() * 1000.0))
        if age > 500:
            raise SystemExit(f"no tracker pose of the robot tools (nearest sample {age:.0f} ms old)")
        log(f"engaged — endo start residual c = {np.round(link.c['endo'] * 1000, 2).tolist()} mm ({a.scene_map}), "
            f"measured start vs ready: endo {np.linalg.norm(vm[0:3] - ready[0:3]):.2f} mm, drill {np.linalg.norm(vm[7:10] - ready[7:10]):.2f} mm")

        async def see(image: bool = True):
            """The recorder page now: (endoscope image | None, reply with meta.bridge) — page_obs.get, a second try after 1 s
            (RuntimeError when the page still gives nothing)."""
            try:
                return await asyncio.to_thread(PG.get, a.obs_url, a.size, image)
            except RuntimeError as e:
                log(f"observation failed, once more: {e}")
                await asyncio.sleep(1.0)
                return await asyncio.to_thread(PG.get, a.obs_url, a.size, image)

        def medium_fault(mm: dict, strict: bool = False) -> str | None:
            """The page must show the medium the policy's training frames were rendered in (field of view 92.9° in air, 67.4°
            in water). None = it does — or, unless `strict`, its camera has neither view angle (yet: before the endoscope view
            is up there may be no camera, or one without the lens model)."""
            want, got = P.get("optics") or "air", PG.medium(mm)
            va = (PG.bridge(mm).get("endoCamera") or {}).get("viewAngle")
            if got == want or (got is None and not strict):
                return None
            shows = f"the {got} medium (view angle {va:.1f}°)" if got else f"a view angle of {va}° (neither air nor water)"
            return (f"the recorder page shows {shows}, the policy was trained on {want} renders "
                    f"({PG.VIEW_ANGLE[want]:.1f}°) — a run with this model needs the step session (no --obs-url)")
        if page:   # the page is recording by now (the arms are released after its start)
            try:
                img, m = await see()
            except RuntimeError as e:
                raise SystemExit(str(e)) from e
            if (why := medium_fault(m)) is not None:
                raise SystemExit(why)
            log(f"recorder page: endoscope view {'on screen' if img is not None else 'not on screen yet'}, medium {PG.medium(m)}, "
                f"removed {float(PG.bridge(m).get('removedMm3') or 0.0):.1f} mm3, one observation {m.get('s')} s")

        pause_s = a.drill_pause_s if a.drill_pause_s is not None else (10.0 if n_state == 21 else 1.0)
        tracker = RO.PhaseTracker(P["portal_endo"], dt_ms / 1000.0, a.drill_target_mm3, a.drill_max_s, pause_s, a.drill_stall_s, a.drill_stall_mm3)
        fd = Path(a.frames_dir) if a.frames_dir else None
        if fd: fd.mkdir(parents=True, exist_ok=True)
        S = {"cmd": [], "meas": [], "kind": [], "phases": [], "vols": [], "vis": [], "dlog": [], "gate_log": [], "chunks": [], "done_log": [], "clamp_log": [],
             "t_inf": [], "t_step": [], "t_step_noimg": [], "t_gate": [], "n_gate": [], "fed": [], "obs": [], "windows": [], "deadline": False, "n_gated": 0, "n_blind_chunks": 0, "n_ph_clamped": 0, "n_clamped": 0, "n_aimed": 0, "n_rcm": 0, "rcm_deg": 0.0, "n_travel": 0, "abort": None}
        segs: list = []
        prev_cmd = ready.copy()
        lim = {"endo": a.travel_mm[0], "drill": a.travel_mm[1]}
        stall = {"endo": 0, "drill": 0}
    except BaseException:
        link.close()
        if ws is not None:
            await ws.close()
        raise
    rc = 0
    try:
        if not page:
            img, m = await RO.step(ws, vm, dt_feed)

        def note(vc: np.ndarray, v: np.ndarray, kind: str) -> int:
            """A measured frame (and the command it answers) into the run's record; returns its index."""
            S["cmd"].append(vc.tolist()); S["meas"].append(v.tolist()); S["kind"].append(kind)
            return len(S["meas"]) - 1

        async def feed(i: int, n_frames: int, want_img: bool) -> None:
            """Frame i of the record into the viewer session, standing for n_frames of policy time (the frames since the last
            one fed). The image is captured only for the observation (want_img): the screenshot is most of a step's cost.
            Drilling frames are all fed — every 5th frame of the first closed-loop run cut 122 mm³ where every frame cut
            278 (10-04): the measured tip jitters ~1 mm and the burr removes what it sweeps."""
            nonlocal img, m
            v = np.asarray(S["meas"][i], float)
            t0 = time.monotonic()
            im, m = await RO.step(ws, v, dt_feed * n_frames, images=want_img)
            if im is not None:
                img = im
            S["t_step" if want_img else "t_step_noimg"].append(round(time.monotonic() - t0, 3))
            dr = ((m.get("meta") or {}).get("bridge") or {}).get("drill") or {}
            px, iv = dr.get("proxy") or {}, dr.get("inView") or {}
            tracker.dt = dt_ms / 1000.0 * n_frames
            S["fed"].append(i)
            S["dlog"].append([px.get("gate"), None if px.get("penMm") is None else round(px["penMm"], 2), int(bool(iv.get("inFov"))),
                              int(bool(iv.get("visible"))), int(bool(px.get("cutting")))])
            S["phases"].append(tracker.update(m, v)); S["vols"].append(tracker.vol)
            S["vis"].append([int(bool(iv.get("inFov") and iv.get("visible"))), iv.get("x"), iv.get("y"), int(v[14] > 0.5), int(RO.drill_on_screen(m, P.get("optics")))])
            if fd and want_img:
                from PIL import Image
                Image.fromarray(img).save(fd / f"{i:05d}.png")

        async def look(i0: int, i1: int) -> bool:
            """--obs-url: the recorder page as it is now (the arms hold, trigger off) is the observation, and frames i0..i1 of
            the record — the ones the arms made since the last look — go through the phase rules. The page cut bone while
            the arms moved; how much is known only now, as a total: the frames with the trigger on share it (bone is removed
            only with the trigger on). Where the endoscope is (inside, its entry) is the page's at the hold for all of them;
            whether the drill was in view is known for the last frame only. False = the page gave nothing (S['abort'])."""
            nonlocal img, m
            t0 = time.monotonic()
            try:
                im, m = await see()
            except RuntimeError as e:
                S["abort"] = str(e)
                return False
            S["t_step"].append(round(time.monotonic() - t0, 3))
            S["obs"].append([i1, round(float(m.get("t_ms") or 0.0)), int(im is not None)])   # frame, wall ms of the look, image?
            if im is not None:
                img = im
            b = PG.bridge(m)
            vol = float(b.get("removedMm3") or 0.0)
            idx = list(range(i0, i1 + 1))
            cuts = PG.chunk_cuts([S["meas"][i][14] > 0.5 for i in idx], vol > tracker.vol + 1e-6)
            px, iv = (b.get("drill") or {}).get("proxy") or {}, (b.get("drill") or {}).get("inView") or {}
            for i, cut in zip(idx, cuts):
                v = np.asarray(S["meas"][i], float)
                tracker.dt = dt_ms / 1000.0
                S["fed"].append(i)
                S["phases"].append(tracker.update(PG.frame_meta(m, vol if cut else tracker.vol, cut), v)); S["vols"].append(tracker.vol)
                if i == i1:
                    S["dlog"].append([px.get("gate"), None if px.get("penMm") is None else round(px["penMm"], 2), int(bool(iv.get("inFov"))),
                                      int(bool(iv.get("visible"))), int(cut)])
                    S["vis"].append([int(bool(iv.get("inFov") and iv.get("visible"))), iv.get("x"), iv.get("y"), int(v[14] > 0.5), int(RO.drill_on_screen(m, P.get("optics")))])
                else:
                    S["dlog"].append([None, None, None, None, int(cut)]); S["vis"].append([None, None, None, int(v[14] > 0.5), None])
            if fd and im is not None:
                from PIL import Image
                Image.fromarray(img).save(fd / f"{i1:05d}.png")
            return True

        async def execute(targets: list, scripted: bool) -> bool:
            """Gate → send → the arms move → the measured frames into the viewer → settle. False = aborted (S['abort'])."""
            nonlocal prev_cmd, vm
            # the arms cannot follow: a follower left its replay (the run ends), or the tracker's frame stamps are stale and
            # the followers skip every frame — the arms hold their last goal, so wait and send the chunk afterwards
            t_f = t_k = time.time(); told = False
            while (why := link.fault()) is not None:
                if "follower stopped" in why or time.time() - t_f > a.fault_wait_s:
                    S["abort"] = why
                    return False
                if not told:
                    log(f"waiting: {why}"); told = True
                if ws is not None and time.time() - t_k > 20.0:   # the viewer session closes after 60 s without a message
                    vk = vm.copy(); vk[14] = 0.0
                    await RO.step(ws, vk, 1.0, images=False); t_k = time.time()
                await asyncio.sleep(0.5)
            if told:
                log(f"the tracker's stamps are back after {time.time() - t_f:.0f} s")
            cmds = []; t_g = time.monotonic(); v_before = vm.copy(); n_chk = 0
            blind = a.visible_trigger and not scripted and RO.trigger_blind(m, P.get("optics"), a.visible_trigger_rule)   # no drilling off screen (as the rollout)
            if blind and any(t[14] > 0.5 for t in targets):
                # the rollout judges this every step, here once per chunk: a chunk that wanted to drill is kept short
                targets = targets[:a.blind_frames]; S["n_blind_chunks"] += 1
            safe, j_safe = prev_cmd.copy(), -1   # the last frame the gate passed, and its index in this chunk
            turn_hold: dict = {}   # gate turn_wait: a tool whose turn was cut keeps that orientation for the rest of the chunk —
            for j, vv in enumerate(targets):   # the arm will not turn further before the next look, and every frame is not re-cut
                vv = np.array(vv, float)
                for off, qh in turn_hold.items():
                    vv[off:off + 4] = qh
                # the gate takes 0.2-0.4 s a frame (two IK solutions + the safe-zone model): a frame is checked when a tip
                # has moved --gate-mm (or the tool turned --gate-deg) since the last checked one, and always the last frame
                far = j == len(targets) - 1 or any(np.linalg.norm(vv[o:o + 3] - safe[o:o + 3]) >= (a.gate_mm_scripted if scripted else a.gate_mm)
                                                   for o in (0, 7)) or any(
                    2 * np.degrees(np.arccos(np.clip(abs(float(np.dot(vv[o:o + 4], safe[o:o + 4]))), 0, 1))) >= a.gate_deg for o in (3, 10))
                if far:
                    n_chk += 1
                    g, gi = gate.filter(link.arm_frame(safe), link.arm_frame(vv))
                    if ws is not None:   # a chunk of refused frames takes 30 s of IK: let the viewer session answer its pings
                        await asyncio.sleep(0)   # (10-07 dry run: "keepalive ping timeout" as soon as the gate began to hold)
                    if not gi["ok"] or gi["scale"] < 1.0:
                        S["gate_log"].append([len(S["meas"]) + j, gi["scale"], (gi["why"] or "")[:80]])
                        for c in cmds[j_safe + 1:]:   # the unchecked frames on the way to a refused one: stay at the last safe pose
                            c[:14] = safe[:14]
                    vv = link.scene_frame(g)
                    if gi.get("turn") is not None:   # the turn was cut (gate turn_wait): the unchecked frames on the way keep their
                        S["gate_log"].append([len(S["meas"]) + j, 1.0, (gi["why"] or "")[:80], f"turn {gi['turn']}"])   # tips, not their turn
                        for off in gi.get("turn_off") or (3, 10):
                            turn_hold[off] = vv[off:off + 4].copy()
                            for c in cmds[j_safe + 1:]:
                                c[off:off + 4] = vv[off:off + 4]
                    safe, j_safe = vv.copy(), j
                for tool, o in (("endo", 0), ("drill", 7)):   # the run's travel limit (guard_pairs.py stops the arms beyond it)
                    if np.linalg.norm(vv[o:o + 3] - ready[o:o + 3]) > lim[tool]:
                        vv[o:o + 7] = prev_cmd[o:o + 7]; vv[14] = 0.0; S["n_travel"] += 1
                if scripted or tracker.ph >= 5:
                    vv[14] = 0.0
                if blind and vv[14] > 0.5:
                    vv[14] = 0.0; S["n_gated"] += 1
                cmds.append(vv); prev_cmd = vv.copy()
            S["t_gate"].append(round((time.monotonic() - t_g) / max(1, n_chk), 3)); S["n_gate"].append(n_chk)
            t0 = time.time() * 1000.0 + a.lead_ms
            times = [t0 + (k + 1) * dt_wall_ms for k in range(len(cmds))]
            link.send_chunk(cmds, times)
            S["windows"].append([round(times[0] - dt_wall_ms - 500.0), round(times[-1] + dt_wall_ms + 1500.0)])   # the arms move: wall ms
            if a.windows_out:
                Path(a.windows_out).write_text(json.dumps(S["windows"]))
            link.trigger_off_at(times[-1] + dt_wall_ms)   # then the arms hold, trigger off, while the viewer is fed
            w = (times[-1] + dt_wall_ms) / 1000.0 + 0.1 - time.time()
            if w > 0:   # no render while the arms move and the drill may cut (the recorder page shares the GPU)
                await asyncio.sleep(w)
            stride = a.feed_stride_scripted if scripted else a.feed_stride
            last_fed, i_first = -1, len(S["meas"])
            for j, (vv, tk) in enumerate(zip(cmds, times)):
                v, age = link.state_at(tk, a.meas_half_ms)
                if age > a.lost_ms:
                    S["abort"] = f"robot tool not tracked at frame {len(S['meas'])} (nearest sample {age:.0f} ms away)"
                    return False
                v[14] = vv[14]
                i = note(vv, v, "scripted" if scripted else "policy")
                if not page and ((j + 1) % stride == 0 or j == len(cmds) - 1):
                    await feed(i, j - last_fed, False); last_fed = j
            # the settled pose = the median over the hold (the tip wanders ~1 mm around its target while the arm holds)
            t_h0 = times[-1] + dt_wall_ms + a.settle_s * 1000.0
            w = (t_h0 + a.settle_min_s * 1000.0) / 1000.0 - time.time()
            if w > 0:
                await asyncio.sleep(w)
            v, age = link.state_at(window=(t_h0, time.time() * 1000.0))
            hold_std = {t: [round(float(x), 2) for x in sd] for t, sd in link.last_std.items()}
            if age > a.lost_ms:
                S["abort"] = f"robot tool not tracked after the chunk (nearest sample {age:.0f} ms old)"
                return False
            dev = [float(np.linalg.norm(v[o:o + 3] - cmds[-1][o:o + 3])) for o in (0, 7)]
            odev = [float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(v[o:o + 4] / np.linalg.norm(v[o:o + 4]),
                                                                   cmds[-1][o:o + 4] / np.linalg.norm(cmds[-1][o:o + 4]))), 0, 1)))) for o in (3, 10)]
            v[14] = cmds[-1][14]
            if page:
                if not await look(i_first, note(cmds[-1], v, "settle")):
                    return False
            else:
                await feed(note(cmds[-1], v, "settle"), 1, True)
            vm = v.copy()   # the next observation: measured poses, the trigger as last commanded (policy time has no hold)
            S["chunks"].append({"frames": len(cmds), "scripted": scripted, "dev_mm": [round(d, 2) for d in dev], "dev_deg": [round(d, 1) for d in odev], "hold_std_mm": hold_std, "phase": tracker.ph,
                                "removed_mm3": round(tracker.vol, 1), "trig": int(sum(c[14] > 0.5 for c in cmds)), "gate_checks": n_chk,
                                "t_s": round(time.time() - link.t_engage, 1)})
            log(f"chunk {len(S['chunks'])} ({'scripted' if scripted else 'policy'}, {len(cmds)} frames, {time.time() - link.t_engage:.0f} s): phase {tracker.ph}, "
                f"trigger {S['chunks'][-1]['trig']}/{len(cmds)}, removed {tracker.vol:.0f} mm3, settled {dev[0]:.1f}/{dev[1]:.1f} mm "
                f"{odev[0]:.1f}/{odev[1]:.1f}° from the command (endo/drill), gate shrunk {gate.n_shrunk} held {gate.n_held}")
            if max(dev) > a.abort_dev_mm:
                S["abort"] = f"the arms are {dev[0]:.1f}/{dev[1]:.1f} mm from the command after settling (limit {a.abort_dev_mm:g})"
                return False
            for tool, o in (("endo", 0), ("drill", 7)):   # an arm that stopped taking goals (server fault, lost lease) stays where it is
                D = float(np.linalg.norm(cmds[-1][o:o + 3] - v_before[o:o + 3])); Mv = float(np.linalg.norm(v[o:o + 3] - v_before[o:o + 3]))
                stall[tool] = stall[tool] + 1 if (D > 3.0 and Mv < 0.3 * D) else 0
                if stall[tool] >= 2:
                    S["abort"] = f"the {tool} arm is not following (commanded {D:.1f} mm, moved {Mv:.1f} mm, two chunks in a row)"
                    return False
            if (why := link.fault()) is not None and "follower stopped" in why:   # also for commands under 3 mm
                S["abort"] = why
                return False
            return True

        async def scripted_path(frames: list, name: str) -> bool:
            segs.append([name, len(S["meas"])])
            for i in range(0, len(frames), a.scripted_chunk):
                if not await execute(frames[i:i + a.scripted_chunk], True):
                    return False
            return True

        # ------------------------------------------------------------------ the policy part without stops (--continuous)
        async def gate_window(targets: list) -> tuple[list, int]:
            """The safety gate and the run's rules over a window's targets: execute()'s first part for policy frames, kept
            apart from it until the continuous loop has run on the robot (execute() is the path every run so far took).
            The gate itself runs in a thread: the looks and the answers go on meanwhile."""
            nonlocal prev_cmd
            cmds, n_chk = [], 0
            blind = a.visible_trigger and RO.trigger_blind(m, P.get("optics"), a.visible_trigger_rule)
            if blind and any(t[14] > 0.5 for t in targets):
                S["n_blind_chunks"] += 1      # not shortened as a chunk is: the next look comes a period later anyway
            safe, j_safe = prev_cmd.copy(), -1
            turn_hold: dict = {}
            for j, vv in enumerate(targets):
                vv = np.array(vv, float)
                for off, qh in turn_hold.items():
                    vv[off:off + 4] = qh
                far = j == len(targets) - 1 or any(np.linalg.norm(vv[o:o + 3] - safe[o:o + 3]) >= a.gate_mm for o in (0, 7)) or any(
                    2 * np.degrees(np.arccos(np.clip(abs(float(np.dot(vv[o:o + 4], safe[o:o + 4]))), 0, 1))) >= a.gate_deg for o in (3, 10))
                if far:
                    n_chk += 1
                    g, gi = await asyncio.to_thread(gate.filter, link.arm_frame(safe), link.arm_frame(vv))
                    if not gi["ok"] or gi["scale"] < 1.0:
                        S["gate_log"].append([len(S["meas"]) + j, gi["scale"], (gi["why"] or "")[:80]])
                        for c in cmds[j_safe + 1:]:
                            c[:14] = safe[:14]
                    vv = link.scene_frame(g)
                    if gi.get("turn") is not None:
                        S["gate_log"].append([len(S["meas"]) + j, 1.0, (gi["why"] or "")[:80], f"turn {gi['turn']}"])
                        for off in gi.get("turn_off") or (3, 10):
                            turn_hold[off] = vv[off:off + 4].copy()
                            for c in cmds[j_safe + 1:]:
                                c[off:off + 4] = vv[off:off + 4]
                    safe, j_safe = vv.copy(), j
                for tool, o in (("endo", 0), ("drill", 7)):
                    if np.linalg.norm(vv[o:o + 3] - ready[o:o + 3]) > lim[tool]:
                        vv[o:o + 7] = prev_cmd[o:o + 7]; vv[14] = 0.0; S["n_travel"] += 1
                if tracker.ph >= 5:
                    vv[14] = 0.0
                if blind and vv[14] > 0.5:
                    vv[14] = 0.0; S["n_gated"] += 1
                cmds.append(vv); prev_cmd = vv.copy()
            return cmds, n_chk

        async def stream_policy() -> bool:
            """A look every P frames while the arms move; each answer's rows L−1 … L+P−2 are gated and sent as the window L
            frames after its look (stream_plan.py, CONTINUOUS.md). Windows follow each other without a gap; when an answer
            comes late the timetable waits for it (the arms hold at their last target, no frame is lost). False = a fault of the arms or
            the tracker (no scripted exit, as execute()); a server that stops answering sets S['abort'] and returns True."""
            nonlocal prev_cmd, vm, n_pol, img, m
            H = int(rinfo["action_horizon"]) if remote is not None else int(pol.config.chunk_size)
            tm = SP.Timing(a.stream_period, a.stream_lead, a.stream_blend, H)
            Pn = tm.period
            q: asyncio.Queue = asyncio.Queue()
            # the frame clock (frame 0 = the first look). It is pushed back when an answer comes late: the arms then wait at
            # their last target for exactly that long and no frame is lost — sent targets keep the times they were sent with
            st = {"clock": SP.Clock(time.time() * 1000.0 + 300.0, dt_wall_ms), "t_of": {}, "sent": {}, "last": None, "last_cmd": None, "trig": [],
                  "stop": None, "fatal": False, "late": 0, "ended": None, "wins": {}, "looks": [0, 0], "t_ready": [],
                  "cmds": [(0.0, prev_cmd.copy())], "started": False, "off": []}
            by_cmd = a.stream_anchor == "command"
            log(f"continuous: a look every {Pn} frames ({Pn * dt_wall_ms / 1000:.1f} s), windows {tm.lead} frames ({tm.lead * dt_wall_ms / 1000:.1f} s) "
                f"after their look, {H} rows an answer; " + (f"plans anchored at the command, hand-over gamma {a.stream_gamma:g}, the first answer from its first row"
                                                             if by_cmd else f"plans anchored at the measured pose, {tm.blend} frames of fade"))

            def cmd_at(t_ms: float) -> np.ndarray:
                """The command the arms had at wall time t_ms: between two sent targets the pose on the way (the followers
                interpolate), before the first one the last target of the entry, after the last one that target."""
                cs = st["cmds"]
                k = max((j for j in range(len(cs)) if cs[j][0] <= t_ms), default=0)
                if k + 1 < len(cs) and cs[k + 1][0] - cs[k][0] <= 2.5 * dt_wall_ms and cs[k][0] > 0.0:
                    return SP.mix(cs[k][1], cs[k + 1][1], (t_ms - cs[k][0]) / (cs[k + 1][0] - cs[k][0]))
                return cs[k][1].copy()

            def win_of(n: int):
                return next((w for w in st["wins"].values() if w["a"] <= n <= w["b"]), None)

            def trigger_off_now() -> None:
                """The targets already sent cannot be taken back (the followers append): the arms run them out, with the
                trigger off from now on."""
                now = time.time() * 1000.0
                with link._lock:
                    link.sched = [(t, v_ and t <= now) for t, v_ in link.sched] + [(now, False)]

            async def note_passed(upto_ms: float, held: np.ndarray | None = None):
                """The sent frames whose time has passed → the record, with the pose measured at their time. (first index,
                last index) of the new record frames (last < first: none), or None = a tool was not tracked.
                held: the pose measured after a hold (the tracker's stamps were stale: the arms stood while these frames
                were due) — a frame without a sample near its time gets it, and the windows are not judged for following."""
                i_first = len(S["meas"])
                for n in sorted(k for k in st["sent"] if st["t_of"][k] + a.meas_half_ms <= upto_ms):
                    v, age = link.state_at(st["t_of"].pop(n), a.meas_half_ms)
                    if age > a.lost_ms:
                        if held is None:
                            S["abort"] = f"robot tool not tracked at frame {len(S['meas'])} (nearest sample {age:.0f} ms away)"
                            return None
                        v = held.copy()
                    c = st["sent"].pop(n); v[14] = c[14]
                    note(c, v, "policy")
                    w = win_of(n)
                    if w is not None:
                        w["dev"].append([float(np.linalg.norm(v[o:o + 3] - c[o:o + 3])) for o in (0, 7)]); w["meas"].append(v.copy())
                        w["odev"].append([float(np.degrees(2 * np.arccos(np.clip(abs(np.dot(v[o:o + 4] / np.linalg.norm(v[o:o + 4]),
                                                                                             c[o:o + 4] / np.linalg.norm(c[o:o + 4]))), 0, 1)))) for o in (3, 10)])
                        if n == w["b"]:        # the window is through: its line in the record, and the two checks execute() makes
                            dev = np.median(np.asarray(w["dev"]), axis=0)
                            odev = np.median(np.asarray(w["odev"]), axis=0)
                            S["chunks"].append({"frames": len(w["dev"]), "scripted": False, "dev_mm": [round(float(d), 2) for d in dev], "dev_deg": [round(float(d), 1) for d in odev],
                                                "hold_std_mm": {}, "phase": tracker.ph, "removed_mm3": round(tracker.vol, 1), "trig": w["trig"],
                                                "gate_checks": w["n_chk"], "late": w["late"], "t_s": round(time.time() - link.t_engage, 1)})
                            if held is not None:
                                continue
                            if max(dev) > a.abort_dev_mm:
                                S["abort"] = f"the arms are {dev[0]:.1f}/{dev[1]:.1f} mm from the command over a window (limit {a.abort_dev_mm:g})"
                                return None
                            for tool, o in (("endo", 0), ("drill", 7)):
                                D = float(np.linalg.norm(w["cmd1"][o:o + 3] - w["cmd0"][o:o + 3])); Mv = float(np.linalg.norm(w["meas"][-1][o:o + 3] - w["meas"][0][o:o + 3]))
                                stall[tool] = stall[tool] + 1 if (D > 3.0 and Mv < 0.3 * D) else 0
                                if stall[tool] >= 2:
                                    S["abort"] = f"the {tool} arm is not following (commanded {D:.1f} mm, moved {Mv:.1f} mm, two windows in a row)"
                                    return None
                return i_first, len(S["meas"]) - 1

            async def look_moving(i0: int, i1: int) -> bool:
                """look() while the arms move: the image from the page's screencast (no screenshot), the frames i0..i1 of the
                record through the phase rules as look() does (the bookkeeping is look()'s, copied — see gate_window)."""
                nonlocal img, m
                t0 = time.monotonic()
                try:
                    im, m = await asyncio.to_thread(PG.get, a.obs_url, a.size, True, 20.0, True)
                except RuntimeError as e:
                    S["abort"] = str(e)
                    return False
                S["t_step"].append(round(time.monotonic() - t0, 3)); st["looks"][0 if m.get("stream") else 1] += 1
                if im is not None:
                    img = im
                if i1 < i0:
                    return True
                S["obs"].append([i1, round(float(m.get("t_ms") or 0.0)), int(im is not None)])
                b = PG.bridge(m)
                vol = float(b.get("removedMm3") or 0.0)
                idx = list(range(i0, i1 + 1))
                cuts = PG.chunk_cuts([S["meas"][i][14] > 0.5 for i in idx], vol > tracker.vol + 1e-6)
                px, iv = (b.get("drill") or {}).get("proxy") or {}, (b.get("drill") or {}).get("inView") or {}
                for i, cut in zip(idx, cuts):
                    v = np.asarray(S["meas"][i], float)
                    tracker.dt = dt_ms / 1000.0
                    S["fed"].append(i)
                    S["phases"].append(tracker.update(PG.frame_meta(m, vol if cut else tracker.vol, cut), v)); S["vols"].append(tracker.vol)
                    if i == i1:
                        S["dlog"].append([px.get("gate"), None if px.get("penMm") is None else round(px["penMm"], 2), int(bool(iv.get("inFov"))),
                                          int(bool(iv.get("visible"))), int(cut)])
                        S["vis"].append([int(bool(iv.get("inFov") and iv.get("visible"))), iv.get("x"), iv.get("y"), int(v[14] > 0.5), int(RO.drill_on_screen(m, P.get("optics")))])
                    else:
                        S["dlog"].append([None, None, None, None, int(cut)]); S["vis"].append([None, None, None, int(v[14] > 0.5), None])
                if fd and im is not None:
                    from PIL import Image
                    Image.fromarray(img).save(fd / f"{i1:05d}.png")
                return True

            async def answer(im: np.ndarray, v: np.ndarray, ph: int) -> np.ndarray:
                """Absolute rows (H, 15 | 16) for the observation — the server's own, or a local chunk-relative chunk undone."""
                t0 = time.monotonic()
                if remote is not None:
                    rows = np.asarray(await asyncio.to_thread(remote.act, im, v, remote.task(ph)), float)
                    RP.abs_to_chunk(v, rows)          # the loop's own check of an answer (finite, quaternions): raises RemoteError
                else:
                    ph_in = min(4 if done_out else 5, max(3, ph))
                    sv = v if n_state == 15 else np.concatenate([v, np.eye(6)[ph_in - 1]])
                    obs = {"observation.state": torch.tensor(sv, dtype=torch.float32)[None],
                           "observation.images.endo": torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0,
                           "task": [RO.PHASES[(ph_in if a.phase_rules else 1) - 1]]}

                    def infer():
                        with torch.no_grad():
                            return RO.predict_chunk(pol, pre, post, obs, step_norm)
                    rows = RP.chunk_to_abs(v, np.asarray(await asyncio.to_thread(infer), float))
                S["t_inf"].append(round(time.monotonic() - t0, 3))
                return rows

            async def observer() -> None:
                i = 0
                while st["stop"] is None:
                    n_o = tm.obs_frame(i)
                    while by_cmd and i >= 1 and st["stop"] is None and not st["started"]:
                        await asyncio.sleep(0.02)              # the first answer is executed from its first row: looks while moving begin after it
                    while st["stop"] is None and (w := st["clock"].t(n_o) / 1000.0 - time.time()) > 0:
                        await asyncio.sleep(min(w, 0.05))      # in short steps: the clock may be pushed back meanwhile
                    if st["stop"] is not None:
                        break
                    if (why := link.fault()) is not None and "follower stopped" in why:
                        S["abort"] = why; st["stop"], st["fatal"] = "abort", True; trigger_off_now()
                        break
                    if why is not None:
                        # the tracker's frame stamps are stale: the followers skip every frame and the arms hold whatever is
                        # sent. A stream cannot wait in the middle of its timetable — it ends here, and when the stamps are
                        # back the run goes on stop-and-go (execute() waits for them before every chunk)
                        st["stop"] = "stale"; trigger_off_now()
                        log(f"continuous: {why} — the stream ends here; stop-and-go from where the arms stand once the stamps are back")
                        break
                    t_look = time.time() * 1000.0
                    if i == 0:
                        im_i, v = img, vm.copy()      # the entry's last look: the arms have stood since
                        anc = SP.anchor_pose(v, prev_cmd)
                    else:
                        r = await note_passed(t_look)
                        if r is None or not await look_moving(*r):
                            st["stop"], st["fatal"] = "abort", r is None; trigger_off_now()
                            break
                        t_img = float(m.get("t_ms") or t_look) if m.get("stream") else t_look
                        v, age = link.state_at(t_img - a.page_lag_ms, a.obs_half_ms)
                        if age > a.lost_ms:
                            S["abort"] = f"robot tool not tracked at a look (nearest sample {age:.0f} ms away)"; st["stop"], st["fatal"] = "abort", True; trigger_off_now()
                            break
                        v[14] = float(next((tg for t_, tg in reversed(st["trig"]) if t_ <= t_look), False))   # the trigger as commanded now
                        im_i = img
                        anc = SP.anchor_pose(v, cmd_at(t_img - a.page_lag_ms))   # the tips of the command of the moment the state was measured at
                    st["off"].append([float(np.linalg.norm(v[o:o + 3] - anc[o:o + 3])) for o in (0, 7)])
                    if tracker.done_drilling:          # a rule of the harness ended the drilling (pause, stall, the 240 s bound)
                        st["stop"] = "rule"; trigger_off_now()
                        break
                    if n_pol >= a.steps:
                        st["stop"] = "steps"
                        break
                    if a.deadline_s and time.time() - link.t_engage > a.deadline_s - a.exit_reserve_s:
                        S["deadline"] = True; st["stop"] = "deadline"; trigger_off_now()
                        log(f"deadline: {time.time() - link.t_engage:.0f} s since engage — the drilling ends here, exit")
                        break
                    S["n_ph_clamped"] += min(4 if done_out else 5, max(3, tracker.ph)) != tracker.ph
                    await q.put((i, t_look, asyncio.create_task(answer(im_i, v, tracker.ph)), v.copy(), anc))
                    i += 1
                await q.put(None)

            async def committer() -> None:
                nonlocal prev_cmd, n_pol
                prev_rows = None
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    i, t_look, task, v_fed, anc = item
                    try:
                        rows = await task
                    except RP.RemoteError as e:
                        # no answer is a fault of the run, not a reason to guess (as the stop-and-go loop): nothing more is
                        # sent, the arms hold after their last target, the scripted exit takes the tools out
                        if st["stop"] is None:
                            S["abort"] = f"the policy server gave no usable answer ({e})"; log(S["abort"]); st["stop"] = "abort"; trigger_off_now()
                        continue
                    if st["stop"] is not None:
                        continue
                    a_fr, _ = tm.window(i); r0 = tm.rows()[0]
                    if not by_cmd:        # the run of 10-12: the rows as answered (anchored at the measured pose), faded in
                        want = SP.desired(rows, prev_rows, tm); prev_rows = rows
                    else:
                        bel = SP.reanchor(rows, v_fed, anc)      # the plan on the command timeline
                        if a.rcm and a.stream_rcm_first:     # its rows on the pivot before the stitch and the limit (10-16): the
                            # last sent target is on it, so the stitch and the limit then compare like with like — only motion
                            bel = np.stack([RO.rcm_project(r, portals, shafts, a.rcm_near_mm) for r in bel])
                        if i == 0:        # asked standing: executed from its first row, up to where the second answer's window begins
                            r0, r1 = SP.first_rows(tm); a_fr = 1
                            want = [bel[k].copy() for k in range(r0, min(r1, len(bel)))]
                        else:
                            want = SP.stitch(bel, anc, prev_cmd, tm, a.stream_gamma)
                    j_done = None
                    if done_out:      # the end rule on the rows this window would execute (the hold may reach beyond them)
                        j_done = RO.done_step(rows[r0:, 15], len(want), level=end_level)
                        S["done_log"].append([len(S["meas"]) + len(st["sent"]), [round(float(x), 2) for x in rows[r0:r0 + len(want), 15]], j_done])
                        if j_done is not None:
                            want = want[:j_done]
                    targets, vc = [], prev_cmd.copy()
                    for kw, wv in enumerate(want):
                        if len(S["clamp_log"]) < 2000:
                            asked = step_size(vc, np.asarray(wv, float)[:15])    # which tool, which limit, where in the window (10-13)
                        vc, hit = limit_step(vc, np.asarray(wv, float)[:15], a.max_step_mm, a.max_step_deg)
                        S["n_clamped"] += hit
                        if hit and len(S["clamp_log"]) < 2000:
                            S["clamp_log"].append([len(S["meas"]) + len(st["sent"]), kw, len(want), *asked])
                        if a.rcm:
                            for tool, off in (("endo", 0), ("drill", 7)):
                                vc, dg = RO.rcm(vc, off, portals[tool], shafts[tool], a.rcm_near_mm, a.rcm_max_deg)
                                S["n_rcm"] += dg > 0; S["rcm_deg"] += float(dg)
                        targets.append(vc.copy())
                    t_g = time.monotonic()
                    cmd0 = prev_cmd.copy()
                    cmds, n_chk = (await gate_window(targets)) if targets else ([], 0)
                    S["t_gate"].append(round((time.monotonic() - t_g) / max(1, n_chk), 3)); S["n_gate"].append(n_chk)
                    now = time.time() * 1000.0
                    st["t_ready"].append(round((now - t_look) / 1000.0, 3))       # look → targets ready: what the lead must cover
                    k = SP.late(st["clock"], a_fr, 10 ** 6, now, a.send_lead_ms) if cmds else 0
                    if k:      # late: the timetable waits — every frame from this window on is due k frames later
                        st["clock"] = SP.Clock(st["clock"].t0_ms + k * dt_wall_ms, dt_wall_ms)
                        if by_cmd and i == 0:      # the first answer is awaited standing, as a stop-and-go chunk: not a wait of the stream
                            k = 0
                        else:
                            st["late"] += k
                            log(f"window {i}: ready {st['t_ready'][-1]:.2f} s after its look — the arms wait {k * dt_wall_ms / 1000:.1f} s at their last target")
                    send = cmds
                    if send and st["stop"] is None:
                        times = [st["clock"].t(a_fr + j) for j in range(len(cmds))]
                        link.send_chunk(send, times)
                        link.trigger_off_at(times[-1] + dt_wall_ms)      # unless the next window goes on from there
                        for c, n, t_n in zip(send, range(a_fr, a_fr + len(cmds)), times):
                            st["sent"][n] = c; st["t_of"][n] = t_n
                        st["last"] = a_fr + len(cmds) - 1; st["last_cmd"] = send[-1].copy(); n_pol += len(send)
                        st["cmds"] = (st["cmds"] + [(t_n, c.copy()) for c, t_n in zip(send, times)])[-600:]
                        # the trigger on the wall clock, as link.sched has it: each frame's value, off after the window's last frame
                        st["trig"] = (st["trig"] + [(t_n, bool(c[14] > 0.5)) for c, t_n in zip(send, times)] + [(times[-1] + dt_wall_ms, False)])[-400:]
                        st["wins"][i] = {"a": a_fr, "b": st["last"], "n_chk": n_chk, "late": k, "trig": int(sum(c[14] > 0.5 for c in send)),
                                         "dev": [], "odev": [], "meas": [], "cmd0": cmd0, "cmd1": send[-1].copy()}
                        if S["windows"] and S["windows"][-1][1] >= times[0] - dt_wall_ms - 500.0:
                            S["windows"][-1][1] = round(times[-1] + dt_wall_ms + 1500.0)      # one stretch of motion, growing
                        else:
                            S["windows"].append([round(times[0] - dt_wall_ms - 500.0), round(times[-1] + dt_wall_ms + 1500.0)])
                        if a.windows_out:
                            Path(a.windows_out).write_text(json.dumps(S["windows"]))
                    st["started"] = True       # the first answer is through (sent, or nothing to send): the looks while moving may begin
                    if j_done is not None and st["stop"] is None:
                        st["ended"] = {"chunk_step": int(j_done), "phase_before": int(min(4, max(3, tracker.ph)))}
                        st["stop"] = "policy"

            await asyncio.gather(observer(), committer())
            stale = st["stop"] == "stale"
            if stale:      # as execute() before a chunk: wait for the stamps (they came back by themselves after 265 s on 10-05)
                t_f = time.time()
                while (why := link.fault()) is not None:
                    if "follower stopped" in why or time.time() - t_f > a.fault_wait_s:
                        S["abort"] = why; st["fatal"] = True
                        break
                    await asyncio.sleep(0.5)
                if not st["fatal"]:
                    log(f"the tracker's stamps are back after {time.time() - t_f:.0f} s")
            # the arms run out what they were sent, then hold: settle and look once, as after a chunk
            if st["last"] is not None and not st["fatal"]:
                t_end = st["clock"].t(st["last"]) + dt_wall_ms
                if stale:
                    t_end = max(t_end, time.time() * 1000.0)      # the arms set out for their last target only now
                w = t_end / 1000.0 + 0.1 - time.time()
                if w > 0:
                    await asyncio.sleep(w)
                held = None
                if stale:      # the frames of the hold have no sample near their time: they get the pose measured after it
                    await asyncio.sleep(a.settle_s + a.settle_min_s)
                    held, age = link.state_at(window=(t_end + a.settle_s * 1000.0, time.time() * 1000.0))
                    if age > a.lost_ms:
                        S["abort"] = f"robot tool not tracked after the stamps came back (nearest sample {age:.0f} ms old)"; st["fatal"] = True
                r = None if st["fatal"] else await note_passed(time.time() * 1000.0 + 1e9, held=held)
                if r is None:
                    st["fatal"] = True
                else:
                    t_h0 = t_end + a.settle_s * 1000.0
                    w = (t_h0 + a.settle_min_s * 1000.0) / 1000.0 - time.time()
                    if w > 0:
                        await asyncio.sleep(w)
                    v, age = link.state_at(window=(t_h0, time.time() * 1000.0))
                    if age > a.lost_ms:
                        S["abort"] = f"robot tool not tracked after the last window (nearest sample {age:.0f} ms old)"; st["fatal"] = True
                    else:
                        v[14] = 0.0
                        keep = S["abort"]          # look() only sets it when the page gives nothing
                        if not await look(r[0], note(st["last_cmd"], v, "settle")):
                            st["fatal"] = True
                        S["abort"] = keep or S["abort"]
                        vm = v.copy()
            tr = st["t_ready"]
            off = np.asarray(st["off"], float).reshape(-1, 2)
            S["continuous"] = {"period": tm.period, "lead": tm.lead, "blend": tm.blend, "rows": H, "windows": len(st["wins"]),
                               "anchor": a.stream_anchor, "gamma": a.stream_gamma if by_cmd else None,
                               "rcm_first": bool(a.rcm and a.stream_rcm_first),
                               "measured_from_command_mm": {"median": [round(float(x), 2) for x in np.median(off, axis=0)],
                                                            "p90": [round(float(x), 2) for x in np.percentile(off, 90, axis=0)]} if len(off) else None,
                               "waited_frames": st["late"], "waited_s": round(st["late"] * dt_wall_ms / 1000.0, 1),
                               "looks_stream": st["looks"][0], "looks_screenshot": st["looks"][1], "stopped_by": st["stop"],
                               "look_to_ready_s": {"median": round(float(np.median(tr)), 3), "max": round(float(np.max(tr)), 3)} if tr else None,
                               "budget_s": round(tm.lead * dt_wall_ms / 1000.0 - a.send_lead_ms / 1000.0, 3)}
            log("continuous:", json.dumps(S["continuous"]))
            if st["ended"] is not None and not st["fatal"] and st["stop"] == "policy":
                took = tracker.policy_done()
                S["policy_done"] = {"frame": len(S["meas"]) - 1, "chunk_step": st["ended"]["chunk_step"], "ended_the_drilling": bool(took), "phase_before": st["ended"]["phase_before"]}
                log(f"the policy says the drilling is over (step {st['ended']['chunk_step']} of a window, {tracker.vol:.0f} mm3 removed, {tracker.drill_s:.0f} s of drilling)"
                    + ("" if took else f" — a rule had already ended it ({tracker.end})"))
            return not st["fatal"]

        ok = await scripted_path([f.copy() for f in cyc["entry"]], "entry")
        segs.append(["policy", len(S["meas"])])
        n_pol = 0
        if ok and page and not a.demo_replay and a.stop_after != "entry":
            # the tools are inside now: without an image, or with one in another medium than the training frames', the policy
            # is not asked — the tools leave the way they came (the scripted exit below) and the run counts as aborted
            S["abort"] = "the recorder page shows no endoscope view after the entry" if img is None else medium_fault(m, strict=True)
            if S["abort"]:
                log("no policy steps:", S["abort"])
        if ok and a.demo_replay:   # the demo's own in-body frames through the same stream (checks the link, no policy)
            ph5 = (meta.get("phases") or {}).get("5")
            k_end = min(len(states) - 1, int(round(ph5[0] * a.fps))) if ph5 else len(states) - 1
            k_end = min(k_end, k0 + a.steps)
            fr = [states[k].copy() for k in range(k0 + 1, k_end + 1)]
            for i in range(0, len(fr), n_exec):
                if not await execute(fr[i:i + n_exec], False):
                    ok = False; break
        if ok and not S["abort"] and a.continuous and not a.demo_replay and a.stop_after != "entry":
            ok = await stream_policy()
        fell_back = bool(a.continuous and (S.get("continuous") or {}).get("stopped_by") == "stale")   # stale stamps ended the stream
        while ok and not S["abort"] and (not a.continuous or fell_back) and not a.demo_replay and n_pol < a.steps and not (tracker.done_drilling or a.stop_after == "entry"):
            if a.deadline_s and time.time() - link.t_engage > a.deadline_s - a.exit_reserve_s:
                # the FE recorder closes a session at 600 s (max_duration): leave in time for the exit to be on the record
                S["deadline"] = True
                log(f"deadline: {time.time() - link.t_engage:.0f} s since engage — the drilling ends here, exit")
                break
            v = vm.copy()
            # the trimmed training data has phases 3-5 only: the other one-hot bits have std 0 and a set bit would be
            # normalised to ~1e8 (review 10-04) — the policy is asked only inside the skin, so clamp
            ph_in = min(4 if done_out else 5, max(3, tracker.ph))   # a done-output policy was never told "the drilling is over"
            S["n_ph_clamped"] += ph_in != tracker.ph
            sv = v if n_state == 15 else np.concatenate([v, np.eye(6)[ph_in - 1]])
            ph = ph_in if a.phase_rules else 1
            obs = {"observation.state": torch.tensor(sv, dtype=torch.float32)[None],
                   "observation.images.endo": torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)[None] / 255.0,
                   "task": [RO.PHASES[ph - 1]]}
            t0 = time.monotonic()
            if remote is not None:
                try:   # in a thread: the tap and the trigger heartbeat have their own, this keeps the event loop free as well
                    ch = await asyncio.to_thread(remote.chunk, img, v, remote.task(tracker.ph))
                except RP.RemoteError as e:
                    # no answer is a fault of the run, not a reason to guess: nothing more is asked, the tools are
                    # withdrawn by the scripted exit below (ok stays True) and the run counts as aborted
                    S["abort"] = f"the policy server gave no usable answer ({e})"
                    log(S["abort"])
                    break
            else:
                with torch.no_grad():
                    ch = RO.predict_chunk(pol, pre, post, obs, step_norm)
            S["t_inf"].append(round(time.monotonic() - t0, 3))
            n_run = n_exec
            j_done = RO.done_step(ch[:, 13], n_exec, level=end_level) if done_out else None
            if done_out:   # what the policy said about the end, chunk by chunk: frame of the observation, the executed steps' values
                S["done_log"].append([len(S["meas"]), [round(float(x), 2) for x in ch[:n_exec, 13]], j_done])
            if j_done is not None:   # the steps before the policy's "done" are executed, then its part is over
                n_run = j_done
            targets, vc, aimed = [], v.copy(), 0.0
            for j, d in enumerate(AC.cum_to_steps(ch[:n_run, :13])):
                vc, hit = limit_step(vc, AC.from_delta(vc, d), a.max_step_mm, a.max_step_deg)
                S["n_clamped"] += hit
                if a.rcm:
                    for tool, off in (("endo", 0), ("drill", 7)):
                        vc, dg = RO.rcm(vc, off, portals[tool], shafts[tool], a.rcm_near_mm, a.rcm_max_deg)
                        S["n_rcm"] += dg > 0; S["rcm_deg"] += float(dg)   # how often and how much the pivot rule turned a tool
                if a.aim and tracker.ph >= 3 and j < a.aim_frames:
                    # the viewer's camera/drill geometry is known only at the observation: the rollout's per-step correction
                    # (gain 0.3 of the angle beyond the dead zone, <= aim_max_deg a step) converges on that angle within a
                    # few steps — here 0.9 of it is spread over the first frames, <= aim_max_deg each
                    vc, deg = RO.aim_endo(vc, m, 0.9, a.aim_max_deg, a.aim_dead_deg + aimed / 0.9, pivot=portals["endo"] if a.rcm else None)
                    aimed += deg; S["n_aimed"] += deg > 0
                targets.append(vc.copy())
            n_pol += len(targets)
            if targets:
                ok = await execute(targets, False)
            if j_done is not None and ok and not S["abort"]:
                took = tracker.policy_done()
                S["policy_done"] = {"frame": len(S["meas"]) - 1, "chunk_step": int(j_done), "ended_the_drilling": bool(took), "phase_before": int(ph_in)}
                log(f"the policy says the drilling is over (step {j_done} of the chunk, {tracker.vol:.0f} mm3 removed, {tracker.drill_s:.0f} s of drilling)"
                    + ("" if took else f" — a rule had already ended it ({tracker.end})"))
                if not took:
                    break
        if ok:
            # scripted exit from where the tools are: straight back along each shaft to outside the skin, then the ready pose
            v_from = vm.copy()
            v_out = RO.retract(v_from, portals, shafts, a.ready_out_mm, own_axis=True)
            n1 = max(2, int(np.ceil(max(np.linalg.norm(v_out[o:o + 3] - v_from[o:o + 3]) for o in (0, 7)) / (a.entry_speed_mms / a.fps))))
            fr = []
            for j in range(1, n1 + 1):
                f = v_from + (v_out - v_from) * (j / n1); f[3:7] = v_from[3:7]; f[10:14] = v_from[10:14]; f[14] = 0.0
                fr.append(f)
            ok = await scripted_path(fr, "exit")
            if ok:
                v_from = vm.copy()
                n2 = max(2, int(np.ceil(max(np.linalg.norm(ready[o:o + 3] - v_from[o:o + 3]) for o in (0, 7)) / (a.entry_speed_mms / a.fps))))
                fr = []
                for j in range(1, n2 + 1):
                    f = v_from + (ready - v_from) * (j / n2)
                    for o in (3, 10):
                        f[o:o + 4] = _slerp(v_from[o:o + 4], ready[o:o + 4], j / n2)
                    f[14] = 0.0; fr.append(f)
                ok = await scripted_path(fr, "return")
        if not ok or S["abort"]:
            log("ABORTED:", S["abort"]); rc = 3
    finally:
        link.close()
        if ws is not None:
            try:
                await ws.send(json.dumps({"op": "close"}))
            except Exception:
                pass
            await ws.close()
        med = lambda x: None if not x else round(float(np.median(x)), 3)

        def per_frame(arr: list) -> list:   # a value per fed frame → a value per frame (each frame takes its group's fed frame)
            out, k = [], 0
            for i in range(len(S["meas"])):
                while k < len(S["fed"]) - 1 and S["fed"][k] < i:
                    k += 1
                out.append(arr[k] if arr else None)
            return out
        out = {"fps": a.fps, "frames": S["meas"], "frames_cmd": S["cmd"], "ref_recording": meta["recording"],
               "meta": {"policy": "demo-replay" if a.demo_replay else a.policy_url or a.policy_path, "live": True, "live_run": a.live_run,
                        "remote_policy": remote.summary() if remote is not None else None,
                        # what the policy looked at: the recorder page (one simulation) or its own step session (a second one)
                        "observation": "page" if page else "step", "page_observations": S["obs"],
                        "dataset": str(ds), "episode": a.episode, "start_frame": k0, "steps": a.steps, "phase_rules": a.phase_rules,
                        "speed": a.speed, "scene_map": a.scene_map, "endo_start_residual_mm": (link.c["endo"] * 1000).round(2).tolist(),
                        "filters": {"n_action_steps": n_exec, "drill_pause_s": pause_s, "start_state": a.start_state},
                        "frame_kind": S["kind"], "chunks": S["chunks"], "abort": S["abort"], "motion_windows_ms": S["windows"],
                        "timing": {"inference_s_median": med(S["t_inf"]), "inference_s": S["t_inf"],
                                   "render_step_s_median": med(S["t_step"]), "render_steps": len(S["t_step"]),
                                   "step_noimg_s_median": med(S["t_step_noimg"]), "steps_noimg": len(S["t_step_noimg"]), "gate_s_per_check_median": med(S["t_gate"]),
                                   "gate_checks": int(sum(S["n_gate"])),
                                   "session_dt_ms": dt_feed, "torch_threads": a.torch_threads},
                        "clamped_frames": S["n_clamped"], "trigger_gated_frames": S["n_gated"], "blind_chunks": S["n_blind_chunks"],
                        "visible_trigger": a.visible_trigger, "visible_trigger_rule": a.visible_trigger_rule,
                        "gate_anchor_state": getattr(a, "gate_anchor_state", None), "gate_pair": a.gate_pair, "gate_pairs_file": a.gate_pairs_file,
                        "phase_clamped_chunks": int(S["n_ph_clamped"]), "aimed_frames": S["n_aimed"], "aim": bool(a.aim), "rcm": bool(a.rcm),
                        "rcm_tool_frames": int(S["n_rcm"]), "rcm_deg_total": round(float(S["rcm_deg"]), 1),
                        "travel_limited_frames": S["n_travel"], "drill_log": json.dumps(per_frame(S["dlog"])),
                        "cycle": {"segments": segs, "ready_out_mm": a.ready_out_mm, "robot_start": q_ready},
                        "registration": P["registration"], "portal_from": a.portal_from,
                        "portals_mm": {t: np.asarray(p_, float).round(2).tolist() for t, p_ in portals.items()},
                        "safety_gate": {"pair_of": a.gate_pair, "checked": gate.n_checked, "shrunk": gate.n_shrunk,
                                        "turn_wait": gate.turn_wait, "turn_cut": gate.n_turn_cut,
                                        "held": gate.n_held, "events": S["gate_log"][:200]},
                        "fed_frames": S["fed"], "deadline_hit": S["deadline"], "wall_s": round(time.time() - link.t_engage, 1),
                        "drill_end": tracker.end or ("deadline" if S["deadline"] else "steps" if tracker.ph == 4 else None), "drill_s": round(tracker.drill_s, 1),
                        "done_output": done_out, "end_output": end_kind, "end_level": end_level,
                        "done_hold_steps": RO.DONE_HOLD if done_out else None, "policy_done": S.get("policy_done"),
                        "done_log": S.get("done_log") or None,
                        # a clamped frame: [frame, row of its window, rows of the window, endoscope mm, deg, drill mm, deg asked for]
                        "clamp_log": S.get("clamp_log") or None,
                        "phases_seen": per_frame(S["phases"]), "removed_mm3": per_frame(S["vols"]), "drill_in_view": per_frame(S["vis"])}}
        out["meta"]["continuous"] = S.get("continuous")      # None: the stop-and-go loop
        Path(a.out).write_text(json.dumps(out))
        log(json.dumps({"out": a.out, "frames": len(S["meas"]), "removed_mm3": round(tracker.vol, 1), "abort": S["abort"],
                        "inference_s": med(S["t_inf"]), "render_step_s": med(S["t_step"])}))
    return rc


def main() -> None:
    ap = RO.build_parser(__doc__)
    ap.add_argument("--prepare-out", default=None, help="write the seed trajectory (ready pose, robot start joints) and stop")
    ap.add_argument("--live-run", default=None, help="the robot run id (robot_origin_run.sh RUN: the arms' latency dirs <RUN>_endo/_drill)")
    ap.add_argument("--speed", type=float, default=0.33, help="arm speed relative to the demos (one step lasts 1/(fps·speed) s)")
    ap.add_argument("--tap-port", type=int, default=9884, help="a copy of the tracker tap (tap_fanout)")
    ap.add_argument("--scene-map", choices=["anchor", "relative"], default="anchor")
    ap.add_argument("--lead-ms", type=float, default=300.0, help="the first target of a chunk is this far ahead of now")
    ap.add_argument("--settle-s", type=float, default=2.0, help="the hold after a chunk counts from this long after its last target …")
    ap.add_argument("--settle-min-s", type=float, default=2.0, help="… and lasts at least this long: the observation is the median pose over it")
    ap.add_argument("--meas-half-ms", type=float, default=60.0, help="a frame's measured pose = the median within ± this of its target time")
    ap.add_argument("--lost-ms", type=float, default=700.0, help="abort if a robot tool has no 4-ball fix this near a target time")
    ap.add_argument("--abort-dev-mm", type=float, default=10.0, help="abort if a settled tip is this far from its command")
    ap.add_argument("--travel-mm", type=float, nargs=2, default=[130.0, 130.0], help="endo / drill tip travel limit from the ready pose (80 / 100 until 10-15; user: maximum = the guard's 130)")
    ap.add_argument("--gate-mm", type=float, default=1.0, help="policy chunks: a frame goes through the safety gate when a tip moved this far "
                    "since the last checked frame (and always the last frame of a chunk)")
    ap.add_argument("--gate-mm-scripted", type=float, default=2.0, help="the same on the scripted entry / exit (straight shaft-axis moves)")
    ap.add_argument("--gate-deg", type=float, default=1.0, help="… or a tool turned this much")
    ap.add_argument("--feed-stride", type=int, default=1, help="policy chunks: every n-th measured frame (and the last) is stepped into the "
                    "viewer — keep 1 while the drill may cut (a coarser path removes much less bone)")
    ap.add_argument("--feed-stride-scripted", type=int, default=8, help="the same on the scripted entry / exit (trigger off)")
    ap.add_argument("--deadline-s", type=float, default=0.0, help="wall seconds since engage after which the run must be over — the FE recorder "
                    "closes a session at 600 s; 0 = no limit")
    ap.add_argument("--exit-reserve-s", type=float, default=95.0, help="the drilling stops this long before --deadline-s (exit + return)")
    ap.add_argument("--aim-frames", type=int, default=6, help="frames of a chunk over which the endoscope aiming correction is spread")
    ap.add_argument("--blind-frames", type=int, default=8, help="a chunk predicted while the drill is hidden is cut to this many frames "
                    "(trigger off), so the visibility is judged again soon")
    ap.add_argument("--scripted-chunk", type=int, default=40, help="frames per chunk on the scripted entry / exit")
    ap.add_argument("--engage-timeout-s", type=float, default=900.0)
    ap.add_argument("--fault-wait-s", type=float, default=300.0,
                    help="how long to wait for stale tracker stamps to come back before a chunk (10-05: the tracker switched "
                         "its stamp source by itself after 265 s)")
    ap.add_argument("--stop-after", choices=["entry", "all"], default="all", help="entry: scripted entry, then exit (link test)")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--windows-out", default=None, help="write the wall-clock windows in which the arms move (render_video.py --windows)")
    ap.add_argument("--render-wait-s", type=float, default=1200.0, help="wait this long for the renderer to be free before opening the session")
    ap.add_argument("--latency-dir", default=None, help="where the arms' run dirs are (default ~/il-robot/runs/latency)")
    ap.add_argument("--session-dt", choices=["policy", "wall"], default="policy",
                    help="the viewer session's clock per fed frame: policy = 1/fps (as the training renders, default) or wall = "
                         "1/(fps·speed), the time the robot really spent (3x the engine ticks per frame at 0.33x)")
    ap.add_argument("--ready-flag", default=None, help="file written once the policy is loaded and the trigger heartbeat runs")
    ap.add_argument("--obs-url", default=None, help="the policy looks at the recorder page (robot_origin_rec.py --obs-port P → "
                    "http://127.0.0.1:P; page_obs.py) instead of a render-API step session of its own")
    ap.add_argument("--continuous", action="store_true",
                    help="the policy part without stops (CONTINUOUS.md): looks while the arms move, the next answer asked for before "
                         "the current one is used up. Needs --obs-url (the recorder started with --obs-stream, or its looks are screenshots "
                         "that stop the page). The scripted entry and exit stay stop-and-go")
    ap.add_argument("--stream-period", type=int, default=8, help="continuous: frames between two looks = frames of a window")
    ap.add_argument("--stream-lead", type=int, default=10, help="continuous: frames from a look to the first frame executed from its "
                    "answer (look + answer + gate + sending must fit: 10 frames = 2.0 s at 0.33x)")
    ap.add_argument("--stream-blend", type=int, default=4, help="continuous, --stream-anchor measured: frames over which a new plan is faded in")
    ap.add_argument("--stream-anchor", choices=["command", "measured"], default="command",
                    help="continuous: what an answer's rows are anchored at. measured = the pose measured at its look, as "
                         "answered (the first continuous robot run, 10-12: both tools swung for the whole run — the arms' "
                         "error at a look came back as a command L frames later). command = the command of that moment: "
                         "the plan moves the command timeline, the measured pose only goes to the model (stream_plan.py)")
    ap.add_argument("--stream-gamma", type=float, default=0.5, help="continuous, --stream-anchor command: how far a window is "
                    "brought from the last sent target's continuation toward the new plan's positions (0 = its motion only, 1 = all)")
    ap.add_argument("--stream-rcm-first", action="store_true", help="continuous: put each answer's rows on the pivot (rcm_project, ≤ 5°) "
                    "before the stitch and the step limit. Without it the limit compares the plan's absolute rotation with the last command, "
                    "which the RCM rule turned ≤ 2° toward the portal — every frame asks that turn back, the limit clamps it "
                    "and RCM turns again (10-16: most of the continuous runs' rotation clamps; L3 ipsi2's endoscope 9–11° off "
                    "its portal). Stop-and-go has no such pull: its steps are relative to the RCM'd pose")
    ap.add_argument("--send-lead-ms", type=float, default=150.0, help="continuous: a target reaches the arms at least this long before its time")
    ap.add_argument("--obs-half-ms", type=float, default=300.0, help="continuous: the observed tool state = the median within ± this of the look "
                    "(the tip wanders ~1 mm around the command; the stop-and-go loop takes the median of a 2 s hold)")
    ap.add_argument("--page-lag-ms", type=float, default=100.0, help="continuous: the page shows the tools about this long after the tracker saw them")
    ap.add_argument("--policy-url", default=None, help="ask a policy server instead of loading a checkpoint (remote_policy.py, protocol "
                    "spine-policy/1): http(s)://host:port — e.g. a GR00T model on a cloud GPU, reached through an SSH tunnel")
    ap.add_argument("--policy-token-file", default=None, help="file with the server's bearer token (chmod 600); never on a command line")
    ap.add_argument("--goal", default=None, help="the goal told to a remote model in words, LEVEL[:EXTENT] — L3 | L4 | L5 and ipsilateral | contralateral "
                                                 "(dataset/instructions.py lang1; default L4:contralateral). Not for models of the legacy phase texts")
    ap.add_argument("--goal-text-file", default=None, help="the drilling sentence, read from this file as written (one line) instead of "
                    "the set's wording of --goal — e.g. a paraphrase the model never met, to show it follows the meaning, not a "
                    "memorised sentence (10-16). The approach sentence stays --goal's")
    ap.add_argument("--policy-image", default="png", help="how the observation is sent to a remote policy: png (lossless, the default) "
                    "or jpeg[:quality] — a quarter of the bytes; only for a model whose answers were measured to be the same from it "
                    "(remote/replay_obs.py --image)")
    ap.add_argument("--policy-ca-file", default=None, help="https with a certificate no authority signed: the server's own certificate (PEM), "
                    "the only one trusted for this connection")
    ap.add_argument("--policy-timeout-s", type=float, default=6.0, help="per request; with one retry this stays under the 15 s after "
                    "which arms without a new target lose their lease")
    ap.add_argument("--policy-retries", type=int, default=1)
    a = ap.parse_args()
    if not a.prepare_out and not a.live_run:
        ap.error("--live-run RUN (or --prepare-out)")
    if not a.prepare_out and not a.policy_path and not a.policy_url and not a.demo_replay:
        ap.error("--policy-path or --policy-url needed (or --demo-replay)")
    if a.continuous and not a.obs_url and not a.prepare_out:
        ap.error("--continuous needs --obs-url: a look while the arms move is the recorder page's (a step session cannot be fed and rendered meanwhile)")
    if a.policy_path and a.policy_url:
        ap.error("--policy-path and --policy-url are two different policies: give one")
    sys.exit(asyncio.run(run(a)))


if __name__ == "__main__":
    main()
