"""A new rev.4 hand recording → robot-origin sessions, end to end (2026-09-29).

    python new_recording.py RID [RID ...] [--reps 2] [--len-s 30] [--no-queue] [--force]

RID = the recording's 4-hex id (…_trk_<RID>.jsonl under viewer-data/sessions/<date>/ or sessions/policy/<date>/
(robot/inference/policy2rec.py output), robot/ excluded).
Per recording:
  1. window: from = first bone cut − 2 s (first_cut.py, rounded like the 09-28 windows), len 30 s → pairstart/windows.json
  2. hand reference for the IoU: replay_to_render.py --hand --pose filtered → ../pd_<RID>000_hand.json (ro_iou.py's hand tag)
  3. paired start poses: pairstart/search_clear.py (tip vector = the recording's, Clearance start ≥ 95 / replay ≥ 90 mm,
     safe-zone paths + 30 s replay) and postpass_clear.py (exact arm-to-arm minimum); log pairstart/search_clear_<RID>.log
  4. gate: replay_unsafe == 0, vec_err ≤ 3 mm, arms_min_replay ≥ 10 mm — else the recording is reported and skipped
  5. cfg/pj_<RID>.env (SPEED 0.33 + the pair's --start per arm; preset j is robot_origin_run.sh's default)
  6. queue: pr_<RID>j1..jN lines appended to ro_jobs.txt → ro_queue2.sh records them (robot relay, headless FE recorder,
     both arms, trigger from the drill arm) and runs ro_iou.py (recorded vs recorded IoU, tolerance, tick-matched re-sim)
Status per recording in new_recordings.json. Robot motion happens only in step 6, through the queue (guard + safe zone).
"""
import argparse, glob, json, subprocess, sys, time
from pathlib import Path

W = Path(__file__).resolve().parent
J = W.parent
PS = W / "pairstart"
PY = str(Path.home() / "il-robot/.venv/bin/python")
ROBOT = Path(__import__("os").environ.get("SPINE_IL_REPO", str(Path.home() / "spine-il"))) / "robot"
M = __import__("os").path.expanduser("~/il-robot/calib/tracker_models")
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
sys.path.insert(0, str(W))
sys.path.insert(0, str(ROBOT))
import first_cut  # noqa: E402
import hand_calib  # noqa: E402


def log(rid, *a):
    print(f"[new] {time.strftime('%T')} {rid}", *a, flush=True)


def find(rid: str) -> str:
    base = Path.home() / "il-robot/viewer-data/sessions"
    hits = sorted(f for pat in (f"*/*_trk_{rid}.jsonl", f"policy/*/*_trk_{rid}.jsonl")   # hand sessions, policy2rec output
                  for f in glob.glob(str(base / pat)) if "/robot/" not in f)
    if not hits:
        raise SystemExit(f"{rid}: no recording …_trk_{rid}.jsonl under viewer-data/sessions/<date>/")
    if len(hits) > 1:   # 10-03: a policy recording reused an old id and the robot replayed the old file
        raise SystemExit(f"{rid}: {len(hits)} recordings share this id ({', '.join(Path(h).name for h in hits)}) — rename one")
    return hits[0]


def update(path: Path, key: str, val) -> None:
    d = json.loads(path.read_text()) if path.exists() else {}
    d[key] = val
    path.write_text(json.dumps(d, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rid", nargs="+")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--len-s", type=float, default=30.0)
    ap.add_argument("--speed", type=float, default=0.33)
    ap.add_argument("--radius-mm", type=float, default=80.0, help="replay sphere (endo_follow --radius-mm); full cycle from the ready pose: 100")
    ap.add_argument("--tag", default="j", help="job tag letter(s): pr_<RID><tag><n>")
    ap.add_argument("--from-s", type=float, default=None, help="window start override (e.g. a rehearsal matched to its demo)")
    ap.add_argument("--no-queue", action="store_true", help="stop after the cfg (no robot jobs)")
    ap.add_argument("--no-iou", action="store_true", help="no hand reference trajectory → the queue skips ro_iou.py after the run "
                    "(inference iterations: policy_eval.py compares the cuts with the demos instead)")
    ap.add_argument("--force", action="store_true", help="redo the pair search even if pair_clear_<RID>.json exists")
    ap.add_argument("--start-from-traj", default=None, help="rollout trajectory JSON (--cycle): use its meta.cycle.robot_start "
                    "as the robot start pose (verified with robot/inference/verify_traj.py) instead of searching")
    ap.add_argument("--pair-of", default=None, help="reuse this recording's verified pair (pairs_by_recording_chosen.json) "
                    "instead of searching — for a policy recording that starts at that pair's window start and passed "
                    "the inference safety gate (robot/inference/safety_gate.py)")
    a = ap.parse_args()
    status = W / "new_recordings.json"
    for rid in a.rid:
        st = {"time": time.strftime("%m-%d %H:%M")}
        try:
            rec = find(rid); st["file"] = rec
            frm = a.from_s if a.from_s is not None else first_cut.window(rec)
            if frm is None:
                raise SystemExit("no bone cut in the recording")
            st["from_s"], st["len_s"] = frm, a.len_s
            update(PS / "windows.json", rid, {"file": rec, "from_s": frm, "len_s": a.len_s})
            log(rid, f"window {frm}–{frm + a.len_s:g} s ({Path(rec).name})")

            hand = J / f"pd_{rid}000_hand.json"
            if not hand.exists() and not a.no_iou:
                subprocess.run([PY, str(ROBOT / "replay_to_render.py"), rec, "--from-s", str(frm), "--len-s", str(a.len_s),
                                "--hand", "--pose", "filtered", "-o", str(hand)], check=True, capture_output=True)
            st["hand_tag"] = hand.name[:-len("_hand.json")]

            out = PS / f"pair_clear_{rid}.json"
            guard = ""
            if a.start_from_traj:
                tj = json.loads(Path(a.start_from_traj).read_text())
                qs = tj["meta"]["cycle"]["robot_start"]
                # guard_pairs.py stops a run 30 mm from its start; a verified cycle travels further → its own limit + 15 mm
                mv = lambda o: max(sum((f[o + i] - tj["frames"][0][o + i]) ** 2 for i in range(3)) ** 0.5 for f in tj["frames"])
                guard = f"RO_GUARD_MM={mv(0) + 15:.0f}:{mv(7) + 15:.0f}"
                out.write_text(json.dumps({"best": {"q_left": qs["left"], "q_right": qs["right"], "vec_err_mm": 0.0, "replay_unsafe": 0,
                                                    "arms_min_mm_replay": 50, "from_traj": a.start_from_traj}}))
                log(rid, "robot start pose from the rollout's ready pose (gate-checked)")
            elif a.pair_of:
                chosen = json.loads((PS / "pairs_by_recording_chosen.json").read_text())[a.pair_of]
                out.write_text(json.dumps({"best": {**chosen, "reused_from": a.pair_of, "replay_unsafe": 0,
                                                    "arms_min_mm_replay": chosen.get("arms_min_mm_replay", 50)}}))
                log(rid, f"pair reused from {a.pair_of} (safety gate passed during inference)")
            elif a.force or not out.exists():
                log(rid, "pair start search (search_clear.py) …")
                env = {**__import__("os").environ, "PS_RADIUS_M": str(a.radius_mm / 1000.0)}
                with open(PS / f"search_clear_{rid}.log", "w") as fh:
                    subprocess.run([PY, "-u", str(PS / "search_clear.py"), rid], cwd=PS, stdout=fh, stderr=subprocess.STDOUT,
                                   check=True, env=env)
                subprocess.run([PY, "-u", str(PS / "postpass_clear.py"), rid], cwd=PS, check=True, capture_output=True, env=env)
            b = json.loads(out.read_text())["best"]
            st["pair"] = {k: b.get(k) for k in ("q_left", "q_right", "vec_err_mm", "sep_mm", "replay_unsafe", "arms_min_mm_replay",
                                                  "joint_margin_min_deg")}
            bad = []
            if b.get("replay_unsafe", 1):
                bad.append(f"replay_unsafe {b.get('replay_unsafe')}")
            if b.get("vec_err_mm", 99) > 3:
                bad.append(f"vec_err {b.get('vec_err_mm')} mm")
            if (b.get("arms_min_mm_replay") or 0) < 10:
                bad.append(f"arms_min {b.get('arms_min_mm_replay')} mm")
            if bad:
                raise SystemExit("no safe pair: " + ", ".join(bad))
            start = lambda q: "--start=" + ",".join(f"{j}={v:.1f}" for j, v in zip(JOINTS, q))
            try:   # the hand tool calibration the recording names (10-06 ≠ 09-28: drill tip 5.1 mm); the run script picks it too
                cal = hand_calib.for_recording(rec)
            except LookupError as e:
                log(rid, f"hand calibration unknown ({e}) — the 09-28 files")
                cal = {"endo": f"{M}/hand_20260928/tool_calib.json", "drill": f"{M}/hand_20260928/tool_calib_drill.json"}
            st["hand_calib"] = {t: str(f) for t, f in cal.items()}
            (W / f"cfg/pj_{rid}.env").write_text(
                f"# pair start {rid} (new_recording.py), preset j (robot_origin_run.sh default)\nSPEED={a.speed:g}\n"
                f'LEFT_EXTRA="--post-weight 0.1 --tool-calib {cal["endo"]} '
                f'--robot-tool-calib {M}/tool_calib_endo_robotL_tip.json {start(b["q_left"])}"\n'
                f'RIGHT_EXTRA="--post-weight 0.1 --tool-calib {cal["drill"]} '
                f'--robot-tool-calib {M}/tool_calib_drill_robotR_tip.json {start(b["q_right"])}"\n')
            log(rid, f"pair ok: vec err {b['vec_err_mm']} mm, arms min {b['arms_min_mm_replay']} mm → cfg/pj_{rid}.env")
            if not a.no_queue:
                with open(W / "ro_jobs.txt", "a") as fh:
                    for n in range(1, a.reps + 1):
                        fh.write(f"pr_{rid}{a.tag}{n} {Path(rec).name} {frm} {a.len_s:g} pj_{rid} drill"
                                 + (" " + ",".join(e for e in (f"RO_RADIUS_MM={a.radius_mm:g}" if a.radius_mm != 80 else "",
                                                               guard) if e) if a.radius_mm != 80 or guard else "") + "\n")
                st["queued"] = [f"pr_{rid}{a.tag}{n}" for n in range(1, a.reps + 1)]
                log(rid, "queued", st["queued"])
            st["ok"] = True
        except (SystemExit, subprocess.CalledProcessError) as e:
            st["ok"], st["error"] = False, str(e)[:300]
            log(rid, "SKIPPED:", st["error"])
        update(status, rid, st)


if __name__ == "__main__":
    main()
