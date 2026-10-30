#!/usr/bin/env python3
"""Both arms moving together through the safe zone, recording quasi-static samples until told to stop (2026-09-27).

  collect_both.py [--out DIR]          # runs until DIR/STOP exists (or SIGINT/SIGTERM), then both arms go home, torque off

Each stop is a pair of poses — one per arm — where every ball of both markers is seen (safezone: both imagers, depth
--z-mm, facing angle, line of sight, the measured detection window, the two markers apart in the image) and the two
models (marker blocks, balls, rods included) stay clear of the floor, each other and themselves. The two arms move
together along one straight line in joint space, checked point by point with both arms moving. A separate process
plans the next stops while the arms move. At every stop: settle, then --rec-s of taps + encoders (events.jsonl,
"record" per arm) and an IR snapshot. A marker lost for --lost-s sends both arms back along the same line; an arm
lagging its goal by --lag-deg holds everything; a servo above --max-temp ends the run.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_two_arm as V  # noqa: E402

sz = V.sz
ARMS = ("left", "right")
SEE = ARMS


def _pools(zone, rng, bnd, z_mm, n_try, keep):
    """Per arm: poses well inside the zone on their own (window margin, depth band), the other arm ignored."""
    lim, cam = zone.lim, zone.cam
    out = {}
    for arm in ARMS:
        model, got = zone.arms[arm], []
        for _ in range(n_try):
            q = np.array([rng.uniform(max(V.RANGES[j][0], bnd[arm][j][0] + 3), min(V.RANGES[j][1], bnd[arm][j][1] - 3))
                          for j in sz.JOINTS[:5]] + [rng.uniform(*V.GRIPPER[arm])])
            B, n = model.balls(q)
            Bc = cam.cam(B)
            if Bc[:, 2].min() * 1000 < z_mm[0] or Bc[:, 2].max() * 1000 > z_mm[1]:
                continue
            px = cam.pixels(Bc)
            u0, u1, v0, v1 = lim.window_px
            if px[:, 0].min() < u0 + 20 or px[:, 0].max() > u1 - 20 or px[:, 2].min() < v0 + 20 or px[:, 2].max() > v1 - 20:
                continue
            to_cam = cam.centres()[0] - B.mean(0)
            if math.degrees(math.acos(float(n @ to_cam / np.linalg.norm(to_cam)))) > lim.target_view_deg:
                continue
            if zone.single(arm, q, strict=True)[0]:
                got.append(q)
                if len(got) >= keep:
                    break
        out[arm] = got
    return out


def _history(events: Path):
    """What the data says so far: poses recorded (10-D, both arms) and targets where a marker was lost."""
    done, bad = [], []
    try:
        for l in open(events):
            if '"pair_target"' not in l:
                continue
            e = json.loads(l)
            v = np.r_[e["q"]["left"][:5], e["q"]["right"][:5]]
            (bad if e.get("lost") else done).append(v)
    except OSError:
        pass
    return done, bad


def planner(start: dict, home: dict, seed: int, z_mm, out_q, stop_ev, log_q, events: str = ""):
    """Separate process: keeps up to 2 checked stops ready, each reachable from the previous one. The next stop is
    steered by the data so far: as far as possible from every pose already recorded, never near a target where a
    marker was lost; fresh candidate poses every 50 stops."""
    rng = np.random.default_rng(seed)
    arms, cam, info = V.load_models()
    zone = sz.Zone(arms, cam)
    zone.set_start(home)
    bnd = {x: V.rpc(x).status()["bounds"] for x in ARMS}
    pools = _pools(zone, rng, bnd, z_mm, 20000, 250)
    log_q.put(f"planner: {len(pools['left'])} left / {len(pools['right'])} right poses in the zone")
    cur = {x: np.asarray(v, float) for x, v in start.items()}
    hist = [np.r_[cur["left"][:5], cur["right"][:5]]]
    fails, made = 0, 0
    while not stop_ev.is_set():
        if made and made % 50 == 0:
            pools = _pools(zone, rng, bnd, z_mm, 20000, 250)
            made += 1
            log_q.put(f"planner: fresh poses {len(pools['left'])}/{len(pools['right'])}")
        if out_q.qsize() >= 2:
            time.sleep(0.3)
            continue
        cands = []
        for _ in range(300):                                   # pairs that are safe together
            q = {"left": pools["left"][rng.integers(len(pools["left"]))],
                 "right": pools["right"][rng.integers(len(pools["right"]))]}
            if zone.check(q, strict=True, see=SEE)[0]:
                cands.append(q)
                if len(cands) >= 16:
                    break
        if not cands:
            fails += 1
            log_q.put("planner: no safe pair found this round")
            if fails > 5:
                log_q.put("planner: giving up")
                return
            continue
        done, bad = _history(Path(events)) if events else ([], [])
        H = np.array(hist + done)
        feats = [np.r_[c["left"][:5], c["right"][:5]] for c in cands]
        if bad:
            B_ = np.array(bad)
            keep = [i for i, f in enumerate(feats) if np.min(np.abs(B_ - f).max(axis=1)) > 10.0]
            cands, feats = [cands[i] for i in keep], [feats[i] for i in keep]
            if not cands:
                continue
        div = [float(np.min(np.linalg.norm(H - f, axis=1))) for f in feats]
        for k in np.argsort(div)[::-1][:6]:                    # the most different pose that a safe move reaches
            if stop_ev.is_set():
                return
            ok, why, n = zone.path(cur, cands[k], 3.0, see=SEE)
            if ok:
                out_q.put({x: cands[k][x].tolist() for x in ARMS})
                made += 1
                cur = cands[k]
                hist.append(np.r_[cur["left"][:5], cur["right"][:5]])
                fails = 0
                break
        else:
            fails += 1
            log_q.put("planner: no reachable pair this round")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--speed", type=float, default=12.0, help="deg/s of the fastest joint of either arm")
    ap.add_argument("--guard-step", type=float, default=8.0)
    ap.add_argument("--rec-s", type=float, default=1.5)
    ap.add_argument("--lost-s", type=float, default=0.5)
    ap.add_argument("--lag-deg", type=float, default=20.0)
    ap.add_argument("--max-temp", type=float, default=60.0)
    ap.add_argument("--z-mm", type=float, nargs=2, default=(330.0, 520.0))
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path.home() / "il-robot/runs/collect" / time.strftime("%Y%m%dT%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ir").mkdir(exist_ok=True)
    home = V.read_q()
    arms, cam, info = V.load_models()
    zone = sz.Zone(arms, cam)
    zone.set_start(home)
    if not zone.check(home)[0]:
        sys.exit(f"start pose not safe: {zone.check(home)[1]}")
    (out / "plan.json").write_text(json.dumps({"start": {k: v.tolist() for k, v in home.items()}, "models": info,
                                                "args": vars(a), "limits": zone.lim.__dict__}, indent=1, default=float))
    log = open(out / "events.jsonl", "a")
    tap = V.Tap(V.TAP_PORT, log)
    time.sleep(0.5)
    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    R = {x: V.rpc(x) for x in ARMS}

    def say(msg):
        print(f"[collect {time.strftime('%H:%M:%S')}] {msg}", flush=True)

    def event(**kw):
        with tap.lock:
            log.write(json.dumps({"k": "event", "t": round(time.time() * 1000, 3), **kw}, default=float) + "\n")
            log.flush()

    def state(x):
        st = R[x].state()
        return np.array([st["pos"][j] for j in sz.JOINTS])

    def enc_log(goal):
        with tap.lock:
            for x in ARMS:
                log.write(json.dumps({"k": "enc", "arm": x, "t": round(time.time() * 1000, 3),
                                      "pos": state(x).round(3).tolist(), "goal": np.round(goal[x], 3).tolist()}) + "\n")

    def stop_requested():
        return stop["flag"] or (out / "STOP").exists()

    tok = {}

    def send(q):
        for x in ARMS:
            R[x].must(cmd="goal", token=tok[x], pos={j: float(v) for j, v in zip(sz.JOINTS, q[x])})

    def move(q0, q1, check_marker=True, allow_stop=True) -> str:
        """Same straight joint-space line as before, now timed by the clock with a minimum-jerk profile (zero speed
        and acceleration at both ends; peak = --speed) and a fresh goal at 60 Hz, faster than the server's 30 Hz tick.
        A slower, index-stepped loop (~26 Hz) left every ~7th server tick with no new goal — a 4.4 Hz stutter."""
        span = max(float(np.abs(q1[x] - q0[x]).max()) for x in ARMS)
        T = max(0.2, 1.875 * span / a.speed)
        lag_since, t0, k = None, time.time(), 0
        while True:
            if allow_stop and stop_requested():
                return "stop"
            tau = min(1.0, (time.time() - t0) / T)
            s = tau ** 3 * (10 - 15 * tau + 6 * tau ** 2)
            qi = {x: q0[x] + (q1[x] - q0[x]) * s for x in ARMS}
            send(qi)
            k += 1
            if k % 2 == 0 or tau >= 1.0:                        # checks and log at 30 Hz
                enc_log(qi)
                lag = max(float(np.abs(state(x) - qi[x])[:5].max()) for x in ARMS)
                lag_since = (lag_since or time.time()) if lag > a.lag_deg else None
                if lag_since and time.time() - lag_since > 0.5:
                    event(what="lag", lag_deg=lag)
                    return "lag"
                if check_marker:
                    for x in ARMS:
                        if tap.age(V.TOOLS[x]) > a.lost_s:
                            event(what="lost", arm=x, at=round(tau, 3))
                            return "lost"
            if tau >= 1.0:
                return "ok"
            time.sleep(max(0.0, t0 + k / 60 - time.time()))

    def dwell(q, k):
        hist, t0 = [], time.time()
        while time.time() - t0 < 4.0:
            send(q)
            enc_log(q)
            hist.append(np.r_[state("left"), state("right")])
            if len(hist) >= 15 and np.ptp(np.array(hist[-15:]), axis=0).max() < 0.3:
                break
            time.sleep(1 / 30)
        t_settle = time.time()
        while time.time() - t_settle < 0.5 + a.rec_s:
            send(q)
            enc_log(q)
            time.sleep(1 / 30)
        for x in ARMS:
            event(what="record", arm=x, t0=round((t_settle + 0.5) * 1000, 3), t1=round(time.time() * 1000, 3),
                  stop=k, kind="pair")
        try:
            import urllib.request
            (out / "ir" / f"pair{k:04d}.jpg").write_bytes(urllib.request.urlopen(V.FRAME_URL, timeout=2).read())
        except OSError:
            pass

    def temps() -> float:
        return max(max(R[x].status().get("temp", {}).values() or [0]) for x in ARMS)

    def go_home(cur) -> bool:
        """Both together if that line is safe, else one after the other (each alone, the other still)."""
        if zone.path(cur, home, 2.0, see=())[0]:
            return move(cur, home, check_marker=False, allow_stop=False) == "ok"
        for x in ("right", "left"):
            nxt = {**cur, x: home[x]}
            if not zone.path(cur, nxt, 2.0, see=())[0]:
                say(f"no checked way home for the {x} arm — holding")
                return False
            if move(cur, nxt, check_marker=False, allow_stop=False) != "ok":
                return False
            cur = nxt
        return True

    ctx = mp.get_context("spawn")
    stop_ev, out_q, log_q = ctx.Event(), ctx.Queue(), ctx.Queue()
    proc = None
    result, k, cur = "ok", 0, None
    try:
        for x in ARMS:
            tok[x] = R[x].lease("collect")
            if R[x].state()["state"] == "OFF":
                R[x].must(cmd="enable", token=tok[x])
            R[x].must(cmd="set", token=tok[x], step_deg=a.guard_step)
        cur = {x: state(x) for x in ARMS}
        event(what="start")

        def start_planner(from_q, seed):
            p = ctx.Process(target=planner, args=({x: v.tolist() for x, v in from_q.items()},
                                                 {x: v.tolist() for x, v in home.items()}, seed, a.z_mm, out_q,
                                                 stop_ev, log_q, str(out / "events.jsonl")), daemon=True)
            p.start()
            return p
        proc = start_planner(cur, a.seed)
        say(f"started → {out}")
        waiting = time.time()
        while not stop_requested():
            while not log_q.empty():
                say(log_q.get())
            try:
                tgt = out_q.get(timeout=1 / 30)
            except queue.Empty:
                send(cur)                                       # hold while the planner works — every tick: the
                # server drops to HOLD after 250 ms without a goal and re-latches, which shook the arms (2026-09-27)
                if not proc.is_alive():
                    say("planner stopped — ending")
                    break
                if time.time() - waiting > 5:
                    enc_log(cur)
                    waiting = time.time()
                continue
            tgt = {x: np.asarray(v, float) for x, v in tgt.items()}
            t_move = time.time()
            r = move(cur, tgt)
            event(what="pair_target", q={x: tgt[x].tolist() for x in ARMS}, lost=(r == "lost"))
            if r == "lost":
                say(f"pair {k}: a marker was lost — back along the same line, planning afresh")
                r = move({x: state(x) for x in ARMS}, cur, check_marker=False)
                stop_ev.set()
                proc.join(5)
                while not out_q.empty():
                    out_q.get()
                stop_ev.clear()
                proc = start_planner(cur, a.seed + k + 1)
                if r != "ok":
                    result = r
                    break
                continue
            if r != "ok":
                result = r
                break
            dwell(tgt, k)
            cur = tgt
            k += 1
            if k % 5 == 0:
                tmax = temps()
                say(f"{k} pairs recorded (last move {time.time() - t_move:.1f} s), hottest servo {tmax} °C")
                if tmax > a.max_temp:
                    say("servo too warm — ending")
                    break
    finally:
        stop_ev.set()
        if proc is not None:
            proc.join(5)
        if result in ("ok", "stop") and cur is not None:
            say(f"going home ({k} pairs recorded)")
            ok = go_home({x: state(x) for x in ARMS})
            for x in ARMS:
                R[x].call(cmd="hold", token=tok.get(x))
            time.sleep(0.5)
            if ok:
                for x in ARMS:
                    off = R[x].call(cmd="torque_off", token=tok.get(x))
                    if off.get("error") == "not_at_neutral":
                        R[x].must(cmd="home", token=tok[x])
                        t0 = time.time()
                        while time.time() - t0 < 30 and R[x].state()["state"] == "HOMING":
                            R[x].call(cmd="renew", token=tok[x])
                            time.sleep(0.3)
                        off = R[x].call(cmd="torque_off", token=tok[x])
                    say(f"{x}: torque {off.get('state')}")
        else:
            for x in ARMS:
                try:
                    R[x].call(cmd="hold", token=tok.get(x))
                except Exception:
                    pass
            say(f"stopped: {result} — both arms hold where they are (torque on)")
        for x in ARMS:
            try:
                R[x].call(cmd="release", token=tok.get(x))
            except Exception:
                pass
        event(what="end", pairs=k, result=result)
        tap.close()
        log.close()
        say(f"done ({result}), {k} pairs → {out}")


if __name__ == "__main__":
    main()
