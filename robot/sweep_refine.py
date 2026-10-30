#!/usr/bin/env python3
"""A slow, continuous sweep of one arm's marker block for the reprojection-refinement recording (rec_blobs.py), 2026-09-28.

  sweep_refine.py plan --arm right            # waypoints + spline, every sample checked in the safe zone → plan json
  sweep_refine.py run  --arm right --plan P --wait-dir DIR
                                              # rest → centre, hold until DIR exists (rec_blobs started), hold --hold-s
                                              # (its GET READY + light alignment + HOLD STILL), sweep --sweep-s, centre,
                                              # home, torque off

The camera belongs to rec_blobs while it records, so there is no marker check while moving: the whole trajectory is
checked offline instead (lerobot_robot_spinearm.safezone, the other arm resting at its neutral pose as an obstacle,
every 30 Hz sample of the spline), and the server's own lag/overload faults stop the arm. Balls stay 35–60 cm from the
camera; wrist_flex / wrist_roll / gripper and the pan vary the block's tilt and roll. One arm at a time.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_two_arm as V  # noqa: E402
from lerobot_robot_spinearm import safezone as sz  # noqa: E402
from lerobot_robot_spinearm.client import ArmRpc  # noqa: E402
from lerobot_robot_spinearm.protocol import DEFAULT_PORTS, JOINTS  # noqa: E402

CENTRE = {"right": [3.5, -49.0, 23.8, 83.4, 8.4, -45.2],   # the replay start poses (validated, used every run)
          "left": [0.0, -60.0, 50.0, 70.0, 0.0, -40.0]}
SPAN = np.array([15.0, 12.0, 15.0, 20.0, 35.0, 15.0])     # sampling half-range around the centre, per joint (deg)
Z_MM = (350.0, 600.0)
RATE = 30.0


def other(arm: str) -> str:
    return "left" if arm == "right" else "right"


def neutral(arm: str) -> np.ndarray:
    st = ArmRpc("127.0.0.1", DEFAULT_PORTS[arm][0]).status()
    return np.array([st["neutral"][j] for j in JOINTS], float)


def catmull(P: np.ndarray, T: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Centripetal-free uniform Catmull-Rom through the points P at knot times T, sampled every dt. The first and
    last segments use a duplicated end point, so the path starts and ends at rest."""
    Q = np.vstack([P[:1], P, P[-1:]])
    t_out, q_out = [], []
    for i in range(len(P) - 1):
        p0, p1, p2, p3 = Q[i], Q[i + 1], Q[i + 2], Q[i + 3]
        n = max(2, int(round((T[i + 1] - T[i]) / dt)))
        for k in range(n):
            s = k / n
            q = 0.5 * ((2 * p1) + (-p0 + p2) * s + (2 * p0 - 5 * p1 + 4 * p2 - p3) * s * s + (-p0 + 3 * p1 - 3 * p2 + p3) * s ** 3)
            t_out.append(T[i] + (T[i + 1] - T[i]) * s)
            q_out.append(q)
    t_out.append(T[-1]); q_out.append(P[-1])
    return np.array(t_out), np.array(q_out)


def plan(a) -> None:
    arms, cam, _ = V.load_models()
    zone = sz.Zone(arms, cam)
    rest = {x: neutral(x) for x in ("left", "right")}
    zone.set_start(rest)
    o, c = other(a.arm), np.array(CENTRE[a.arm])
    st = ArmRpc("127.0.0.1", DEFAULT_PORTS[a.arm][0]).status()
    lo = np.array([st["bounds"][j][0] + 5 for j in JOINTS]); hi = np.array([st["bounds"][j][1] - 5 for j in JOINTS])

    def ok(q, strict):
        g, why, m = zone.check({a.arm: q, o: rest[o]}, strict=strict, see=(a.arm,))
        z = m.get(a.arm, {}).get("z_mm") if m else None
        return g and z is not None and Z_MM[0] <= min(z) and max(z) <= Z_MM[1], why, m

    g0, why0, m0 = ok(c, False)
    print(f"[plan] centre {c.tolist()}: {'ok' if g0 else why0} z {[round(v) for v in m0[a.arm]['z_mm']]} "
          f"view {m0[a.arm]['view_deg']:.0f} deg")
    if not g0:
        sys.exit("[plan] the centre pose itself fails the zone check")
    rng = np.random.default_rng(a.seed)
    cands = []
    for _ in range(a.n_try):
        q = np.clip(c + rng.uniform(-1, 1, 6) * SPAN, lo, hi)
        g, _, m = ok(q, True)
        if g:
            cands.append((q, m[a.arm]["view_deg"]))
    print(f"[plan] {len(cands)} / {a.n_try} strict candidates")
    if len(cands) < a.n_way:
        sys.exit("[plan] too few candidates")
    # spread: farthest-point sampling on the joints that tilt / roll the block (weights favour orientation)
    w = np.array([0.6, 0.6, 0.6, 1.0, 1.0, 1.0])
    pick = [int(np.argmax([np.abs((q - c) * w).sum() for q, _ in cands]))]
    while len(pick) < a.n_way:
        d = [min(np.abs((q - cands[p][0]) * w).sum() for p in pick) for q, _ in cands]
        pick.append(int(np.argmax(d)))
    way = [cands[p][0] for p in pick]
    # tour: nearest neighbour from the centre, each hop a checked straight line (fallback if the spline fails)
    tour, cur, left_ = [c], c, list(range(len(way)))
    while left_:
        nxt = min(left_, key=lambda k: np.abs(way[k] - cur).max())
        if zone.path({a.arm: cur, o: rest[o]}, {a.arm: way[nxt], o: rest[o]}, 2.0, (a.arm,))[0]:
            tour.append(way[nxt]); cur = way[nxt]
        left_.remove(nxt)
    tour.append(c)
    P = np.array(tour)
    seg = np.maximum(np.abs(np.diff(P, axis=0)).max(1) / a.vmax, a.min_seg_s)
    T = np.concatenate([[0.0], np.cumsum(seg)])
    t, Q = catmull(P, T, 1.0 / RATE)
    bad = []
    for k in range(0, len(Q), 3):
        g, why, _ = ok(Q[k], False)
        if not g:
            bad.append((round(float(t[k]), 2), why[:2]))
    lim_ok = bool(np.all(Q >= lo - 5 + 3) and np.all(Q <= hi + 5 - 3))
    vmax = float(np.abs(np.diff(Q, axis=0)).max() * RATE)
    print(f"[plan] tour {len(P)} poses, {T[-1]:.0f} s per lap, peak joint speed {vmax:.1f} deg/s, "
          f"inside limits {lim_ok}, unsafe samples {len(bad)} / {len(Q) // 3 + 1}", bad[:3])
    if bad or not lim_ok:
        sys.exit("[plan] rejected — re-run with another --seed")
    out = Path(a.out or Path.home() / f"il-robot/runs/verify/sweep_{a.arm}_{time.strftime('%Y%m%dT%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"arm": a.arm, "centre": c.tolist(), "tour": P.tolist(), "knots_s": T.tolist(),
                               "rate_hz": RATE, "vmax_dps": a.vmax, "z_mm": Z_MM, "rest": {k: v.tolist() for k, v in rest.items()},
                               "t": t.tolist(), "q": Q.tolist()}, indent=0))
    print(f"[plan] → {out}")


def run(a) -> None:
    P = json.loads(Path(a.plan).read_text())
    arm, o = P["arm"], other(P["arm"])
    t_lap, Q_lap = np.array(P["t"]), np.array(P["q"])
    R = ArmRpc("127.0.0.1", DEFAULT_PORTS[arm][0])
    so = ArmRpc("127.0.0.1", DEFAULT_PORTS[o][0]).state()["state"]
    if so != "OFF":
        sys.exit(f"[run] the other arm ({o}) must be resting with torque off — it is {so}")
    stop = {"flag": False, "why": ""}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True, why="sigint"))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True, why="sigterm"))
    log = open(Path(a.plan).with_suffix(f".run_{time.strftime('%H%M%S')}.jsonl"), "w")
    tok = R.lease("sweep_refine")
    t_renew = 0.0

    def goal(q):
        nonlocal t_renew
        R.must(cmd="goal", token=tok, pos={j: float(v) for j, v in zip(JOINTS, q)})
        if time.time() - t_renew > 3.0:
            R.call(cmd="renew", token=tok); t_renew = time.time()

    def healthy() -> bool:
        s = R.state()
        log.write(json.dumps({"t": time.time(), "state": s.get("state"), "pos": [s["pos"][j] for j in JOINTS]}) + "\n")
        if s.get("state") not in ("MOVING", "HOLD"):
            stop.update(flag=True, why=f"arm state {s.get('state')}")
            return False
        return True

    def ramp(q0, q1, dps):
        n = max(1, int(np.ceil(np.abs(q1 - q0).max() / (dps / RATE))))
        for i in range(1, n + 1):
            if stop["flag"]:
                return
            s = i / n
            goal(q0 + (q1 - q0) * (10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5))
            time.sleep(1 / RATE)

    try:
        R.must(cmd="enable", token=tok)
        q_now = np.array([R.state()["pos"][j] for j in JOINTS])
        c = np.array(P["centre"])
        print(f"[run] {arm}: rest → centre ({np.abs(c - q_now).max():.0f} deg at {a.approach_dps:g} deg/s)", flush=True)
        ramp(q_now, c, a.approach_dps)
        print(f"[run] at centre — waiting for {a.wait_dir} (start rec_blobs now)", flush=True)
        t0 = time.time()
        while not stop["flag"] and not Path(a.wait_dir).exists():
            goal(c); time.sleep(1 / RATE)
            if int((time.time() - t0) * RATE) % 30 == 0 and not healthy():
                break
            if time.time() - t0 > a.wait_timeout_s:
                stop.update(flag=True, why="rec_blobs did not start")
        if not stop["flag"]:
            print(f"[run] recording started — holding {a.hold_s:g} s (GET READY + light alignment + HOLD STILL)", flush=True)
            t0 = time.time()
            while not stop["flag"] and time.time() - t0 < a.hold_s:
                goal(c); time.sleep(1 / RATE)
        if not stop["flag"]:
            print(f"[run] sweeping {a.sweep_s:g} s ({t_lap[-1]:.0f} s per lap)", flush=True)
            t0 = time.time(); k = 0
            while not stop["flag"]:
                el = time.time() - t0
                if el >= a.sweep_s and (el % t_lap[-1]) < 1.0 / RATE * 2:   # finish on a lap end (= the centre)
                    break
                q = np.array([np.interp(el % t_lap[-1], t_lap, Q_lap[:, i]) for i in range(6)])
                goal(q); k += 1
                if k % 10 == 0 and not healthy():
                    break
                time.sleep(max(0.0, t0 + k / RATE - time.time()))
        print(f"[run] sweep ended ({stop['why'] or 'done'})", flush=True)
    finally:
        try:
            q_now = np.array([R.state()["pos"][j] for j in JOINTS])
            if R.state().get("state") in ("MOVING", "HOLD"):
                stop["flag"] = False
                ramp(q_now, np.array(P["centre"]), a.approach_dps)
            R.call(cmd="hold", token=tok); time.sleep(0.5)
            off = R.call(cmd="torque_off", token=tok)
            if off.get("error") == "not_at_neutral":
                R.must(cmd="home", token=tok); t1 = time.time()
                while time.time() - t1 < 40 and R.state()["state"] == "HOMING":
                    R.call(cmd="renew", token=tok); time.sleep(0.3)
                off = R.call(cmd="torque_off", token=tok)
            print(f"[run] {arm}: home, torque {off.get('state')} {off.get('error') or ''}", flush=True)
        finally:
            R.call(cmd="release", token=tok)
            log.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--arm", choices=["left", "right"], required=True)
    p.add_argument("--n-try", type=int, default=600)
    p.add_argument("--n-way", type=int, default=8)
    p.add_argument("--vmax", type=float, default=8.0, help="deg/s, largest joint")
    p.add_argument("--min-seg-s", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default=None)
    r = sub.add_parser("run")
    r.add_argument("--plan", required=True)
    r.add_argument("--wait-dir", required=True, help="rec_blobs --out directory: the sweep starts once it exists")
    r.add_argument("--wait-timeout-s", type=float, default=300.0)
    r.add_argument("--hold-s", type=float, default=30.0)
    r.add_argument("--sweep-s", type=float, default=150.0)
    r.add_argument("--approach-dps", type=float, default=12.0)
    a = ap.parse_args()
    plan(a) if a.cmd == "plan" else run(a)


if __name__ == "__main__":
    main()
