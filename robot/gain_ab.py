#!/usr/bin/env python3
"""Servo gain A/B on both arms: the same slow test moves under each gain set, shake and hold error compared (2026-09-27).

  gain_ab.py plan                          # per arm: home → A → B → C → home, all checked in the safe zone
  gain_ab.py set --arm right --gains P32   # write a gain set (torque off first; the arm is left off, at rest)
  verify_two_arm.py run <dir> --plan plan_<arm>.json --tag <arm>_<gains>     # the moves (slow, recorded)
  gain_ab.py report <dir>                  # shake while moving (encoders 30 Hz, marker 90 Hz), hold error at stops
  gain_ab.py fold <dir>                    # plan_fold_<arm>.json: home → the most stretched safe pose → home (the big
                                           # fold home where the left arm shook, 2026-09-27)

Gain sets (servo EEPROM, gravity joints = shoulder_lift, elbow_flex, wrist_flex):
  P16  LeRobot's values, all joints P16/D32 — the right arm as set up
  P32  Feetech default, all joints P32/D32
  P40  all joints P32/D32, gravity joints P40/D48 — the left arm since 2026-09-24 (MEASURE-1 §7)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_two_arm as V  # noqa: E402

J = V.sz.JOINTS
GRAV = ("shoulder_lift", "elbow_flex", "wrist_flex")
SETS = {"P16": ({j: 16 for j in J}, {j: 32 for j in J}),
        "P32": ({j: 32 for j in J}, {j: 32 for j in J}),
        "P40": ({j: 40 if j in GRAV else 32 for j in J}, {j: 48 if j in GRAV else 32 for j in J})}
RUNS = Path.home() / "il-robot/runs/gain_ab"


def plan(a) -> None:
    rng = np.random.default_rng(a.seed)
    arms, cam, info = V.load_models()
    zone = V.sz.Zone(arms, cam)
    start = V.read_q()
    zone.set_start(start)
    bnd = {x: V.rpc(x).status()["bounds"] for x in V.PORTS}
    out_dir = RUNS / time.strftime("%Y%m%dT%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in V.PORTS:
        other = next(x for x in V.PORTS if x != arm)
        cand = []
        for _ in range(a.tries):                                 # safe (not necessarily optimal) poses, well seen
            q = np.array([rng.uniform(max(V.RANGES[j][0], bnd[arm][j][0] + 3), min(V.RANGES[j][1], bnd[arm][j][1] - 3))
                          for j in J[:5]] + [start[arm][5]])     # gripper stays where it rests
            B, n = arms[arm].balls(q)
            px = cam.pixels(cam.cam(B))
            u0, u1, v0, v1 = zone.lim.window_px
            if px[:, 0].min() < u0 + 20 or px[:, 0].max() > u1 - 20 or px[:, 2].min() < v0 + 20 or px[:, 2].max() > v1 - 20:
                continue
            qq = {arm: q, other: start[other]}
            if zone.check(qq, strict=True, see=(arm,))[0]:
                cand.append(q)
            if len(cand) >= a.pool:
                break
        # three far-apart poses that chain home → A → B → C → home with safe moves; prefer a stretched one (load)
        best = None
        for _ in range(60):
            idx = rng.choice(len(cand), 3, replace=False)
            seq = [start[arm]] + [cand[i] for i in idx] + [start[arm]]
            spread = min(np.abs(seq[i] - seq[i + 1])[:5].max() for i in range(4))
            if best is not None and spread <= best[0]:
                continue
            if all(zone.path({arm: seq[i], other: start[other]}, {arm: seq[i + 1], other: start[other]},
                             2.0, (arm,))[0] for i in range(4)):
                best = (spread, seq)
        if best is None:
            sys.exit(f"{arm}: no safe test chain found")
        seq = best[1]
        stops = [{"kind": "target", "q": {arm: q.tolist(), other: start[other].tolist()}} for q in seq[1:4]]
        stops.append({"kind": "home", "q": {x: v.tolist() for x, v in start.items()}})
        P = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "start": {k: v.tolist() for k, v in start.items()},
             "phases": [{"arm": arm, "stops": stops}], "step_deg": 2.0, "models": info,
             "camera": {"T_world_cam": cam.T_world_cam.tolist()}}
        (out_dir / f"plan_{arm}.json").write_text(json.dumps(P, indent=1))
        print(f"[gain_ab] {arm}: A/B/C = {[np.round(q, 0).astype(int).tolist() for q in seq[1:4]]} "
              f"(min move {best[0]:.0f}°)", flush=True)
    print(f"[gain_ab] → {out_dir}")


def fold(a) -> None:
    """The most stretched pose (largest move from home in lift/elbow/wrist_flex) that is safe and seen, with safe
    straight moves home → it → home."""
    rng = np.random.default_rng(a.seed)
    arms, cam, info = V.load_models()
    zone = V.sz.Zone(arms, cam)
    start = V.read_q()
    zone.set_start(start)
    bnd = {x: V.rpc(x).status()["bounds"] for x in V.PORTS}
    gi = [J.index(j) for j in GRAV]
    for arm in a.arms.split(","):
        other = next(x for x in V.PORTS if x != arm)
        cand = []
        for _ in range(a.tries):
            q = np.array([rng.uniform(max(V.RANGES[j][0], bnd[arm][j][0] + 3), min(V.RANGES[j][1], bnd[arm][j][1] - 3))
                          for j in J[:5]] + [start[arm][5]])
            fold_deg = float(np.abs(q - start[arm])[gi].sum())
            if fold_deg < 120:
                continue
            B, n = arms[arm].balls(q)
            px = cam.pixels(cam.cam(B))
            u0, u1, v0, v1 = zone.lim.window_px
            if px[:, 0].min() < u0 + 20 or px[:, 0].max() > u1 - 20 or px[:, 2].min() < v0 + 20 or px[:, 2].max() > v1 - 20:
                continue
            if zone.check({arm: q, other: start[other]}, strict=True, see=(arm,))[0]:
                cand.append((fold_deg, q))
        cand.sort(key=lambda c: -c[0])
        best = next(((f, q) for f, q in cand[:40] if zone.path(start, {arm: q, other: start[other]}, 2.0, (arm,))[0]), None)
        if best is None:
            print(f"[gain_ab] {arm}: no stretched pose reachable inside the zone")
            continue
        f, q = best
        stops = [{"kind": "target", "q": {arm: q.tolist(), other: start[other].tolist()}},
                 {"kind": "home", "q": {x: v.tolist() for x, v in start.items()}}]
        P = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "start": {k: v.tolist() for k, v in start.items()},
             "phases": [{"arm": arm, "stops": stops}], "step_deg": 2.0, "models": info}
        (Path(a.dir) / f"plan_fold_{arm}.json").write_text(json.dumps(P, indent=1))
        print(f"[gain_ab] {arm}: fold from {np.round(q, 0).astype(int).tolist()} (lift+elbow+flex {f:.0f}° from home)")


def set_gains(a) -> None:
    P, D = SETS[a.gains]
    r = V.rpc(a.arm)
    tok = r.lease("gain_ab", force=True)
    try:
        st = r.state()["state"]
        if st != "OFF":
            r.call(cmd="hold", token=tok)
            time.sleep(0.4)
            off = r.call(cmd="torque_off", token=tok)
            if off.get("error") == "not_at_neutral":
                r.must(cmd="home", token=tok)
                t0 = time.time()
                while r.state()["state"] == "HOMING" and time.time() - t0 < 30:
                    time.sleep(0.2)
                off = r.call(cmd="torque_off", token=tok)
            if r.state()["state"] != "OFF":
                sys.exit(f"torque_off refused: {off}")
        for name, vals in (("P_Coefficient", P), ("D_Coefficient", D)):
            got = r.must(cmd="set_regs", token=tok, name=name, values=vals)["regs"][name]
            if got != vals:
                sys.exit(f"{name} read back {got} ≠ {vals}")
        print(f"[gain_ab] {a.arm}: {a.gains} written (P {P}, D {D}); torque stays off")
    finally:
        r.call(cmd="release", token=tok)


def _osc(t, x):
    """rms of the part of x (n × k) that a 0.4 s moving average does not follow, per column."""
    g = np.arange(t[0], t[-1], 1 / 30.0)
    X = np.array([np.interp(g, t, x[:, i]) for i in range(x.shape[1])]).T
    k = 12
    S = np.array([np.convolve(X[:, i], np.ones(k) / k, mode="same") for i in range(X.shape[1])]).T
    return np.sqrt(((X - S)[k:-k] ** 2).mean(0))


def _band(t, P, lo, hi, fs=90.0):
    g = np.arange(t[0], t[-1], 1 / fs)
    X = np.array([np.interp(g, t, P[:, i]) for i in range(P.shape[1])]).T
    F = np.fft.rfft(X - X.mean(0), axis=0)
    f = np.fft.rfftfreq(len(g), 1 / fs)
    F[(f < lo) | (f > hi)] = 0
    return float(np.linalg.norm(np.sqrt((np.fft.irfft(F, n=len(g), axis=0) ** 2).mean(0))))


def report(a) -> None:
    d = Path(a.dir)
    rows = []
    for f in sorted(d.glob("events_*.jsonl")):
        tag, gains = f.stem.split("_")[1:3]
        test, arm = tag.split("-") if "-" in tag else ("chain", tag)
        ev = [json.loads(l) for l in open(f)]
        evs = [e for e in ev if e["k"] == "event"]
        enc = [e for e in ev if e["k"] == "enc" and e["arm"] == arm and e.get("goal")]
        rec = [e for e in evs if e.get("what") == "record"]
        done = next((e["t"] for e in evs if e.get("what") == "phase_done"), enc[-1]["t"] if enc else 0)
        lost = sum(1 for e in evs if e.get("what") == "lost")
        lag = sum(1 for e in evs if e.get("what") == "lag")
        # moving stretches = outside the dwell windows (record t0−settle … t1)
        windows = [(e["t0"] - 1500, e["t1"]) for e in rec]
        mov = [e for e in enc if not any(w0 <= e["t"] <= w1 for w0, w1 in windows)]
        t = np.array([e["t"] for e in mov]) / 1000
        x = np.array([e["pos"] for e in mov])
        g = np.array([e["goal"] for e in mov])
        osc = _osc(t, x) if len(t) > 60 else np.full(6, np.nan)
        lagm = np.abs(g - x).mean(0)
        # the home move alone: from the last record to phase_done
        th0 = rec[-1]["t1"] if rec else enc[0]["t"]
        hm = [e for e in enc if th0 + 1000 <= e["t"] <= done]
        th = np.array([e["t"] for e in hm]) / 1000
        osc_home = _osc(th, np.array([e["pos"] for e in hm])) if len(hm) > 60 else np.full(6, np.nan)
        tools = V.TOOLS[arm]
        taps = [m for m in ev if m["k"] == "tap" and m.get("tool") in tools and not m.get("lost")]
        tp = [m for m in taps if th0 + 1000 <= m["trecv"] <= done]
        vib_home = (_band(np.array([m["trecv"] for m in tp]) / 1000, np.array([m["raw_pos"] for m in tp]), 5, 15)
                    if len(tp) > 90 else float("nan"))
        tm = [m for m in taps if not any(w0 <= m["trecv"] <= w1 for w0, w1 in windows) and m["trecv"] < th0]
        vib_move = (_band(np.array([m["trecv"] for m in tm]) / 1000, np.array([m["raw_pos"] for m in tm]), 5, 15)
                    if len(tm) > 90 else float("nan"))
        # at the stops: steady error (goal − encoder) and marker jitter
        hold, jit = [], []
        for e in rec:
            w = [x_ for x_ in enc if e["t0"] <= x_["t"] <= e["t1"]]
            if w:
                hold.append(np.abs(np.array([x_["goal"] for x_ in w]) - np.array([x_["pos"] for x_ in w])).mean(0))
            s = [m for m in taps if e["t0"] <= m["trecv"] <= e["t1"]]
            if len(s) > 30:
                jit.append(float(np.linalg.norm(np.array([m["raw_pos"] for m in s]).std(0))))
        hold = np.mean(hold, 0) if hold else np.full(6, np.nan)
        rows.append((f"{test}:{arm}", gains, osc, osc_home, vib_move, vib_home, lagm, hold,
                     np.mean(jit) if jit else np.nan, len(rec), lost, lag))
    gi = [J.index(j) for j in GRAV]
    print(f"{'test:arm':10s} {'gains':5s} | {'shake moving':>12s} {'shake home':>10s} (encoder °, lift/elbow/flex rms) | "
          f"marker 5-15 Hz mm (moving / home) | lag ° | hold err ° | stop jitter mm | stops lost lag")
    for arm, gains, osc, oh, vm, vh, lg, hold, jit, n, lost, lag in sorted(rows):
        print(f"{arm:10s} {gains:5s} | {np.mean(osc[gi]):12.3f} {np.mean(oh[gi]):10.3f}"
              f"                                  | {vm:6.2f} / {vh:5.2f}                     | {np.mean(lg[gi]):5.2f}"
              f" | {np.mean(hold[gi]):9.2f} | {jit:14.3f} | {n:5d} {lost:4d} {lag:3d}")
    (d / "report.json").write_text(json.dumps(
        [{"arm": r[0], "gains": r[1], "shake_moving_deg": r[2].round(3).tolist(), "shake_home_deg": r[3].round(3).tolist(),
          "marker_vib_moving_mm": r[4], "marker_vib_home_mm": r[5], "lag_deg": r[6].round(2).tolist(),
          "hold_err_deg": r[7].round(2).tolist(), "stop_jitter_mm": r[8], "stops": r[9], "lost": r[10], "lag_events": r[11]}
         for r in rows], indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--tries", type=int, default=20000)
    p.add_argument("--pool", type=int, default=60)
    p.add_argument("--seed", type=int, default=3)
    s = sub.add_parser("set")
    s.add_argument("--arm", required=True, choices=list(V.PORTS))
    s.add_argument("--gains", required=True, choices=list(SETS))
    r = sub.add_parser("report")
    r.add_argument("dir")
    f = sub.add_parser("fold")
    f.add_argument("dir")
    f.add_argument("--tries", type=int, default=40000)
    f.add_argument("--seed", type=int, default=4)
    f.add_argument("--arms", default="left,right")
    a = ap.parse_args()
    {"plan": plan, "set": set_gains, "report": report, "fold": fold}[a.cmd](a)


if __name__ == "__main__":
    main()
