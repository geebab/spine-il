#!/usr/bin/env python3
"""How much the arm replay changes a recorded hand motion — offline, no hardware (the numbers in REPLAY_FIDELITY.md).

  python robot/replay_budget.py <recorder v2 .jsonl> [--tool endo|drill|both] [--now-registration <json|file>]
                                [--from-s S] [--max-gap-s 0.25] [--json out.json]

Every number is a tool-tip distance in mm between what `endo_follow.py --mode replay` would command (before the
arm's own tracking error) and the recorded motion = the raw optical fixes, linearly interpolated at the command
times. The replay loop is simulated at 90 Hz (the tracker frames that drive it), random phase. The tip model is the
recording's own (hdr.toolCalib), so no robot files are needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROBOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROBOT), str(ROBOT / "lerobot_robot_spinearm")]
import endo_follow as ef  # noqa: E402  (TargetShaper, clean_replay: the replay's own code)
from lerobot_robot_spinearm import handeye as he  # noqa: E402
from lerobot_robot_spinearm import recording as rec  # noqa: E402

PRESETS = {"follow_left_smooth": 3.0, "follow_left_best": 8.0}  # robot/presets: --filter shape, --shape-hz
SETTLE_S = 1.0  # s of recording after a hole (and before it, for the zero-phase clean) left out of the stats
LOOP_HZ = 90.0
LAGS = np.arange(-0.10, 0.50, 0.005)  # s, searched for the lag-free RMS


def st(e) -> dict:
    e = np.asarray(e, float)
    return {"rms": round(float(np.sqrt((e ** 2).mean())), 3), "p50": round(float(np.percentile(e, 50)), 3),
            "p99": round(float(np.percentile(e, 99)), 3), "max": round(float(e.max()), 3)}


def tips(rp, tip) -> np.ndarray:
    return np.array([R @ tip + p for R, p in zip(rp["R"], rp["pos"])]) * 1000.0  # mm


def at(ts, X, tq) -> np.ndarray:
    return np.array([np.interp(tq, ts, X[:, k]) for k in range(X.shape[1])]).T


def shaper(hz: float):
    """The preset shaper (robot/presets/follow_left_*.args): shape, vmax 300, amax 1500, jump 8, deadband 1 mm / 1°,
    lead 60 ms."""
    return ef.TargetShaper("shape", hz, 300.0, 1500.0, 8.0, 1.0, 1.0, np.eye(3), None, None, 0.060)


def lag_free(tq, out, ts, X, lags=LAGS) -> tuple[float, float]:
    """(lag s, RMS mm) of the output against the recording shifted by the lag that fits best."""
    best = min(((lag, float(np.sqrt((np.linalg.norm(out - at(ts, X, tq - lag), axis=1) ** 2).mean())))
                for lag in lags), key=lambda z: z[1])
    return round(float(best[0]), 3), round(best[1], 3)


def budget(path: str, tool: str, speeds, T_now, rng, from_s=None, gap=rec.MAX_GAP_S) -> dict:
    rp = rec.load_recording(path, tool, t_start_s=from_s, max_gap_s=gap)
    tip, ts = rp["tip_local_m"], rp["t"]
    X = tips(rp, tip)
    H0_tip, H0_R = rp["H0_tip"], rp["H0_R"]
    c = rp["counts"]
    holes = np.flatnonzero(np.diff(ts) > gap) if gap else np.array([], int)

    def outside(tq, T, after=0.0, before=0.0):
        """No ground truth inside a hole (the recording only has its two ends), and a shaper still catches up
        after the jump: those spans are left out of the statistics (and counted)."""
        ok = np.ones(len(tq), bool)
        for k in holes:
            ok &= ~((tq > T[k] - before) & (tq < T[k + 1] + after))
        return ok

    out: dict = {"n": len(ts), "duration_s": round(float(ts[-1]), 2), "pose": rp["pose_source"],
                 "time": rp["time_source"], "vis_lt4": int(sum(v < 4 for v in rp["vis"])),
                 "gaps": {"max_ms": c["gap_max_ms"], f"over_{gap * 1000:g}ms": c["gaps_over_max"],
                          "jump_max_mm": c["gap_jump_max_mm"],
                          "left_out_s": round(float(np.diff(ts)[holes].sum()), 2), "settle_s": SETTLE_S}}

    # 1. sampling between recorded frames: zoh (the old replay) vs linear, in the replay loop (holes > gap held)
    for sp in speeds:
        T = ts / sp
        live = np.arange(rng.uniform(0, 1 / LOOP_HZ), T[-1], 1 / LOOP_HZ)
        ref = at(T, X, live)
        rps = {**rp, "t": T}
        for mode in ("zoh", "linear"):
            got = np.array([R @ tip + p for R, p, _ in (rec.sample_pose(rps, x, mode, gap / sp) for x in live)])
            out[f"sample_{mode}_x{sp:g}"] = st(np.linalg.norm(got * 1000 - ref, axis=1)[outside(live, T)])
    half = {"t": ts[::2], "pos": rp["pos"][::2], "R": rp["R"][::2], "vis": rp["vis"][::2]}
    for mode in ("zoh", "linear"):  # every other frame removed and predicted from the rest (2× the spacing; pure
        # interpolation, no hole rule)
        got = np.array([R @ tip + p for R, p, _ in (rec.sample_pose(half, x, mode) for x in ts[1::2])]) * 1000
        odd = np.arange(1, len(ts), 2)
        ok = ~(np.isin(odd - 1, holes) | np.isin(odd, holes))  # a held-out frame next to a hole has no truth
        out[f"holdout_{mode}"] = st(np.linalg.norm(got - X[1::2], axis=1)[ok])

    # 2. --replay-clean-hz: the offline zero-phase clean (endo_follow.clean_replay)
    keep = [i for i, v in enumerate(rp["vis"]) if v >= 4]
    for fc in (2.0, 4.0):  # the zero-phase clean bridges holes linearly and smears both sides
        e = np.linalg.norm(tips(ef.clean_replay(rp, fc), tip) - X[keep], axis=1)
        out[f"clean_{fc:g}hz"] = st(e[outside(ts[keep], ts, SETTLE_S, SETTLE_S)])

    # 3. the preset shaper: on the recorded frames (as the 2026-09-26 audit), then in the replay loop fed by each
    #    sampling mode — the command the IK gets
    for name, hz in PRESETS.items():
        sh, o = shaper(hz), []
        for k in range(len(ts)):
            dp, _ = sh(X[k] / 1000 - H0_tip, rp["R"][k] @ H0_R.T, rp["vis"][k], ts[k])
            o.append((dp + H0_tip) * 1000)
        o = np.array(o)
        ok = outside(ts, ts, SETTLE_S)
        lag, rms = lag_free(ts[ok], o[ok], ts, X)
        out[f"shaper_{name}_frames"] = {**st(np.linalg.norm(o - X, axis=1)[ok]), "lag_s": lag, "rms_lag_free": rms}
        for sp in speeds:
            T = ts / sp
            live = np.arange(rng.uniform(0, 1 / LOOP_HZ), T[-1], 1 / LOOP_HZ)
            ref = at(T, X, live)
            rps = {**rp, "t": T}
            for mode in ("zoh", "linear"):
                sh, o = shaper(hz), []
                for x in live:
                    R, p, vis = rec.sample_pose(rps, x, mode, gap / sp)
                    pos_rel = (lambda dRh, p=p: (dRh @ H0_R) @ tip + p - H0_tip)
                    dp, _ = sh(R @ tip + p - H0_tip, R @ H0_R.T, vis, x, pos_rel)
                    o.append((dp + H0_tip) * 1000)
                o = np.array(o)
                ok = outside(live, T, SETTLE_S / sp)
                lag, rms = lag_free(live[ok], o[ok], T, X)
                out[f"loop_{name}_{mode}_x{sp:g}"] = {**st(np.linalg.norm(o - ref, axis=1)[ok]), "lag_s": lag,
                                                      "rms_lag_free": rms}

    # 4. the relative mapping: displacements in the recording camera's axes, turned by today's R_bc
    dp = X - X[0]
    reach = np.linalg.norm(dp, axis=1)
    out["reach_from_H0_mm"] = st(reach)
    out["radius_clamp_160mm_hits"] = int((reach > 160.0).sum())
    for deg in (5.0, 11.85):  # worst case: the camera turned about an axis ⟂ the displacement
        out[f"relative_worst_{deg:g}deg"] = round(float(2 * np.sin(np.radians(deg) / 2) * reach.max()), 2)
    if T_now is not None and rp["recording"]["registration"]["dynamic"]:
        out["relative_vs_absolute"] = "dynamic registration: not computed (T0 only)"
    elif T_now is not None:
        hdr = rec.read_header(path)
        Rr, tr = rec.parse_rigid(hdr["registration"]["T0"])
        Rd = T_now[0].T @ Rr  # recording camera → today's camera, through the scene
        e = np.linalg.norm(dp @ Rd.T - dp, axis=1)
        rot = [he.R_angle_deg((Rd @ R @ H0_R.T @ Rd.T).T @ (R @ H0_R.T)) for R in rp["R"]]
        out["relative_vs_absolute"] = {"camera_turn_deg": round(he.R_angle_deg(Rd), 2),
                                       "camera_shift_mm": round(float(np.linalg.norm(T_now[1] - tr)), 1),
                                       "tip_mm": st(e), "rotation_deg": st(rot)}

    # 5. raw vs filtered: the loader's pose choice (rigid tips), and the tracker's separately filtered tip field
    rf = rec.load_recording(path, tool, pose="filtered", t_start_s=from_s)
    if len(rf["t"]) == len(ts) and np.allclose(rf["t"], ts):
        out["raw_vs_filtered_rigid"] = st(np.linalg.norm(tips(rf, tip) - X, axis=1))
    _, rows, _ = rec.read_rows(path)
    tf, lo = rp["time_source"], rp["t0_ms"]
    hi = lo + float(ts[-1]) * 1000.0 + 1e-6  # the loaded window
    fr = [d for d in rows if d.get("k") == "f" and (d.get("tool") or "endo") == tool and not d.get("lost")
          and not d.get("hold") and d.get("tip") and isinstance(d.get(tf), (int, float)) and lo <= d[tf] <= hi]
    if fr:
        field = np.array([d["tip"] for d in fr])
        rigid = np.array([rec.quat_to_R(d["quat"]) @ (tip * 1000) + d["pos"] for d in fr])
        out["filtered_tip_field_vs_rigid"] = st(np.linalg.norm(field - rigid, axis=1))
        rw = [(k, d["raw"]) for k, d in enumerate(fr) if isinstance(d.get("raw"), dict) and d["raw"].get("quat")]
        if rw:  # what the bridge renders (the tip field) vs what the replay now plays (raw rigid tip)
            raw_tip = np.array([rec.quat_to_R(r["quat"]) @ (tip * 1000) + r["pos"] for _, r in rw])
            out["raw_vs_filtered_tip_field"] = st(np.linalg.norm(raw_tip - field[[k for k, _ in rw]], axis=1))
    return out


def show(tool: str, b: dict) -> None:
    f = lambda s: f"{s['rms']:.2f} / {s['p99']:.2f} / {s['max']:.2f}"  # noqa: E731
    print(f"\n== {tool}: {b['n']} frames, {b['duration_s']} s, pose {b['pose']}, time {b['time']} "
          f"(RMS / p99 / max mm)")
    for k, v in b.items():
        if isinstance(v, dict) and "rms" in v:
            extra = f"   lag {v['lag_s'] * 1000:.0f} ms → {v['rms_lag_free']:.2f}" if "lag_s" in v else ""
            print(f"  {k:<40} {f(v)}{extra}")
        elif isinstance(v, dict):
            print(f"  {k:<40} {json.dumps(v)}")
        elif k not in ("n", "duration_s", "pose", "time"):
            print(f"  {k:<40} {v}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", nargs="?", default=os.environ.get("SPINE_REC_SAMPLE"),
                    help="a recorder v2 .jsonl (default $SPINE_REC_SAMPLE)")
    ap.add_argument("--tool", choices=["endo", "drill", "both"], default="both")
    ap.add_argument("--now-registration", default=None,
                    help="today's T_scene←cam (as endo_follow --replay-registration): the relative-mode error of the "
                         "camera move between the recording and today")
    ap.add_argument("--speeds", default="1,0.2", help="replay speeds (endo_follow --replay-speed; default 0.2)")
    ap.add_argument("--from-s", type=float, default=None, help="as endo_follow --replay-from-s (H0 moves with it)")
    ap.add_argument("--max-gap-s", type=float, default=rec.MAX_GAP_S, help="as endo_follow --replay-max-gap-s")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="write every number here")
    a = ap.parse_args()
    if not a.recording:
        sys.exit("a recording (or SPINE_REC_SAMPLE) is needed")
    T_now = rec.load_registration(a.now_registration)
    speeds = [float(s) for s in a.speeds.split(",")]
    res = {"recording": a.recording, "speeds": speeds, "loop_hz": LOOP_HZ, "seed": a.seed, "from_s": a.from_s,
           "max_gap_s": a.max_gap_s, "numpy": np.__version__, "tools": {}}
    t0 = time.time()
    for tool in (("endo", "drill") if a.tool == "both" else (a.tool,)):
        res["tools"][tool] = budget(a.recording, tool, speeds, T_now, np.random.default_rng(a.seed), a.from_s,
                                    a.max_gap_s)
        show(tool, res["tools"][tool])
    res["wall_s"] = round(time.time() - t0, 1)
    print(f"\n({res['wall_s']} s)")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
