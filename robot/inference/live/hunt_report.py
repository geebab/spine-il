#!/usr/bin/env python3
"""How much the arms' tips hunt in a closed-loop run, from the arms' own logs (<latency-dir>/<run>_<tool>/events.jsonl).

    python robot/inference/live/hunt_report.py RUN [RUN ...] [--latency-dir DIR] [--json OUT] [--until-s S]

A joint of these arms stays put until its servo error is a few tenths of a degree and then lets go by 2-5 encoder steps
(0.18-0.44°, 1-3 mm at the tip). While it is stuck both integrators of endo_follow.py keep collecting the same error — the
joint one (sag_I) and the camera one (vis_I) — and when the joint lets go the tip overshoots by what vis_I collected:
with a still target the tip wanders 1.0-1.6 mm (rms) at about 2 mm/s (10-05, preset j, runs D_*).

holds  (target still for > 4 s; the first 1.5 s left out):
    hold_radius_mm   rms distance of the measured tip from its own mean      hold_speed_mms   measured tip path / time
    hold_offset_mm   |mean tip − target|                                     vis_I_swing_mm, sag_I_swing_deg   range in a hold
slow motion  (target path < 0.45 mm per policy frame = policy chunks; one frame = 0.2 s of wall time at 0.33x):
    path_ratio       measured tip path / target path, both sampled once per frame
    motion_err_sd_mm scatter of (tip − the target 150 ms earlier) about its mean
All values are medians over the holds / motions of a run, per arm.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(d: Path, tool: str):
    tt, tip, ct, tg, vi, sg = [], [], [], [], [], []
    for l in open(d / "events.jsonl"):
        if l.startswith('{"k":"tap"'):
            m = json.loads(l)
            if (tool == "drill") != (m.get("tool") == "drill") or m.get("tip") is None or int(m.get("vis", 4)) < 4:
                continue
            tt.append(m["ts"]); tip.append(m["tip"])
        elif l.startswith('{"k":"cmd"'):
            m = json.loads(l)
            ct.append(m["trecv"]); tg.append(m["target_tip_cam_mm"]); vi.append(m["vis_I_mm"]); sg.append(m["sag_I"])
    return [np.asarray(x, float) for x in (tt, tip, ct, tg, vi, sg)]


def report(d: Path, tool: str, frame_ms: float = 200.0, until_s: float | None = None) -> dict | None:
    tt, tip, ct, tg, vi, sg = load(d, tool)
    if len(ct) < 100 or len(tt) < 100:
        return None
    if until_s:   # only the first part of a run (e.g. before the arms lost their lease)
        n = int(np.searchsorted(ct, ct[0] + until_s * 1000.0))
        ct, tg, vi, sg = ct[:n], tg[:n], vi[:n], sg[:n]
    still = np.linalg.norm(np.diff(tg, axis=0), axis=1) < 1e-3
    edges = np.concatenate([[0], np.flatnonzero(~still) + 1, [len(tg)]])
    H = []
    for a, b in zip(edges[:-1], edges[1:]):
        if ct[b - 1] - ct[a] < 4000:
            continue
        t0, t1 = ct[a] + 1500, ct[b - 1]
        i0, i1 = np.searchsorted(tt, [t0, t1])
        if i1 - i0 < 80:
            continue
        e = tip[i0:i1] - tg[a]; a2 = np.searchsorted(ct, t0)
        H.append([np.sqrt(((e - e.mean(0)) ** 2).sum(1).mean()), np.linalg.norm(np.diff(e, axis=0), axis=1).sum() / ((t1 - t0) / 1000),
                  np.linalg.norm(e.mean(0)), np.linalg.norm(np.ptp(vi[a2:b], axis=0)), np.abs(np.ptp(sg[a2:b], axis=0)).max(), (t1 - t0) / 1000])
    M, mv = [], np.flatnonzero(~still)
    for s in (np.split(mv, np.flatnonzero(np.diff(mv) > 30) + 1) if len(mv) else []):   # > 30 cmd rows (~0.6 s) apart = two motions
        a, b = s[0], s[-1] + 1
        if ct[b - 1] - ct[a] < 2500:
            continue
        g = np.arange(ct[a], ct[b - 1], frame_ms)
        jt = np.clip(np.searchsorted(tt, g), 0, len(tt) - 1); jc = np.clip(np.searchsorted(ct, g - 150.0), 0, len(ct) - 1)
        e = tip[jt] - tg[jc]
        M.append([np.linalg.norm(np.diff(tip[jt], axis=0), axis=1).sum(), np.linalg.norm(np.diff(tg[jc], axis=0), axis=1).sum(),
                  np.sqrt(((e - e.mean(0)) ** 2).sum(1).mean()), len(g)])
    H, M = np.array(H), np.array(M)
    out = {"tool": tool, "holds": len(H)}
    if len(H):
        md = np.median(H, axis=0)
        out.update(hold_radius_mm=md[0], hold_speed_mms=md[1], hold_offset_mm=md[2], vis_I_swing_mm=md[3], sag_I_swing_deg=md[4], hold_s=md[5])
    slow = M[M[:, 1] / M[:, 3] < 0.45] if len(M) else M
    if len(slow):
        out.update(slow_motions=len(slow), path_ratio=slow[:, 0].sum() / slow[:, 1].sum(), meas_mm_per_frame=slow[:, 0].sum() / slow[:, 3].sum(),
                   target_mm_per_frame=slow[:, 1].sum() / slow[:, 3].sum(), motion_err_sd_mm=float(np.median(slow[:, 2])))
    return {k: (round(float(v), 3) if isinstance(v, (float, np.floating)) else v) for k, v in out.items()}


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("runs", nargs="+"); ap.add_argument("--latency-dir", default="~/il-robot/runs/latency"); ap.add_argument("--json", default=None)
ap.add_argument("--until-s", type=float, default=None, help="use only this many seconds from the first goal")
a = ap.parse_args()
res = {}
for run in a.runs:
    run = Path(run).name
    res[run] = [r for r in (report(Path(a.latency_dir).expanduser() / f"{run}_{t}", t, until_s=a.until_s) for t in ("endo", "drill")) if r]
    print(run)
    for r in res[run]:
        print("  " + "  ".join(f"{k} {v}" for k, v in r.items()))
if a.json:
    Path(a.json).write_text(json.dumps(res, indent=1))
