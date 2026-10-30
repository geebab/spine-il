#!/usr/bin/env python3
"""Re-check a rollout trajectory the way the robot will play it: gate anchored at its FIRST frame with the robot start pose.

    python robot/inference/verify_traj.py traj.json [--radius-mm 120]

rollout.py --cycle checks every frame with the gate anchored at the working pose; the playback (endo_follow, relative
replay) is anchored at the first frame — the ready pose — with the joint solution found there (meta.cycle.robot_start).
This runs safety_gate.RobotGate from that anchor over all frames and prints the failures; exit 0 only if none.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
import safety_gate as G

ap = argparse.ArgumentParser(); ap.add_argument("traj"); ap.add_argument("--radius-mm", type=float, default=120.0)
a = ap.parse_args()
t = json.loads(Path(a.traj).read_text()); F = np.asarray(t["frames"], float)
qs = (t["meta"].get("cycle") or {}).get("robot_start")
if not qs:
    raise SystemExit("no meta.cycle.robot_start (run rollout.py with --cycle and --gate-pair)")
hdr = next(d for d in map(json.loads, open(t["ref_recording"])) if d.get("k") == "hdr")
G.RADIUS_M = a.radius_mm / 1000.0
g = G.RobotGate(F[0], hdr["registration"]["T0"], qs["left"], qs["right"])
bad = []
for i, v in enumerate(F):
    ok, why, q = g.check(v)
    if q:
        g.q.update(q)
    if not ok:
        bad.append([i, why[:90]])
d = [float(np.linalg.norm(F[:, o:o + 3] - F[0, o:o + 3], axis=1).max()) for o in (0, 7)]
print(json.dumps({"frames": len(F), "unsafe": len(bad), "first": bad[:5], "max_disp_mm": {"endo": round(d[0]), "drill": round(d[1])},
                  "radius_mm": a.radius_mm, "robot_start": qs}, ensure_ascii=False))
sys.exit(1 if bad else 0)
