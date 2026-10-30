#!/usr/bin/env python3
"""How far outside the skin can the two arms hold the tools? — the robot-feasible ready pose, per recording.

    python robot/inference/ready_sweep.py --dataset ~/il-robot/datasets/spine_rev4_step_delta [-o ready_sweep.json]

For each hand recording with a verified robot start pair (eval/pairstart/pairs_by_recording_chosen.json, anchored at the
recording's window start = its working pose), both tools are pulled back along their own shafts (rollout.retract) in 2 mm
frames and every frame goes through the inference safety gate (safety_gate.RobotGate.check: safe zone with both markers
seen, Clearance, arm distance, IK ≤ 5 mm / 10°, joint limits). Reported per recording:
  out_max_mm    the largest distance outside the skin the whole retraction stays safe (searched to --max-out-mm)
  why           what stops it there
  travel_mm     tip travel from the working pose to that ready pose (endo, drill)
  human_ready   where the hand's own ready pose (frame 0) is from that robot ready pose: tip distance and shaft angle
The recordings' own ready pose (tools resting on the flank) is outside the robot's safe zone; this is the guide for a
ready pose the robot can start from and return to.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout as RO                     # noqa: E402
import safety_gate as G                  # noqa: E402

WIN = {"3868": 17, "91cb": 15, "bef5": 17, "d340": 27, "bad1": 12, "0338": 18, "919d": 10, "6639": 14}   # pairstart/recs.py


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pairs", default=str(Path(__file__).resolve().parents[1] / "eval/pairstart/pairs_by_recording_chosen.json"))
    ap.add_argument("--max-out-mm", type=float, default=80.0)
    ap.add_argument("--step-mm", type=float, default=2.0)
    ap.add_argument("--radius-mm", type=float, default=150.0, help="replay sphere for the sweep (travel is reported)")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    G.RADIUS_M = a.radius_mm / 1000.0
    ds = Path(a.dataset).expanduser()
    pairs = json.loads(Path(a.pairs).expanduser().read_text())
    eps = [json.loads(l) for l in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()]
    portals = RO.demo_portals(ds)
    res = {}
    for ep, e in enumerate(eps):
        rid = Path(e["recording"]).stem.split("_trk_")[-1]
        if rid not in pairs or rid not in WIN:
            continue
        meta, states = RO.episode(ds, ep)
        with open(meta["recording"]) as fh:
            hdr = next(d for d in map(json.loads, fh) if d.get("k") == "hdr")
        shafts = RO.shaft_axes(meta["recording"])
        k0 = int(round(WIN[rid] * a.fps))
        work = states[k0].copy(); work[14] = 0.0
        gate = G.RobotGate(work, hdr["registration"]["T0"], pairs[rid]["q_left"], pairs[rid]["q_right"])
        full = RO.retract(work, portals, shafts, a.max_out_mm)
        skin = RO.retract(work, portals, shafts, 0.0)            # tips at the portal
        depth = {t: float(np.linalg.norm(skin[o:o + 3] - work[o:o + 3])) for t, o in (("endo", 0), ("drill", 7))}
        n = int(np.ceil(max(np.linalg.norm(full[o:o + 3] - work[o:o + 3]) for o in (0, 7)) / a.step_mm))
        out_ok, why, last = None, "", work
        for j in range(1, n + 1):
            f = work + (full - work) * (j / n)
            f[3:7], f[10:14] = work[3:7], work[10:14]
            ok, w, q = gate.check(f)
            if not ok:
                why = w
                break
            gate.q.update(q); last = f
            # distance outside the skin of the tool that is least far out
            out_ok = min(float(np.linalg.norm(f[o:o + 3] - work[o:o + 3])) - depth[t] for t, o in (("endo", 0), ("drill", 7)))
        travel = {t: round(float(np.linalg.norm(last[o:o + 3] - work[o:o + 3])), 1) for t, o in (("endo", 0), ("drill", 7))}
        human = {}
        for t, o in (("endo", 0), ("drill", 7)):
            u0 = RO.q2R(states[0][o + 3:o + 7]) @ shafts[t]; u1 = RO.q2R(last[o + 3:o + 7]) @ shafts[t]
            human[t] = {"tip_mm": round(float(np.linalg.norm(states[0][o:o + 3] - last[o:o + 3])), 1),
                        "shaft_deg": round(float(np.degrees(np.arccos(np.clip(np.dot(u0, u1), -1, 1)))), 1)}
        res[rid] = {"episode": ep, "window_from_s": WIN[rid], "depth_mm": {t: round(d, 1) for t, d in depth.items()},
                    "out_max_mm": None if out_ok is None else round(out_ok, 1), "reached_max": not why, "why": why,
                    "travel_mm": travel, "human_ready": human}
        print(rid, json.dumps(res[rid], ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
