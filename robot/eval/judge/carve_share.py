#!/usr/bin/env python3
"""Approximate "share past the midline" from a live_loop.py live.json (no voxel export needed): the removed volume gained on
each frame is booked to the drill tip's x on that frame; past the L4 midline = x < 21.9 mm (scene frame,
dataset/instructions.py MIDLINE_X_MM). Also the reach (midline − p5 x of the tip while the volume grows).
The paper's figure (26.8 % for run 17) is a voxel metric — run this on run 17 itself to see how far the proxy is from it.
    python carve_share.py live.json [--midline 21.9]"""
import json, sys
import numpy as np
p = sys.argv[1]; mid = float(sys.argv[3]) if len(sys.argv) > 3 else 21.9
d = json.load(open(p)); m = d["meta"]; F = np.asarray(d["frames"], float)
rem = m.get("removed_mm3")
if not isinstance(rem, list) or len(rem) != len(F):
    print(json.dumps({"file": p, "error": f"removed_mm3 is {type(rem).__name__} len {len(rem) if isinstance(rem, list) else '-'} vs {len(F)} frames",
                      "meta_keys": [k for k in m][:80]})); sys.exit(0)
r = np.asarray([x if x is not None else np.nan for x in rem], float); r = np.fmax.accumulate(np.nan_to_num(r))
dv = np.diff(r, prepend=0.0); x = F[:, 7]; cut = dv > 1e-6
past = dv[cut & (x < mid)].sum(); tot = dv.sum()
print(json.dumps({"file": p.split("/")[-2], "frames": len(F), "removed_mm3": round(float(r[-1]), 1),
                  "share_past_midline_proxy": round(float(past / tot), 3) if tot else None,
                  "reach_past_midline_mm": round(mid - float(np.percentile(x[cut], 5)), 1) if cut.any() else None,
                  **{k: m.get(k) for k in ("drill_end", "abort", "policy_done", "end_level", "wall_s", "drill_s") if k in m}}))
