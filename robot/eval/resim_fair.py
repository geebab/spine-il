"""Tick-rate matched simulation: the hand recording at 15 Hz (= the live page) vs the robot session at 15·speed Hz (the
same ticks per second of RECORDING time), plus IoU/ext metrics of each vs the hand's recorded cut (hrec cache)."""
import json, sys, glob, numpy as np, iou_eval as E, metrics_ext as X
rob, speed = sys.argv[1], float(sys.argv[2])
hr = [f for f in glob.glob(str(E.OUT / "iou_cache_hrec_*.npz"))]
# the 3868 17–47 hand recorded set = the cache whose final count matches 434.2 mm³
Hr = None
for f in hr:
    d = np.load(f, allow_pickle=True)
    if abs(len(d["final"]) * 0.512 - float(sys.argv[3])) < 0.3:
        Hr = {"hdr": json.loads(str(d["hdr"])), "final": d["final"]}
for hz in (15.0 * speed, 15.0):
    v = E.voxels(E.render(rob, {**E.OPTS, "tick_hz": hz}))
    j, dc, *_ = E.iou(v["final"], Hr["final"])
    print(json.dumps({"robot_tick_hz": hz, "mm3": round(len(v["final"]) * 0.512, 1), "iou_vs_hand_rec": round(j, 3),
                      **X.ext_metrics(v["final"], Hr["final"], Hr["hdr"])}), flush=True)
