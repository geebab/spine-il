#!/usr/bin/env python3
"""Download the robot-run files the judge scripts need from the Hugging Face dataset (plain HTTPS, stdlib only):

    robot_runs/sessions/<run 17 and run 19 recorded sessions>.jsonl   ~36 MB + ~38 MB
    robot_runs/eval_cache/iou_cache_rofull_<session>.npz             the evaluation's removed voxels (to compare with)
    robot_runs/run17_live/live.json + frames/*.png                   what the policy saw in run 17 (8 drilling frames)

    python robot/eval/judge/fetch_robot_runs.py            → $SPINE_IL_DATA (default: <repo>/data/robot_runs)"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, HF_BASE, RUN17, RUN19  # noqa: E402

FRAMES = [116, 412, 708, 1004, 1308, 1604, 1900, 2196]
FILES = ([f"sessions/{s}.jsonl" for s in (RUN17, RUN19)] + [f"eval_cache/iou_cache_rofull_{s}.npz" for s in (RUN17, RUN19)]
         + ["run17_live/live.json"] + [f"run17_live/frames/{f:05d}.png" for f in FRAMES])

for rel in FILES:
    dst = DATA / rel
    if dst.exists() and dst.stat().st_size > 0:
        print("have", rel); continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    print("get ", rel, flush=True)
    urllib.request.urlretrieve(f"{HF_BASE}/{rel}", dst)
print("data in", DATA)
