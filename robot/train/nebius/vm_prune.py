#!/usr/bin/env python3
"""ON THE GPU MACHINE: keep the disk from filling up during a training — the optimizer state of every checkpoint older
than the newest complete one is removed (13 GB each for GR00T N1.7; a checkpoint is 24 GB with it, 12 GB without). The
model of every checkpoint stays (evaluation and serving need only pretrained_model); the newest complete checkpoint keeps
its state, so the training can still be resumed from where it is. User, 10-12: "지워" (the states of past checkpoints).

    python vm_prune.py NAME [--once]      # without --once: every minute until the training's DONE file appears
"""
import re
import shutil
import sys
import time
from pathlib import Path

name = sys.argv[1]; once = "--once" in sys.argv
out = Path.home() / "il-robot/out" / name
ck = out / "run/checkpoints"
while True:
    last = (ck / "last").resolve().name if (ck / "last").exists() else None      # LeRobot moves this link after a save is complete
    if last and re.fullmatch(r"\d{6}", last):
        for d in sorted(x for x in ck.iterdir() if x.is_dir() and not x.is_symlink() and re.fullmatch(r"\d{6}", x.name)):
            ts = d / "training_state"
            if d.name < last and ts.is_dir():
                size = sum(f.stat().st_size for f in ts.rglob("*") if f.is_file())
                shutil.rmtree(ts)
                print(time.strftime("%m-%d %T"), f"removed {ts} ({size / 2**30:.1f} GiB); newest complete checkpoint {last}", flush=True)
    if once or (out / "DONE").exists():
        break
    time.sleep(60)
