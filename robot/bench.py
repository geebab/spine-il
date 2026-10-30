#!/usr/bin/env python3
"""Benchmark a follower setting on the recorded-hand replay: run it N times, append metrics to a CSV.

  python robot/bench.py --name smith2 --repeat 2 -- --smith-model … --smith-k 2.0

Everything after `--` goes to endo_follow.py (on top of BASE below). Waits for the servos to cool below --max-temp
and stops on any arm fault. Metrics are vs the human's raw motion (what the robot should reproduce).
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path

from lerobot_robot_spinearm.client import ArmRpc

HOME = Path.home()
RUNS = HOME / "il-robot/runs/latency"
CSV = RUNS / "bench.csv"
PY = str(HOME / "il-robot/.venv/bin/python")
ROBOT = Path(__file__).resolve().parent
REPLAY = RUNS / "20260923T230604_follow"
BASE = (f"--mode replay --replay-run {REPLAY} --replay-max-s 33 "
        "--start shoulder_pan=0,shoulder_lift=-60,elbow_flex=50,wrist_flex=70,wrist_roll=0,gripper=-40")
FIELDS = ["time", "name", "run", "rms", "p95", "max", "rms_after_delay", "delay_ms", "jit_x", "jit_y", "jit_z",
          "base_x", "base_y", "base_z", "cmds", "disengages", "temp_max", "args"]


def cool(rpc, max_temp: int) -> int:
    while True:
        st = rpc.call(cmd="state")
        if st.get("faults"):
            sys.exit(f"arm fault {st['faults']} — stopping")
        t = max(st.get("temp", {}).values() or [0])
        if t < max_temp:
            return t
        print(f"[bench] servo {t} °C ≥ {max_temp} — cooling 60 s", flush=True)
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--max-temp", type=int, default=48)
    ap.add_argument("--replay", default=str(REPLAY))
    ap.add_argument("--max-s", type=float, default=33)
    ap.add_argument("--ilc", type=int, default=0, help="iterative learning: each repeat learns from the previous run")
    ap.add_argument("--ilc-gain", type=float, default=0.6)
    ap.add_argument("--ilc-init", default=None, help="start the learning from this correction file")
    ap.add_argument("--ilc-q-hz", type=float, default=2.0)
    ap.add_argument("--ilc-shift-ms", type=float, default=70.0)
    ap.add_argument("--speed", type=float, default=0.2,
                    help="replay speed (default 0.2, the project default for reproduced motion); recorded in args")
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    extra = [x for x in a.rest if x != "--"]
    if not any(x.startswith("--replay-speed") for x in extra):
        extra += ["--replay-speed", f"{a.speed:g}"]   # explicit, so the CSV row says which speed it ran at
    if not any(x.startswith("--replay-interp") for x in extra):  # rows before 2026-09-27 ran zoh (no such flag)
        extra += ["--replay-interp", "linear"]
    speed = float(extra[extra.index("--replay-speed") + 1]) if "--replay-speed" in extra else a.speed
    base = BASE.replace(str(REPLAY), a.replay).replace("--replay-max-s 33", f"--replay-max-s {a.max_s}")
    rpc = ArmRpc("127.0.0.1", 5601, timeout_ms=3000)
    rows = []
    ilc_file = a.ilc_init
    n_runs = a.ilc if a.ilc > 0 else a.repeat
    for i in range(n_runs):
        temp = cool(rpc, a.max_temp)
        before = set(RUNS.glob("*_replay"))
        cmd = [PY, str(ROBOT / "endo_follow.py")] + shlex.split(base) + extra
        if ilc_file:
            cmd += ["--ilc-file", ilc_file]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=int(120 + a.max_s / speed))
        new = sorted(set(RUNS.glob("*_replay")) - before)
        if p.returncode != 0 or not new:
            print(p.stdout[-800:], p.stderr[-1500:], flush=True)
            sys.exit(f"[bench] run failed ({p.returncode})")
        run = new[-1]
        subprocess.run([PY, str(ROBOT / "latency_report.py"), str(run), "--no-plots"], capture_output=True, timeout=300)
        s = json.loads((run / "summary.json").read_text())
        e = s.get("errors_mm", {})
        h = e.get("vs_human_same_time") or {}
        jit = e.get("robot_jitter_base_xyz") or [None] * 3
        bx = e.get("base_xyz_rms") or [None] * 3
        row = {"time": time.strftime("%m-%d %H:%M"), "name": a.name, "run": run.name, "rms": h.get("rms"),
               "p95": h.get("p95"), "max": h.get("max"),
               "rms_after_delay": (e.get("vs_human_after_delay") or {}).get("rms"),
               "delay_ms": (s.get("delays_ms", {}).get("human_raw_to_robot") or {}).get("tau_ms"),
               "jit_x": jit[0], "jit_y": jit[1], "jit_z": jit[2], "base_x": bx[0], "base_y": bx[1], "base_z": bx[2],
               "cmds": s.get("n", {}).get("cmds"), "disengages": ";".join(s.get("n", {}).get("disengages", [])),
               "temp_max": temp, "args": " ".join(extra)}
        if a.ilc > 0:
            row["name"] = f"{a.name}_ilc{i}"
            nxt = str(RUNS / f"ilc_{a.name}_{i + 1}.npz")
            u = subprocess.run([PY, str(ROBOT / "ilc_update.py"), str(run), "--out", nxt, "--gain", str(a.ilc_gain),
                                "--q-hz", str(a.ilc_q_hz), "--shift-ms", str(a.ilc_shift_ms)]
                               + (["--prev", ilc_file] if ilc_file else []), capture_output=True, text=True)
            print(u.stdout.strip(), flush=True)
            ilc_file = nxt
        rows.append(row)
        new_file = not CSV.exists()
        with open(CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(f"[bench] {a.name} #{i + 1}: RMS {row['rms']} (p95 {row['p95']}, max {row['max']}), after-delay "
              f"{row['rms_after_delay']}, delay {row['delay_ms']} ms, jitter z {row['jit_z']}, {temp} °C", flush=True)
    rms = [r["rms"] for r in rows if r["rms"] is not None]
    if rms:
        print(f"[bench] {a.name}: mean RMS {sum(rms) / len(rms):.2f} over {len(rms)}", flush=True)


if __name__ == "__main__":
    main()
