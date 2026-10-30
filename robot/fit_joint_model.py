#!/usr/bin/env python3
"""Fit the per-joint FOPDT + gravity-bias model (server ticks: written command → encoder) used by the Smith predictor.

pos[k+1] = a·pos[k] + b·cmd[k−d] + c, least squares per joint for d = 0…5, d chosen on held-out data (the last
third of the ticks, or --val runs). Writes {kind, dt_s, joints{a, b, c, delay_ticks, tau_s, gain, bias_deg}}.

  python robot/fit_joint_model.py RUN [RUN …] --out ~/il-robot/calib/joint_model_left.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot_robot_spinearm import latency as L
from lerobot_robot_spinearm.protocol import JOINTS


def series(run):
    ev = L.load(Path(run) / "events.jsonl")
    st = sorted([s for s in ev["state"] if s.get("cmd") and s.get("tr") and s["state"] == "MOVING"],
                key=lambda s: s["tick"])
    return (np.array([s["tick"] for s in st]), np.array([[s["pos"][j] for j in JOINTS] for s in st]),
            np.array([[s["cmd"][j] for j in JOINTS] for s in st]))


def rows(tick, pos, cmd, j, d, sl=slice(None)):
    X, Y = [], []
    idx = range(len(tick) - 1)[sl]
    for k in idx:
        if k < d or tick[k + 1] - tick[k] != 1 or tick[k] - tick[k - d] != d:
            continue
        X.append([pos[k, j], cmd[k - d, j], 1.0])
        Y.append(pos[k + 1, j])
    return np.array(X).reshape(-1, 3), np.array(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--val", nargs="*", default=None, help="held-out runs (default: last third of each fit run)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dt", type=float, default=1 / 30)
    a = ap.parse_args()
    S = [series(r) for r in a.runs]
    V = [series(r) for r in a.val] if a.val else None
    model = {}
    for j, name in enumerate(JOINTS):
        best = None
        for d in range(6):
            fit, val = [], []
            for s in S:
                n = len(s[0])
                cut = n if V else int(n * 2 / 3)
                fit.append(rows(*s, j, d, slice(0, cut)))
                if not V:
                    val.append(rows(*s, j, d, slice(cut, None)))
            if V:
                val = [rows(*v, j, d) for v in V]
            X = np.vstack([x for x, _ in fit])
            Y = np.concatenate([y for _, y in fit])
            Xv = np.vstack([x for x, _ in val])
            Yv = np.concatenate([y for _, y in val])
            if len(Y) < 50 or len(Yv) < 20:
                continue
            th, *_ = np.linalg.lstsq(X, Y, rcond=None)
            rv = float(np.sqrt(np.mean((Xv @ th - Yv) ** 2)))
            if best is None or rv < best[0]:
                best = (rv, d, th)
        rv, d, (aa, bb, cc) = best
        tau = -a.dt / np.log(aa) if 0 < aa < 1 else float("nan")
        model[name] = {"a": float(aa), "b": float(bb), "c": float(cc), "delay_ticks": int(d), "tau_s": float(tau),
                       "gain": float(bb / (1 - aa)), "bias_deg": float(cc / (1 - aa)), "val_resid_deg": rv}
        print(f"{name:13s} d={d} tau={tau * 1000:6.1f} ms gain={bb / (1 - aa):5.3f} bias={cc / (1 - aa):6.2f}° "
              f"val 1-step {rv:.3f}°", flush=True)
    Path(a.out).write_text(json.dumps({"kind": "spinearm-joint-fopdt", "dt_s": a.dt, "source": a.runs,
                                       "validated_on": a.val or "last third", "joints": model}, indent=1))
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
