#!/usr/bin/env python3
"""Iterative learning control for replaying one recorded trajectory: next correction from a replay run.

L_next(t) = Q[ L(t) + γ · e(t + δ) ],  e = recorded (raw human) target − observed arm tip, camera frame.
δ shifts the error by the arm's lag (so the correction is applied early enough), Q is a zero-phase low-pass
(forward–backward first order) that keeps the learning from amplifying noise. Standard P-type ILC with a Q filter.

  python robot/ilc_update.py RUN --prev prev.npz --out next.npz --gain 0.6 --q-hz 2 --shift-ms 70
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from lerobot_robot_spinearm import latency as L


def zero_phase_lp(x: np.ndarray, hz: float, dt: float) -> np.ndarray:
    a = 1.0 - np.exp(-2 * np.pi * hz * dt)
    y = x.copy()
    for i in range(1, len(y)):
        y[i] = y[i - 1] + a * (y[i] - y[i - 1])
    for i in range(len(y) - 2, -1, -1):
        y[i] = y[i + 1] + a * (y[i] - y[i + 1])
    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--prev", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gain", type=float, default=0.6)
    ap.add_argument("--q-hz", type=float, default=2.0)
    ap.add_argument("--shift-ms", type=float, default=70.0)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--max-mm", type=float, default=30.0)
    ap.add_argument("--clip-mm", type=float, default=15.0, help="errors larger than this are glitches, not learned")
    a = ap.parse_args()
    ev = L.load(Path(a.run) / "events.jsonl")
    t0 = ev["engage"][0]["t"]
    cmds = [c for c in ev["cmd"] if c.get("t_rel") is not None]
    tc = np.array([c["t_rel"] for c in cmds])
    target = np.array([c["target_raw_tip_cam_mm"] for c in cmds]) / 1000.0
    role = {f["fn"]: f["robot"] for f in ev["frame"]}
    rob = sorted((m for m in ev["tap"] if role.get(m.get("fn")) == m["tool"] and not m.get("lost")),
                 key=lambda m: m["ts"])
    tr = np.array([(m["ts"] - t0) / 1000.0 for m in rob])
    xr = np.array([m["raw_tip"] for m in rob]) / 1000.0
    grid = np.arange(0.0, tc.max(), a.dt)
    tgt = np.array([np.interp(grid, tc, target[:, i]) for i in range(3)]).T
    obs = L.interp3(grid * 1000.0, tr * 1000.0, xr * 1000.0) / 1000.0  # NaN across marker gaps
    e = tgt - obs
    ok = np.isfinite(e).all(1)
    ok &= np.linalg.norm(np.where(np.isfinite(e), e, 0.0), axis=1) * 1000 <= a.clip_mm  # tracking glitches
    e[~ok] = 0.0
    shift = int(round(a.shift_ms / 1000.0 / a.dt))
    e_sh = np.vstack([e[shift:], np.zeros((shift, 3))])  # e(t + δ)
    prev = np.zeros_like(tgt)
    if a.prev:
        p = np.load(a.prev)
        prev = np.array([np.interp(grid, p["t"], p["corr"][:, i]) for i in range(3)]).T
    new = prev + a.gain * e_sh
    new = np.array([zero_phase_lp(new[:, i], a.q_hz, a.dt) for i in range(3)]).T
    n = np.linalg.norm(new, axis=1, keepdims=True)
    new = np.where(n * 1000 > a.max_mm, new * (a.max_mm / 1000.0 / np.maximum(n, 1e-9)), new)
    np.savez(a.out, t=grid, corr=new)
    print(f"[ilc] {Path(a.run).name}: error RMS {np.sqrt((e[ok] ** 2).sum(1).mean()) * 1000:.2f} mm → correction RMS "
          f"{np.sqrt((new ** 2).sum(1).mean()) * 1000:.2f} mm (max {np.linalg.norm(new, axis=1).max() * 1000:.1f}) → {a.out}")


if __name__ == "__main__":
    main()
