"""Latency / accuracy analysis of an `endo_follow.py` run (pure functions over the logged events).

Clocks: tracker and follower stamp wall-clock ms (time.time()); the arm server stamps CLOCK_MONOTONIC s. The
follower logs `clock` rows (wall − mono) so server times map onto wall time. Positions are mm in the tracker
camera frame unless named `_base`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(path: Path | str) -> dict[str, list[dict]]:
    ev: dict[str, list[dict]] = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            ev[d["k"]].append(d)
    return ev


def stats(v) -> dict | None:
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not v.size:
        return None
    return {"n": int(v.size), "median": round(float(np.median(v)), 2), "p95": round(float(np.percentile(v, 95)), 2),
            "max": round(float(v.max()), 2), "mean": round(float(v.mean()), 2)}


def interp3(t_q, t, x) -> np.ndarray:
    """Linear interpolation of (N,3) samples at t_q; NaN outside [t0, t1] or across gaps > 50 ms."""
    t, x, t_q = np.asarray(t, float), np.asarray(x, float), np.asarray(t_q, float)
    out = np.full((t_q.size, x.shape[1]), np.nan)
    if t.size < 2:
        return out
    i = np.searchsorted(t, t_q)
    ok = (i > 0) & (i < t.size)
    i1 = np.clip(i, 1, t.size - 1)
    i0 = i1 - 1
    gap = t[i1] - t[i0]
    ok &= gap <= 50.0
    w = np.where(gap > 0, (t_q - t[i0]) / np.where(gap > 0, gap, 1), 0.0)
    out[ok] = (x[i0] + (x[i1] - x[i0]) * w[:, None])[ok]
    return out


def smooth(x, k: int = 5) -> np.ndarray:
    """Centred moving average over k samples (zero phase — does not shift τ); edges use the shorter window."""
    x = np.asarray(x, float)
    if k <= 1 or len(x) < k:
        return x
    c = np.cumsum(np.vstack([np.zeros((1, x.shape[1])), x]), axis=0)
    h = k // 2
    i = np.arange(len(x))
    lo, hi = np.clip(i - h, 0, len(x)), np.clip(i + h + 1, 0, len(x))
    return (c[hi] - c[lo]) / (hi - lo)[:, None]


def delay_fit(t_ref, x_ref, t_obs, x_obs, max_ms: float = 600.0, demean: bool = True, k_smooth: int = 5) -> dict:
    """τ that best aligns obs(t + τ) with ref(t): least RMS over a 1 ms grid, parabolic refinement.

    demean removes a constant offset first (calibration bias should not bias τ). Both series are smoothed with
    the same centred window first: both markers come from the same 90 fps frames, so obs(t_ref + τ) is
    interpolated at a phase that repeats every frame, and the interpolation's noise averaging would otherwise
    pull τ toward mid-frame (≈1 ms bias at 0.2 mm noise). Returns τ, the RMS at τ and at 0, the motion content
    (ref std) and the number of samples used.
    """
    t_ref, x_ref = np.asarray(t_ref, float), smooth(x_ref, k_smooth)
    x_obs = smooth(x_obs, k_smooth)
    taus = np.arange(0.0, max_ms + 1.0, 1.0)
    rms = np.full(taus.size, np.nan)
    for k, tau in enumerate(taus):
        y = interp3(t_ref + tau, t_obs, x_obs)
        m = np.isfinite(y).all(1)
        if m.sum() < 20:
            continue
        a, b = x_ref[m], y[m]
        if demean:
            a, b = a - a.mean(0), b - b.mean(0)
        rms[k] = float(np.sqrt(((b - a) ** 2).sum(1).mean()))
    if not np.isfinite(rms).any():
        return {"tau_ms": None}
    k = int(np.nanargmin(rms))
    tau = taus[k]
    if 0 < k < taus.size - 1 and np.isfinite(rms[k - 1]) and np.isfinite(rms[k + 1]):
        d = rms[k - 1] - 2 * rms[k] + rms[k + 1]
        if d > 0:
            tau += 0.5 * (rms[k - 1] - rms[k + 1]) / d
    return {"tau_ms": round(float(tau), 1), "rms_at_tau": round(float(rms[k]), 3),
            "rms_at_0": round(float(rms[0]), 3) if np.isfinite(rms[0]) else None,
            "motion_std_mm": round(float(np.linalg.norm(x_ref.std(0))), 2), "n": int(len(t_ref)),
            "edge": bool(k == taus.size - 1)}


def err_stats(e: np.ndarray) -> dict | None:
    e = np.asarray(e, float)
    e = e[np.isfinite(e).all(1)] if e.ndim == 2 else e[np.isfinite(e)]
    if not len(e):
        return None
    n = np.linalg.norm(e, axis=1) if e.ndim == 2 else np.abs(e)
    out = {"n": int(len(n)), "rms": round(float(np.sqrt((n ** 2).mean())), 2),
           "median": round(float(np.median(n)), 2), "p95": round(float(np.percentile(n, 95)), 2),
           "max": round(float(n.max()), 2)}
    if e.ndim == 2:
        out["bias_xyz"] = np.round(e.mean(0), 2).tolist()
    return out


def still_windows(t, x, speed_mm_s: float = 5.0, min_s: float = 0.5) -> list[tuple[float, float]]:
    """Intervals where |dx/dt| stays below speed_mm_s for at least min_s (t ms, x (N,3) mm)."""
    t, x = np.asarray(t, float), np.asarray(x, float)
    if t.size < 3:
        return []
    v = np.linalg.norm(np.gradient(x, t / 1000.0, axis=0), axis=1)
    slow = v < speed_mm_s
    out, s = [], None
    for i, f in enumerate(slow):
        if f and s is None:
            s = i
        if (not f or i == len(slow) - 1) and s is not None:
            e = i if f else i - 1
            if t[e] - t[s] >= min_s * 1000:
                out.append((t[s], t[e]))
            s = None
    return out


def step_metrics(t_cmd, x_cmd, t_obs, x_obs, axis: int, t_change: float, t_end: float, amp: float) -> dict:
    """One step along `axis` commanded at t_change (wall ms): dead time (obs moves > 1 mm toward the new target),
    10–90 % rise, overshoot, settling to ±1 mm, steady-state error (last 300 ms)."""
    t_obs, x_obs = np.asarray(t_obs, float), np.asarray(x_obs, float)
    m = (t_obs >= t_change - 100) & (t_obs <= t_end)
    to, xo = t_obs[m], x_obs[m, axis]
    if to.size < 10:
        return {}
    pre = xo[to < t_change]
    x0 = float(np.median(pre)) if pre.size else float(xo[0])
    tgt = float(np.asarray(x_cmd)[np.searchsorted(np.asarray(t_cmd), t_end - 1) - 1, axis])
    d = tgt - x0
    if abs(d) < 0.3 * amp:
        return {}
    s = np.sign(d)
    prog = (xo - x0) * s
    aft = to >= t_change

    def first(cond):
        idx = np.nonzero(aft & cond)[0]
        return float(to[idx[0]] - t_change) if idx.size else None

    t10, t90 = first(prog >= 0.1 * abs(d)), first(prog >= 0.9 * abs(d))
    tail = xo[to >= t_end - 300]
    ss = float(np.mean(tail) - tgt) if tail.size else None
    within = np.abs(xo - tgt) <= 1.0
    settle = None
    for i in range(len(to)):
        if to[i] >= t_change and within[i:].all():
            settle = float(to[i] - t_change)
            break
    r1 = (lambda v: None if v is None else round(v, 1))
    return {"dead_ms": r1(first(prog >= 1.0)), "rise_ms": r1((t90 - t10) if t10 is not None and t90 is not None else None),
            "overshoot_pct": round(float(max(0.0, prog[aft].max() - abs(d)) / abs(d) * 100), 1),
            "settle_ms": r1(settle), "ss_err_mm": round(ss, 2) if ss is not None else None,
            "size_mm": round(abs(d), 1)}


def sine_fit(t_ms, x, f_hz):
    """Amplitude and phase (deg) of the f_hz component (least squares a·sin + b·cos + c)."""
    t = np.asarray(t_ms, float) / 1000.0
    A = np.column_stack([np.sin(2 * np.pi * f_hz * t), np.cos(2 * np.pi * f_hz * t), np.ones_like(t)])
    (a, b, _), *_ = np.linalg.lstsq(A, np.asarray(x, float), rcond=None)
    return float(np.hypot(a, b)), float(np.degrees(np.arctan2(b, a)))
