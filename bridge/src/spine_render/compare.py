"""Shown pose vs the recording's drawn pose, per captured frame (audit M2).

`seek` returns, with each frame, the recorded `drawn` objects for that moment (sessionRecordSchemaV2T0.ts DrawnV2,
flat as written in the rows): endo {tip, viewDir, depth, roll, entry}, drill {B, u, entry, entryFallback}. The viewer's
bridge metrics say what it actually shows: drill.B, endoPose {entry, viewDir, depth, roll} (endo tip = entry +
depth·viewDir, as drawn). In `replay` the two must agree (a check that the viewer applied the drawn poses); in
`simulate` (a recording re-run through the current engine) the gap is the simulation's deviation from what the
operator saw at the same screen moment.
"""

from __future__ import annotations

import math

import numpy as np


def _vec(v) -> np.ndarray | None:
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        return None
    a = np.asarray(v, float)
    return a if np.all(np.isfinite(a)) else None


def endo_tip(p: dict | None) -> np.ndarray | None:
    """Tip of an endoscope pose record: `tip`, else entry + depth·viewDir."""
    if not isinstance(p, dict):
        return None
    tip = _vec(p.get("tip"))
    if tip is not None:
        return tip
    e, d, v = _vec(p.get("entry")), p.get("depth"), _vec(p.get("viewDir"))
    if e is None or v is None or not isinstance(d, (int, float)):
        return None
    return e + float(d) * v


def _angle_deg(a, b) -> float | None:
    a, b = _vec(a), _vec(b)
    if a is None or b is None or np.linalg.norm(a) < 1e-12 or np.linalg.norm(b) < 1e-12:
        return None
    c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def frame_deviation(metrics: dict | None, drawn: dict | None) -> dict:
    """Shown (bridge metrics) vs drawn, per tool: drillBMm, endoTipMm, endoViewDirDeg (only what both have)."""
    out: dict = {}
    if not isinstance(metrics, dict) or not isinstance(drawn, dict):
        return out
    d_drill, d_endo = drawn.get("drill"), drawn.get("endo")
    b = _vec(((metrics.get("drill") or {}).get("B")))
    rb = _vec((d_drill or {}).get("B")) if isinstance(d_drill, dict) else None
    if b is not None and rb is not None:
        out["drillBMm"] = round(float(np.linalg.norm(b - rb)), 4)
    pose = metrics.get("endoPose")
    t, rt = endo_tip(pose), endo_tip(d_endo)
    if t is not None and rt is not None:
        out["endoTipMm"] = round(float(np.linalg.norm(t - rt)), 4)
    ang = _angle_deg((pose or {}).get("viewDir") if isinstance(pose, dict) else None,
                     (d_endo or {}).get("viewDir") if isinstance(d_endo, dict) else None)
    if ang is not None:
        out["endoViewDirDeg"] = round(ang, 4)
    return out


def summarize(values: list[float]) -> dict | None:
    if not values:
        return None
    a = np.asarray(values, float)
    return {"n": int(a.size), "p50": round(float(np.percentile(a, 50)), 4), "p90": round(float(np.percentile(a, 90)), 4),
            "max": round(float(a.max()), 4)}


class DeviationTally:
    def __init__(self):
        self.values: dict[str, list[float]] = {}

    def add(self, dev: dict) -> None:
        for k, v in dev.items():
            self.values.setdefault(k, []).append(v)

    def summary(self) -> dict:
        return {k: summarize(v) for k, v in self.values.items()}
