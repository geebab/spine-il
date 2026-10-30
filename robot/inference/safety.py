"""Per-frame motion limits for policy output before it reaches the viewer or the robot.

A chunked policy (ACT, GR00T: 40 steps per chunk) jumps where one chunk ends and the next begins — the ACT smoke rollout
(09-30) moved the drill tip up to 31 mm in one 1/15 s frame at every 40th step. `limit_step` moves each tool at most
`max_mm` (tip) and `max_deg` (rotation) from its previous pose per frame; at 15 fps the defaults (2 mm, 3°) are 30 mm/s and
45 °/s — the FE rev.4 drill feed limit is 30 mm/s. Frame layout = the dataset's 15-vector (robot/dataset/README.md).
"""
from __future__ import annotations

import numpy as np


def _slerp(q0: np.ndarray, q1: np.ndarray, f: float) -> np.ndarray:
    q0, q1 = q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + f * (q1 - q0)
        return q / np.linalg.norm(q)
    th = np.arccos(np.clip(d, -1, 1))
    return (np.sin((1 - f) * th) * q0 + np.sin(f * th) * q1) / np.sin(th)


def limit_step(prev: np.ndarray, nxt: np.ndarray, max_mm: float = 2.0, max_deg: float = 3.0) -> tuple[np.ndarray, bool]:
    """The next frame with each tool's tip move and rotation clamped relative to `prev`; (frame, clamped?)."""
    out, hit = np.array(nxt, float), False
    for off in (0, 7):
        d = out[off:off + 3] - prev[off:off + 3]
        n = float(np.linalg.norm(d))
        if n > max_mm:
            out[off:off + 3] = prev[off:off + 3] + d * (max_mm / n); hit = True
        q0, q1 = np.asarray(prev[off + 3:off + 7], float), out[off + 3:off + 7]
        ang = 2 * np.degrees(np.arccos(np.clip(abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)))), 0, 1)))
        if ang > max_deg:
            out[off + 3:off + 7] = _slerp(q0, q1, max_deg / ang); hit = True
        else:
            out[off + 3:off + 7] = q1 / np.linalg.norm(q1)
    return out, hit


def step_size(prev: np.ndarray, nxt: np.ndarray) -> list[float]:
    """[endoscope mm, endoscope deg, drill mm, drill deg] of the step prev → nxt — what limit_step measures against its limits."""
    out = []
    for off in (0, 7):
        q0, q1 = np.asarray(prev[off + 3:off + 7], float), np.asarray(nxt[off + 3:off + 7], float)
        out += [float(np.linalg.norm(np.asarray(nxt[off:off + 3], float) - np.asarray(prev[off:off + 3], float))),
                float(2 * np.degrees(np.arccos(np.clip(abs(float(np.dot(q0 / np.linalg.norm(q0), q1 / np.linalg.norm(q1)))), 0, 1))))]
    return [round(x, 2) for x in out]


def limit_traj(frames: np.ndarray, max_mm: float = 2.0, max_deg: float = 3.0) -> tuple[np.ndarray, int]:
    """Whole trajectory, frame by frame from the first; (limited frames, number of clamped frames)."""
    out = [np.asarray(frames[0], float)]
    n = 0
    for f in frames[1:]:
        g, hit = limit_step(out[-1], np.asarray(f, float), max_mm, max_deg)
        out.append(g); n += hit
    return np.asarray(out), n
