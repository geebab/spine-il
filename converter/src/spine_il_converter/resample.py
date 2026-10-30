"""Fixed-rate grid, pose interpolation (linear position, slerp rotation) and nearest-time matching."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import matrix_to_quat, quat_to_matrix, slerp
from .session import PoseTrack


@dataclass
class Resampled:
    tip: np.ndarray  # (T, 3)
    rot: np.ndarray  # (T, 3, 3)
    on: np.ndarray  # (T,) zero-order hold
    valid: np.ndarray  # (T,) bool
    bridged: np.ndarray  # (T,) bool — interpolated across missing samples (≤ max_gap)


def phase_locked_grid(anchor_ms: np.ndarray, t_min: float, t_max: float, fps: int) -> np.ndarray:
    """Grid at `fps` starting at the first anchor (capture time) inside [t_min, t_max].

    Locking the grid phase to the captures keeps image↔state offsets well inside the match tolerance.
    """
    step = 1000.0 / fps
    inside = anchor_ms[(anchor_ms >= t_min) & (anchor_ms <= t_max)]
    if inside.size == 0:
        return np.empty(0)
    start, end = inside[0], min(inside[-1], t_max)
    n = int(np.floor((end - start) / step + 1e-9)) + 1
    return start + step * np.arange(n)


def interpolate_track(track: PoseTrack, times: np.ndarray, max_gap_ms: float) -> Resampled:
    t = track.t_src
    n = len(t)
    i0 = np.clip(np.searchsorted(t, times, side="right") - 1, 0, n - 1)
    i1 = np.minimum(i0 + 1, n - 1)
    dt = t[i1] - t[i0]
    exact = times == t[i0]
    in_range = (times >= t[0]) & (times <= t[-1])
    valid = in_range & (exact | (dt <= max_gap_ms))
    w = np.where(dt > 0, (times - t[i0]) / np.where(dt > 0, dt, 1.0), 0.0)
    w = np.clip(w, 0.0, 1.0)
    tip = track.tip[i0] + w[:, None] * (track.tip[i1] - track.tip[i0])
    quats = matrix_to_quat(track.rot)
    rot = quat_to_matrix(slerp(quats[i0], quats[i1], w))
    nominal = np.median(np.diff(t)) if n > 2 else np.inf
    bridged = valid & ~exact & (dt > 2.5 * nominal)
    return Resampled(tip=tip, rot=rot, on=track.on[i0], valid=valid, bridged=bridged)


def match_nearest(query: np.ndarray, reference: np.ndarray, tol_ms: float) -> np.ndarray:
    """Index into `reference` of the nearest time for each query, or -1 if farther than tol_ms."""
    if reference.size == 0:
        return np.full(query.shape, -1, dtype=np.int64)
    order = np.argsort(reference, kind="stable")
    ref = reference[order]
    j = np.clip(np.searchsorted(ref, query), 1, len(ref) - 1) if len(ref) > 1 else np.zeros(query.shape, int)
    left = np.maximum(j - 1, 0)
    pick = np.where(np.abs(query - ref[left]) <= np.abs(ref[j] - query), left, j)
    out = order[pick]
    return np.where(np.abs(reference[out] - query) <= tol_ms, out, -1)
