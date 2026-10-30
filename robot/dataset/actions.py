"""Action representations for the tool-space policy (robot/dataset/README.md).

absolute (15)  action = the next frame's observation.state: endo xyz, quat wxyz, drill xyz, quat wxyz, trigger
delta    (13)  per tool: dpos = next xyz − xyz (scene mm) and drot = rotation vector of R·ᵀ·R_next (tool frame, rad),
               then the trigger (absolute, next frame). One frame moves 0.2–0.4 mm; regressing that as an absolute pose in
               workspace-normalised coordinates is hard (ACT smoke 09-30: open-loop error 10× the "stay still" baseline),
               as a delta it is the quantity itself. to_delta / from_delta are exact inverses.
chunk-relative (13 per chunk step)  the pose j+1 frames ahead relative to the current state — train/lerobot-train-rel.py;
               cum_to_steps() turns a predicted chunk into per-step deltas for the rollout.
"""
from __future__ import annotations

import numpy as np

DELTA_NAMES = [f"{tool}.{c}" for tool in ("endo", "drill") for c in ("dx", "dy", "dz", "rx", "ry", "rz")] + ["drill.trigger"]


def q2R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R2q(R: np.ndarray, ref=None) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1) * 2; q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.asarray(q, float); q /= np.linalg.norm(q)
    return -q if ref is not None and np.dot(q, ref) < 0 else q


def log_R(R: np.ndarray) -> np.ndarray:
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9:
        return np.zeros(3)
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def exp_R(w) -> np.ndarray:
    w = np.asarray(w, float); th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def to_delta(state: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    out = []
    for off in (0, 7):
        out += list(nxt[off:off + 3] - state[off:off + 3])
        out += list(log_R(q2R(state[off + 3:off + 7]).T @ q2R(nxt[off + 3:off + 7])))
    return np.asarray(out + [nxt[14]], np.float32)


def from_delta(state: np.ndarray, act: np.ndarray) -> np.ndarray:
    out = np.array(state, float)
    for off, a in ((0, 0), (7, 6)):
        out[off:off + 3] = state[off:off + 3] + act[a:a + 3]
        out[off + 3:off + 7] = R2q(q2R(state[off + 3:off + 7]) @ exp_R(act[a + 3:a + 6]), state[off + 3:off + 7])
    out[14] = float(act[12])
    return out


def cum_to_steps(chunk: np.ndarray) -> list[np.ndarray]:
    """A chunk-relative (cumulative) action chunk (T, 13) — each row the pose relative to the state the chunk was predicted
    from (train/lerobot-train-rel.py) — as per-step delta actions, so the rollout applies them like a delta policy's and the
    corrections made between steps (aiming, RCM, safety gate) are kept."""
    out, prev = [], np.zeros(13)
    for c in np.asarray(chunk, float):
        d = np.zeros(13)
        for a in (0, 6):
            d[a:a + 3] = c[a:a + 3] - prev[a:a + 3]
            d[a + 3:a + 6] = log_R(exp_R(prev[a + 3:a + 6]).T @ exp_R(c[a + 3:a + 6]))
        d[12] = c[12]
        out.append(d); prev = c
    return out
