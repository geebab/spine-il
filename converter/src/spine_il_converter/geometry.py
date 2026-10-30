"""Pure-numpy geometry shared by the converter and (later) the robot client.

Conventions (docs: CONVERTER-1 contract):
- Scene/world coordinates are LPS millimetres: +x Left, +y Posterior, +z Superior (cranial).
- A rotation matrix maps tool-frame vectors into its parent frame; its columns are the tool axes.
- Tool frame: z = shaft direction pointing toward the tip, x = "up" reference orthogonalised to z,
  y = z × x.
- rot6d = first two columns of the rotation matrix, column-major: [R00, R10, R20, R01, R11, R21].
- Quaternions are (w, x, y, z).
"""

from __future__ import annotations

import numpy as np

SUPERIOR = np.array([0.0, 0.0, 1.0])
# Up references tried in order when the provided one is (nearly) parallel to the shaft: S, A, L.
_FALLBACK_UPS = (SUPERIOR, np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
_DEGENERATE = 1e-6


class GeometryError(ValueError):
    pass


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < _DEGENERATE):
        raise GeometryError("zero-length vector")
    return v / n


def basal_frame(entry, target) -> tuple[np.ndarray, np.ndarray]:
    """PLAN-0 §6.3: origin = skin entry, z = entry→target, x = cranial (+S) projected ⟂ z, y = z × x.

    Returns (origin, R) where R's columns are the basal x, y, z axes in scene coordinates.
    """
    entry = np.asarray(entry, dtype=np.float64)
    z = normalize(np.asarray(target, dtype=np.float64) - entry)
    x = SUPERIOR - SUPERIOR.dot(z) * z
    if np.linalg.norm(x) < _DEGENERATE:
        raise GeometryError("basal vector is parallel to the cranial axis")
    x = x / np.linalg.norm(x)
    return entry, np.column_stack([x, np.cross(z, x), z])


def points_to_frame(points, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    """p_frame = Rᵀ (p − origin), for (..., 3) arrays."""
    return (np.asarray(points, dtype=np.float64) - origin) @ R


def points_from_frame(points, origin: np.ndarray, R: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=np.float64) @ R.T + origin


def tool_rotation(direction, up=None) -> np.ndarray:
    """Rotation(s) whose z axis is `direction` and x axis is `up` orthogonalised to it.

    Accepts (3,) or (N, 3). Rows whose up is missing or parallel to the shaft fall back to S, A, L.
    """
    single = np.ndim(direction) == 1
    z = normalize(np.atleast_2d(direction))
    ups = np.broadcast_to(SUPERIOR, z.shape) if up is None else np.atleast_2d(np.asarray(up, dtype=np.float64))
    x = ups - np.sum(ups * z, axis=1, keepdims=True) * z
    norms = np.linalg.norm(x, axis=1)
    for fallback in _FALLBACK_UPS:
        bad = norms < _DEGENERATE
        if not bad.any():
            break
        x[bad] = fallback - (z[bad] @ fallback)[:, None] * z[bad]
        norms = np.linalg.norm(x, axis=1)
    x = x / norms[:, None]
    R = np.stack([x, np.cross(z, x), z], axis=-1)
    return R[0] if single else R


def rot6d(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def rot6d_to_matrix(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    b1 = normalize(v[..., 0:3])
    a2 = v[..., 3:6]
    b2 = normalize(a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-1)


def matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix (..., 3, 3) → unit quaternion (..., 4) as (w, x, y, z), w ≥ 0."""
    R = np.asarray(R, dtype=np.float64)
    m = R.reshape(-1, 3, 3)
    m00, m11, m22 = m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]
    t = np.stack([1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], axis=1)
    k = np.argmax(t, axis=1)
    s = 2.0 * np.sqrt(np.maximum(t[np.arange(len(k)), k], 1e-300))
    d21, d02, d10 = m[:, 2, 1] - m[:, 1, 2], m[:, 0, 2] - m[:, 2, 0], m[:, 1, 0] - m[:, 0, 1]
    s01, s02, s12 = m[:, 0, 1] + m[:, 1, 0], m[:, 0, 2] + m[:, 2, 0], m[:, 1, 2] + m[:, 2, 1]
    candidates = np.stack(
        [
            np.stack([s / 4, d21 / s, d02 / s, d10 / s], axis=1),
            np.stack([d21 / s, s / 4, s01 / s, s02 / s], axis=1),
            np.stack([d02 / s, s01 / s, s / 4, s12 / s], axis=1),
            np.stack([d10 / s, s02 / s, s12 / s, s / 4], axis=1),
        ],
        axis=1,
    )
    q = candidates[np.arange(len(k)), k]
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(q[:, :1] < 0, -q, q)
    return q.reshape(R.shape[:-2] + (4,))


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=-2,
    )


def slerp(q0: np.ndarray, q1: np.ndarray, t) -> np.ndarray:
    """Shortest-path spherical interpolation, vectorised over leading dimensions."""
    q0 = normalize(q0)
    q1 = normalize(q1)
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    near = sin_theta < 1e-6
    safe = np.where(near, 1.0, sin_theta)
    w0 = np.where(near, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    w1 = np.where(near, t, np.sin(t * theta) / safe)
    return normalize(w0 * q0 + w1 * q1)
