"""Rotation helpers (quaternions are [w, x, y, z]) and the viewer's endoscope-roll convention.

Frames: `cam` = D435 optical frame (+Y down, mm); `scene` = viewer LPS (mm). The registration T0 maps
cam → scene: p_scene = R0 · p_cam + t0, so t0 is the camera centre in scene coordinates.
A tool pose is (pos, R): pos = marker-constellation origin, R = tool → frame rotation; tip = pos + R·tipLocal,
shaft direction u = R·shaftLocal (unit, pointing handle → tip).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < EPS:
        raise ValueError("zero-length vector")
    return v / n


def perp(v, axis) -> np.ndarray:
    """Component of v perpendicular to the unit `axis`, normalised."""
    v, axis = np.asarray(v, float), np.asarray(axis, float)
    return unit(v - np.dot(v, axis) * axis)


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = unit(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quat(R) -> np.ndarray:
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = unit(q)
    return q if q[0] >= 0 else -q


def rotvec_to_matrix(rv) -> np.ndarray:
    """Rodrigues: rotation vector (radians) → matrix."""
    rv = np.asarray(rv, float)
    th = np.linalg.norm(rv)
    if th < EPS:
        return np.eye(3)
    k = rv / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def slerp(q0, q1, s: float) -> np.ndarray:
    q0, q1 = unit(q0), unit(q1)
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        return unit(q0 + s * (q1 - q0))
    th = np.arccos(d)
    return unit((np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th))


def roll_reference(u) -> np.ndarray:
    """Roll zero direction a ⟂ u: scene +Z projected (or +Y when u is nearly ±Z) — trackerEndoMap.ts."""
    ref = np.array([0.0, 1.0, 0.0]) if abs(u[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
    return perp(ref, u)


def endo_roll_deg(R_s, pos_s, u, cam_centre_s) -> float:
    """Viewer roll (deg) for a tool pose in scene frame: the marker-plane normal (tool z) flipped to face the
    camera, projected ⟂ u, measured from `roll_reference(u)` about u (filmPlaneFacingSign → filmPoseToEndoAbs)."""
    u = unit(u)
    n0 = np.asarray(R_s, float)[:, 2]
    sign = 1.0 if np.dot(n0, np.asarray(cam_centre_s, float) - np.asarray(pos_s, float)) > 0 else -1.0
    tgt = perp(sign * n0, u)
    a = roll_reference(u)
    return float(np.degrees(np.arctan2(np.dot(np.cross(a, tgt), u), np.dot(a, tgt))))


def triad(local_a, local_b, frame_a, frame_b) -> np.ndarray:
    """Rotation R with R·local_a = frame_a and R·(local_b ⟂ local_a) ∥ (frame_b ⟂ frame_a)."""
    b1 = unit(local_a)
    b2 = perp(local_b, b1)
    c1 = unit(frame_a)
    c2 = perp(frame_b, c1)
    B = np.column_stack([b1, b2, np.cross(b1, b2)])
    C = np.column_stack([c1, c2, np.cross(c1, c2)])
    return C @ B.T


def pose_from_tip_u_roll(tip_s, u, roll_deg, tip_local, shaft_local, cam_centre_s) -> tuple[np.ndarray, np.ndarray, float]:
    """Scene-frame (pos_s, R_s) whose tip, shaft direction and viewer roll are the given ones.

    The roll fixes the marker-plane normal's direction ⟂ u only up to the camera-facing flip, so both
    candidates are tried and the one that reproduces the roll is returned with its residual (deg)."""
    u = unit(u)
    a = roll_reference(u)
    th = np.radians(roll_deg)
    tgt = a * np.cos(th) + np.cross(u, a) * np.sin(th)
    z_local = np.array([0.0, 0.0, 1.0])
    best = None
    for w in (tgt, -tgt):
        R_s = triad(shaft_local, z_local, u, w)
        pos_s = np.asarray(tip_s, float) - R_s @ np.asarray(tip_local, float)
        err = abs(((endo_roll_deg(R_s, pos_s, u, cam_centre_s) - roll_deg) + 180.0) % 360.0 - 180.0)
        if best is None or err < best[2]:
            best = (pos_s, R_s, err)
    return best


class Registration:
    """T_scene←cam."""

    def __init__(self, R, t):
        self.R = np.asarray(R, float).reshape(3, 3)
        self.t = np.asarray(t, float)

    @classmethod
    def from_dict(cls, d: dict) -> "Registration":
        return cls(d["R"], d["t"])

    def to_dict(self) -> dict:
        return {"R": [float(x) for x in self.R.ravel()], "t": [float(x) for x in self.t]}

    def to_scene(self, p):
        return self.R @ np.asarray(p, float) + self.t

    def to_cam(self, p):
        return self.R.T @ (np.asarray(p, float) - self.t)

    @property
    def cam_centre(self) -> np.ndarray:
        return self.t
