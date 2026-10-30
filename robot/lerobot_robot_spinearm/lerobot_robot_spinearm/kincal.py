"""Parametric forward kinematics of the SO-101 chain (parsed from the URDF) for kinematic calibration.

Each joint: T_parent←child(q) = T(origin xyz + δxyz, origin rpy) · Rot(axis, q + offset). Calibrating δxyz (link
lengths / joint positions) and offsets on top of the hand-eye X, Z follows the optical-tracker calibration literature
(joint offsets dominate, link lengths next; 2–10× accuracy on industrial arms).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .handeye import compose, rotvec_to_R
from .protocol import JOINTS


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


class Chain:
    """Serial chain base_link → … → `tip` (fixed joints folded in). Angles in degrees (model angles)."""

    def __init__(self, urdf: Path | str, tip: str = "marker_frame_link"):
        root = ET.parse(str(urdf)).getroot()
        joints = {j.find("child").get("link"): j for j in root.findall("joint")}
        path, link = [], tip
        while link in joints:
            j = joints[link]
            path.append(j)
            link = j.find("parent").get("link")
        path.reverse()
        self.elems = []  # (T_origin, axis or None, joint index or None)
        for j in path:
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()]
            T = compose(_rpy(*rpy), xyz)
            if j.get("type") == "revolute":
                ax = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                self.elems.append((T, ax / np.linalg.norm(ax), JOINTS.index(j.get("name"))))
            else:
                self.elems.append((T, None, None))
        self.n_dxyz = 3 * sum(1 for e in self.elems if e[2] is not None)

    def fk(self, q_deg, dxyz=None) -> np.ndarray:
        """dxyz: (6, 3) metres added to each revolute joint's origin (JOINTS order), or None."""
        q = np.deg2rad(np.asarray(q_deg, float))
        T = np.eye(4)
        for To, ax, i in self.elems:
            if i is not None and dxyz is not None:
                To = To.copy()
                To[:3, 3] = To[:3, 3] + dxyz[i]
            T = T @ To
            if ax is not None:
                T = T @ compose(rotvec_to_R(ax * q[i]), np.zeros(3))
        return T
