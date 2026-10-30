"""ROS-free conversions shared by the bridge nodes (unit-tested without ROS).

Tracker tap: mm, quaternion [w, x, y, z] (tool → camera), wall-clock ms. Arm server: degrees, CLOCK_MONOTONIC s.
ROS: metres, quaternion (x, y, z, w), radians, wall-clock stamps (ROS time = system clock by default).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
PORTS = {"left": (5601, 5602), "right": (5611, 5612)}


def split_stamp(t_s: float) -> tuple[int, int]:
    """Wall seconds → (sec, nanosec)."""
    sec = int(np.floor(t_s))
    return sec, int(round((t_s - sec) * 1e9)) % 1_000_000_000


def split_stamp_ms(t_ms: float) -> tuple[int, int]:
    """Wall milliseconds (tracker: µs decimals) → (sec, nanosec) without losing the µs to float seconds."""
    sec = int(t_ms // 1000)
    return sec, int(round((t_ms - sec * 1000) * 1e6)) % 1_000_000_000


def mono_offset_s() -> float:
    """wall − monotonic (s) on this host; median of a few reads."""
    v = []
    for _ in range(5):
        a = time.time()
        m = time.monotonic()
        b = time.time()
        v.append((a + b) / 2 - m)
    return float(np.median(v))


def quat_wxyz_to_xyzw(q) -> tuple[float, float, float, float]:
    w, x, y, z = (float(v) for v in q)
    n = (w * w + x * x + y * y + z * z) ** 0.5 or 1.0
    return x / n, y / n, z / n, w / n


def R_to_xyzw(R) -> tuple[float, float, float, float]:
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[i] = 0.25 * s
        q[3] = (R[k, j] - R[j, k]) / s
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
        x, y, z, w = q
    return float(x), float(y), float(z), float(w)


def tap_to_fields(m: dict) -> dict:
    """Tap message → plain fields for TrackedPose / PoseStamped (metres, xyzw, wall seconds)."""
    out = {"tool": m.get("tool", ""), "frame": int(m.get("fn") or 0), "lost": bool(m.get("lost")),
           "stamp": (m.get("ts") or m.get("tsend") or 0.0) / 1000.0,
           "stamp_ms": float(m.get("ts") or m.get("tsend") or 0.0),
           "t_arrival": (m.get("ta") or 0.0) / 1000.0, "t_pose": (m.get("tp") or 0.0) / 1000.0,
           "t_send": (m.get("tsend") or 0.0) / 1000.0, "visible": int(m.get("vis") or 0),
           "rmse": float(m.get("rmse") or 0.0), "still": bool(m.get("still")), "held": bool(m.get("held"))}
    if not out["lost"] and m.get("pos") is not None:
        out["pos"] = [v / 1000.0 for v in m["pos"]]
        out["quat"] = quat_wxyz_to_xyzw(m["quat"])
        out["pos_raw"] = [v / 1000.0 for v in m.get("raw_pos", m["pos"])]
        out["quat_raw"] = quat_wxyz_to_xyzw(m.get("raw_quat", m["quat"]))
        out["tip"] = [v / 1000.0 for v in m["tip"]] if m.get("tip") else None
        out["tip_raw"] = [v / 1000.0 for v in m["raw_tip"]] if m.get("raw_tip") else None
    return out


def state_to_fields(s: dict, off_s: float, joint_offset_deg=None) -> dict:
    """Arm-server snapshot → fields for ArmState / JointState. JointState positions are *model* angles in radians
    (arm angle + calibration joint offset), so robot_state_publisher's TF matches the calibrated kinematics."""
    t_read = ((s["t"] + s["tr"]) / 2 if s.get("tr") else s["t"]) + off_s
    nan = float("nan")
    jo = np.zeros(6) if joint_offset_deg is None else np.asarray(joint_offset_deg, float)
    pos = [float(s["pos"].get(j, nan)) for j in JOINTS] if s.get("pos") else [nan] * 6
    return {"stamp": t_read, "tick": int(s.get("tick") or 0), "state": s.get("state", ""),
            "torque": bool(s.get("torque")), "bus_ok": bool(s.get("bus_ok")), "faults": list(s.get("faults") or []),
            "position": pos,
            "goal": [float(s["goal"][j]) if s.get("goal") and j in s["goal"] else nan for j in JOINTS],
            "command": [float(s["cmd"][j]) if s.get("cmd") and j in s["cmd"] else nan for j in JOINTS],
            "load": [int((s.get("load") or {}).get(j, 0)) for j in JOINTS],
            "temperature": [int((s.get("temp") or {}).get(j, 0)) for j in JOINTS],
            "step_deg": float(s.get("step_deg") or 0.0), "accel": int(s.get("accel") or 0),
            "t_read": (s["tr"] + off_s) if s.get("tr") else 0.0, "t_write": (s["tw"] + off_s) if s.get("tw") else 0.0,
            "goal_seq": int(s["gseq"]) if s.get("gseq") is not None else -1,
            "goal_rx": (s["grx"] + off_s) if s.get("grx") else 0.0,
            "joint_rad": [float(np.deg2rad(p + o)) for p, o in zip(pos, jo)]}


def load_calib(path: str | Path) -> dict:
    """Hand-eye / kincal JSON → {Z: T_base←cam, X: T_link←const, joint_offset_deg (JOINTS order)}."""
    d = json.loads(Path(path).expanduser().read_text())
    off = d.get("joint_offset_deg") or {}
    return {"Z": np.asarray(d["T_base_cam"], float), "X": np.asarray(d["T_link_const"], float),
            "joint_offset_deg": [float(off.get(j, 0.0)) for j in JOINTS]}


def command_to_goal(names, positions_rad, joint_offset_deg=None) -> dict:
    """/<arm>/joint_command (JointState, model angles in rad — the same convention as joint_states) → arm-server
    goal {joint: arm deg}. Inverse of `state_to_fields`: arm = model − calibration joint offset.
    Unknown joint names raise (a silent drop would leave that joint where it was)."""
    if len(names) != len(positions_rad):
        raise ValueError(f"{len(names)} names but {len(positions_rad)} positions")
    jo = dict(zip(JOINTS, [0.0] * 6 if joint_offset_deg is None else joint_offset_deg))
    out = {}
    for n, p in zip(names, positions_rad):
        if n not in jo:
            raise ValueError(f"unknown joint {n!r}")
        if not math.isfinite(p):
            raise ValueError(f"{n}: non-finite position")
        out[n] = float(np.rad2deg(p)) - jo[n]
    return out
