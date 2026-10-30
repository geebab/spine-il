"""Shared constants and message shapes for the arm server and its clients.

Wire format: JSON, one request → one reply on a ZMQ REP socket; a ZMQ PUB socket streams
state snapshots. Units are degrees for all six joints (joint 6, the former gripper, carries the
tracked marker and is a real axis).
"""

from __future__ import annotations

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# 5555/5556 are LeKiwi's defaults in LeRobot — stay clear of them.
DEFAULT_PORTS = {"left": (5601, 5602), "right": (5611, 5612)}

STATES = ("OFF", "HOLD", "MOVING", "HOMING", "FAULT", "ESTOP")

# Faults that only a 12 V power-cycle (then `enable`) can clear.
POWER_CYCLE_FAULTS = {"overload_protection", "power_cycled"}

ERRORS = (
    "bad_request",
    "bad_token",
    "lease_held",
    "no_lease",
    "estop",
    "fault",
    "off",
    "busy",
    "not_at_neutral",
    "no_neutral",
    "bounds",
    "bus_lost",
    "timeout",
    "sim_only",
)

# Fallback joint bounds (degrees, symmetric) when no calibration file is available (pure sim).
DEFAULT_BOUNDS = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-95.0, 95.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-150.0, 150.0),
    "gripper": (-65.0, 65.0),
}
WRIST_ROLL_SOFT_LIMIT = 150.0  # calibration records the full turn (0..4095); ±180° is the wrap point


def cmd_endpoint(host: str, port: int) -> str:
    return f"tcp://{host}:{port}"


def error(code: str, **detail) -> dict:
    return {"ok": False, "error": code, **detail}


def ok(**payload) -> dict:
    return {"ok": True, **payload}
