"""eval/fit_registration.py: the rotation that puts a recording's two tools onto the arms' tools, and the registration it gives."""

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "eval"))

import fit_registration as FR  # noqa: E402


def rot(axis, deg) -> np.ndarray:
    a = np.radians(deg); x, y, z = np.asarray(axis, float) / np.linalg.norm(axis)
    c, s, C = np.cos(a), np.sin(a), 1 - np.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


HAND = {"left": rot((1, 2, 3), 40.0), "right": rot((3, -1, 2), 75.0)}   # tool → the recording's camera
T0 = {"R": rot((0, 1, 0), 20.0).reshape(-1).tolist(), "t": [10.0, 20.0, 1000.0]}


def test_the_same_tools_seen_from_another_camera_are_matched_exactly():
    Q0 = rot((2, -1, 1), 15.0)
    arms = {k: Q0 @ R for k, R in HAND.items()}   # the arms hold the tools as the hand did; only the camera differs
    T_now, Q, info = FR.registration(T0, HAND, arms)
    assert np.allclose(Q, Q0, atol=1e-9) and info["turn_deg"] == 15.0
    assert info["off_before_deg"] == {"left": 15.0, "right": 15.0} and info["off_after_deg"] == {"left": 0.0, "right": 0.0}
    # the scene does not move: a pose seen by this rig's camera through T_now is where T0 had it
    R0, Rn = np.asarray(T0["R"]).reshape(3, 3), np.asarray(T_now["R"]).reshape(3, 3)
    p_rec = np.array([0.1, -0.2, 0.5])
    assert np.allclose(Rn @ (Q @ p_rec), R0 @ p_rec) and T_now["t"] == T0["t"]
    for k in HAND:
        assert np.allclose(Rn @ arms[k], R0 @ HAND[k])   # the arm's tool, in the scene, is the hand's tool


def test_a_different_arrangement_of_the_two_tools_is_shared_between_them():
    Q0 = rot((0, 0, 1), 12.0)
    arms = {"left": Q0 @ HAND["left"], "right": Q0 @ rot((1, 0, 0), 8.0) @ HAND["right"]}   # the drill held 8° differently
    _, Q, info = FR.registration(T0, HAND, arms)
    a = info["off_after_deg"]
    assert abs(a["left"] - 4.0) < 0.05 and abs(a["right"] - 4.0) < 0.05   # half each: no rotation of the bone removes it
    assert info["off_before_deg"]["left"] == 12.0 and a["left"] < info["off_before_deg"]["left"]


def test_no_rotation_is_needed_on_the_recordings_own_rig():
    _, Q, info = FR.registration(T0, HAND, dict(HAND))
    assert np.allclose(Q, np.eye(3), atol=1e-9) and info["turn_deg"] == 0.0
