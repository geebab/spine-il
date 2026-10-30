"""10-16: rcm_project and the order of the step limit and the RCM rule in the continuous loop (live_loop --stream-rcm-first).
A plan whose rotation is off the pivot by more than the RCM rule's 2° a frame, at the limit's 3°: limit → rcm keeps asking
the turn back (a clamp almost every frame) and leaves the shaft off its portal; rcm_project → limit → rcm clamps nothing."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rollout as RO          # noqa: E402
from safety import limit_step  # noqa: E402

PORTALS = {"endo": np.array([0.0, 0.0, 0.0]), "drill": np.array([30.0, 0.0, 0.0])}
SHAFTS = {"endo": np.array([0.0, 0.0, 1.0]), "drill": np.array([0.0, 0.0, 1.0])}


def frame(tilt_deg: float, roll_deg: float = 0.0, z: float = 40.0) -> np.ndarray:
    """Both tools' tips z mm beyond their portals along +z, shafts tilted tilt_deg about x (off the pivot), rolled roll_deg."""
    v = np.zeros(15)
    R = RO._rotvec_R(np.array([np.radians(tilt_deg), 0, 0])) @ RO._rotvec_R(np.array([0, 0, np.radians(roll_deg)]))
    for tool, o in (("endo", 0), ("drill", 7)):
        v[o:o + 3] = PORTALS[tool] + [0, 0, z]
        v[o + 3:o + 7] = RO.R2q(R, np.array([1.0, 0, 0, 0]))
    return v


def miss(v: np.ndarray) -> float:
    out = 0.0
    for tool, o in (("endo", 0), ("drill", 7)):
        to = v[o:o + 3] - PORTALS[tool]; u = RO.q2R(v[o + 3:o + 7]) @ SHAFTS[tool]
        out = max(out, float(np.degrees(np.arccos(np.clip(np.dot(u, to / np.linalg.norm(to)), -1, 1)))))
    return out


def test_project_puts_the_shaft_through_the_portal_and_keeps_the_roll():
    v = RO.rcm_project(frame(4.0, roll_deg=20.0), PORTALS, SHAFTS, 70.0)
    assert miss(v) < 0.25
    R = RO.q2R(v[3:7]); x = R @ np.array([1.0, 0, 0])
    assert abs(np.degrees(np.arctan2(x[1], x[0])) - 20.0) < 1.0     # the roll about the shaft stays the plan's


def test_a_tool_far_off_the_pivot_turns_at_most_5_degrees_and_continuously():
    m = [miss(RO.rcm_project(frame(t), PORTALS, SHAFTS, 70.0)) for t in (4.0, 5.5, 7.0, 40.0)]
    assert m[0] < 0.25 and abs(m[1] - 0.5) < 0.1 and abs(m[2] - 2.0) < 0.1 and abs(m[3] - 35.0) < 0.1


def test_order_limit_then_rcm_clamps_rcm_first_does_not():
    # every window (8 frames) a new answer, 4° off the pivot to one side or the other (each answer is a new sample)
    plans = [frame(4.0 if (k // 8) % 2 else -4.0, z=40.0 + 0.1 * k) for k in range(64)]
    res = {}
    for first in (False, True):
        vc = RO.rcm_project(plans[0], PORTALS, SHAFTS, 70.0); hits = 0; worst = 0.0
        for w in plans[1:]:
            tgt = RO.rcm_project(w, PORTALS, SHAFTS, 70.0) if first else w
            vc, h = limit_step(vc, tgt, 2.0, 3.0); hits += h
            for tool, o in (("endo", 0), ("drill", 7)):
                vc, _ = RO.rcm(vc, o, PORTALS[tool], SHAFTS[tool], 70.0, 2.0)
            worst = max(worst, miss(vc))
        res[first] = (hits, worst)
    assert res[True][0] == 0 and res[True][1] < 0.25
    assert res[False][0] >= 7 and res[False][1] > 1.5          # a clamp at each new answer, the shaft left ~2° off
