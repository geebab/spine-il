"""safety_gate.only_turn: which refused steps may keep their position (turn_wait) — only an IK orientation miss with the tip in place."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from safety_gate import only_turn  # noqa: E402


def test_only_an_orientation_miss_lets_the_tip_go_on():
    assert only_turn("left IK 0.0 mm / 11.5°")
    assert only_turn("left IK 0.3 mm / 12.0°; right IK 0.1 mm / 10.4°")
    assert not only_turn("left IK 6.2 mm / 3.0°")                      # the tip itself is not reached
    assert not only_turn("left IK 0.0 mm / 11.5°; right clearance 84 mm")
    assert not only_turn("left IK 0.0 mm / 11.5°; arms 8 mm apart")
    assert not only_turn("left joint limit")
    assert not only_turn("drill 83 mm from its start (> 80 mm replay sphere)")
    assert not only_turn("") and not only_turn("left marker hidden")   # nothing to wait for / a safe-zone reason
