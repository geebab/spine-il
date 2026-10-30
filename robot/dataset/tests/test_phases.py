"""phases.py on synthetic tracks with known boundaries."""

import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import phases as P  # noqa: E402
import synth  # noqa: E402
from lerobot_robot_spinearm.recording import read_rows  # noqa: E402


def _tracks(tmp_path):
    p = tmp_path / "s_hand_aaaa.jsonl"
    synth.write_hand(p, "s_hand_aaaa")
    _, rows, _ = read_rows(p)
    return P.from_rows(rows, synth.BASE, (np.eye(3), np.zeros(3)))


def test_hand_six_phases_in_order(tmp_path):
    e, d, cuts = _tracks(tmp_path)
    res = P.segment(e, d, cuts, 0.0, 20.0)
    b = res.bounds
    assert b[1][0] == 0.0 and b[1][1] == pytest.approx(1.0, abs=0.1)      # still until the motion starts at 1 s
    assert b[3][0] == pytest.approx(2.0, abs=0.05)                        # skin entry at the portal
    assert (b[4][0], b[4][1]) == pytest.approx((5.0, 12.0), abs=1e-6)     # first → last cut, pause kept inside
    assert b[6][0] == pytest.approx(18.5, abs=0.05)                       # both tools out
    assert res.flags == []
    assert res.info["cut_gaps_over_1s"] == 1
    ph = res.phase_at(np.arange(0, 20, 1 / 15))
    assert np.all(np.diff(ph) >= 0) and set(ph.tolist()) == {1, 2, 3, 4, 5, 6}


def test_resting_on_flank_is_not_inside(tmp_path):
    # depth > 0 at the ready pose (entry 50 mm off the portal) must not start phase 3
    e, d, cuts = _tracks(tmp_path)
    res = P.segment(e, d, cuts, 0.0, 20.0)
    assert res.t_in > 1.5


def test_blip_and_robot_window():
    t = np.arange(0, 10, 1 / 30)
    tip = np.tile([50.0, -10.0, 0.0], (len(t), 1))
    tr = P.ToolTrack(t, tip, np.full(len(t), 10.0), np.tile(synth.PORTAL, (len(t), 1)))
    cuts = np.r_[0.5, np.arange(3.0, 6.0, 0.1)]  # a lone cut row at 0.5 s = blip
    res = P.segment(tr, tr, cuts, 0.0, 10.0, expect_full=False)
    assert res.t_cut0 == pytest.approx(3.0)
    assert res.bounds[1] == (0.0, 0.0) and res.bounds[2] == (0.0, 0.0)   # inside from the start: no 1/2
    assert res.t_out == pytest.approx(10.0)                              # still inside at the end: no 6
    assert any("블립" in f for f in res.flags)


def test_cut_bursts_merge():
    assert P.cut_bursts(np.array([0.0, 0.5, 0.9, 3.0, 3.2])) == [(0.0, 0.9, 3), (3.0, 3.2, 2)]


def test_no_cuts_raises():
    t = np.arange(0, 2, 0.1)
    tr = P.ToolTrack(t, np.zeros((len(t), 3)), np.zeros(len(t)), np.zeros((len(t), 3)))
    with pytest.raises(ValueError):
        P.segment(tr, tr, np.array([]), 0.0, 2.0)
