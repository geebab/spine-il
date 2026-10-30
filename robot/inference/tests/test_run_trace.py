"""run_trace.py: the pure pieces — trigger bursts, the depth/lateral split, the gate event kinds."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import run_trace as T   # noqa: E402


def test_bursts_join_short_pauses_and_count_volume():
    fps = 10.0
    trig = np.array([0, 1, 1, 0, 1, 1] + [0] * 20 + [1, 1, 1, 0], float)       # 0.1 s pause joins, 2 s pause splits
    vol = np.cumsum(trig) * 2.0
    b = T.bursts(trig, vol, fps)
    assert [x["start_s"] for x in b] == [0.1, 2.6]
    assert [x["frames"] for x in b] == [4, 3]
    assert [x["removed_mm3"] for x in b] == [8.0, 6.0]
    assert b[0]["len_s"] == 0.5


def test_bursts_none_without_trigger():
    assert T.bursts(np.zeros(10), np.zeros(10), 15.0) == []


def test_split_depth_and_lateral():
    p0, u = np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.0, 1.0])
    d, lat = T.split(p0 + np.array([3.0, 4.0, -2.0]), p0, u)
    assert abs(d + 2.0) < 1e-9 and abs(lat - 5.0) < 1e-9


def test_ending_counts_the_idle_trigger_and_the_distance_to_the_pause_rule():
    fps = 10.0   # 10 s: cut for 4 s, the trigger held 3 s more with nothing cut, then off for 3 s
    trig = np.r_[np.ones(70), np.zeros(30)]
    cut = np.r_[np.ones(40), np.zeros(60)]
    vol = np.cumsum(cut) * 0.5
    e = T.ending(trig, vol, cut, fps, pause_s=15.0, stall_s=5.0)
    assert e == {"no_cut_s": 6.0, "trigger_held_without_cut_s": 3.0, "trigger_off_at_end_s": 3.0, "pause_rule_in_s": 9.0, "removed_last_window_mm3": 0.0}
    assert T.ending(trig, vol, None, fps, pause_s=15.0, stall_s=5.0) == e            # without the log's flags: where the volume rose
    busy = T.ending(np.ones(100), np.arange(100.0), np.ones(100), fps)              # cutting to the last frame
    assert busy["no_cut_s"] == 0.0 and busy["trigger_held_without_cut_s"] == 0.0 and busy["trigger_off_at_end_s"] == 0.0 and busy["pause_rule_in_s"] == 15.0
    late = T.ending(np.zeros(300), np.zeros(300), np.zeros(300), fps)                  # never cut, never triggered: 30 s
    assert late["no_cut_s"] == 30.0 and late["pause_rule_in_s"] == 0.0 and late["trigger_off_at_end_s"] == 30.0


def test_tool_motion_path_and_turn():
    n = 31; S = np.zeros((n, 15)); S[:, 0] = np.arange(n) * 0.5            # the endoscope tip moves 0.5 mm a frame along x
    ang = np.radians(np.arange(n) * 1.0)                                   # and turns 1° a frame about z
    S[:, 3:7] = np.c_[np.zeros(n), np.zeros(n), np.sin(ang / 2), np.cos(ang / 2)]
    S[:, 10:14] = [0, 0, 0, 1]
    m = T.tool_motion(S, 0, fps=15.0)
    assert m["tip_path_mm"] == 15.0 and abs(m["turn_path_deg"] - 30.0) < 0.2 and abs(m["turn_speed_deg_s_median"] - 15.0) < 0.1
    assert T.tool_motion(S, 7, fps=15.0) == {"tip_path_mm": 0.0, "turn_path_deg": 0.0, "turn_speed_deg_s_median": 0.0}


def test_gate_kinds():
    assert T.gate_kind([10, 1.0, "left IK 0.0 mm / 10.8°", "turn 0.5"]) == ("left", "turn")
    assert T.gate_kind([11, 0.25, "right IK 0.0 mm / 10.0°", ""]) == ("right", "shrink")
    assert T.gate_kind([12, 0.0, "right IK 0.0 mm / 10.1°", ""]) == ("right", "hold")
    assert T.gate_kind([13, 1.0, "left IK 0.0 mm / 11.6°; right IK 0.0 mm / 10.1°", "turn 0.0"]) == ("left+right", "turn")
