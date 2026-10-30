"""end_gesture.py — the recording rule of 10-09 ("끝내는 동작을 같게") on made-up demonstrations: one that ends as the rule
says, one that ends as the 24 recordings before it do, and the ways of missing it."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import end_gesture as EG      # noqa: E402

FPS = 15.0


def demo(end="rule", pause_away=False, repull=False):
    """Drilling for 20 s (cutting while the trigger is on), then the end. state: endo xyz·quat, drill xyz·quat, trigger."""
    rng = np.random.default_rng(0)
    rows, prog, p = [], [], 0.0
    drill = np.array([20.0, -50.0, 1020.0]); endo = np.array([38.0, -33.0, 1027.0])

    def put(d, e, trig, cut=0.0):
        nonlocal p
        p += cut
        rows.append(np.r_[e, 1, 0, 0, 0, d, 1, 0, 0, 0, float(trig)]); prog.append(p)
    for _ in range(15):                                   # approach
        put(drill, endo, 0)
    for k in range(300):                                  # drilling: small moves on the bone, cutting
        drill = drill + rng.normal(0, 0.15, 3)
        if pause_away and 150 <= k < 195:                 # a pause with the drill drawn back 8 mm for 3 s: looks like the end
            put(drill + np.array([0, 0, 8.0]), endo, 0)
        elif 100 <= k < 110:                              # an ordinary pause: trigger off, the drill stays at the bone
            put(drill, endo, 0)
        else:
            put(drill, endo, 1, cut=1.0)
    last = len(rows) - 1
    if end == "rule":
        for k in range(8):                                # trigger off, drawn back 10 mm in half a second
            put(drill + np.array([0, 0, 10.0 * (k + 1) / 8]), endo, 0)
        for _ in range(45):                               # 3 s at rest
            put(drill + np.array([0, 0, 10.0]) + rng.normal(0, 0.1, 3), endo + rng.normal(0, 0.1, 3), int(repull))
        for k in range(30):                               # out
            put(drill + np.array([0, 0, 10.0 + 1.5 * k]), endo, 0)
    else:                                                 # as before the rule: straight out, 45 mm away after 4 s
        for k in range(60):
            put(drill + np.array([0, 0, 0.75 * (k + 1)]), endo + np.array([0, 0, 0.1 * k]), 0)
    prog = np.asarray(prog) / max(prog)
    return prog, np.asarray(rows), 15, last


def test_the_gesture_of_the_rule_is_recognised():
    r = EG.check(*demo("rule"), FPS)
    assert r["follows_the_rule"] and r["misses"] == []
    assert r["trigger_pulled_again"] == 0 and r["trigger_off_after_s"] <= 0.1
    assert 2.5 <= r["rest"]["for_s"] <= 3.6 and 8 <= r["rest"]["drawn_back_mm"] <= 11 and r["rest"]["from_s"] < 1.0
    assert r["end_like_rests_while_drilling"] == 0        # the ordinary pause at the bone is not the gesture


def test_the_recordings_before_the_rule_have_no_rest():
    r = EG.check(*demo("straight out"), FPS)
    assert not r["follows_the_rule"] and r["rest"] is None and r["misses"] == ["2-3 no rest of 2 s with the drill 5–25 mm back"]
    assert r["drill_away_at_end_mm"] > 40 and r["after_last_cut_s"] == 4.0


def test_a_trigger_pulled_again_and_an_early_gesture_are_named():
    r = EG.check(*demo("rule", repull=True), FPS)
    assert not r["follows_the_rule"] and r["trigger_pulled_again"] == 1 and any(m.startswith("1 ") for m in r["misses"])
    r = EG.check(*demo("rule", pause_away=True), FPS)
    assert not r["follows_the_rule"] and r["end_like_rests_while_drilling"] == 1 and r["rest"] is not None
    assert r["misses"] == ["while drilling: 1 rest(s) that look like the end"]


def test_rests_are_ranges_where_both_tips_stay_put():
    tip = np.zeros((90, 3)); tip[30:, 0] = 5.0; tip[60:, 0] = np.linspace(5, 50, 30)
    other = np.zeros((90, 3))
    assert EG.rests(tip, other, FPS) == [(0, 30), (30, 62)]      # frame 61 is still within 2 mm of where the rest began
    other[40:50, 1] = 9.0                                  # the other tool moves in the middle of the second rest
    assert EG.rests(tip, other, FPS) == [(0, 30)]
