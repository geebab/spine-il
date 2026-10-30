"""stream_plan.py — the continuous loop's timetable: windows without gaps, the fade at a window's start, the step limit
from the last sent target, and what a late answer costs; the anchor of a plan (the command, not the measured pose) and
the hand-over between plans — with stream_sim.py, the loop in miniature, for what the first continuous robot run showed."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1] / "dataset"))
import stream_plan as SP      # noqa: E402
import stream_sim as SS       # noqa: E402

Q = np.array([0.6114, -0.626, -0.4173, 0.2452]); Q /= np.linalg.norm(Q)


def answer(n_obs: int, off_mm: float = 0.0, speed: float = 0.12, H: int = 40, progress: bool = True) -> np.ndarray:
    """Rows of an answer observed at frame n_obs: the drill on the line x = speed·frame (+ off_mm sideways, the anchor's
    noise), the endoscope still, trigger on; row j = frame n_obs + 1 + j."""
    rows = np.zeros((H, 16 if progress else 15))
    for j in range(H):
        n = n_obs + 1 + j
        rows[j, 0:3] = [38.0, -33.0, 1027.0]; rows[j, 3:7] = Q
        rows[j, 7:10] = [20.0 + speed * n, -50.0 + off_mm, 1024.0]; rows[j, 10:14] = Q
        rows[j, 14] = 1.0
        if progress:
            rows[j, 15] = min(1.0, n / 1000.0)
    return rows


def test_windows_follow_each_other_without_a_gap_and_stay_inside_the_answer():
    tm = SP.Timing(period=8, lead=10, blend=4, horizon=40)
    assert tm.window(0) == (10, 18) and tm.window(1) == (18, 26) and tm.window(5, n0=100) == (150, 158)
    assert tm.rows() == (9, 17) and tm.fade_rows() == (17, 21)
    for i in range(6):
        assert tm.window(i)[1] == tm.window(i + 1)[0]
        a, b = tm.window(i)                                   # the rows of answer i are exactly these frames
        assert [tm.obs_frame(i) + 1 + j for j in range(*tm.rows())] == list(range(a, b))
        fa, fb = tm.fade_rows()                               # … and the previous answer's fade rows are the window's first frames
        assert [tm.obs_frame(i - 1) + 1 + j for j in range(fa, fb)] == list(range(a, a + tm.blend))
    with pytest.raises(ValueError):
        SP.Timing(period=16, lead=30, horizon=40)             # rows up to 44 of 40
    assert SP.Timing(period=16, lead=20, blend=8, horizon=40).fade_rows() is None      # the previous answer is too short for the fade


def test_a_new_plan_is_faded_in_from_the_previous_plans_continuation():
    tm = SP.Timing(period=8, lead=10, blend=4)
    prev, new = answer(0), answer(8, off_mm=1.0)              # the new observation's anchor is 1 mm off (measurement noise)
    want = SP.desired(new, prev, tm)
    assert len(want) == 8
    y = [w[8] + 50.0 for w in want]                           # the sideways offset comes in over four frames …
    assert y[:4] == pytest.approx([0.2, 0.4, 0.6, 0.8]) and y[4:] == pytest.approx([1.0] * 4)
    x = [w[7] for w in want]                                  # … while the motion along the line is untouched
    assert x == pytest.approx([20.0 + 0.12 * n for n in range(18, 26)])
    last_prev = prev[16]                                      # the previous window's last target (frame 17)
    steps = np.linalg.norm(np.diff(np.array([last_prev[7:10]] + [w[7:10] for w in want]), axis=0), axis=1)
    assert steps.max() < 0.25                                 # no frame jumps by the 1 mm of the anchor
    unfaded = SP.desired(new, None, tm)
    assert np.linalg.norm(unfaded[0][7:10] - last_prev[7:10]) > 1.0     # what the first frame would have been without it
    assert all(w[14] == 1.0 for w in want) and want[0][15] == new[9, 15]   # trigger and progress are the new plan's


def test_rotations_fade_along_the_shorter_arc_and_stay_unit():
    tm = SP.Timing(period=8, lead=10, blend=4)
    prev, new = answer(0), answer(8)
    new[:, 10:14] = -Q                                        # the same rotation with the other sign
    want = SP.desired(new, prev, tm)
    for w in want:
        assert np.linalg.norm(w[10:14]) == pytest.approx(1.0) and abs(float(w[10:14] @ Q)) == pytest.approx(1.0, abs=1e-9)


def test_the_window_is_followed_from_the_last_sent_target_with_the_step_limit():
    tm = SP.Timing(period=8, lead=10, blend=0)
    new = answer(8, off_mm=5.0)                               # a plan 5 mm to the side: no fade here, the limit alone
    last = answer(0)[16]
    got, hit = SP.follow(last, SP.desired(new, None, tm), max_mm=2.0, max_deg=3.0)
    steps = np.linalg.norm(np.diff(np.array([last[7:10]] + [g[7:10] for g in got]), axis=0), axis=1)
    assert steps.max() <= 2.0 + 1e-6 and hit == 2              # two frames at the limit, the third reaches the plan
    assert np.allclose(got[-1][7:10], new[16, 7:10]) and got[0][14] == 1.0 and got[0][15] == new[9, 15]


def test_a_late_answer_is_late_by_whole_frames():
    tm = SP.Timing(period=8, lead=10); ck = SP.Clock(t0_ms=1000.0, dt_ms=200.0)
    a, b = tm.window(0)                                       # frames 10 … 17 due at 3000 … 4400 ms
    assert ck.t(a) == 3000.0 and ck.frame(3000.0) == 10 and ck.frame(2999.0) == 9
    assert SP.late(ck, a, b - a, now_ms=2700.0, send_lead_ms=150.0) == 0        # ready 300 ms ahead
    assert SP.late(ck, a, b - a, now_ms=2900.0, send_lead_ms=150.0) == 1        # frame 10 would get 100 ms: one frame late
    assert SP.late(ck, a, 10 ** 6, now_ms=3500.0, send_lead_ms=150.0) == 4      # 650 ms short → four frames
    pushed = SP.Clock(ck.t0_ms + 4 * ck.dt_ms, ck.dt_ms)                         # the loop's answer: the clock goes back by them
    assert SP.late(pushed, a, 10 ** 6, now_ms=3500.0, send_lead_ms=150.0) == 0 and pushed.t(a) == 3800.0


def test_the_timetable_waits_for_a_late_answer_and_loses_no_frame():
    tm = SP.Timing(period=8, lead=10)
    ok = SP.simulate(tm, 0.2, [1.2, 1.5, 1.84] * 10)           # look + answer + gate inside 1.85 s
    assert ok["budget_s"] == pytest.approx(1.85) and ok["waited_frames"] == 0 and ok["s_per_frame"] == pytest.approx(0.2)
    one = SP.simulate(tm, 0.2, [1.2] * 9 + [2.5])              # one answer 0.65 s over: one wait of four frames, 80 frames all sent
    assert one["waited_frames"] == 4 and one["longest_wait_s"] == pytest.approx(0.8) and one["frames"] == 80
    slow = SP.simulate(tm, 0.2, [2.33] * 40)                   # a server that is always 0.5 s over (the dry run of 10-12)
    assert slow["waited_frames"] == 3 * 40 and slow["s_per_frame"] == pytest.approx(0.275)
    assert SP.simulate(SP.Timing(period=8, lead=14), 0.2, [2.33] * 40)["waited_frames"] == 0      # a longer lead buys it back


def pose(rng) -> np.ndarray:
    def q():
        v = rng.normal(size=4); return v / np.linalg.norm(v)
    return np.concatenate([rng.normal(size=3) * 20, q(), rng.normal(size=3) * 20, q(), [0.0]])


def test_a_reanchored_row_is_the_loops_own_relative_form_applied_at_the_anchor():
    import remote_policy as RP
    rng = np.random.default_rng(3)
    fed, anchor = pose(rng), pose(rng)
    rows = np.stack([np.concatenate([pose(rng), [0.4]]) for _ in range(6)]); rows[:, 14] = [0, 1, 1, 0, 1, 1]
    got = SP.reanchor(rows, fed, anchor); ref = RP.chunk_to_abs(anchor, RP.abs_to_chunk(fed, rows))
    assert np.allclose(got[:, [0, 1, 2, 7, 8, 9]], ref[:, [0, 1, 2, 7, 8, 9]], atol=1e-4)
    for o in (3, 10):
        assert (np.abs((got[:, o:o + 4] * ref[:, o:o + 4]).sum(1)) > 1 - 1e-9).all()        # the same rotations
    assert np.array_equal(got[:, 14:], rows[:, 14:])                                         # trigger and progress untouched
    assert np.allclose(SP.reanchor(rows, fed, fed)[:, :14], rows[:, :14], atol=1e-9)         # anchored where it was fed: unchanged


def test_the_first_answer_runs_from_its_first_row_to_where_the_second_window_begins():
    for L in (10, 14, 16):
        tm = SP.Timing(period=8, lead=L, blend=4, horizon=40)
        a, b = SP.first_rows(tm)
        assert a == 0 and 1 + (b - a) == tm.window(1)[0]          # frames 1 … L+P−1, then the window of the look at frame P


def test_a_window_continues_from_the_last_sent_target_and_gamma_brings_it_to_the_plan():
    tm = SP.Timing(period=8, lead=16, blend=4, horizon=40); a, b = tm.rows()
    bel = answer(0); anchor = bel[0].copy()
    last = bel[a - 1].copy(); last[7:10] += [0.0, 1.5, -0.7]; last[14] = 0.0          # the command is 1.7 mm off the plan's place for it
    own = SP.stitch(bel, anchor, last, tm, 0.0)                                         # gamma 0: the plan's motion, from where the command is
    assert len(own) == tm.period
    assert np.allclose(own[0][7:10] - last[7:10], bel[a][7:10] - bel[a - 1][7:10])
    assert np.allclose(own[-1][7:10] - last[7:10], bel[b - 1][7:10] - bel[a - 1][7:10])
    full = SP.stitch(bel, anchor, last, tm, 1.0)                                        # gamma 1: at the plan's positions by the window's end
    assert np.allclose(full[-1][7:10], bel[b - 1][7:10])
    half = SP.stitch(bel, anchor, last, tm, 0.5)
    left = [float(np.linalg.norm(t[7:10] - bel[a + j][7:10])) for j, t in enumerate(half)]
    assert left == sorted(left, reverse=True) and abs(left[-1] - 0.5 * float(np.linalg.norm([1.5, -0.7]))) < 1e-9
    assert all(t[14] == 1.0 and t[15] == bel[a + j][15] for j, t in enumerate(half))    # trigger and progress are the plan's
    assert all(abs(np.linalg.norm(t[o:o + 4]) - 1) < 1e-9 for t in half for o in (3, 10))


@pytest.mark.parametrize("lead,full", [(10, True), (14, False), (16, True)])
def test_a_plan_anchored_at_the_measured_pose_echoes_and_one_anchored_at_the_command_does_not(lead, full):
    """The first continuous robot run (10-12, anchor = measured, lead 16): command path 2.0 m against a few 0.1 m, 605 of
    2018 frames at the per-frame limit, the commands' steps repeating L+P frames later (correlation 0.75–0.88). In
    miniature, with the arms' measured response and a policy that answers "the state I was given + a displacement".
    How hard the echo is depends on where a look falls in the window being executed: with lead 16 a look sees the end of
    the window three before its own (the robot run's 24 frames), with lead 10 the middle of the one two before; with lead
    14 it falls into a window's fade and the echo is weak — still there, and nothing to rely on."""
    old = SS.run("measured", lead=lead, windows=200, policy=SS.Policy(sample_mm=0.5))
    new = SS.run("command", 0.5, lead=lead, windows=200, policy=SS.Policy(sample_mm=0.5))
    for tool in ("endo", "drill"):
        o, n = old[tool], new[tool]
        if full:
            assert o["command_path_mm"] > 5 * o["policy_path_mm"] and o["swing_command_mm"] > 2.0
        assert o["command_path_mm"] > 1.3 * o["policy_path_mm"] and o["swing_command_mm"] > 3 * n["swing_command_mm"]
        assert n["command_path_mm"] < 1.15 * n["policy_path_mm"] and n["swing_command_mm"] < 0.3
        assert abs(n["net_mm"] - n["policy_path_mm"]) < 0.1 * n["policy_path_mm"]       # … and it gets where the policy goes
    assert new["clamped_frames"] == 0
    if full:
        assert old["clamped_frames"] > 0.1 * old["frames"] and old["drill"]["step_corr_L+P"] > 0.5


def test_the_echo_needs_no_resonance_a_plain_lag_and_the_arms_wander_are_enough():
    old = SS.run("measured", lead=16, windows=200, arm=SS.Arm(bump=0.0), policy=SS.Policy(sample_mm=0.5))
    assert old["drill"]["command_path_mm"] > 3 * old["drill"]["policy_path_mm"] and old["drill"]["step_corr_L+P"] > 0.5


@pytest.mark.parametrize("lead", [10, 16])
def test_a_correction_the_plan_makes_in_its_first_rows_is_executed_only_through_gamma(lead):
    """Something new in the image after 60 answers: the line the tips should follow lies 3 mm further. The plan corrects in its
    first five rows — rows a window never executes. With gamma 0 only the plan's motion counts and the tips stay off the
    line; with gamma 0.5 they get there, without the loop swinging."""
    def go(gamma):
        return SS.run("command", gamma, lead=lead, windows=200, policy=SS.Policy(sample_mm=0.3, goal_gain=1.0, line_moves=(60, 3.0)))
    none, half = go(0.0), go(0.5)
    assert abs(none["drill"]["off_line_mm_end"]) > 2.0
    assert abs(half["drill"]["off_line_mm_end"]) < 0.8 and half["drill"]["swing_command_mm"] < 0.5 and half["clamped_frames"] == 0


def test_the_anchor_takes_the_tips_from_the_command_and_the_rotations_from_the_measured_pose():
    """The arms make about a tenth of a turn they are asked for: a rotation carried on from command to command grows away
    from the tool (the dry run with the real model: 10° commanded, 1.3° turned, the gate held 262 frames). So the plan's
    rotations are the answer's own (measured pose + the model's turn) and a window ends at them."""
    rng = np.random.default_rng(5)
    measured, command = pose(rng), pose(rng)
    anc = SP.anchor_pose(measured, command)
    assert np.array_equal(anc[[0, 1, 2, 7, 8, 9]], command[[0, 1, 2, 7, 8, 9]]) and np.array_equal(anc[3:7], measured[3:7]) and np.array_equal(anc[10:14], measured[10:14])
    rows = np.stack([np.concatenate([pose(rng), [0.2]]) for _ in range(40)])
    bel = SP.reanchor(rows, measured, anc)
    for o in (3, 10):                                             # the rotations are as answered …
        assert (np.abs((bel[:, o:o + 4] * (rows[:, o:o + 4] / np.linalg.norm(rows[:, o:o + 4], axis=1, keepdims=True))).sum(1)) > 1 - 1e-9).all()
    assert np.allclose(bel[:, 7:10] - rows[:, 7:10], command[7:10] - measured[7:10])        # … the tips moved by command − measured
    tm = SP.Timing(period=8, lead=16, blend=4, horizon=40); a, b = tm.rows()
    last = pose(rng)
    win = SP.stitch(bel, anc, last, tm, 0.5)
    for o in (3, 10):                                             # the window ends at the plan's rotation, the tips half way
        assert abs(float(np.dot(win[-1][o:o + 4], bel[b - 1][o:o + 4]))) > 1 - 1e-9
    own_end = last[7:10] + bel[b - 1][7:10] - bel[a - 1][7:10]
    assert np.allclose(win[-1][7:10], 0.5 * own_end + 0.5 * bel[b - 1][7:10])

