"""The policy ends the drilling itself (a --done-output policy): where a predicted chunk says so, and what the phase
tracker makes of it."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rollout as RO  # noqa: E402


def chunk(*runs):
    """done values of a 40-step chunk from (value, count) runs; the rest is 0."""
    d = np.concatenate([np.full(n, v, float) for v, n in runs])
    return np.r_[d, np.zeros(40 - len(d))]


def test_a_chunk_without_done_goes_on():
    assert RO.done_step(np.zeros(40), 20) is None
    assert RO.done_step(chunk((0.4, 40)), 20) is None                       # never above one half


def test_done_is_the_first_step_from_which_it_stays_up():
    assert RO.done_step(chunk((0, 12), (1, 28)), 20) == 12
    assert RO.done_step(chunk((1, 40)), 20) == 0                             # already over at the chunk's first step
    assert RO.done_step(chunk((0, 3), (0.9, 2), (0, 5), (0.8, 30)), 20) == 10  # a two-step blip does not count, the later rise does


def test_only_the_steps_about_to_be_executed_can_end_it():
    assert RO.done_step(chunk((0, 25), (1, 15)), 20) is None                 # the rise lies beyond the executed steps: next chunk
    assert RO.done_step(chunk((0, 18), (1, 22)), 20) == 18                   # the hold may reach into the steps not executed
    assert RO.done_step(chunk((0, 18), (1, 3)), 20) is None                  # … but it has to hold
    assert RO.done_step(np.r_[np.zeros(37), np.ones(3)], 40) == 37           # at the chunk's very end: as many as are left


def tracker(ph=4):
    t = RO.PhaseTracker([0.0, 0.0, 0.0], 1 / 15, drill_max_s=240.0, pause_s=15.0, stall_s=25.0)
    t.ph = ph
    return t


def test_the_policys_done_ends_the_drilling():
    t = tracker(4)
    assert t.policy_done() and t.ph == 5 and t.done_drilling and t.end == "policy"
    assert not t.policy_done() and t.end == "policy"                         # once


def test_done_before_the_first_cut_is_the_policys_too():
    t = tracker(3)
    assert t.policy_done() and t.end == "policy" and t.ph == 5


def test_a_rule_that_came_first_keeps_the_end():
    t = tracker(4)
    t.ph = 5; t.done_drilling = True; t.end = "stall"                        # the stall bound ended the drilling
    assert not t.policy_done() and t.end == "stall"
    assert not tracker(1).policy_done() and not tracker(2).policy_done()    # outside the skin there is nothing to end


# ---- done_eval.py: the same rule walked along a demonstration ----
import done_eval as DE  # noqa: E402


def test_the_demo_ends_at_the_first_frame_after_its_last_cut():
    assert DE.demo_end(np.array([3, 3, 4, 4, 4, 5, 5])) == 5
    assert DE.demo_end(np.array([3, 4, 4])) is None


def test_fold_phase_matches_the_training_input():
    s = np.zeros(21); s[15 + 4] = 1                                          # phase 5
    f = DE.fold_phase(s)
    assert f[15:].tolist() == [0, 0, 0, 1, 0, 0] and s[19] == 1             # folded into drilling, the input untouched
    assert DE.fold_phase(np.arange(15.0)).tolist() == list(range(15))


def test_walk_ends_where_the_first_chunk_says_done():
    end = 130                                                                # the frame after the last cut
    done_of = lambda t: (np.arange(t + 1, t + 41) >= end).astype(float)      # a perfect predictor: step j leads to frame t+1+j
    k, trace = DE.walk(done_of, 200, 20, 0)
    assert k == end and trace[-1][0] == 120 and trace[-1][1] == 1.0 and trace[0] == [0, 0.0]
    k, _ = DE.walk(done_of, 200, 20, 7)                                      # another chunk grid: the same end
    assert k == end
    late = lambda t: (np.arange(t + 1, t + 41) >= end + 45).astype(float)
    assert DE.walk(late, 200, 20, 0)[0] == end + 45
    assert DE.walk(lambda t: np.zeros(40), 200, 20, 0)[0] is None            # a policy that never says it


def test_the_end_value_has_two_forms_and_each_its_level():
    """User 10-09: the end output becomes a progress value (bone removed / the demonstration's total)."""
    assert RO.end_output({}) == (None, None) and RO.end_output(None) == (None, None)
    assert RO.end_output({"done_output": True}) == ("done", 0.5)                               # the checkpoints of 10-08
    # 0.91: computed by end_level.py (10-09) — not a number to change by feel; compute it again on the model that is used
    assert RO.END_LEVEL["progress"] == 0.91 and RO.end_output({"done_output": True, "end_output": "progress"}) == ("progress", 0.91)
    assert RO.end_output({"done_output": True, "end_output": "progress", "end_level": 0.9}) == ("progress", 0.9)
    prog = np.r_[np.linspace(0.5, 0.94, 30), np.full(10, 0.97)]                                # rises, passes 0.95 at step 30
    assert RO.done_step(prog, 40, level=0.95) == 30 and RO.done_step(prog, 20, level=0.95) is None
    assert RO.done_step(prog, 40) == 1                                                         # as a flag it would be "done" from the start
    blip = np.r_[np.full(10, 0.9), [0.96, 0.96], np.full(28, 0.9)]                             # two steps above the level: not the end
    assert RO.done_step(blip, 40, level=0.95) is None
