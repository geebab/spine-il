"""run_criteria.py: who ended the drilling, and what the harness changed in the policy's part of a run."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
import run_criteria as R   # noqa: E402


def test_a_policy_with_the_done_output_has_to_end_the_drilling_itself():
    assert R.ended_by_policy("policy", None, done_output=True)
    for end in ("pause", "stall", "time", "steps", "volume", "deadline", None):   # every rule of the harness is only a bound
        assert not R.ended_by_policy(end, None, done_output=True), end
    assert not R.ended_by_policy("policy", "robot tool not tracked", done_output=True)


def test_a_policy_without_it_keeps_the_legacy_reading():
    assert R.ended_by_policy("pause", None)
    for end in ("stall", "time", "steps", "volume", "deadline", "time (old 40 s limit)", "policy", None):
        assert not R.ended_by_policy(end, None), end


def test_an_abort_fails_whatever_the_end():
    assert not R.ended_by_policy("pause", "the arms are 12.0/3.0 mm from the command after settling (limit 10)")


def test_interventions_count_what_the_harness_changed():
    meta = {"frame_kind": ["scripted"] * 3 + ["policy"] * 40 + ["settle"] * 2,
            "safety_gate": {"checked": 20, "shrunk": 2, "held": 3, "turn_cut": 4},
            "trigger_gated_frames": 8, "travel_limited_frames": 1, "clamped_frames": 5}
    h = R.interventions(meta)
    assert h["policy_frames"] == 40 and h["gate_checked"] == 20
    assert (h["gate_held"], h["gate_shrunk"], h["gate_turn_cut"]) == (3, 2, 4)
    assert h["blocked"] == 3 + 1              # gate held + travel stop: a motion replaced by a stop
    assert h["overrides"] == 3 + 2 + 1 + 5    # criterion 6: held + shrunk + travel stop + clamped — a motion the harness changed
    assert h["trigger_off_frames"] == 8       # counted and shown, not a mark: the trigger off while the drill is off screen is a rule of the run


def test_what_the_gate_refused_on_scripted_frames_is_not_the_policys():
    fk = ["scripted"] * 5 + ["policy"] * 20 + ["scripted"] * 5
    ev = [[8, 1.0, "left IK 0.0 mm / 10.4°", "turn 0.5"], [12, 0.5, "right IK"], [26, 0.25, "drill 81 mm"], [27, 0.0625, "drill 81 mm"], [28, 0.0, "drill 81 mm"]]
    h = R.interventions({"frame_kind": fk, "safety_gate": {"checked": 30, "shrunk": 3, "held": 1, "turn_cut": 1, "events": ev}})
    assert (h["gate_held"], h["gate_shrunk"], h["gate_turn_cut"]) == (0, 1, 1)          # the policy's frames
    assert h["gate_on_scripted_frames"] == {"held": 1, "shrunk": 2, "turn_cut": 0}
    assert h["overrides"] == 1 and h["blocked"] == 0
    # a log that is not complete (cut at 200 events, or an old run): everything is counted, as before
    h = R.interventions({"frame_kind": fk, "safety_gate": {"checked": 30, "shrunk": 3, "held": 1, "turn_cut": 1, "events": ev[:2]}})
    assert (h["gate_held"], h["gate_shrunk"]) == (1, 3) and h["gate_on_scripted_frames"] is None


def test_interventions_of_an_old_run_without_the_counters():
    h = R.interventions({"safety_gate": {"checked": 10, "shrunk": 0, "held": 0}})
    assert h["blocked"] == 0 and h["gate_turn_cut"] == 0 and h["policy_frames"] is None
