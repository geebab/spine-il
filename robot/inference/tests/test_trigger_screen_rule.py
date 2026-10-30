"""The trigger rule and the evaluation's visibility measure key on the drill tip leaving the endoscope's SCREEN — a tip
hidden behind bone but inside the picture is on the screen (user, 2026-10-07)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "inference"))
import scope_screen as SS   # noqa: E402

VA = 67.4013   # the water view angle the step session reports


def iv(in_fov, visible, x, y):
    return {"inFov": in_fov, "visible": visible, "x": x, "y": y}


def test_a_hidden_tip_inside_the_picture_is_on_the_screen():
    assert SS.tip_on_screen(iv(True, False, 0.05, 0.10), VA, "water")
    assert not SS.tip_seen(iv(True, False, 0.05, 0.10), VA, "water")     # the stricter test of before


def test_out_of_the_field_of_view_or_outside_the_water_circle_is_off_the_screen():
    assert not SS.tip_on_screen(iv(False, False, 1.4, 0.2), VA, "water")
    assert SS.tip_on_screen(iv(True, True, 0.0, 0.93), None, None)        # air / unknown medium: the bridge's frame alone
    assert not SS.tip_on_screen(iv(True, True, 0.0, 0.93), VA, "water")   # inside the pinhole frame, beyond the image circle


def test_trigger_rule_screen_blocks_only_off_screen():
    import rollout as RO
    meta = lambda v: {"meta": {"bridge": {"drill": {"inView": v}, "endoCamera": {"viewAngle": VA}}}}
    hidden, off, seen = meta(iv(True, False, 0.05, 0.10)), meta(iv(False, False, 1.4, 0.2)), meta(iv(True, True, 0.0, 0.0))
    assert not RO.trigger_blind(hidden, "water", "screen") and RO.trigger_blind(hidden, "water", "tip")
    assert RO.trigger_blind(off, "water", "screen") and RO.trigger_blind(off, "water", "tip")
    assert not RO.trigger_blind(seen, "water", "screen") and not RO.trigger_blind(seen, "water", "tip")
    assert RO.build_parser().get_default("visible_trigger_rule") == "screen"


def test_view_stats_counts_the_screen_share_and_keeps_the_old_one():
    import policy_eval as PE
    row = lambda v, cutting=True: {"bridge": {"drill": {"proxy": {"cutting": cutting}, "inView": {**v, "zMm": 20.0}}, "endoCamera": {"viewAngle": VA}}}
    rows = [row(iv(True, True, 0.0, 0.0)), row(iv(True, False, 0.05, 0.1)), row(iv(True, False, 0.1, 0.0)), row(iv(False, False, 1.4, 0.2)),
            row(iv(False, False, 1.5, 0.0), cutting=False)]
    st = PE.view_stats(rows, "water")
    assert st["cutting_frames"] == 4
    assert st["visible_while_cutting"] == 0.25          # seen: 1 of 4 (the measure until 10-07)
    assert st["on_screen_while_cutting"] == 0.75        # on the screen: 3 of 4
    assert (st["hidden_on_screen_frames"], st["off_screen_frames"]) == (2, 1)
