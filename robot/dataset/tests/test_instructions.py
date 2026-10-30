"""dataset/instructions.py — the texts a frame's task comes from (2026-10-13)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import instructions as I  # noqa: E402
import phases as PH  # noqa: E402


def test_phase1_is_the_legacy_text():
    assert all(I.text("phase1", p, legacy=PH.PHASES) == PH.PHASES[p] for p in PH.PHASES)


def test_extent_from_the_medial_reach():
    assert I.extent_of([28, 27, 25, 24, 30]) == "ipsilateral"
    assert I.extent_of(np.r_[np.full(50, 27.0), np.full(10, 20.5)]) == "ipsilateral"     # at the midline, not across it
    assert I.extent_of(np.r_[np.full(50, 27.0), np.full(10, 13.0)]) == "contralateral"
    assert I.extent_of(np.r_[np.full(50, 27.0), np.full(10, 20.3)], "L3") == "contralateral"     # L3's midline lies at 22.6 mm
    assert I.extent_of(np.r_[np.full(50, 27.0), np.full(10, 20.3)], "L4") == "ipsilateral"
    with pytest.raises(ValueError):
        I.extent_of([])


def test_the_words_tell_the_goals_apart():
    drilling = [w for lv in I.LEVELS for ex in I.EXTENTS for w in I.wordings(4, lv, ex)]
    assert len(drilling) == len(set(drilling))          # no sentence stands for two goals
    assert all("left" in w.lower() for w in drilling)   # every recording so far is a left laminectomy
    assert "L3" in I.text("lang1", 3, "L3") and "L5" in I.text("lang1", 4, "L5", "ipsilateral")


def test_one_wording_per_episode_and_all_wordings_used():
    assert I.text("lang1", 4, "L4", "contralateral", key="a") == I.text("lang1", 4, "L4", "contralateral", key="a")
    assert {I.text("lang1", 4, "L4", "contralateral", key=f"e{i}") for i in range(60)} == set(I.wordings(4, "L4", "contralateral"))
    assert I.text("lang1", 4, "L4", "contralateral") == I.wordings(4, "L4", "contralateral")[0]   # what the loop sends by default


def test_tables_are_json():
    json.dumps(I.table("lang1")); json.dumps(I.table("phase1", PH.PHASES))
    with pytest.raises(ValueError):
        I.text("nope", 4)


def test_level_from_the_nearest_mesh():
    d = Path.home() / "il-robot/calib/level_meshes"
    if not (d / "L4.stl").exists():
        pytest.skip("the level meshes are not on this machine")
    l4 = np.array([[28.0, -50.0, 1025.0], [22.0, -48.0, 1028.0], [30.0, -52.0, 1024.0]])   # cut centres of the L4 recordings
    lv, info = I.level_of(l4, d, every=1)
    assert lv == "L4" and info["share"]["L4"] == 1.0
    pts = {n: I._stl_points(d / f"{n}.stl") for n in ("L3", "L5")}
    for n, p in pts.items():                           # points on a level's own surface are that level's
        assert I.level_of(p[::50], d, every=1)[0] == n
    with pytest.raises(ValueError):                    # half on L3, half on L5: no level holds 80 %
        I.level_of(np.vstack([pts["L3"][::200][:20], pts["L5"][::200][:20]]), d, every=1)


def test_lang2_names_only_a_clear_extent():
    """lang2 (10-14): the medial reach on the midline or short of it = ipsilateral, more than 5 mm past = contralateral,
    between = the sentence that names no extent; lang1 keeps its two classes for the same cuts."""
    mid = I.MIDLINE_X_MM["L4"]
    reach = lambda x: np.r_[np.full(90, 30.0), np.full(10, x)]    # the 5th percentile = x
    assert I.extent_label("lang2", reach(mid + 0.5)) == "ipsilateral" and I.extent_label("lang2", reach(mid)) == "ipsilateral"
    assert I.extent_label("lang2", reach(mid - 1.0)) == I.NEUTRAL and I.extent_label("lang1", reach(mid - 1.0)) == "ipsilateral"
    assert I.extent_label("lang2", reach(mid - 4.0)) == I.NEUTRAL and I.extent_label("lang1", reach(mid - 4.0)) == "contralateral"
    assert I.extent_label("lang2", reach(mid - 5.5)) == "contralateral"
    assert I.extent_label("lang2", reach(I.MIDLINE_X_MM["L5"] - 4.8), "L5") == I.NEUTRAL      # bd4c (10-14): 16.7 mm on L5
    with pytest.raises(ValueError):
        I.extent_label("lang2", [])


def test_the_neutral_sentence_is_lang2s():
    neutral = I.wordings(4, "L4", I.NEUTRAL)
    named = [w for ex in I.EXTENTS for w in I.wordings(4, "L4", ex)]
    assert len(set(neutral)) == 3 and not set(neutral) & set(named)
    assert all("ipsilateral" not in w and "contralateral" not in w and "midline" not in w for w in neutral)
    assert I.text("lang2", 4, "L4", I.NEUTRAL) == neutral[0] and I.text("lang2", 4, "L4", "ipsilateral") == I.text("lang1", 4, "L4", "ipsilateral")
    with pytest.raises(ValueError):          # a lang1 model never met it
        I.text("lang1", 4, "L4", I.NEUTRAL)
    assert I.text("lang1", 3, "L4", I.NEUTRAL) == I.text("lang1", 3, "L4")        # the approach names no extent anyway
    t1, t2 = I.table("lang1"), I.table("lang2")
    assert t1["extents"] == list(I.EXTENTS) and I.NEUTRAL not in t1["drilling"]["L4"] and "edge_mm" not in t1
    assert t2["extents"] == [*I.EXTENTS, I.NEUTRAL] and t2["drilling"]["L4"][I.NEUTRAL] == neutral and t2["edge_mm"] == [0.0, 5.0]
    assert set(I.texts(t2)) - set(I.texts(t1)) == {w for lv in I.LEVELS for w in I.wordings(4, lv, I.NEUTRAL)}
    json.dumps(t2)
    with pytest.raises(ValueError):
        I.table("lang9")


def test_goal_parsing_knows_the_neutral_sentence():
    assert I.parse_goal("L4:unspecified") == {"level": "L4", "extent": I.NEUTRAL}
    assert I.parse_goal("neutral") == {"level": "L4", "extent": I.NEUTRAL}
    assert I.parse_goal("L5:ipsi") == {"level": "L5", "extent": "ipsilateral"}
    with pytest.raises(ValueError):
        I.parse_goal("L4:sideways")


def test_lang3_more_wordings_and_the_test_paraphrases_stay_unseen():
    """10-17: lang3 = lang2's labels with 15 wordings per extent (10 neutral); its first three are lang2's, so the robot's
    sentence (the first) is unchanged; none of the held-out proof paraphrases, nor their distinctive words, is trained."""
    for lv in I.LEVELS:
        for ex, n in (("ipsilateral", 15), ("contralateral", 15), (I.NEUTRAL, 10)):
            w3 = I.wordings(4, lv, ex, "lang3")
            assert len(w3) == len(set(w3)) == n and w3[:3] == I.wordings(4, lv, ex, "lang2")
            assert I.text("lang3", 4, lv, ex) == I.text("lang2", 4, lv, ex)
    assert I.extent_label("lang3", [I.MIDLINE_X_MM["L4"] - 1.0] * 20) == I.NEUTRAL and "lang3" in I.SETS
    t = I.texts(I.table("lang3"))
    assert set(I.texts(I.table("lang2"))) < set(t)
    banned = ("right", "bilateral", "unilateral", "take down", "carry on", "undercut", "untouched", "approach", "keep to", "other side")
    drilling = [w for lv in I.LEVELS for ex in (*I.EXTENTS, I.NEUTRAL) for w in I.wordings(4, lv, ex, "lang3")[3:]]
    assert not [w for w in drilling for b in banned if b in w.lower()]
