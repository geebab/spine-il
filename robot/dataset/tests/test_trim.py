"""trim.py: the span rule on synthetic masks, and export_lerobot.py --trim robot end to end (synthetic sessions)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import export_lerobot as E  # noqa: E402
import sources as S  # noqa: E402
import synth  # noqa: E402
import trim as TR  # noqa: E402

HAND = "s_20260928T000000_altician_lumbar1_trk_aaaa"
ROBOT = "s_20260929T000000_altician_lumbar1_trk_bbbb"
RUN = "20260929T000000_replay_pr_aaaa01"


def _masks(t, endo=(2.0, 30.0), drill=(6.0, 28.0)):
    return np.stack([(t >= endo[0]) & (t < endo[1]), (t >= drill[0]) & (t < drill[1])], 1)


def test_span_rule():
    t = np.arange(0, 40, 1 / 15)
    ins = _masks(t)
    # first cut 20 s, both out at 30 s: start = cut − 8 = 12 s (both inside since 6 s), end = 30 s
    ia, ib, info = TR.span(t, ins, 20.0, 30.0, pre_s=8.0, post_s=0.0)
    assert t[ia] == pytest.approx(12.0, abs=1 / 15) and t[ib] == pytest.approx(30.0, abs=1 / 15)
    assert info["t_both_inside"] == pytest.approx(6.0, abs=1 / 15) and info["frames_kept"] == ib - ia + 1
    # a long pre_s is limited by the later tool's skin entry
    ia, _, _ = TR.span(t, ins, 20.0, 30.0, pre_s=60.0)
    assert t[ia] == pytest.approx(6.0, abs=1 / 15)
    # the default cuts the last second before both tools are out; post_s moves the end; a gate-verified span narrows both ends and reports that the rule was outside it
    assert t[TR.span(t, ins, 20.0, 30.0)[1]] == pytest.approx(30.0 + TR.POST_S, abs=1 / 15)
    _, ib, _ = TR.span(t, ins, 20.0, 30.0, pre_s=8.0, post_s=2.0)
    assert t[ib] == pytest.approx(32.0, abs=1 / 15)
    ia, ib, info = TR.span(t, ins, 20.0, 30.0, pre_s=8.0, gate_span=(14.0, 27.5))
    assert t[ia] == pytest.approx(14.0, abs=1 / 15) and t[ib] == pytest.approx(27.5, abs=1 / 15)
    assert info["rule_inside_gate"] is False
    assert TR.span(t, ins, 20.0, 30.0, pre_s=8.0, gate_span=(5.0, 35.0))[2]["rule_inside_gate"] is True


def test_span_flicker_and_too_short():
    t = np.arange(0, 40, 1 / 15)
    ins = _masks(t)
    ins[(t > 19.9) & (t < 20.1), 1] = False          # the drill reads outside at the first cut
    ia, _, info = TR.span(t, ins, 20.0, 30.0, pre_s=8.0)
    assert "flag" in info and t[ia] == pytest.approx(12.0, abs=1 / 15)
    with pytest.raises(ValueError):
        TR.span(t, ins, 20.0, 30.0, pre_s=8.0, gate_span=(25.0, 25.05))


@pytest.fixture()
def world(tmp_path, monkeypatch):
    sessions, runs = tmp_path / "sessions", tmp_path / "runs"
    synth.write_hand(sessions / "2026-09-28" / f"{HAND}.jsonl", HAND)
    synth.write_robot(sessions, runs, ROBOT, HAND, RUN)
    monkeypatch.setattr(S, "RUNS_ROOT", runs)
    return {"sessions": sessions, "tmp": tmp_path}


def _load(out):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/test", root=out)
    hf = ds.hf_dataset.with_format("numpy")
    side = [json.loads(x) for x in (out / "meta/spine_il/episodes.jsonl").read_text().splitlines()]
    return ds, hf, side


def test_export_trim(world, monkeypatch):
    out = world["tmp"] / "ds"
    monkeypatch.setattr(E, "TRIM", [None])
    E.main(["--out", str(out), "--sessions", str(world["sessions"]), "--images", "none", "--repo-id", "local/test",
            "--fps", "15", "--trim", "robot", "--trim-pre-s", "2", "--trim-post-s", "0", "--action-mode", "delta"])
    monkeypatch.setattr(E, "ACTION_MODE", ["absolute"])
    ds, hf, side = _load(out)
    hand, robot = side
    ep = np.asarray(hf["episode_index"]).reshape(-1)
    h = ep == 0
    # synthetic hand: both tools inside 2.0–18.5 s, cuts 5–12 s → kept 3.0 s … ~18.5 s
    tr = hand["trim"]
    assert tr["t_a"] == pytest.approx(3.0, abs=0.07) and tr["t_b"] == pytest.approx(18.5, abs=0.1)
    assert tr["cut_front_s"] == pytest.approx(3.0, abs=0.07) and hand["n_frames"] == int(h.sum()) == tr["frames_kept"] - 1
    ts = np.asarray(hf["time.src_s"]).reshape(-1)[h]
    assert ts[0] == pytest.approx(3.0, abs=0.07) and np.allclose(np.diff(ts), 1 / 15, atol=1e-3)
    st = np.asarray(hf["observation.state"])[h]
    np.testing.assert_allclose(st[0, :3], synth.hand_tip(float(ts[0])), atol=0.6)
    ph = np.asarray(hf["phase.index"]).reshape(-1)
    assert set(ph[h].tolist()) == {3, 4, 5} and np.all(np.diff(ph[h]) >= 0)
    # delta actions still lead to the next state inside the kept part
    import actions as AC
    ac = np.asarray(hf["action"])[h]
    for i in (0, 50, len(st) - 2):
        np.testing.assert_allclose(AC.from_delta(st[i], ac[i])[:3], st[i + 1][:3], atol=1e-3)
    assert hand["trigger_duty_dataset"] == pytest.approx(hand["trigger_duty_source"], abs=0.01)
    # robot episodes are not trimmed
    assert "trim" not in robot and robot["n_frames"] == 149


def test_export_trim_spans(world, monkeypatch):
    out = world["tmp"] / "ds2"
    spans = world["tmp"] / "spans.json"
    spans.write_text(json.dumps({"aaaa": [4.0, 15.0]}))
    monkeypatch.setattr(E, "TRIM", [None])
    E.main(["--out", str(out), "--sessions", str(world["sessions"]), "--images", "none", "--repo-id", "local/test",
            "--fps", "15", "--include", "hand", "--trim", "robot", "--trim-pre-s", "2", "--trim-spans", str(spans)])
    _, hf, side = _load(out)
    tr = side[0]["trim"]
    assert tr["t_a"] == pytest.approx(4.0, abs=0.07) and tr["t_b"] == pytest.approx(15.0, abs=0.07)
    assert tr["rule_inside_gate"] is False
