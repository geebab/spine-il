import json
from dataclasses import replace

import numpy as np
import pytest

from spine_il_converter import schema
from spine_il_converter.episode import build_episode
from spine_il_converter.features import STATE_NAMES, dataset_features
from spine_il_converter.fixtures import (
    HUMAN_PHASE_SHIFT_S,
    PHASE_FRACTIONS,
    T0_SRC_MS,
    standard_specs,
    tool_truth,
    write_session,
)
from spine_il_converter.geometry import basal_frame, points_to_frame, rot6d
from spine_il_converter.session import load_session

SPECS = {s.session_id: s for s in standard_specs()}


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    for spec in SPECS.values():
        write_session(root, spec)
    return root


def convert(root, sid):
    return build_episode(load_session(root / "sessions" / "2026-09-15" / f"{sid}.jsonl"), root)


@pytest.mark.parametrize("sid", [s for s, spec in SPECS.items() if spec.expected_reason])
def test_rejections(data_root, sid):
    result = convert(data_root, sid)
    assert not result.included
    assert result.reason == SPECS[sid].expected_reason, result.detail


@pytest.mark.parametrize("sid", ["s_good_0", "s_good_recovery", "s_good_bridged"])
def test_state_matches_analytic_truth(data_root, sid):
    spec = SPECS[sid]
    ep = convert(data_root, sid).episode
    assert ep is not None and ep.n_frames >= 80
    t = ep.t0_src_ms + np.arange(ep.n_frames) * 1000.0 / 30  # frame k sits on the 30 fps grid
    origin, R_b = basal_frame(spec.entry, spec.target)
    expected = []
    for tool in schema.TOOLS:
        tip, R = tool_truth(spec, tool, t)
        expected += [points_to_frame(tip, origin, R_b), rot6d(np.einsum("ji,njk->nik", R_b, R))]
    expected = np.concatenate(expected, axis=1)
    np.testing.assert_allclose(ep.state[:, :18], expected, atol=2e-4)  # float32 storage of ~100 mm values
    np.testing.assert_array_equal(ep.action[:-1], ep.state[1:])  # action = next frame's absolute state
    assert ep.state.dtype == np.float32 and ep.state.shape[1] == len(STATE_NAMES) == 19


def test_frame_clock_and_phases(data_root):
    spec = SPECS["s_good_0"]
    ep = convert(data_root, "s_good_0").episode
    np.testing.assert_allclose(np.diff(ep.tracker_time[:, 0]), 1 / 30, atol=1e-5)
    assert ep.tracker_time[0, 0] == 0.0
    assert (ep.phase.sum(axis=1) == 1).all()
    ids = ep.phase.argmax(axis=1)
    assert set(ids.tolist()) == {0, 1, 2, 3, 4}
    # Footer (human) boundary for phase 3 is HUMAN_PHASE_SHIFT_S later than the auto boundary.
    t_rel = (ep.t0_src_ms - T0_SRC_MS) / 1000.0 + ep.tracker_time[:, 0]
    first_p3 = t_rel[np.argmax(ids == 3)]
    boundary = PHASE_FRACTIONS[3] * spec.duration_s + HUMAN_PHASE_SHIFT_S
    assert boundary - 0.05 <= first_p3 <= boundary + 0.05
    drill_on = ep.state[:, -1]
    assert drill_on[ids == 0].max() == 0 and drill_on[ids == 3].mean() > 0.8
    assert ep.task == "Left L4-5 HNP: left L4-5 laminectomy (ipsilateral)"


def test_goal_is_area_in_basal_frame(data_root):
    spec = SPECS["s_good_0"]
    ep = convert(data_root, "s_good_0").episode
    origin, R_b = basal_frame(spec.entry, spec.target)
    center = np.asarray(spec.target) + [0.0, -6.0, 0.0]
    np.testing.assert_allclose(ep.goal[:3], points_to_frame(center, origin, R_b), atol=1e-4)
    np.testing.assert_allclose(ep.goal[3:], np.array([0.0, 1.0, 0.0]) @ R_b, atol=1e-6)


def test_bridged_session_reports_stale_captures(data_root):
    stats = convert(data_root, "s_good_bridged").episode.stats
    assert stats["endo_bridged_frames"] > 0 and stats["stale_capture_frames"] > 0


def test_missing_up_falls_back_with_warning(tmp_path):
    spec = replace(SPECS["s_good_0"], session_id="s_no_up", omit_up=True)
    path = write_session(tmp_path, spec)
    session = load_session(path)
    assert any("without scene.up" in i.message for i in session.issues)
    assert build_episode(session, tmp_path).included


def test_schema_errors_carry_line_numbers(tmp_path):
    path = write_session(tmp_path, replace(SPECS["s_good_0"], session_id="s_broken"))
    lines = path.read_text().splitlines()
    m_line = next(i for i, l in enumerate(lines) if json.loads(l)["k"] == "m")
    row = json.loads(lines[m_line])
    del row["exposureEndo"]
    lines[m_line] = json.dumps(row)
    lines.insert(3, "{not json")
    path.write_text("\n".join(lines) + "\n")
    errors = [str(e) for e in load_session(path).errors]
    assert any("line 4: invalid JSON" in e for e in errors)
    assert any(f"line {m_line + 2}: m missing ['exposureEndo']" in e for e in errors)


def test_replay_infeasible_filter(data_root):
    from spine_il_converter.episode import ConvertConfig

    session = load_session(data_root / "sessions" / "2026-09-15" / "s_good_0.jsonl")
    result = build_episode(session, data_root, ConvertConfig(replay_infeasible=frozenset({"s_good_0"})))
    assert result.reason == "replay_infeasible"


def test_drill_on_name_is_unique_substring():
    names = dataset_features()["action"]["names"]
    assert [n for n in names if "drill_on" in n] == ["drill_on"]
