"""Real recorder v2 input (rec-2.x) → episodes: poses on the tExp axis, masks and gates, drill channels, basal, pieces,
malformed input, and the LeRobot dataset."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import rec2_synth as syn

from spine_il_converter.cli import main as cli_main
from spine_il_converter.episode import ConvertConfig, build_episode
from spine_il_converter.episode_v2 import REC_STATE_NAMES, RecPieces, build_rec_episode
from spine_il_converter.fixtures import FixtureSpec
from spine_il_converter.fixtures import write_session as write_plan0_session
from spine_il_converter.geometry import points_to_frame, rot6d, rot6d_to_matrix
from spine_il_converter.recorder_v2 import FORMAT_PLAN0, FORMAT_REC2, Recording, detect_format, load_recording, wrap_deg
from spine_il_converter.session import Session, load_session

SPEC = syn.RecSpec()
STEP = 1000.0 / 30


def convert(tmp_path, spec=SPEC, cfg=ConvertConfig(), **write_kw):
    path = syn.write_session(tmp_path, spec, **write_kw)
    rec = load_session(path)
    return rec, build_episode(rec, tmp_path, cfg)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    rec, res = convert(tmp_path_factory.mktemp("clean"))
    assert res.included, (res.reason, res.detail)
    return rec, res.episode


def grid_of(ep):
    return ep.t0_ms + np.arange(ep.n_frames) * STEP


def basal_of(ep):
    b = ep.stats["basal"]
    return np.asarray(b["origin"]), np.column_stack([b["x"], b["y"], b["z"]])


def rows_valid(spec, tool):
    """Row times and validity straight from the spec (independent of the converter)."""
    times = syn.frame_times(spec)
    bad = [w for w in spec.invalid if w[0] == tool]
    s = (times - syn.T_EXP0) / 1000.0
    valid = np.array([not any(w[2] <= x < w[3] for w in bad) for x in s])
    return times, valid


def expected_mask(times, valid, grid, max_gap_ms=250.0):
    out = []
    for g in grid:
        i = np.searchsorted(times, g, side="right") - 1
        if i < 0:
            out.append(False)
        elif times[i] == g:
            out.append(bool(valid[i]))
        else:
            out.append(bool(i + 1 < len(times) and valid[i] and valid[i + 1] and times[i + 1] - times[i] <= max_gap_ms))
    return np.array(out)


def expected_state(spec, times_by_tool, ep):
    """State rows from the analytic truth at per-tool sample times, in the episode's basal frame."""
    origin, R_b = basal_of(ep)
    parts = []
    for tool in ("endo", "drill"):
        tip, R, _ = syn.truth(tool, times_by_tool[tool], spec)
        parts += [points_to_frame(tip, origin, R_b), rot6d(np.einsum("ji,njk->nik", R_b, R))]
    return np.concatenate(parts, axis=1)


def rows_with(spec, mutate):
    rows = syn.build_rows(spec)
    mutate(rows)
    return rows


# ---------------------------------------------------------------- format and row order


def test_format_detection_keeps_plan0_fixtures(tmp_path):
    plan0 = write_plan0_session(tmp_path, FixtureSpec(session_id="s_plan0"))
    real = syn.write_session(tmp_path, SPEC)
    assert detect_format(plan0) == FORMAT_PLAN0 and isinstance(load_session(plan0), Session)
    assert detect_format(real) == FORMAT_REC2 and isinstance(load_session(real), Recording)


def test_shuffled_resent_file_gives_the_same_episode(tmp_path, clean):
    _, ep = clean
    _, res = convert(tmp_path, shuffle=True, resend=True)
    for key in ("state", "action", "endo_roll", "drill_cut", "mask", "t_src_lag"):
        np.testing.assert_array_equal(getattr(res.episode, key), getattr(ep, key), err_msg=key)


def test_unknown_recorder_patch_version_is_kept_as_metadata(tmp_path):
    _, res = convert(tmp_path, replace(SPEC, recorder_version="rec-2.9.7-t0"))
    assert res.included and res.episode.stats["recorder_version"] == "rec-2.9.7-t0"


# ---------------------------------------------------------------- poses on the tExp axis


def test_state_matches_truth_on_the_texp_axis(clean):
    rec, ep = clean
    assert ep.time_axis == "tExp" and ep.t0_ms == syn.T_EXP0  # frame 0 = the first camera frame both tools share
    grid = grid_of(ep)
    np.testing.assert_allclose(np.diff(ep.tracker_time[:, 0]), 1 / 30, atol=1e-5)
    ok = ep.mask[:, 0].astype(bool) & ep.mask[:, 1].astype(bool)
    assert ok.sum() > 100
    want = expected_state(SPEC, {"endo": grid, "drill": grid}, ep)
    np.testing.assert_allclose(ep.state[ok][:, [0, 1, 2, 9, 10, 11]], want[ok][:, [0, 1, 2, 9, 10, 11]], atol=2e-3)
    np.testing.assert_allclose(ep.state[ok][:, 3:9], want[ok][:, 3:9], atol=1e-4)
    np.testing.assert_allclose(ep.state[ok][:, 12:18], want[ok][:, 12:18], atol=1e-4)
    np.testing.assert_array_equal(ep.action[:-1], ep.state[1:])
    # the same poses stamped with tSrc (~24.5 ms later) would be off by the tools' speed × lag
    shifted = expected_state(SPEC, {t: grid + syn.TSRC_LAG_MS for t in ("endo", "drill")}, ep)
    assert np.abs(want[ok][:, :3] - shifted[ok][:, :3]).max() > 0.3
    assert 20 < ep.stats["endo_tsrc_minus_texp_ms_p50"] < 35
    assert ep.state.dtype == np.float32 and ep.state.shape[1] == len(REC_STATE_NAMES) == 19
    assert np.isfinite(ep.state).all()


def test_roll_is_the_viewer_roll_and_lives_in_rot6d(clean):
    _, ep = clean
    grid = grid_of(ep)
    ok = ep.mask[:, 0].astype(bool)
    _, _, roll = syn.truth("endo", grid)
    np.testing.assert_allclose(ep.endo_roll[ok, 0], roll[ok], atol=2e-3)
    assert np.ptp(ep.endo_roll[:, 0]) > 50  # twisting 15°/s for 4 s — never a constant 0
    assert ep.stats["endo_roll_rule_err_max_deg"] < 1e-3 and 0 < ep.stats["endo_roll_viewer_frac"] <= 1
    # rot6d keeps the roll: the endoscope's x axis is the viewer's camera up (marker normal ⟂ shaft)
    _, R_b = basal_of(ep)
    R = R_b @ rot6d_to_matrix(ep.state[:, 3:9].astype(np.float64))
    x, z = R[:, :, 0], R[:, :, 2]
    a = np.array([0.0, 0.0, 1.0]) - z[:, 2:3] * z
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    from_rot6d = np.degrees(np.arctan2(np.sum(np.cross(a, x) * z, axis=1), np.sum(a * x, axis=1)))
    np.testing.assert_allclose(from_rot6d[ok], ep.endo_roll[ok, 0], atol=1e-3)


def test_roll_is_unwrapped_within_the_episode(tmp_path):
    rec = load_recording(syn.write_session(tmp_path, SPEC))
    endo = rec.tools["endo"]
    endo.roll[:] = wrap_deg(np.linspace(150.0, 250.0, len(endo.roll)))  # a scope turned through ±180°
    roll = build_rec_episode(rec).episode.endo_roll[:, 0]
    assert roll.max() > 200 and np.abs(np.diff(roll)).max() < 10.0  # continuous, not folded back by 360°


# ---------------------------------------------------------------- validity masks and their gates


def test_invalid_rows_are_masked_and_held_not_interpolated(clean):
    rec, ep = clean
    grid = grid_of(ep)
    for col, tool in enumerate(("endo", "drill")):
        times, valid = rows_valid(SPEC, tool)
        want = expected_mask(times, valid, grid)
        np.testing.assert_array_equal(ep.mask[:, col].astype(bool), want, err_msg=tool)
        masked = np.flatnonzero(~want)
        assert masked.size >= 2, tool
        last_valid = [times[valid & (times <= g)][-1] for g in grid[masked]]
        held = expected_state(SPEC, {"endo": last_valid, "drill": last_valid}, ep)
        cols = slice(0, 3) if tool == "endo" else slice(9, 12)
        np.testing.assert_allclose(ep.state[masked, cols], held[:, cols], atol=2e-3)
    assert ep.stats["endo_masked_frames"] == int((ep.mask[:, 0] == 0).sum())
    assert ep.stats["endo_masked_frac"] == round(ep.stats["endo_masked_frames"] / ep.n_frames, 4)
    counts = {t: rec.tools[t].counts for t in ("endo", "drill")}
    assert counts["endo"]["lost"] == 4 and counts["endo"]["vis_lt4"] == 2
    assert counts["drill"]["hold"] == 2 and counts["drill"]["no_scene"] == 2
    still = (syn.frame_times(SPEC) - syn.T_EXP0 >= 1000) & (syn.frame_times(SPEC) - syn.T_EXP0 < 1500)
    assert rec.tools["endo"].valid[still].all()  # still rows are valid measurements


def test_masked_fraction_inside_the_episode_is_gated(tmp_path):
    # 3-ball bursts of 90 ms every 200 ms: never lost, no run > 250 ms, yet a third of the endoscope frames are held
    bursts = tuple(("endo", "vis", 0.5 + 0.2 * k, 0.59 + 0.2 * k) for k in range(15))
    _, res = convert(tmp_path, replace(SPEC, invalid=bursts))
    assert res.reason == "masked" and res.detail.startswith("endo masked in")
    _, res = convert(tmp_path, replace(SPEC, invalid=bursts), ConvertConfig(max_masked_frac=0.5))
    assert res.included and 0.25 < res.episode.stats["endo_masked_frac"] <= 0.5


def test_lost_rows_outside_the_episode_only_warn(tmp_path):
    spec = replace(SPEC, invalid=(("endo", "lost", 0.0, 0.5),))  # 12 % of the recording, all before the episode
    rec, res = convert(tmp_path, spec)
    assert rec.tools["endo"].lost_frac > 0.1
    assert res.included and res.episode.stats["endo_masked_frames"] == 0
    assert any("rows are lost" in w for w in res.episode.stats["warnings"])


def test_long_masked_run_excludes_or_splits_on_request(tmp_path):
    spec = replace(SPEC, duration_s=8.0, invalid=(("endo", "lost", 3.4, 3.8),))
    _, res = convert(tmp_path, spec)
    assert res.reason == "gap"  # a masked run longer than max_gap_ms is never bridged
    _, res = convert(tmp_path, spec, ConvertConfig(split_on_gap=True))
    pieces = res.episode
    assert isinstance(pieces, RecPieces) and [p.piece for p in pieces.pieces] == [0, 1]
    first, second = pieces.pieces
    gap = (syn.T_EXP0 + 3400.0, syn.T_EXP0 + 3800.0)  # lost rows in [3.4 s, 3.8 s)
    assert first.t0_ms + first.n_frames * STEP < gap[0] and second.t0_ms >= gap[1]  # the gap is in neither piece
    assert all(p.mask[:, :2].all() for p in pieces.pieces) and all(p.session_id == SPEC.session_id for p in pieces.pieces)
    assert min(p.stats["duration_s"] for p in pieces.pieces) >= 3.0
    np.testing.assert_array_equal(second.action[:-1], second.state[1:])
    # a piece shorter than 3 s is dropped, the rest kept
    _, res = convert(tmp_path, replace(spec, invalid=(("endo", "lost", 2.0, 2.4),)), ConvertConfig(split_on_gap=True))
    assert [p.piece for p in res.episode.pieces] == [1]
    assert res.episode.stats["dropped_pieces"][0]["reason"] == "too_short"


# ---------------------------------------------------------------- drill channels


def test_drill_trig_holds_last_known_value(clean):
    _, ep = clean
    grid = grid_of(ep)
    times = syn.frame_times(SPEC)
    s = (times - syn.T_EXP0) / 1000.0
    trig_rows, known_rows, last = [], [], 0.0
    for x in s:
        if SPEC.btn_unknown[0] <= x < SPEC.btn_unknown[1]:
            known_rows.append(0.0)
        else:
            last = float(SPEC.trig[0] <= x < SPEC.trig[1])
            known_rows.append(1.0)
        trig_rows.append(last)
    idx = np.searchsorted(times, grid, side="right") - 1
    np.testing.assert_array_equal(ep.state[:, 18], np.asarray(trig_rows)[idx])
    np.testing.assert_array_equal(ep.mask[:, 2], np.asarray(known_rows)[idx])
    t_rel = (grid - syn.T_EXP0) / 1000.0
    held = (t_rel >= 2.5) & (t_rel < 2.7)  # released during the stall, but the last known state was pressed
    assert held.any() and (ep.state[held, 18] == 1).all() and (ep.mask[held, 2] == 0).all()


def _expected_cut_frames(spec, grid):
    cut_t = np.array([row["tExp"] for row in syn.build_rows(spec) if row["k"] == "cut"])
    return np.array([float(np.any((cut_t > g - STEP) & (cut_t <= g))) for g in grid]), cut_t


def test_drill_cut_from_cut_rows_or_proxy_stamped(tmp_path, clean):
    _, ep = clean
    want, cut_t = _expected_cut_frames(SPEC, grid_of(ep))
    np.testing.assert_array_equal(ep.drill_cut[:, 0], want)
    assert ep.stats["drill_cut_source"] == "cut_rows" and ep.stats["cut_rows"] == len(cut_t) > 10
    _, res = convert(tmp_path, replace(SPEC, cut_rows=False))  # rec-2.2: no cut rows, engine snapshots only
    assert res.episode.stats["drill_cut_source"] == "proxy_stamped"
    np.testing.assert_array_equal(res.episode.drill_cut[:, 0], want)


# ---------------------------------------------------------------- basal frame


def test_basal_from_the_inserted_stable_entry(clean, tmp_path):
    _, ep = clean
    basal = ep.stats["basal"]
    assert basal["source"] == "drawn" and basal["rule"] == "inserted_stable_window" and basal["values"] >= 8
    np.testing.assert_allclose(basal["origin"], syn.PORTAL, atol=0.3)  # not the pre-insertion skin hit
    np.testing.assert_allclose(basal["z"], syn.ENDO.u, atol=1e-5)
    assert not any(w.startswith("basal") for w in ep.stats["warnings"])
    _, res = convert(tmp_path, replace(SPEC, drawn=False))  # real entries (~6 per second): the inserted median
    assert res.episode.stats["basal"]["source"] == "real"
    np.testing.assert_allclose(res.episode.stats["basal"]["origin"], syn.PORTAL, atol=0.3)
    _, res = convert(tmp_path, replace(SPEC, drawn=False, real_entries=False, header_basal=True))
    assert res.episode.stats["basal"]["rule"] == "header"
    assert any(w.startswith("basal origin from header") for w in res.episode.stats["warnings"])
    _, res = convert(tmp_path, replace(SPEC, drawn=False, real_entries=False))
    assert res.reason == "basal"


def test_basal_never_uses_pre_insertion_or_frozen_entries(tmp_path):
    far = np.linalg.norm(syn.PRE_INSERT_ENTRY - syn.PORTAL)
    # a shaky portal (±2.5 mm): no stable inserted window, while the pre-insertion entry is stable — take the
    # inserted median and warn, never the earlier window (0551 perturbed this way used to move 80 mm)
    _, res = convert(tmp_path, replace(SPEC, entry_alternate_mm=2.5))
    basal = res.episode.stats["basal"]
    assert basal["rule"] == "inserted_median" and np.linalg.norm(np.asarray(basal["origin"]) - syn.PORTAL) < 2.6 < far
    assert any(w.startswith("basal origin from drawn/inserted_median") for w in res.episode.stats["warnings"])
    # an entry frozen 30 mm off for 1.2 s after insertion (a re-used entry, 1170-like): no evidence, skipped
    _, res = convert(tmp_path, replace(SPEC, frozen_entry=(0.0, 1.7, (30.0, 0.0, 0.0))))
    basal = res.episode.stats["basal"]
    assert basal["rule"] == "inserted_stable_window" and basal["window_s"][0] >= 1.7
    np.testing.assert_allclose(basal["origin"], syn.PORTAL, atol=0.3)
    # the scope moved to a second portal 30 mm away (1170-like): keep the first, but say so
    _, res = convert(tmp_path, replace(SPEC, reinserted=(1.6, (0.0, 0.0, -30.0))))
    basal = res.episode.stats["basal"]
    assert basal["rule"] == "inserted_stable_window" and basal["inserted_median_dist_mm"] > 10
    np.testing.assert_allclose(basal["origin"], syn.PORTAL, atol=0.3)
    assert any("inserted at more than one place" in w for w in res.episode.stats["warnings"])


# ---------------------------------------------------------------- gates


def test_gates_reject_unless_allowed(tmp_path):
    rec, res = convert(tmp_path, replace(SPEC, timing_spec="v2-core"))
    assert res.reason == "timing_core" and rec.time_axis == "tSrc"
    _, res = convert(tmp_path, replace(SPEC, timing_spec="v2-core"), ConvertConfig(allow_core=True))
    assert res.included and res.episode.time_axis == "tSrc" and res.episode.stats["drill_trig_known_frac"] == 0.0
    for robot in (replace(SPEC, robot_tools=True), replace(SPEC, drill_calib_file="tool_calib_drill_robotR.json"),
                  replace(SPEC, drill_sha12="162a00282c56")):
        rec, res = convert(tmp_path, robot)
        assert res.reason == "robot_tools" and rec.robot_tools, rec.robot_tools
        _, res = convert(tmp_path, robot, ConvertConfig(allow_robot_tools=True))
        assert res.included
    _, res = convert(tmp_path, replace(SPEC, calib_mismatch=True, t0_override=True))
    assert res.included and "t0Override: the start gate was overridden by hand" in res.episode.stats["warnings"]
    _, res = convert(tmp_path, replace(SPEC, verdict="bad"))
    assert res.reason == "verdict_bad"
    _, res = convert(tmp_path, replace(SPEC, footer=False))
    assert res.reason == "no_footer"
    no_drill = [row for row in syn.build_rows(SPEC) if row.get("tool") != "drill"]
    rec = load_recording(syn.write_session(tmp_path, SPEC, rows=no_drill))
    assert build_rec_episode(rec).reason == "no_rows"


def test_malformed_rows_are_masked_not_fatal(tmp_path):
    def endo_pose_rows(rows):
        return [r for r in rows if r["k"] == "f" and r["tool"] == "endo" and "quat" in r]

    def breaks(rows):
        endo = endo_pose_rows(rows)
        endo[100]["quat"] = [0.0, 0.0, 0.0, 0.0]
        endo[140]["scene"]["u"] = [0.0, 0.0, 0.0]
        endo[150]["seq"] = [1]  # an unhashable message key
        endo[160]["drawn"] = "?"
        rows[0]["stream"]["tools"] = [{"name": "endo"}, {"name": "drill"}]
        rows.insert(5, {"k": "f", "r": 0, "tool": ["endo"], "tExp": syn.T_EXP0})
        for r, row in enumerate(rows):
            row["r"] = r

    rec = load_recording(syn.write_session(tmp_path, SPEC, rows=rows_with(SPEC, breaks)))
    assert not rec.errors and rec.tools["endo"].counts["bad_pose"] == 2 and rec.stats["unknown_tool_rows"] == 1
    res = build_rec_episode(rec)
    assert res.included and np.isfinite(res.episode.state).all() and np.isfinite(res.episode.endo_roll).all()

    def shear(rows):
        R = np.asarray(rows[0]["registration"]["T0"]["R"]).reshape(3, 3) @ np.array([[1, 0.3, 0], [0, 1, 0], [0, 0, 1.0]])
        rows[0]["registration"]["T0"]["R"] = R.ravel().tolist()  # det 1, not a rotation

    rec = load_recording(syn.write_session(tmp_path, SPEC, rows=rows_with(SPEC, shear)))
    assert build_rec_episode(rec).reason == "schema" and "not a rotation" in str(rec.errors[0])


@pytest.mark.parametrize("sid", ["../../outside", "..", "s id with spaces"])
def test_unsafe_session_ids_are_rejected(tmp_path, sid):
    rows = rows_with(SPEC, lambda rows: rows[0].update(sessionId=sid))
    rec = load_recording(syn.write_session(tmp_path, SPEC, rows=rows))
    assert build_rec_episode(rec).reason == "schema" and "not a safe id" in str(rec.errors[0])


# ---------------------------------------------------------------- LeRobot dataset


def test_feature_names_and_policy_inputs():
    from lerobot.utils.feature_utils import dataset_to_policy_features

    from spine_il_converter.writer_v2 import rec_dataset_features

    features = rec_dataset_features()
    names = features["action"]["names"]
    assert [n for n in names if "drill_trig" in n] == ["drill_trig"]  # GR00T relative-action exclusion
    policy = dataset_to_policy_features(features)
    assert {"observation.state", "action", "observation.endo_roll_deg", "observation.tracker_time"} == set(policy)
    assert "label.drill_cut" in features  # a label on pose time, never a policy input


@pytest.mark.slow
def test_rec2_dataset_round_trip(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "data"
    good = syn.write_session(root, SPEC)
    syn.write_session(root, replace(SPEC, session_id="s_20260926T120500_synth_trk_bad0", verdict="bad"))
    out = tmp_path / "ds"
    assert cli_main(["build", "--data-root", str(root), "--out", str(out), "--repo-id", "local/rec2"]) == 0
    ds = LeRobotDataset("local/rec2", root=out)
    ep = build_episode(load_session(good), root).episode
    assert ds.meta.total_episodes == 1 and ds.meta.total_frames == ep.n_frames == len(ds)
    assert ds.meta.features["observation.state"]["names"] == REC_STATE_NAMES
    for k in (0, ep.n_frames // 2, ep.n_frames - 1):
        item = ds[k]
        np.testing.assert_allclose(item["observation.state"].numpy(), ep.state[k], atol=1e-6)
        np.testing.assert_allclose(item["action"].numpy(), ep.action[k], atol=1e-6)
        np.testing.assert_allclose(item["observation.endo_roll_deg"].numpy(), ep.endo_roll[k], atol=1e-4)
        np.testing.assert_allclose(item["label.drill_cut"].numpy(), ep.drill_cut[k])
        np.testing.assert_allclose(item["mask"].numpy(), ep.mask[k])
        assert item["task"] == ep.task
    side = out / "meta/spine_il"
    record = json.loads((side / "episodes.jsonl").read_text().splitlines()[0])
    assert record["t0_ms"] == syn.T_EXP0 and record["time_axis"] == "tExp" and record["piece"] is None
    src_lines = good.read_text().splitlines()
    assert (side / "recordings" / SPEC.session_id / "hdr.json").read_text().splitlines() == [src_lines[0]]
    cut_lines = [line for line in src_lines if json.loads(line)["k"] == "cut"]
    assert (side / "recordings" / SPEC.session_id / "cut.jsonl").read_text().splitlines() == cut_lines  # verbatim
    report = json.loads((side / "conversion_report.json").read_text())
    assert report["format"] == "rec2" and [e["reason"] for e in report["excluded"]] == ["verdict_bad"]
    assert report["warnings"] == []


@pytest.mark.slow
def test_pieces_duplicates_and_mixed_models_in_one_dataset(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "data"
    split_spec = replace(SPEC, session_id="s_20260926T121000_synth_trk_sp01", duration_s=8.0,
                         invalid=(("endo", "lost", 3.4, 3.8),))
    syn.write_session(root, split_spec)
    syn.write_session(root, replace(SPEC, session_id="s_20260926T121500_synth_trk_md01", drill_sha12="aaaaaaaaaaaa"))
    copy = syn.write_session(root, SPEC)
    dup = copy.with_name("zz_copy_of_" + copy.name)  # the same sessionId in a second file
    dup.write_text(copy.read_text())
    out = tmp_path / "ds"
    assert cli_main(["build", "--data-root", str(root), "--out", str(out), "--repo-id", "local/rec2-pieces",
                     "--split-on-gap"]) == 0
    ds = LeRobotDataset("local/rec2-pieces", root=out)
    side = out / "meta/spine_il"
    records = [json.loads(line) for line in (side / "episodes.jsonl").read_text().splitlines()]
    assert ds.meta.total_episodes == len(records) == 4  # 2 pieces + 2 whole recordings
    pieces = [r for r in records if r["session_id"] == split_spec.session_id]
    assert [r["piece"] for r in pieces] == [0, 1] and len({r["split"] for r in pieces}) == 1
    assert sum(r["n_frames"] for r in records) == ds.meta.total_frames
    report = json.loads((side / "conversion_report.json").read_text())
    assert [(e["reason"], Path(e["source"]).name) for e in report["excluded"]] == [("duplicate_session", dup.name)]
    assert any("drill: the dataset mixes tool models" in w for w in report["warnings"])
    assert (side / "recordings" / split_spec.session_id / "cut.jsonl").exists()
