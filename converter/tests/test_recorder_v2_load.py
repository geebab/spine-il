"""Loading real recorder v2 files (rec-2.x): row order, re-sends, validity, poses, viewer roll, drill buttons, header gates."""

import json
from dataclasses import replace

import numpy as np
import pytest
import rec2_synth as syn

from spine_il_converter.recorder_v2 import FORMAT_REC2, detect_format, load_recording

SPEC = syn.RecSpec()


def load(tmp_path, spec=SPEC, **write_kw):
    return load_recording(syn.write_session(tmp_path, spec, **write_kw))


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    rec = load(tmp_path_factory.mktemp("clean"))
    assert not rec.errors, [str(e) for e in rec.errors]
    return rec


def warnings_of(rec) -> str:
    return " | ".join(i.message for i in rec.warnings)


def test_row_order_is_r_and_first_copy_wins(tmp_path, clean):
    path = syn.write_session(tmp_path, SPEC, shuffle=True, resend=True)
    assert json.loads(path.read_text().splitlines()[0])["k"] != "hdr"  # hdr is not line 1
    assert detect_format(path) == FORMAT_REC2
    rec = load_recording(path)
    assert not rec.errors and rec.header == clean.header
    assert rec.stats["duplicate_r"] > 30 and rec.stats["duplicate_r_conflicts"] == 1
    assert "re-sent rows" in warnings_of(rec)
    for tool in ("endo", "drill"):
        a, b = rec.tools[tool], clean.tools[tool]
        for key in ("t", "r", "valid", "tip", "rot", "roll", "trig", "trig_known", "stamped_t", "entry"):
            np.testing.assert_array_equal(getattr(a, key), getattr(b, key), err_msg=f"{tool}.{key}")
    assert [line for line, _ in rec.cuts] == [line for line, _ in clean.cuts]


def test_lost_chunks_unknown_rows_and_keys(tmp_path, clean):
    assert clean.stats["unknown_kinds"] == 1 and clean.stats["r_gaps"] == 0  # the future `img` row is ignored
    ev_rs = [row["r"] for row in syn.build_rows(SPEC) if row["k"] == "ev"][:3]
    rec = load(tmp_path, drop_r=tuple(ev_rs))
    assert rec.stats["r_gaps"] == 3 and "rows missing" in warnings_of(rec)
    np.testing.assert_array_equal(rec.tools["endo"].tip, clean.tools["endo"].tip)


@pytest.mark.parametrize("version", ["rec-2.3.0-t0", "rec-2.3.1-t0", "rec-2.9.7-t0"])
def test_recorder_version_is_metadata_only(tmp_path, version):
    rec = load(tmp_path, replace(SPEC, recorder_version=version))
    assert not rec.errors and rec.time_axis == "tExp" and not rec.is_core


def test_rows_on_the_texp_axis_match_truth(clean):
    for tool in ("endo", "drill"):
        tr = clean.tools[tool]
        np.testing.assert_array_equal(tr.t, syn.frame_times(SPEC))  # tExp, not tSrc
        assert np.all(tr.t_src - tr.t > 20)
        tip, R, roll = syn.truth(tool, tr.t[tr.valid])
        np.testing.assert_allclose(tr.tip[tr.valid], tip, atol=6e-4)
        np.testing.assert_allclose(tr.rot[tr.valid], R, atol=5e-6)
        if tool == "endo":
            np.testing.assert_allclose(tr.roll[tr.valid], roll, atol=2e-3)


def test_tool_models_and_marker_geometry_are_recorded(clean):
    assert clean.tool_models["drill"] == {"file": "tool_calib_drill.json", "sha12": "340b2ac02ad2",
                                          "relay_sha12": "340b2ac02ad2"}
    assert clean.tools["endo"].normal_shaft_deg == pytest.approx(90.0, abs=0.01)  # shaft in the marker plane
    assert clean.tools["drill"].normal_shaft_deg == pytest.approx(69.42, abs=0.01)  # like the rig's drill model


def test_viewer_roll_is_preferred_and_matches_the_rule(clean):
    e = clean.tools["endo"]
    assert 0.5 < e.roll_viewer[e.valid].mean() <= 1  # drawn lags 3 frames, so most messages were drawn
    assert np.nanmax(e.roll_rule_err) < 1e-3
    assert np.ptp(e.roll[e.valid]) > 50  # 15°/s twist for 4 s


def test_invalid_rows_are_flagged_and_still_rows_are_valid(clean):
    e, d = clean.tools["endo"], clean.tools["drill"]
    assert e.counts["lost"] == 4 and e.counts["vis_lt4"] == 2
    assert d.counts["hold"] == 2 and d.counts["no_scene"] == 2
    assert int((~e.valid).sum()) == 6 and int((~d.valid).sum()) == 4
    s = (e.t - syn.T_EXP0) / 1000.0
    assert e.valid[(s >= 1.0) & (s < 1.5)].all()  # still = 1 rows


def test_drill_trigger_rows_hold_the_last_known_state(clean):
    d = clean.tools["drill"]
    s = (d.t - syn.T_EXP0) / 1000.0
    unknown = (s >= SPEC.btn_unknown[0]) & (s < SPEC.btn_unknown[1])
    np.testing.assert_array_equal(d.trig_known, ~unknown)
    pressed = (s >= SPEC.trig[0]) & (s < SPEC.trig[1])
    np.testing.assert_array_equal(d.trig[~unknown], pressed[~unknown].astype(float))
    assert (d.trig[unknown] == 1).all()  # the stall starts while ZR is held


def test_dynamic_registration_uses_each_rows_T(tmp_path):
    spec = replace(SPEC, dynamic=True, no_T=(("drill", 3.0, 3.012),))
    rec = load(tmp_path, spec)
    s = (syn.frame_times(spec) - syn.T_EXP0) / 1000.0
    assert rec.tools["drill"].counts["no_T"] == int(((s >= 3.0) & (s < 3.012)).sum()) > 0
    for tool in ("endo", "drill"):
        tr = rec.tools[tool]
        np.testing.assert_allclose(tr.rot[tr.valid], syn.truth(tool, tr.t[tr.valid], spec)[1], atol=5e-6)


def test_header_gates_warn(tmp_path, clean):
    assert "baseline is null" in warnings_of(clean) and not clean.robot_tools
    rec = load(tmp_path, replace(SPEC, calib_mismatch=True, t0_override=True, robot_tools=True))
    msg = warnings_of(rec)
    assert "drill: toolCalib sha256_12 ffffffffffff != tracker.cfg calib sha12 340b2ac02ad2" in msg
    assert "t0Override" in msg and rec.robot_tools == ["endo2"]
    rec = load(tmp_path, replace(SPEC, baseline={"trackerCfgHash": "8d76f7834c32", "feCfgHash": "a4c7b9adfc50",
                                                 "setAt": 1, "match": True}))
    assert "baseline" not in warnings_of(rec)
    rec = load(tmp_path, replace(SPEC, timing_spec="v2-core"))
    assert rec.is_core and rec.time_axis == "tSrc" and not rec.tools["drill"].trig_known.any()
