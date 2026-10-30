"""End-to-end on a real rig recording. Recordings are not committed: point SPINE_REC_SAMPLE at one, e.g.
SPINE_REC_SAMPLE=…/sessions/2026-09-24/s_20260924T000500_altician_lumbar1_trk_0551.jsonl pytest tests/test_recorder_v2_sample.py
"""

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from spine_il_converter.cli import main as cli_main
from spine_il_converter.episode import build_episode
from spine_il_converter.episode_v2 import build_rec_episode
from spine_il_converter.session import load_session

SAMPLE = Path(os.environ.get("SPINE_REC_SAMPLE", "")).expanduser()
pytestmark = pytest.mark.skipif(not os.environ.get("SPINE_REC_SAMPLE") or not SAMPLE.is_file(),
                                reason="set SPINE_REC_SAMPLE to a real recorder v2 session (.jsonl)")
STEP = 1000.0 / 30


@pytest.fixture(scope="module")
def converted():
    rec = load_session(SAMPLE)
    return rec, build_episode(rec, SAMPLE.parent)


def test_real_session_converts_on_the_texp_axis(converted):
    rec, res = converted
    assert rec.format == "rec2" and rec.errors == [], [str(e) for e in rec.errors]
    assert res.included, (res.reason, res.detail)
    ep = res.episode
    assert ep.n_frames >= 30 and ep.time_axis == "tExp"
    t_lo = max(tr.t[tr.valid][0] for tr in rec.tools.values())  # the first camera frame both tools share
    k = (ep.t0_ms - t_lo) / STEP
    assert k >= 0 and abs(k - round(k)) < 1e-6  # frame 0 sits on the tExp grid anchored there, not on tSrc
    np.testing.assert_allclose(np.diff(ep.tracker_time[:, 0]), 1 / 30, atol=1e-4)
    assert np.isfinite(ep.state).all() and np.isfinite(ep.endo_roll).all()
    roll = ep.endo_roll[:, 0]
    assert np.ptp(roll) > 1.0 and np.abs(roll).max() > 1.0  # the roll is not the old constant 0
    assert ep.stats["endo_roll_rule_err_max_deg"] < 0.01  # replicated viewer rule == recorded drawn/real roll
    assert ep.stats["cut_rows_in_episode"] <= ep.stats["cut_rows"] == len(rec.cuts)
    if ep.stats["cut_rows_in_episode"] < ep.stats["cut_rows"]:
        assert any("stamps (cut_rows) fall outside" in w for w in ep.stats["warnings"])
    assert max(ep.stats["endo_masked_frac"], ep.stats["drill_masked_frac"]) <= 0.05


def test_real_session_basal_survives_a_shaky_portal(converted):
    """±2.5 mm alternating on the inserted entries leaves no stable window; the portal must stay put (it used to
    fall back to a pre-insertion window 80 mm away on 0551)."""
    rec, res = converted
    endo = rec.tools["endo"]
    inserted = np.flatnonzero(endo.entry_depth > 0)
    if inserted.size < 20:
        pytest.skip("the endoscope is never inserted in this recording")
    entry = endo.entry.copy()
    entry[inserted, 0] += 2.5 * np.where(np.arange(inserted.size) % 2 == 0, 1.0, -1.0)
    shaken = dataclasses.replace(rec, tools={**rec.tools, "endo": dataclasses.replace(endo, entry=entry)})
    basal = build_rec_episode(shaken).episode.stats["basal"]
    moved = np.linalg.norm(np.asarray(basal["origin"]) - res.episode.stats["basal"]["origin"])
    assert basal["rule"].startswith("inserted") and moved < 10.0, (basal, moved)


@pytest.mark.slow
def test_real_session_writes_a_lerobot_dataset(converted, tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    rec, res = converted
    ep = res.episode
    root = tmp_path / "data"
    link = root / "sessions" / SAMPLE.parent.name / SAMPLE.name
    link.parent.mkdir(parents=True)
    link.symlink_to(SAMPLE.resolve())
    out = tmp_path / "ds"
    assert cli_main(["build", "--data-root", str(root), "--out", str(out), "--repo-id", "local/rec2-sample"]) == 0
    ds = LeRobotDataset("local/rec2-sample", root=out)
    assert ds.meta.total_episodes == 1 and ds.meta.total_frames == ep.n_frames == len(ds)
    for k in (0, ep.n_frames // 2, ep.n_frames - 1):
        np.testing.assert_allclose(ds[k]["observation.state"].numpy(), ep.state[k], atol=1e-6)
        np.testing.assert_allclose(ds[k]["label.drill_cut"].numpy(), ep.drill_cut[k])
    cut = (out / "meta/spine_il/recordings" / rec.session_id / "cut.jsonl").read_text().splitlines()
    assert cut == [line for line, _ in rec.cuts]  # verbatim, full-precision stamp centres
    record = json.loads((out / "meta/spine_il/episodes.jsonl").read_text().splitlines()[0])
    assert record["time_axis"] == "tExp" and record["n_frames"] == ep.n_frames
