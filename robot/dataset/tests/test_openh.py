"""export_lerobot.py --layout openh on synthetic sessions: block layout and units against our own layout, stamps, texts,
splits, the files written after finalize() (no render API, no real files)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import export_lerobot as E  # noqa: E402
import openh as OH  # noqa: E402
import render_cache as RC  # noqa: E402
import sources as S  # noqa: E402
import synth  # noqa: E402

IDS = ["aaaa", "bbbb", "cccc", "dddd"]


@pytest.fixture()
def world(tmp_path, monkeypatch):
    sessions = tmp_path / "sessions"
    for k, i in enumerate(IDS):   # recording order = the order of the ids
        sid = f"s_2026092{8 + k // 2}T00000{k}_altician_lumbar1_trk_{i}"
        synth.write_hand(sessions / f"2026-09-2{8 + k // 2}" / f"{sid}.jsonl", sid)
    monkeypatch.setattr(S, "RUNS_ROOT", tmp_path / "runs")
    return {"sessions": sessions, "tmp": tmp_path}


def _ds(out):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/test", root=out)
    return ds, ds.hf_dataset.with_format("numpy")


def _side(out):
    return [json.loads(x) for x in (out / "meta/spine_il/episodes.jsonl").read_text().splitlines()]


def test_blocks_units_stamps_texts(world):
    out_s, out_o = world["tmp"] / "spine", world["tmp"] / "openh"
    common = ["--sessions", str(world["sessions"]), "--images", "none", "--repo-id", "local/test", "--include", "hand", "--fps", "20"]
    E.main(["--out", str(out_s)] + common)
    E.main(["--out", str(out_o), "--layout", "openh"] + common)
    _, a = _ds(out_s)
    ds, b = _ds(out_o)
    s15, s16, a16 = np.asarray(a["observation.state"]), np.asarray(b["observation.state"]), np.asarray(b["action"])
    assert s16.shape == (len(s15), 16) and ds.meta.fps == 20 and ds.meta.total_episodes == 4
    # the same frames as our layout (the default split keeps the recording order): metres, xyzw, a gripper slot per tool
    np.testing.assert_allclose(s16[:, 0:3], s15[:, 0:3] / 1000, atol=1e-7)
    np.testing.assert_allclose(s16[:, 3:7], s15[:, [4, 5, 6, 3]], atol=1e-7)
    np.testing.assert_allclose(s16[:, 8:11], s15[:, 7:10] / 1000, atol=1e-7)
    np.testing.assert_allclose(s16[:, 11:15], s15[:, [11, 12, 13, 10]], atol=1e-7)
    assert not s16[:, 7].any() and not a16[:, 7].any()
    np.testing.assert_array_equal(s16[:, 15], s15[:, 14])
    ep = np.asarray(b["episode_index"]).reshape(-1)
    for e in range(4):
        np.testing.assert_array_equal(a16[ep == e][:-1], s16[ep == e][1:])
    # stamps: frame k at BASE + k · 50 ms (Unix ns); the pose starts from the 30 Hz tracker sample at or before it
    ns = np.asarray(b["observation.meta.host_stamp_ns"]).reshape(-1)[ep == 0]
    assert ns.dtype == np.int64 and ns[0] == int(synth.BASE) * 1_000_000 and np.all(np.diff(ns) == 50_000_000)
    src = np.asarray(b["observation.meta.drill_stamp_ns"]).reshape(-1)[ep == 0]
    assert np.all(src <= ns) and np.all(ns - src < 34_000_000)
    lost = np.asarray(b["observation.meta.endoscope_stamp_ns"]).reshape(-1)[ep == 0]
    assert (ns - lost).max() > 34_000_000          # the endoscope is lost for 0.1 s at 10 s: held from the last sample
    ph = np.asarray(b["observation.meta.phase_index"]).reshape(-1)
    assert set(ph[ep == 0].tolist()) == {1, 2, 3, 4, 5, 6}
    # texts: one episode-level task, the phase's instruction per frame, the instruments
    item = ds[0]
    assert item["task"] == "drill the L4 L5 lamina under endoscope view" and item["instruction.text"] == "hold the ready pose"
    assert "burr, radius 2 mm (medium)" in item["observation.meta.tool.drill"] and len(ds.meta.tasks) == 1
    mid = ds[int(np.flatnonzero((ep == 0) & (ph == 4))[0])]
    assert mid["instruction.text"] == "drill the L4-5 lamina"
    # after finalize(): splits as index ranges, the README, no camera file without images; the sidecar names no one
    info = json.loads((out_o / "meta/info.json").read_text())
    assert info["splits"] == {"train": "0:2", "val": "2:3", "test": "3:4"}
    assert info["features"]["observation.state"]["names"] == OH.STATE_NAMES == info["features"]["action"]["names"]
    readme = (out_o / "meta/README.md").read_text()
    assert "synchron" in readme.lower() and "$" not in readme and "4 hand-held demonstration(s)" in readme
    assert OH.RENDERING[None] in readme   # no images in this export
    assert not (out_o / "meta/calibration").exists()
    side = _side(out_o)
    assert [m["split"] for m in side] == ["train", "train", "val", "test"]
    assert side[0]["operator"] == "surgeon1" and "/" not in side[0]["recording"]
    assert json.loads((out_o / "meta/spine_il/export.json").read_text())["layout"] == "openh"
    E.main(["--out", str(out_o), "--validate-only", "--repo-id", "local/test"])   # reads the dataset's own layout


def test_named_splits_images_intrinsics(world):
    from PIL import Image
    cache = world["tmp"] / "cache"
    for s in S.discover(world["sessions"]):
        plan = RC.plan_for(s, 20.0)
        d = RC.chunk_dir(cache, plan, ("endo",), 16, plan.k0, plan.k1)
        (d / "frames" / "endo").mkdir(parents=True)
        for i in range(plan.k1 - plan.k0 + 1):
            Image.new("RGB", (16, 16), (i % 256, 0, 0)).save(d / "frames" / "endo" / f"{i:06d}.png")
        (d / "DONE").write_text("{}")
        (d / "manifest.json").write_text(json.dumps({"optics": "water", "cuts": "recorded"}))   # a replay render's manifest
    out = world["tmp"] / "dsi"
    E.main(["--out", str(out), "--layout", "openh", "--sessions", str(world["sessions"]), "--cache", str(cache), "--size", "16",
            "--image-dtype", "image", "--repo-id", "local/test", "--include", "hand", "--fps", "20", "--val", "aaaa", "--test", "cccc"])
    assert json.loads((out / "meta/info.json").read_text())["splits"] == {"train": "0:2", "val": "2:3", "test": "3:4"}
    assert [S.short_id(m["session_id"]) for m in _side(out)] == ["bbbb", "dddd", "aaaa", "cccc"]   # written train, val, test
    cal = json.loads((out / "meta/calibration/camera_intrinsics.json").read_text())
    c = cal["observation.images.endoscope"]
    assert list(cal) == ["observation.images.endoscope"] and c["width"] == c["height"] == 16 and c["medium"] == "water"
    assert c["distortion_model"] == "opencv_fisheye" and len(c["distortion_coeffs"]) == 4
    ds, _ = _ds(out)
    img = ds[7]["observation.images.endoscope"]
    assert tuple(img.shape) == (3, 16, 16) and float(img[0].mean()) == pytest.approx(7 / 255, abs=1e-3)
    readme = (out / "meta/README.md").read_text()
    assert "camera_intrinsics.json" in readme
    # whose bone the images show comes from the cache: the recording's own cuts here, a re-simulation for a step render
    sids = sorted(m["session_id"] for m in _side(out))
    assert E.image_cuts(cache, sids) == "recorded" and OH.RENDERING["recorded"] in readme
    assert json.loads((out / "meta/spine_il/export.json").read_text())["image_cuts"] == "recorded"
    f = next(cache.glob(f"{sids[0]}/*/c*/manifest.json")); f.write_text(json.dumps({"optics": "water"}))   # render_step.py's
    assert E.image_cuts(cache, sids[:1]) == "simulated" and E.image_cuts(cache, sids) == "mixed"
    assert E.image_cuts(cache, ["s_none"]) is None


def test_intrinsics_scale():
    # the viewer session's numbers for the water profile (10-06): 256 px → 256.95 256.98 128.81 111.31, 480 px → 481.78 481.84 241.96 209.14
    w = OH.intrinsics("water", 256)
    assert (w["fx"], w["fy"], w["cx"], w["cy"]) == pytest.approx((256.95, 256.98, 128.81, 111.31), abs=0.03)
    w = OH.intrinsics("water", 480)
    assert (w["fx"], w["fy"], w["cx"], w["cy"]) == pytest.approx((481.78, 481.84, 241.96, 209.14), abs=0.05)
    assert w["image_circle"] == {"cx": 239.5, "cy": 239.5, "r": pytest.approx(230.8, abs=0.01)}
    assert OH.intrinsics("air", 480)["fx"] < w["fx"]            # the air profile is the wider one


def test_robot_episode_joints_in_rad(tmp_path, monkeypatch):
    hand, robot, run = "s_20260928T000000_altician_lumbar1_trk_aaaa", "s_20260929T000000_altician_lumbar1_trk_bbbb", "20260929T000000_replay_pr_aaaa01"
    sessions, runs = tmp_path / "sessions", tmp_path / "runs"
    synth.write_hand(sessions / "2026-09-28" / f"{hand}.jsonl", hand)
    synth.write_robot(sessions, runs, robot, hand, run)
    monkeypatch.setattr(S, "RUNS_ROOT", runs)
    out = tmp_path / "ds"
    E.main(["--out", str(out), "--layout", "openh", "--sessions", str(sessions), "--images", "none", "--repo-id", "local/test"])
    ds, hf = _ds(out)
    ep = np.asarray(hf["episode_index"]).reshape(-1)
    q, ok = np.asarray(hf["observation.meta.joint_positions"]), np.asarray(hf["observation.meta.joint_positions_valid"]).reshape(-1)
    assert not ok[ep == 0].any() and ok[ep == 1].all()
    tq = synth.BASE + np.arange(8, 157) * 1000.0 / 7.5           # as test_export: q_j(t) = j + 0.001 (t − BASE) deg
    np.testing.assert_allclose(np.degrees(q[ep == 1][:, 0]), 0.001 * (tq - synth.BASE), atol=2e-3)
    assert "on an SO-101 arm" in ds[int(np.flatnonzero(ep == 1)[0])]["observation.meta.tool.endoscope"]
    assert json.loads((out / "meta/info.json").read_text())["splits"] == {"train": "0:2"}   # two episodes: no val / test
    assert "1 robot episode(s)" in (out / "meta/README.md").read_text()


def test_openh_refuses_other_representations(world):
    for extra in (["--action-mode", "delta"], ["--state-phase"]):
        with pytest.raises(SystemExit):
            E.main(["--out", str(world["tmp"] / "x"), "--layout", "openh", "--sessions", str(world["sessions"]), "--images", "none"] + extra)
