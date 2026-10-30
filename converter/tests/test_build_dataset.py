"""End-to-end: fixtures → LeRobot v3 dataset on disk → reload with LeRobot → (optionally) train ACT."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from spine_il_converter.cli import main as cli_main
from spine_il_converter.episode import build_episode
from spine_il_converter.fixtures import image_value, write_standard_set
from spine_il_converter.session import load_session
from spine_il_converter.splits import assign_splits

pytestmark = pytest.mark.slow
REPO_ID = "local/spine-lam-fixture"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixture_data")
    write_standard_set(root)
    out = tmp_path_factory.mktemp("datasets") / "spine-lam-fixture"
    assert cli_main(["build", "--data-root", str(root), "--out", str(out), "--repo-id", REPO_ID, "--overwrite"]) == 0
    return root, out


def test_dataset_reloads_with_lerobot(built):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root, out = built
    ds = LeRobotDataset(REPO_ID, root=out)
    assert ds.meta.total_episodes == 3
    assert ds.meta.robot_type == "tracked_bi_so101_dummy_v1"
    assert ds.fps == 30
    assert ds.meta.info.splits["train"].startswith("0:")

    episodes = [json.loads(line) for line in (out / "meta/spine_il/episodes.jsonl").read_text().splitlines()]
    assert sum(e["n_frames"] for e in episodes) == ds.meta.total_frames == len(ds)

    offset = 0
    for rec in episodes:
        ep = build_episode(load_session(Path(rec["source"])), root).episode
        for k in (0, ep.n_frames // 2, ep.n_frames - 1):
            item = ds[offset + k]
            np.testing.assert_allclose(item["observation.state"].numpy(), ep.state[k], atol=1e-6)
            np.testing.assert_allclose(item["action"].numpy(), ep.action[k], atol=1e-6)
            assert item["task"] == ep.task
            assert tuple(item["observation.images.endoscope"].shape) == (3, 256, 256)
            for cam in ("endoscope", "scene"):
                frame_idx = int(Path(ep.image_paths[cam][k]).stem)
                decoded = item[f"observation.images.{cam}"].numpy().mean() * 255
                assert abs(decoded - image_value(cam, frame_idx)) < 6, (cam, k)
        offset += ep.n_frames

    report = json.loads((out / "meta/spine_il/conversion_report.json").read_text())
    assert sorted(e["reason"] for e in report["excluded"]) == sorted(
        ["verdict_bad", "fail_flags", "no_phase3", "tracker_lost", "gap", "basal_parity", "no_footer", "schema"]
    )


def test_overwrite_guard(built):
    root, out = built
    with pytest.raises(FileExistsError):
        cli_main(["build", "--data-root", str(root), "--out", str(out), "--repo-id", REPO_ID])


def test_splits_are_stable_when_sessions_are_added():
    ids = [f"s{i:03d}" for i in range(100)]
    before = assign_splits(ids)
    after = assign_splits(ids + [f"new{i}" for i in range(10)])
    moved = sum(before[s] != after[s] for s in ids)
    assert list(before.values()).count("train") == 80 and moved <= 10


LEROBOT_TRAIN = Path(sys.executable).with_name("lerobot-train")


@pytest.mark.skipif(not LEROBOT_TRAIN.exists(), reason="lerobot-train not installed next to this interpreter")
def test_act_trains_two_steps_on_converted_dataset(built, tmp_path):
    _, out = built
    exe = LEROBOT_TRAIN
    cmd = [
        str(exe), f"--dataset.repo_id={REPO_ID}", f"--dataset.root={out}",
        "--policy.type=act", "--policy.device=cpu", "--policy.push_to_hub=false",
        "--policy.chunk_size=20", "--policy.n_action_steps=20",
        "--batch_size=2", "--steps=2", "--num_workers=0", "--log_freq=1",
        "--save_checkpoint=false", "--wandb.enable=false", f"--output_dir={tmp_path / 'act'}",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "End of training" in proc.stdout + proc.stderr
