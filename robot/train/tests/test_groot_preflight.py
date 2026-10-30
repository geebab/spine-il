"""groot_preflight.py — what LeRobot's GR00T N1.7 code makes of a dataset, on made-up datasets with one defect each: the
report must name the defect (and a dataset without one must pass)."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parents[1] / "dataset"))
import groot_preflight as GP      # noqa: E402
import global_quantiles as GQ     # noqa: E402

NAMES = [f"{t}.{c}" for t in ("endo", "drill") for c in ("x", "y", "z", "qw", "qx", "qy", "qz")] + ["drill.trigger"]


def build(tmp: Path, *, places=(0.0, 0.0), state_pad=True, after=60, flip=False, progress_stop=None, aggregate=True) -> Path:
    """Two episodes of 400 frames: tools near `places[e]` on x (two sittings), drilling from frame 40, the last cut `after`
    frames before the end. aggregate=True writes the quantiles as LeRobot does (mean of per-episode quantiles)."""
    rng = np.random.default_rng(0); n = 400
    S, A, E, K, T, P = [], [], [], [], [], []
    per_ep = []
    for e, x0 in enumerate(places):
        pos = np.cumsum(rng.normal(0, 0.3, (n + 1, 6)), axis=0) + np.array([x0, -30, 1020, x0 - 10, -40, 1015])
        q = np.tile([0.6, -0.6, -0.4, 0.3], (n + 1, 1)) + rng.normal(0, 0.002, (n + 1, 4))
        if flip and e == 0:
            q[200:] *= -1                                    # the same rotation, the other sign
        last = n - after
        trig = ((np.arange(n + 1) >= 40) & (np.arange(n + 1) <= last)).astype(float)
        prog = np.clip((np.arange(n + 1) - 40) / (last - 40), 0, 1) * (progress_stop if (progress_stop and e == 1) else 1.0)
        s = np.c_[pos[:, :3], q, pos[:, 3:], q, trig]
        st = np.c_[s[:-1], np.zeros(n)] if state_pad else s[:-1]
        ac = np.c_[s[1:], prog[1:]]
        S.append(st); A.append(ac); E += [e] * n; K += list(range(n)); T += [0 if k < 40 else 1 for k in range(n)]; P += list(prog[:-1])
        per_ep.append((st, ac))
    S, A = np.concatenate(S), np.concatenate(A)
    ds = tmp / "ds"; (ds / "data/chunk-000").mkdir(parents=True); (ds / "meta").mkdir()
    pq.write_table(pa.table({"episode_index": E, "frame_index": K, "observation.state": [list(r) for r in S], "action": [list(r) for r in A],
                             "task_index": T, "cut.progress": [[v] for v in P]}), ds / "data/chunk-000/file-000.parquet")
    pq.write_table(pa.table({"task_index": [0, 1], "task": ["3 advance", "4 drill"]}), ds / "meta/tasks.parquet")
    sn = NAMES + (["pad.zero"] if state_pad else []); an = NAMES + ["drill.progress"]
    info = {"fps": 15, "features": {"observation.state": {"dtype": "float32", "shape": [len(sn)], "names": sn}, "action": {"dtype": "float32", "shape": [len(an)], "names": an},
                                    "cut.progress": {"dtype": "float32", "shape": [1], "names": None},
                                    "observation.images.endo": {"dtype": "video", "shape": [256, 256, 3], "names": ["height", "width", "channels"], "info": {"video.codec": "h264"}}}}
    (ds / "meta/info.json").write_text(json.dumps(info))

    def stats(parts):
        allx = np.concatenate(parts); w = np.array([len(p) for p in parts])[:, None]
        out = {"min": allx.min(0).tolist(), "max": allx.max(0).tolist(), "mean": allx.mean(0).tolist(), "std": allx.std(0).tolist(), "count": [len(allx)]}
        for k, p in GQ.QS.items():
            out[k] = ((np.stack([np.quantile(x, p, axis=0) for x in parts]) * w).sum(0) / w.sum()).tolist() if aggregate else np.quantile(allx, p, axis=0).tolist()
        return out
    (ds / "meta/stats.json").write_text(json.dumps({"observation.state": stats([p[0] for p in per_ep]), "action": stats([p[1] for p in per_ep]),
                                                    "cut.progress": stats([np.asarray(P[:n])[:, None], np.asarray(P[n:])[:, None]])}))
    return ds


def run(ds: Path, chunk=16, held=()):
    d = GP.load(ds); d["path"] = str(ds)
    return GP.report(d, chunk, ["trigger", "progress"], list(held), 0.91)


def test_the_groups_are_the_ones_lerobot_makes():
    g = GP.groups_of(NAMES + ["drill.progress"], ["trigger", "progress"])
    assert [(x["key"], len(x["indices"]), x["relative"]) for x in g] == [("single_arm", 14, True), ("trigger", 1, False), ("progress", 1, False)]
    try:
        from lerobot.policies.groot.processor_groot import _infer_n1_7_action_groups
    except Exception:
        return
    theirs = _infer_n1_7_action_groups(NAMES + ["drill.progress"], action_dim=16, exclude_joints=["trigger", "progress"])
    assert [(x.key, x.indices, x.relative) for x in theirs] == [(x["key"], x["indices"], x["relative"]) for x in g]
    assert [x["key"] for x in GP.groups_of(["a.pos", "gripper.pos", "b.pos"], ["gripper"])] == [x.key for x in _infer_n1_7_action_groups(["a.pos", "gripper.pos", "b.pos"], action_dim=3, exclude_joints=["gripper"])]


def test_a_sound_dataset_passes(tmp_path):
    r = run(build(tmp_path, aggregate=False))
    assert r["problems"] == [], r["problems"]
    assert r["sampling"]["dropped_tail_frames_per_episode"] == 15 and r["sampling"]["training_samples"] == 2 * (400 - 15)
    assert r["progress"]["largest_value_the_model_can_output"] == pytest.approx(1.0) and r["relative"]["windows"] == 2 * (400 - 39)


def test_two_sittings_and_lerobots_mean_of_quantiles_clip_the_state(tmp_path):
    """The defect of 10-10: episodes recorded at two places, q01/q99 = the mean of the two episodes' ranges."""
    ds = build(tmp_path, places=(0.0, 40.0))
    r = run(ds)
    assert any("'endo.x'" in p and "clipped" in p for p in r["problems"]) and any("'drill.x'" in p for p in r["problems"])
    row = next(x for x in r["scaling"]["observation.state"] if x["name"] == "endo.x")
    assert row["clipped_share"] > 0.9 and row["episodes_more_than_half_clipped"] == 2
    GQ.fix(ds, log=lambda *_: None)
    assert run(ds)["problems"] == []                          # global_quantiles.py mends exactly this


def test_state_and_action_of_different_length_stop_the_training(tmp_path):
    r = run(build(tmp_path, state_pad=False, aggregate=False))
    assert any("state has 15 values, action 16" in p for p in r["problems"])


def test_a_progress_output_that_cannot_reach_the_end_level(tmp_path):
    """One episode that stops at 0.3 pulls LeRobot's averaged q99 under the level at which the loop ends the drilling."""
    r = run(build(tmp_path, progress_stop=0.3, after=3))
    assert r["progress"]["largest_value_the_model_can_output"] < 0.94
    assert any("could never end the drilling" in p for p in r["problems"])


def test_a_quaternion_that_flips_sign_is_named(tmp_path):
    r = run(build(tmp_path, flip=True, aggregate=False))
    assert r["relative"]["max_quaternion_component_change"] > 1.0 and any("sign flip" in p for p in r["problems"])


def test_the_tail_that_is_never_an_input(tmp_path):
    ds = build(tmp_path, after=30, aggregate=False)           # the last cut is frame 370 of 0 … 399: 29 frames follow it
    by = run(ds, chunk=16)["sampling"]["input_frames_after_the_last_cut_by_chunk_size"]
    assert by[16]["min"] == 29 - 15 and by[40]["min"] == 0 and by[40]["episodes_with_none"] == 2
    assert any("no input frame after their last cut" in p for p in run(ds, chunk=40)["problems"])
    assert run(ds, chunk=16)["problems"] == []


def test_held_out_episodes_are_not_counted(tmp_path):
    r = run(build(tmp_path, aggregate=False), held=(1,))
    assert r["sampling"]["episodes_for_training"] == 1 and r["sampling"]["training_samples"] == 400 - 15
