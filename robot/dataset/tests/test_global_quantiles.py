"""global_quantiles.py — the dataset's q01 … q99 as quantiles of all frames, not LeRobot's mean of per-episode quantiles.
A two-episode dataset recorded at two places: the aggregate range covers neither, the rewritten one covers both."""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import global_quantiles as GQ      # noqa: E402


def make(tmp: Path) -> tuple[Path, np.ndarray]:
    """Episode 0 around x = 10, episode 1 around x = 60 (two sittings); the second value is the same in both."""
    rng = np.random.default_rng(0)
    a = np.c_[rng.normal(10, 1, 400), rng.normal(5, 1, 400)]; b = np.c_[rng.normal(60, 1, 600), rng.normal(5, 1, 600)]
    x = np.r_[a, b]
    ds = tmp / "ds"; (ds / "data/chunk-000").mkdir(parents=True); (ds / "meta").mkdir()
    pq.write_table(pa.table({"observation.state": [list(r) for r in x], "episode_index": [0] * 400 + [1] * 600,
                             "cut.progress": [[v] for v in np.r_[np.linspace(0, 1, 400), np.linspace(0, 0.3, 600)]]}), ds / "data/chunk-000/file-000.parquet")

    def agg(col, cut=400):      # as LeRobot: count-weighted mean of the two episodes' quantiles
        col = col.reshape(len(col), -1)
        q = {k: ((np.quantile(col[:cut], p, axis=0) * cut + np.quantile(col[cut:], p, axis=0) * (len(col) - cut)) / len(col)).tolist() for k, p in GQ.QS.items()}
        return {"min": col.min(0).tolist(), "max": col.max(0).tolist(), "mean": col.mean(0).tolist(), "std": col.std(0).tolist(), "count": [len(col)], **q}
    stats = {"observation.state": agg(x), "cut.progress": agg(np.r_[np.linspace(0, 1, 400), np.linspace(0, 0.3, 600)]),
             "observation.images.endo": {"min": [[[0.0]]], "max": [[[1.0]]], "mean": [[[0.4]]], "std": [[[0.2]]], "count": [1000], "q01": [[[0.01]]], "q99": [[[0.9]]]}}
    (ds / "meta/stats.json").write_text(json.dumps(stats))
    return ds, x


def test_the_aggregate_range_covers_neither_place_and_the_global_one_both(tmp_path):
    ds, x = make(tmp_path)
    old = json.loads((ds / "meta/stats.json").read_text())
    lo, hi = old["observation.state"]["q01"][0], old["observation.state"]["q99"][0]
    assert 30 < lo < 45 and 30 < hi < 45                         # the mean of [8, 12] and [58, 62]: where no value is
    assert ((x[:, 0] < lo) | (x[:, 0] > hi)).mean() == 1.0       # every frame would be clipped
    r = GQ.fix(ds, log=lambda *_: None)
    assert r["written"] and set(r["changed"]) == {"observation.state", "cut.progress"}
    new = json.loads((ds / "meta/stats.json").read_text())
    q01, q99 = new["observation.state"]["q01"], new["observation.state"]["q99"]
    assert q01[0] < 9 and q99[0] > 61 and abs(((x[:, 0] < q01[0]) | (x[:, 0] > q99[0])).mean() - 0.02) < 0.005
    assert np.allclose(q01, np.quantile(x, 0.01, axis=0)) and np.allclose(new["observation.state"]["q50"], np.quantile(x, 0.5, axis=0))
    assert new["cut.progress"]["q99"][0] > 0.95 > old["cut.progress"]["q99"][0]      # the model's progress output is scaled with this
    for k in ("min", "max", "mean", "std", "count"):             # LeRobot aggregates these exactly: untouched
        assert new["observation.state"][k] == old["observation.state"][k]
    assert new["observation.images.endo"] == old["observation.images.endo"]         # image statistics are not tabular


def test_the_first_file_is_kept_and_a_second_run_changes_nothing(tmp_path):
    ds, _ = make(tmp_path)
    first = (ds / "meta/stats.json").read_text()
    assert GQ.fix(ds, check=True, log=lambda *_: None)["written"] is False and (ds / "meta/stats.json").read_text() == first      # --check writes nothing
    GQ.fix(ds, log=lambda *_: None)
    assert (ds / "meta/stats.lerobot_aggregate.json").read_text() == first
    note = json.loads((ds / "meta/spine_il/quantiles.json").read_text())
    assert "observation.state" in note["features"] and note["features"]["observation.state"]["share_outside_the_old_q01_q99_worst_dim"] == 1.0
    again = GQ.fix(ds, log=lambda *_: None)
    assert again["changed"] == [] and again["written"] is False
    assert (ds / "meta/stats.lerobot_aggregate.json").read_text() == first           # the backup is LeRobot's file, not the rewritten one
    assert GQ.main([str(ds), "--check"]) == 0
