#!/usr/bin/env python3
"""meta/stats.json's quantiles as what they claim to be: quantiles of the whole dataset.

LeRobot writes a dataset's q01 … q99 as the count-weighted MEAN of the per-episode quantiles (datasets/compute_stats.py,
aggregate_feature_stats). For data recorded at one place that is close enough; ours is recorded at several — three
sittings whose portals lie 14–17 mm apart, tools held at different angles — and the mean of per-episode ranges is far
narrower than the data. GR00T N1.7 (lerobot policies/groot) scales the state and the absolute action groups to [-1, 1]
with exactly these q01/q99 and CLIPS what lies outside: in the 24-demonstration dataset 50 % of the endoscope's depth
values and 49 % of one of its quaternion components sat on the clip limit, 16 of 23 training episodes for more than half
of their frames (train/groot_preflight.py, 10-10) — the model was not shown where the endoscope was. The progress output
was capped at 0.989 for the same reason (one episode ends at 0.32).

    python dataset/global_quantiles.py DS            # rewrite q01 q10 q50 q90 q99 of the tabular features, keep a backup
    python dataset/global_quantiles.py DS --check    # only say what differs (exit 1 if anything does)

Tabular features only (the columns of data/*.parquet); image statistics are left alone. min / max / mean / std / count are
not touched — LeRobot aggregates those exactly. The first run keeps LeRobot's file as meta/stats.lerobot_aggregate.json
and notes what it did in meta/spine_il/quantiles.json. export_lerobot.py calls this at the end of every export.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

QS = {"q01": 0.01, "q10": 0.10, "q50": 0.50, "q90": 0.90, "q99": 0.99}


def columns(ds: Path, keys: list[str]) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq
    files = sorted((ds / "data").glob("*/*.parquet"))
    have = set(pq.read_schema(files[0]).names) if files else set()
    out = {}
    for k in (k for k in keys if k in have):
        vals = []
        for f in files:
            vals += pq.read_table(f, columns=[k]).column(k).to_pylist()
        a = np.asarray(vals, np.float64)
        out[k] = a.reshape(len(a), -1)
    return out


def global_quantiles(ds: Path, keys: list[str] | None = None) -> tuple[dict, dict]:
    """(new stats, report): the stats of meta/stats.json with the quantiles of the listed (default: all tabular) features
    replaced by those of all frames."""
    stats = json.loads((ds / "meta/stats.json").read_text())
    info = ds / "meta/info.json"
    feats = json.loads(info.read_text()).get("features", {}) if info.exists() else {}
    tabular = lambda k: (feats.get(k) or {}).get("dtype") not in ("image", "video", "string")      # noqa: E731 — images kept in the parquet are not numbers
    want = [k for k, s in stats.items() if any(q in s for q in QS) and (keys is None or k in keys) and tabular(k)]
    cols = columns(ds, want)
    new = json.loads(json.dumps(stats)); rep = {}
    for k, x in cols.items():
        shape = np.asarray(stats[k].get("mean", stats[k].get("min"))).shape
        worst = 0.0
        for q, p in QS.items():
            if q not in stats[k]:
                continue
            v = np.quantile(x, p, axis=0).reshape(shape)
            old = np.asarray(stats[k][q], np.float64)
            span = np.maximum(np.asarray(stats[k]["max"], np.float64) - np.asarray(stats[k]["min"], np.float64), 1e-12).reshape(shape)
            worst = max(worst, float(np.max(np.abs(v - old) / span)))
            new[k][q] = v.tolist()
        lo, hi = np.asarray(stats[k].get("q01", stats[k]["min"]), np.float64).reshape(1, -1), np.asarray(stats[k].get("q99", stats[k]["max"]), np.float64).reshape(1, -1)
        tol = 0.01 * np.maximum(hi - lo, 1e-12)                       # outside by more than a hundredth of the range
        rep[k] = {"dims": int(x.shape[1]), "frames": int(x.shape[0]), "largest_quantile_change_as_share_of_min_max": round(worst, 4),
                  "share_outside_the_old_q01_q99_worst_dim": round(float(((x < lo - tol) | (x > hi + tol)).mean(0).max()), 4)}
    return new, rep


def fix(ds: Path, keys: list[str] | None = None, check: bool = False, log=print) -> dict:
    ds = Path(ds); new, rep = global_quantiles(ds, keys)
    changed = {k: r for k, r in rep.items() if r["largest_quantile_change_as_share_of_min_max"] > 1e-6}
    for k, r in rep.items():
        log(f"[quantiles] {k}: {r['dims']} dims, largest change {r['largest_quantile_change_as_share_of_min_max']:.1%} of the min–max range, "
            f"{r['share_outside_the_old_q01_q99_worst_dim']:.1%} of the frames outside the old q01–q99 in the worst dimension")
    if check or not changed:
        return {"changed": sorted(changed), "report": rep, "written": False}
    backup = ds / "meta/stats.lerobot_aggregate.json"
    if not backup.exists():
        backup.write_text((ds / "meta/stats.json").read_text())
    tmp = ds / "meta/stats.json.tmp"; tmp.write_text(json.dumps(new, indent=4)); tmp.replace(ds / "meta/stats.json")
    note = ds / "meta/spine_il/quantiles.json"; note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(json.dumps({"what": "q01 q10 q50 q90 q99 of the tabular features are quantiles over all frames of the dataset; "
                                        "LeRobot's file (count-weighted mean of per-episode quantiles) is kept as meta/stats.lerobot_aggregate.json",
                                "why": "GR00T N1.7 scales state and absolute action groups with q01/q99 and clips outside (robot/dataset/global_quantiles.py)",
                                "when": time.strftime("%Y-%m-%d %H:%M:%S"), "features": rep}, indent=1, ensure_ascii=False))
    return {"changed": sorted(changed), "report": rep, "written": True}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--keys", default=None, help="comma separated features (default: every tabular feature that has quantiles)")
    a = ap.parse_args(argv)
    r = fix(Path(a.dataset).expanduser(), a.keys.split(",") if a.keys else None, a.check)
    print(("would change: " if a.check else "rewritten: " if r["written"] else "nothing to change: ") + (", ".join(r["changed"]) or "—"))
    return 1 if a.check and r["changed"] else 0


if __name__ == "__main__":
    sys.exit(main())
