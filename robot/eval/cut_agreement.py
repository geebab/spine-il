"""How much do closed-loop robot runs agree with each other, and how high can a cut score against the 8 demos? (2026-10-06)

    python cut_agreement.py LIVE_RUN_DIR [LIVE_RUN_DIR ...] [--at 33] [--json OUT]

LIVE_RUN_DIR = a folder of live_run.sh (it has a robot_session file), named <tag>_<model>_s<start>_<variant>. Only the
cached removed-voxel sets are read (policy_eval.py wrote them: iou_cache_rofull_<session>.npz, iou_cache_hfull_<sid>.npz,
with a snapshot every 3 s) — nothing is rendered.

  demos × demos   IoU between the demos' cuts: whole sessions, and cut off --at seconds after their first cut (a robot run
                  drills 25-38 s of policy time under the recorder's 600 s limit, the demos cut for 24-111 s)
  ceiling         the single cut that scores best against the demos is about their consensus (voxels cut by ≥ k of them):
                  its mean IoU vs the 8 demos, and leave-one-out (the consensus of 7 vs the eighth)
  runs × runs     IoU between the runs' cuts: same model and start, same model from other starts, different models
  runs × demos    each run vs the whole demos (= policy_eval.py's mean IoU) and vs the demos cut off at --at
  along a run     mean IoU vs the demos, coverage and precision of the consensus (k = 4) at 50/67/83/100 % of the run's
                  cutting time: is the cut still getting better when the drilling is ended?
  like for like   a demo T s after its first cut vs the other 7 (their whole cuts; consensus = cut by ≥ 4 of the 7), and the
                  runs against the same reference (each demo left out in turn): which drilling time of a demo a run equals
"""
import argparse
import itertools
import json
import re
from pathlib import Path

import numpy as np

import iou_eval as E
from policy_eval import DEMOS, demo_cut

V = 0.512   # mm³ per voxel


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return E.iou(a, b)[0]


def stat(v: list) -> dict:
    return {"mean": round(float(np.mean(v)), 3), "min": round(float(min(v)), 3), "max": round(float(max(v)), 3), "n": len(v)} if v else {}


def robot_cut(run: Path) -> dict | None:
    f = run / "robot_session"
    key = "rofull_" + Path(f.read_text().strip()).stem if f.exists() else ""
    return E.cached(key, None) if key and (E.OUT / f"iou_cache_{key}.npz").exists() else None


def series(c: dict) -> tuple[np.ndarray, list]:
    """Snapshot times counted from the first one with a cut (> 1 mm³), and the snapshots."""
    ts = np.array(sorted(c["snaps"])); n = np.array([len(c["snaps"][t]) for t in ts])
    i0 = int(np.argmax(n * V > 1.0)) if (n * V > 1.0).any() else 0
    return ts[i0:] - ts[i0], [c["snaps"][t] for t in ts[i0:]]


def cut_at(c: dict, T: float) -> np.ndarray:
    t, s = series(c)
    return s[int(np.argmin(np.abs(t - T)))] if len(s) else c["final"]


def consensus(cuts: list, k: int) -> np.ndarray:
    u, n = np.unique(np.concatenate(cuts).astype(np.int64), return_counts=True)
    return u[n >= k]


def vs_others(x: np.ndarray, fin: dict, skip: str | None = None) -> np.ndarray:
    """[mean IoU, coverage, precision] of a cut against 7 demos and their consensus (≥ 4 of 7): the others of demo `skip`,
    or — for a cut that is not a demo — the mean over leaving each demo out in turn."""
    rows = []
    for h in ([skip] if skip else list(fin)):
        oth = [v for q, v in fin.items() if q != h]; c = consensus(oth, 4)
        rows.append([np.mean([iou(x, o) for o in oth]), np.isin(c, x).mean(), np.isin(x, c).mean() if len(x) else np.nan])
    return np.mean(rows, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--at", type=float, default=33.0, help="cut the demos off this many seconds after their first cut")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    dem = {r: demo_cut(r) for r in DEMOS}
    fin = {r: d["final"].astype(np.int64) for r, d in dem.items()}
    part = {r: cut_at(d, a.at).astype(np.int64) for r, d in dem.items()}
    out = {"at_s": a.at, "demo_mm3": {r: round(len(v) * V) for r, v in fin.items()}, "demo_mm3_at": {r: round(len(v) * V) for r, v in part.items()}}
    out["demo_x_demo"] = stat([iou(fin[x], fin[y]) for x, y in itertools.combinations(DEMOS, 2)])
    out["demo_x_demo_at"] = stat([iou(part[x], part[y]) for x, y in itertools.combinations(DEMOS, 2)])
    print(f"demos × demos: whole {out['demo_x_demo']} | cut off at +{a.at:g} s {out['demo_x_demo_at']}")
    ceil = {}
    for k in range(1, 9):
        c = consensus(list(fin.values()), k)
        loo = [iou(consensus([v for q, v in fin.items() if q != r], k), fin[r]) for r in DEMOS] if k < 8 else []
        ceil[k] = {"mm3": round(len(c) * V), "mean_iou": round(float(np.mean([iou(c, v) for v in fin.values()])), 3),
                   "loo_mean_iou": round(float(np.mean(loo)), 3) if loo else None}
    out["consensus"] = ceil
    print("ceiling — the demos' consensus (cut by ≥ k of 8) vs the demos: " + "  ".join(
        f"k{k} {c['mm3']} mm³ {c['mean_iou']:.3f} (loo {c['loo_mean_iou']})" for k, c in ceil.items()))
    cons = consensus(list(fin.values()), 4)
    out["demo_at"] = {}
    for T in (24, 30, 33, 36, 42, 48, 1e9):   # 1e9 = the whole demo
        x = {r: cut_at(d, T).astype(np.int64) for r, d in dem.items()}
        m = np.mean([vs_others(x[r], fin, r) for r in DEMOS], axis=0)
        out["demo_at"]["end" if T > 1e8 else str(T)] = {"mm3": round(float(np.mean([len(v) for v in x.values()])) * V), "mean_iou": round(float(m[0]), 3),
                                                        "coverage": round(float(m[1]), 3), "precision": round(float(m[2]), 3)}
    print("a demo T s after its first cut vs the other 7 (mm³ · mean IoU · coverage · precision): " + "  ".join(
        f"{'+' + k + ' s' if k != 'end' else k} {v['mm3']} · {v['mean_iou']:.3f} · {v['coverage']:.2f} · {v['precision']:.2f}" for k, v in out["demo_at"].items()))
    runs = {}
    for r in a.runs:
        c = robot_cut(Path(r).expanduser())
        if c is None:
            print(f"{Path(r).name}: no cached robot cut — skipped")
        else:
            runs[Path(r).name] = c
    if runs:
        rf = {n: c["final"].astype(np.int64) for n, c in runs.items()}
        tag = {n: ((m.group(1), m.group(2)) if (m := re.match(r"[^_]+_([^_]+)_s(\d+)", n)) else (n, "")) for n in rf}
        pairs = {"same model, same start": [], "same model, other start": [], "different models": []}
        for x, y in itertools.combinations(rf, 2):
            g = "different models" if tag[x][0] != tag[y][0] else "same model, same start" if tag[x][1] == tag[y][1] else "same model, other start"
            pairs[g].append(iou(rf[x], rf[y]))
        out["run_x_run"] = {g: stat(v) for g, v in pairs.items()} | {"all": stat([x for v in pairs.values() for x in v])}
        print(f"runs × runs ({len(rf)} runs): " + " | ".join(f"{g} {s}" for g, s in out["run_x_run"].items() if s))
        per = {}
        for n, c in runs.items():
            t, s = series(c); vol = np.array([len(x) for x in s])
            te = float(t[int(np.argmax(vol >= 0.995 * vol[-1]))]) if len(s) else 0.0   # wall seconds from the first to the last cut
            along = []
            for fr in (0.5, 0.67, 0.83, 1.0):
                x = s[int(np.argmin(np.abs(t - fr * te)))].astype(np.int64)
                along.append({"mm3": round(len(x) * V), "mean_iou": round(float(np.mean([iou(x, v) for v in fin.values()])), 3),
                              "coverage": round(float(np.isin(cons, x).mean()), 3), "precision": round(float(np.isin(x, cons).mean()), 3) if len(x) else None})
            per[n] = {"mm3": round(len(rf[n]) * V), "mean_iou": round(float(np.mean([iou(rf[n], v) for v in fin.values()])), 3),
                      "mean_iou_at": round(float(np.mean([iou(rf[n], v) for v in part.values()])), 3),
                      "iou_consensus4": round(iou(rf[n], cons), 3), "cutting_wall_s": te, "along": along}
            print(f"  {n:22s} {per[n]['mm3']:4d} mm³ | vs demos {per[n]['mean_iou']:.3f}, cut off at +{a.at:g} s {per[n]['mean_iou_at']:.3f} | along the run "
                  f"IoU {' '.join(format(x['mean_iou'], '.3f') for x in along)} cover {' '.join(format(x['coverage'], '.2f') for x in along)} "
                  f"prec {' '.join(format(x['precision'], '.2f') for x in along)}")
        out["runs"] = per
        out["run_x_demo"] = stat([p["mean_iou"] for p in per.values()]); out["run_x_demo_at"] = stat([p["mean_iou_at"] for p in per.values()])
        m = np.mean([vs_others(x, fin) for x in rf.values()], axis=0)
        out["runs_like_a_demo"] = {"mm3": round(float(np.mean([len(x) for x in rf.values()])) * V), "mean_iou": round(float(m[0]), 3),
                                   "coverage": round(float(m[1]), 3), "precision": round(float(m[2]), 3)}
        print(f"the runs against the same reference as a demo (7 demos, consensus ≥ 4 of 7): {out['runs_like_a_demo']}")
        out["along_mean"] = [{k: round(float(np.mean([p["along"][i][k] for p in per.values()])), 3) for k in ("mm3", "mean_iou", "coverage", "precision")}
                             for i in range(4)]
        print(f"runs × demos: whole {out['run_x_demo']} | demos cut off at +{a.at:g} s {out['run_x_demo_at']}")
        print("along a run (50/67/83/100 % of its cutting time), mean of the runs: " + " | ".join(
            f"{k} " + " ".join(format(x[k], ".0f" if k == "mm3" else ".3f") for x in out["along_mean"]) for k in ("mm3", "mean_iou", "coverage", "precision")))
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
