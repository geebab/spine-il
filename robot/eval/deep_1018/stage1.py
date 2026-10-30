"""deep_1018 stage 1 (PREREG.md v1): the robot's voxel metric applied to the hand demos and every evaluated H90a / H90g run.
Demos: the 90 training recordings of H90g (user 10-23: the paper and the hackathon use these 90 only; the 09-28 eight, all
L4 ipsilateral, are left out everywhere — the first version used the 98 of spine_hand98 lang2).
1a reach / share per level × lang2 label with bootstrap CIs and the ①-consistency rule; voxel vs cut-centre reach.
1b midline from the level meshes.  1c error budget.  1d contra-target coverage and IoU with the contra-demo union.

  python stage1.py OUT_DIR [--runs-from STAGE1_JSON]   (--runs-from: only the runs listed there, e.g. to change the demos
                                                         and keep the run set of an earlier report)"""
import glob, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dataset import instructions as INS          # noqa: E402

H = os.path.expanduser("~")
C = f"{H}/il-robot/runs/jitter_iter"
LV = f"{H}/il-robot/runs/act_smoke/live"
EPI = "/srv/il-robot/datasets/spine_hand90_replay_entry_water_progress_lang3/meta/spine_il/episodes.jsonl"
MID = INS.MIDLINE_X_MM
OUT = sys.argv[1]
ONLY = {w["run"] for w in json.load(open(sys.argv[sys.argv.index("--runs-from") + 1]))["runs"]} if "--runs-from" in sys.argv else None
rng = np.random.default_rng(1018)
GRID = None


def load(f):
    global GRID
    z = np.load(f, allow_pickle=True); h = json.loads(str(z["hdr"]))
    g = (h["sdfSha256"], h["gridSha256"])
    if GRID is None:
        GRID = (g, h)
    elif g != GRID[0]:
        raise SystemExit(f"{f}: other grid — comparison void")
    return z["final"].astype(np.int64), h


def xyz(idx, h):
    nx, ny = h["dims"][0], h["dims"][1]; o = np.asarray(h["origin"]); s = h["spacing"][0]
    return np.stack([idx % nx, (idx // nx) % ny, idx // (nx * ny)], 1) * s + o


def metric(idx, h, lv):
    x = xyz(idx, h)[:, 0]
    return {"reach": float(MID[lv] - np.percentile(x, 5)), "past": float(100 * np.mean(x < MID[lv])), "mm3": float(len(idx) * h["voxelMm3"])}


def boot(v, n=10000):
    v = np.asarray(v, float)
    if len(v) < 2:
        return [None, None]
    m = rng.choice(v, (n, len(v))).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ---- demos ----
demos = {}
for line in open(EPI):
    e = json.loads(line); sid = e["session_id"]
    if sid in demos or e.get("source") != "hand":
        continue
    f = f"{C}/iou_cache_hfull_{sid}.npz"
    demos[sid] = {"sid": sid, "day": sid[2:10], "level": e["goal"]["level"], "label": e["goal"]["extent"],
                  "cut_reach": e["goal"]["past_midline_mm"], "file": f if os.path.exists(f) else None}
missing = [d["sid"] for d in demos.values() if not d["file"]]
idx_of = {}
for d in demos.values():
    if d["file"]:
        idx, h = load(d["file"]); idx_of[d["sid"]] = idx; d.update(metric(idx, h, d["level"]))
h = GRID[1]

# ---- runs ----
runs = []
for r in sorted(glob.glob(f"{LV}/R_H90[ag]_*")):
    if not (os.path.exists(f"{r}/live.json") and os.path.exists(f"{r}/robot_session")) or (ONLY is not None and os.path.basename(r) not in ONLY):
        continue
    sid = os.path.basename(open(f"{r}/robot_session").read().strip()).replace(".jsonl", "")
    f = f"{C}/iou_cache_rofull_{sid}.npz"
    if not os.path.exists(f):
        continue
    name = os.path.basename(r); p = name.split("_")
    lv = p[2][:2] if p[2][:2] in MID else "L4"
    grp = (p[2][2:] if lv != "L4" or p[2].startswith("L4") else p[3]).rstrip("0123456789")
    m = json.load(open(f"{r}/live.json"))["meta"]
    idx, _ = load(f); w = {"run": name, "model": p[1], "level": lv, "group": grp, "sid": sid, "end": m.get("drill_end"), **metric(idx, h, lv)}
    fr = f"{C}/iou_cache_ro_{sid}.npz"                       # live recount of the same run (1c)
    if os.path.exists(fr):
        try:
            i2, _ = load(fr); w["reach_ro"] = metric(i2, h, lv)["reach"]
        except SystemExit:
            pass
    d = json.load(open(f"{r}/live.json")); a = np.asarray(d["frames"], float); x = a[a[:, 14] > 0.5, 7]
    if len(x):
        w["tip_reach"] = float(MID[lv] - x.min()); w["tip_p2"] = float(MID[lv] - np.percentile(x, 2))
    idx_of[name] = idx; runs.append(w)

# ---- 1a ----
res = {"grid": {"sdf": h["sdfSha256"][:12], "grid": h["gridSha256"][:12], "spacing": h["spacing"]}, "demos_missing_cache": missing, "cells": {}}
for lv in MID:
    for lab in ("contralateral", "unspecified", "ipsilateral"):
        ds = [d for d in demos.values() if d["level"] == lv and d["label"] == lab and "reach" in d]
        if not ds:
            continue
        r_ = [d["reach"] for d in ds]; s_ = [d["past"] for d in ds]
        res["cells"][f"{lv} {lab}"] = {"n": len(ds), "reach_median": float(np.median(r_)), "reach_iqr": [float(np.percentile(r_, 25)), float(np.percentile(r_, 75))],
                                       "reach_mean": float(np.mean(r_)), "reach_ci": boot(r_), "reach_min_max": [float(min(r_)), float(max(r_))],
                                       "share_mean": float(np.mean(s_)), "share_ci": boot(s_), "mm3_median": float(np.median([d["mm3"] for d in ds])),
                                       "gt5": int(sum(v > 5 for v in r_))}
rule = {}
for lv in MID:
    c = res["cells"].get(f"{lv} contralateral")
    if c:
        rv = [d["reach"] for d in demos.values() if d["level"] == lv and d["label"] == "contralateral" and "reach" in d]
        rule[lv] = {"share_gt5": c["gt5"] / c["n"], "consistent": c["gt5"] / c["n"] >= 0.8, "demo_q25": float(np.percentile(rv, 25))}
res["rule_1"] = rule
dv = [(d["reach"], d["cut_reach"]) for d in demos.values() if "reach" in d and d["cut_reach"] is not None]
a_, b_ = np.array(dv).T
rk = lambda v: np.argsort(np.argsort(v))
res["voxel_vs_cut"] = {"n": len(dv), "spearman": float(np.corrcoef(rk(a_), rk(b_))[0, 1]), "median_voxel_minus_cut": float(np.median(a_ - b_)),
                       "iqr": [float(np.percentile(a_ - b_, 25)), float(np.percentile(a_ - b_, 75))]}

# ---- 1b ----
mid = {}
for lv in MID:
    f = os.path.join(os.environ.get("UBE_LEVEL_STL", f"{H}/il-robot/calib/level_meshes"), f"{lv}.stl")
    if not os.path.exists(f):
        cand = glob.glob(os.path.join(os.path.dirname(f), f"*{lv}*.stl")); f = cand[0] if cand else None
    if not f:
        mid[lv] = "mesh not found"; continue
    P = INS._stl_points(f); post = P[:, 1]                          # LPS: +y posterior
    mid[lv] = {"file": os.path.basename(f), "registered": MID[lv]}
    for q in (1, 2, 5):
        sel = P[post >= np.percentile(post, 100 - q)]; mid[lv][f"top{q}%"] = round(float(sel[:, 0].mean()), 2)
    mid[lv]["flag"] = any(abs(mid[lv][k] - MID[lv]) > 1 for k in ("top1%", "top2%", "top5%"))
res["midline"] = mid

# ---- 1c ----
ro = [w["reach"] - w["reach_ro"] for w in runs if "reach_ro" in w]
tp = [w["tip_reach"] - w["reach"] for w in runs if "tip_reach" in w]
res["error"] = {"voxel_quantisation_sd": 0.8 / np.sqrt(12), "rofull_minus_ro": {"n": len(ro), "mean": float(np.mean(ro)) if ro else None, "sd": float(np.std(ro, ddof=1)) if len(ro) > 1 else None},
                "tip_max_minus_voxel": {"n": len(tp), "mean": float(np.mean(tp)), "sd": float(np.std(tp, ddof=1))}}
sds = [res["error"]["voxel_quantisation_sd"]] + ([res["error"]["rofull_minus_ro"]["sd"]] if res["error"]["rofull_minus_ro"]["sd"] else [])
res["error"]["min_meaningful_mm"] = float(2 * np.sqrt(np.sum(np.square(sds))))

# ---- 1d ----
for lv in MID:
    cs = [s for s, d in demos.items() if d["level"] == lv and d["label"] == "contralateral" and s in idx_of]
    if not cs:
        continue
    allc = np.concatenate([idx_of[s] for s in cs]); u, n = np.unique(allc, return_counts=True)
    xs = xyz(u, h)[:, 0]
    target = set(u[(n >= 0.5 * len(cs)) & (xs < MID[lv])].tolist()); union = set(u.tolist())
    res.setdefault("target", {})[lv] = {"n_demos": len(cs), "target_mm3": len(target) * h["voxelMm3"], "union_mm3": len(union) * h["voxelMm3"]}
    for d in [d for d in demos.values() if d["level"] == lv and "reach" in d] + [w for w in runs if w["level"] == lv]:
        s = set(idx_of[d.get("run", d["sid"])].tolist())
        d["target_cov"] = len(s & target) / len(target) if target else None
        d["iou_contra_union"] = len(s & union) / len(s | union)

json.dump({"res": res, "demos": list(demos.values()), "runs": runs}, open(f"{OUT}/stage1.json", "w"), indent=1, default=lambda o: None)

# ---- text ----
L = [f"grid sdf {res['grid']['sdf']} · grid {res['grid']['grid']} · spacing {h['spacing'][0]} mm (one grid for all demos and runs)",
     f"demos {len(demos)}, missing voxel cache {len(missing)} {missing[:4]}", "", "1a  demos under the robot's voxel metric (reach = midline − p5 x, mm)"]
for k, c in res["cells"].items():
    L.append(f"  {k:<18} n {c['n']:2}  reach median {c['reach_median']:+5.1f} (IQR {c['reach_iqr'][0]:+.1f}…{c['reach_iqr'][1]:+.1f}, range {c['reach_min_max'][0]:+.1f}…{c['reach_min_max'][1]:+.1f})"
             f"  mean {c['reach_mean']:+.1f} CI {c['reach_ci'][0]:+.1f}…{c['reach_ci'][1]:+.1f}  > 5 mm {c['gt5']}/{c['n']}  past {c['share_mean']:.1f} %  {c['mm3_median']:.0f} mm3")
for lv, r_ in rule.items():
    L.append(f"  ① vs demos {lv}: {100 * r_['share_gt5']:.0f} % of contra demos > 5 mm → {'consistent' if r_['consistent'] else 'NOT consistent'}; demo contra q25 = {r_['demo_q25']:+.1f} mm")
v = res["voxel_vs_cut"]; L.append(f"  voxel vs cut-centre reach: Spearman {v['spearman']:.2f}, voxel − cut median {v['median_voxel_minus_cut']:+.1f} mm (IQR {v['iqr'][0]:+.1f}…{v['iqr'][1]:+.1f}), n {v['n']}")
L += ["", "1b  midline from the meshes (mean x of the most posterior vertices)"] + [f"  {lv}: {m}" for lv, m in mid.items()]
e = res["error"]; L += ["", "1c  error budget", f"  voxel quantisation SD {e['voxel_quantisation_sd']:.2f} mm; rofull − ro {e['rofull_minus_ro']}; tip max − voxel {e['tip_max_minus_voxel']}",
                        f"  smallest difference used as evidence: {e['min_meaningful_mm']:.1f} mm"]
L += ["", "1d  contra target (past midline, removed by ≥ 50 % of the level's contra demos)"] + [f"  {lv}: {t}" for lv, t in res.get("target", {}).items()]
L += ["", "runs (voxel reach · past % · mm3 · target coverage · IoU with contra-demo union · tip max reach)"]
for w in sorted(runs, key=lambda w: (w["model"], w["level"], w["group"], w["run"])):
    L.append(f"  {w['run']:<34} {w['level']} {w['group']:<8} end {str(w['end']):<7} reach {w['reach']:+5.1f}  past {w['past']:4.1f} %  {w['mm3']:5.0f}  cov {w.get('target_cov') or 0:.2f}  IoU {w.get('iou_contra_union') or 0:.2f}  tip {w.get('tip_reach', float('nan')):+5.1f}")
open(f"{OUT}/stage1.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
