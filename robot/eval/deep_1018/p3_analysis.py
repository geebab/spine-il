"""deep_1018 P3 (PREREG.md v3): do the novel extent sentences split the drilling side from other start poses?
H90g 020000, L4, C1006 starts r1 · r2 · r4 (the 10-17 r0 series is a secondary stratum), novel contra / novel ipsi ×2 and the
neutral sentence ×1 per start. Per run, from the evaluation cache's removed voxels (as proof_analysis.py): the share of the
removed volume past the L4 midline, medial reach = midline − x p5, the removed volume, who ended the drilling.
Primary: stratified permutation test (relabelling only within each start), one-sided, of the share past the midline,
novel contra > novel ipsi, over the new starts. Secondary: the same for the reach, separation per start, the neutral runs,
the pooled test with r0, criterion ① for the record.

  python p3_analysis.py OUT_DIR [--series p3] [--starts r1,r2,r4] [--pooled r0]
  (check of the code: --series pf --starts r0 --pooled "" reproduces the 10-17 series, p 0.004)"""
import argparse, glob, json, os, re, sys
from itertools import combinations, product
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dataset import instructions as INS          # noqa: E402

MID = INS.MIDLINE_X_MM["L4"]
LV = os.path.expanduser("~/il-robot/runs/act_smoke/live"); C = os.path.expanduser("~/il-robot/runs/jitter_iter")
ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--series", default="p3"); ap.add_argument("--starts", default="r1,r2,r4")
ap.add_argument("--pooled", default="r0", help="extra stratum from the 10-17 series (R_H90g_pf_*), '' for none")
a = ap.parse_args()
STARTS = [s for s in a.starts.split(",") if s]; POOL = [s for s in a.pooled.split(",") if s]


def run_row(r):
    sid = os.path.basename(open(f"{r}/robot_session").read().strip()).replace(".jsonl", "")
    m = json.load(open(f"{r}/live.json"))["meta"]; name = os.path.basename(r)
    row = {"run": name, "group": re.match(r"(ncontra|nipsi|neutral|contra|ipsi)", name.split("_")[3]).group(1), "start": name.split("_")[-1],
           "mm3_live": (m.get("removed_mm3") or [0])[-1], "end": m.get("drill_end"), "abort": m.get("abort"), "reach": None}
    f = f"{C}/iou_cache_rofull_{sid}.npz"
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True); h = json.loads(str(z["hdr"])); idx = z["final"].astype(np.int64)
        x = h["origin"][0] + (idx % h["dims"][0]) * h["spacing"][0]
        row.update(reach=float(MID - np.percentile(x, 5)), past=float(100 * np.mean(x < MID)), mm3=float(len(idx) * h["voxelMm3"]))
    return row


def rows_of(series, starts):
    out = []
    for st in starts:
        for r in sorted(glob.glob(f"{LV}/R_H90g_{series}_*_C1006_{st}"), key=os.path.getmtime):
            if os.path.isdir(r) and os.path.exists(f"{r}/live.json") and os.path.exists(f"{r}/robot_session"):
                out.append(run_row(r))
    return out


def strat_perm(strata, key):
    """strata: [(values of novel contra, values of novel ipsi)] per start. Statistic: the mean over the starts of the per-start
    difference of means. One-sided P(stat ≥ observed) over every relabelling within the starts."""
    def stat(splits):
        return float(np.mean([np.mean(c) - np.mean(i) for c, i in splits]))
    obs = stat(strata); per = []
    for c, i in strata:
        allv = list(c) + list(i); n = len(c)
        per.append([([allv[k] for k in s], [allv[k] for k in range(len(allv)) if k not in s]) for s in combinations(range(len(allv)), n)])
    k = t = 0
    for combo in product(*per):
        k += stat(combo) >= obs - 1e-9; t += 1
    return obs, k / t, t


new = rows_of(a.series, STARTS); old = rows_of("pf", POOL) if POOL else []
L = [f"P3 — H90g 020000, L4, C1006 starts {', '.join(STARTS)} (series {a.series}); L4 midline x = {MID} mm", ""]
for w in new + old:
    ev = "  (no evaluation yet)" if w["reach"] is None else f"  past midline {w['past']:4.1f} %  reach {w['reach']:+5.1f} mm  {w['mm3']:5.0f} mm3"
    L.append(f"  {w['run']:<34} {w['start']:<3} {w['group']:<8} ended by {str(w['end']):<7}{ev}" + (f"  ABORT {w['abort']}" if w["abort"] else ""))
ev_new = [w for w in new if w["reach"] is not None and not w["abort"]]
res = {"runs": new + old, "starts": STARTS}


def strata_of(rows, starts, key):
    s = []
    for st in starts:
        c = [w[key] for w in rows if w["start"] == st and w["group"] == "ncontra"]
        i = [w[key] for w in rows if w["start"] == st and w["group"] == "nipsi"]
        if c and i:
            s.append((c, i))
    return s


L.append("")
L.append("per start: novel contra | novel ipsi | neutral — share past the midline (%), reach (mm); complete separation (every contra > every ipsi)")
per_start = {}
for st in STARTS + POOL:
    rs = ev_new if st in STARTS else [w for w in old if w["reach"] is not None]
    c = [w for w in rs if w["start"] == st and w["group"] == "ncontra"]; i = [w for w in rs if w["start"] == st and w["group"] == "nipsi"]
    nu = [w for w in rs if w["start"] == st and w["group"] == "neutral"]
    if not (c and i):
        L.append(f"  {st}: incomplete (novel contra {len(c)}, novel ipsi {len(i)})"); continue
    f = lambda ws, k: "/".join(f"{w[k]:.1f}" for w in ws) or "—"
    d = np.mean([w["past"] for w in c]) - np.mean([w["past"] for w in i])
    sep = min(w["past"] for w in c) > max(w["past"] for w in i)
    per_start[st] = {"diff_past": float(d), "separated": bool(sep)}
    L.append(f"  {st}: share {f(c, 'past')} | {f(i, 'past')} | {f(nu, 'past')}  → difference {d:+.1f} points, separated {sep}"
             f"   reach {f(c, 'reach')} | {f(i, 'reach')} | {f(nu, 'reach')}")
res["per_start"] = per_start

L.append("")
S1 = strata_of(ev_new, STARTS, "past")
if len(S1) == len(STARTS) and S1:
    d, p, t = strat_perm(S1, "past")
    every = all(per_start[st]["diff_past"] > 0 for st in STARTS)
    verdict = ("the split holds at every new start" if p < 0.05 and every else
               "the split holds on average, not at every start" if p < 0.05 else "the split is not confirmed at the new starts")
    res["primary"] = {"diff": d, "p": p, "relabellings": t, "verdict": verdict}
    L.append(f"PRIMARY (v3): share past the midline, novel contra − novel ipsi, mean over {len(S1)} starts {d:+.1f} points,"
             f" stratified permutation p {p:.4f} (one-sided, {t} relabellings) → {verdict}")
    dr, pr, _ = strat_perm(strata_of(ev_new, STARTS, "reach"), "reach")
    res["reach"] = {"diff": dr, "p": pr}
    L.append(f"  secondary — reach: {dr:+.1f} mm, p {pr:.4f}")
    sep_n = sum(per_start[st]["separated"] for st in STARTS)
    L.append(f"  secondary — starts with complete separation: {sep_n}/{len(STARTS)}")
else:
    L.append(f"PRIMARY: not computable yet — starts with both groups evaluated: {len(S1)}/{len(STARTS)}")
nu = [w for w in ev_new if w["group"] == "neutral"]
if nu:
    mc = np.mean([w["past"] for w in ev_new if w["group"] == "ncontra"]); mi = np.mean([w["past"] for w in ev_new if w["group"] == "nipsi"])
    mn = np.mean([w["past"] for w in nu])
    L.append(f"  secondary — neutral (n {len(nu)}): share {mn:.1f} % vs novel contra {mc:.1f} / novel ipsi {mi:.1f} → "
             f"{'between' if mi < mn < mc else 'not between'} (descriptive)")
if POOL and old:
    S2 = strata_of(ev_new + [w for w in old if w["reach"] is not None], STARTS + POOL, "past")
    if len(S2) == len(STARTS) + len(POOL):
        d2, p2, t2 = strat_perm(S2, "past"); res["pooled"] = {"diff": d2, "p": p2}
        L.append(f"  secondary — pooled with {', '.join(POOL)} (10-17): {d2:+.1f} points, p {p2:.4f} ({t2} relabellings)")
k5 = sum(w["reach"] > 5 for w in ev_new if w["group"] == "ncontra"); n5 = sum(w["group"] == "ncontra" for w in ev_new)
L.append(f"  for the record — criterion ① (contra > 5 mm past the midline): novel contra {k5}/{n5}")
h = [w["run"] for w in new if w["end"] in ("time", "stall", "volume")]
L.append(f"  ended by the harness (failed runs, analysed for the side): {len(h)}" + (f" — {', '.join(h)}" if h else ""))
os.makedirs(a.out, exist_ok=True)
json.dump(res, open(f"{a.out}/p3_analysis_{a.series}.json", "w"), indent=1)
open(f"{a.out}/p3_analysis_{a.series}.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
