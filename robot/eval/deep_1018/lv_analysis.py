"""deep_1018 PREREG v4: the extent-sentence split at each level with L4's design — H90g 020000, end level 0.85, rcm-first.
Per level four start poses: start A with trained contra / ipsi ×5 and novel contra / ipsi ×5, three more starts with novel
contra ×2, novel ipsi ×2, neutral ×1 (L4: r0 = the 10-17 R_H90g_pf_* series, r1 · r2 · r4 = P3; L3: rL3_0 + rL3_1 · rL3_2 · rL3_3;
L5: rL5_2 + rL5_3 · rL5_6 · rL5_7). Per run, from the evaluation cache's removed voxels (as proof_analysis.py / p3_analysis.py):
the share of the removed volume past the level's midline (instructions.MIDLINE_X_MM), medial reach = midline − x p5, removed
volume, who ended the drilling.
Primary per level: stratified permutation test (relabelling only within each start), one-sided, of the share past the midline,
novel contra > novel ipsi, over the level's four starts. Secondary: the trained pair at start A, the reach, separation per start,
the neutral runs, criterion ① for the record, the runs the harness ended.

  python lv_analysis.py OUT_DIR [--levels L3,L4,L5]"""
import argparse, glob, json, os, re, sys
from itertools import combinations, product
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dataset import instructions as INS          # noqa: E402

LV = os.path.expanduser("~/il-robot/runs/act_smoke/live"); C = os.path.expanduser("~/il-robot/runs/jitter_iter")
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--levels", default="L3,L4,L5"); a = ap.parse_args()
STARTS = {"L4": ["r0", "r1", "r2", "r4"], "L3": ["rL3_0", "rL3_1", "rL3_2", "rL3_3"], "L5": ["rL5_2", "rL5_3", "rL5_6", "rL5_7"]}
PAT = re.compile(r"^R_H90g_(?:pf_|p3_)?(L[35])?(ncontra|nipsi|neutral|contra|ipsi)(\d+)b?_C1006_(r\w+)$")


def run_row(r, lv, g, st):
    sid = os.path.basename(open(f"{r}/robot_session").read().strip()).replace(".jsonl", "")
    m = json.load(open(f"{r}/live.json"))["meta"]; mid = INS.MIDLINE_X_MM[lv]
    row = {"run": os.path.basename(r), "level": lv, "group": g, "start": st, "end": m.get("drill_end"), "abort": m.get("abort"),
           "mm3_live": (m.get("removed_mm3") or [0])[-1], "reach": None}
    f = f"{C}/iou_cache_rofull_{sid}.npz"
    if os.path.exists(f):
        z = np.load(f, allow_pickle=True); h = json.loads(str(z["hdr"])); idx = z["final"].astype(np.int64)
        x = h["origin"][0] + (idx % h["dims"][0]) * h["spacing"][0]
        row.update(reach=float(mid - np.percentile(x, 5)) if len(idx) else None, past=float(100 * np.mean(x < mid)) if len(idx) else 0.0,
                   mm3=float(len(idx) * h["voxelMm3"]))
    return row


def strat_perm(strata):
    def stat(sp):
        return float(np.mean([np.mean(c) - np.mean(i) for c, i in sp]))
    obs = stat(strata); per = []
    for c, i in strata:
        al = list(c) + list(i); n = len(c)
        per.append([([al[k] for k in s], [al[k] for k in range(len(al)) if k not in s]) for s in combinations(range(len(al)), n)])
    k = t = 0
    for combo in product(*per):
        k += stat(combo) >= obs - 1e-9; t += 1
    return obs, k / t, t


rows = []
for r in sorted(glob.glob(f"{LV}/R_H90g_*_C1006_r*")):
    m = PAT.match(os.path.basename(r))
    if not m or not os.path.isdir(r) or not os.path.exists(f"{r}/live.json") or not os.path.exists(f"{r}/robot_session"):
        continue
    lv = m.group(1) or "L4"; st = m.group(4)
    if lv in a.levels.split(",") and st in STARTS[lv]:
        rows.append(run_row(r, lv, m.group(2), st))

L = ["PREREG v4 — the extent-sentence split per level, L4's design (H90g 020000, end level 0.85, rcm-first, base C1006)", ""]
res = {}
for lv in a.levels.split(","):
    rs = [w for w in rows if w["level"] == lv and not w["abort"]]; ev = [w for w in rs if w["reach"] is not None]
    L.append(f"== {lv} (midline x {INS.MIDLINE_X_MM[lv]} mm): {len(rs)} runs, {len(ev)} evaluated")
    per = {}
    for st in STARTS[lv]:
        f = lambda g: [w for w in ev if w["start"] == st and w["group"] == g]
        c, i, nu, tc, ti = f("ncontra"), f("nipsi"), f("neutral"), f("contra"), f("ipsi")
        fmt = lambda ws: "/".join(f"{w['past']:.1f}" for w in ws) or "—"
        line = f"  {st:<6} novel contra {fmt(c)} | novel ipsi {fmt(i)} | neutral {fmt(nu)}"
        if tc or ti:
            line += f" | trained contra {fmt(tc)} | trained ipsi {fmt(ti)}"
        if c and i:
            d = np.mean([w["past"] for w in c]) - np.mean([w["past"] for w in i]); sep = min(w["past"] for w in c) > max(w["past"] for w in i)
            per[st] = {"diff": float(d), "separated": bool(sep)}; line += f"  → novel diff {d:+.1f} points, separated {sep}"
        L.append(line)
    strata = [([w["past"] for w in ev if w["start"] == st and w["group"] == "ncontra"], [w["past"] for w in ev if w["start"] == st and w["group"] == "nipsi"]) for st in STARTS[lv]]
    strata = [s for s in strata if s[0] and s[1]]
    out = {"per_start": per, "n_runs": len(rs), "n_eval": len(ev)}
    if len(strata) == len(STARTS[lv]):
        d, p, t = strat_perm(strata); every = all(per[st]["diff"] > 0 for st in STARTS[lv])
        verdict = ("the split holds at every start" if p < 0.05 and every else "the split holds on average, not at every start" if p < 0.05
                   else "the split is not confirmed")
        out["primary"] = {"diff": d, "p": p, "relabellings": t, "verdict": verdict}
        L.append(f"  PRIMARY: share past the midline, novel contra − novel ipsi, mean over 4 starts {d:+.1f} points, p {p:.5f} ({t} relabellings) → {verdict}")
        rstr = [([w["reach"] for w in ev if w["start"] == st and w["group"] == "ncontra"], [w["reach"] for w in ev if w["start"] == st and w["group"] == "nipsi"]) for st in STARTS[lv]]
        dr, pr, _ = strat_perm(rstr); out["reach"] = {"diff": dr, "p": pr}
        L.append(f"  secondary — reach {dr:+.1f} mm, p {pr:.5f}; starts with complete separation {sum(per[s]['separated'] for s in STARTS[lv])}/4")
    else:
        L.append(f"  PRIMARY: not computable yet — starts with both novel groups evaluated {len(strata)}/4")
    A = STARTS[lv][0]; tc = [w["past"] for w in ev if w["start"] == A and w["group"] == "contra"]; ti = [w["past"] for w in ev if w["start"] == A and w["group"] == "ipsi"]
    if len(tc) >= 2 and len(ti) >= 2:
        d, p, t = strat_perm([(tc, ti)]); out["trained"] = {"diff": d, "p": p, "n": [len(tc), len(ti)]}
        L.append(f"  secondary — trained wording at {A}: {np.mean(tc):.1f} vs {np.mean(ti):.1f} % ({len(tc)} vs {len(ti)}), {d:+.1f} points, p {p:.4f}")
    nu = [w["past"] for w in ev if w["group"] == "neutral"]
    if nu:
        mc = np.mean([w["past"] for w in ev if w["group"] == "ncontra"] or [np.nan]); mi = np.mean([w["past"] for w in ev if w["group"] == "nipsi"] or [np.nan])
        L.append(f"  secondary — neutral (n {len(nu)}) {np.mean(nu):.1f} % vs novel contra {mc:.1f} / novel ipsi {mi:.1f} (descriptive)")
    k5 = sum(w["reach"] > 5 for w in ev if w["group"] in ("ncontra", "contra")); n5 = sum(w["group"] in ("ncontra", "contra") for w in ev)
    h = [w["run"] for w in rs if w["end"] in ("time", "stall", "volume")]
    L.append(f"  for the record — criterion ① (contra > 5 mm past the midline) {k5}/{n5}; ended by the harness (failed runs, analysed for the side) {len(h)}")
    res[lv] = out; L.append("")
L.append("summary: " + " | ".join(f"{lv} {res[lv]['primary']['diff']:+.1f} points p {res[lv]['primary']['p']:.4f}" if "primary" in res[lv] else f"{lv} incomplete" for lv in res))
os.makedirs(a.out, exist_ok=True)
json.dump({"rows": rows, "levels": res}, open(f"{a.out}/lv_analysis.json", "w"), indent=1)
open(f"{a.out}/lv_analysis.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
