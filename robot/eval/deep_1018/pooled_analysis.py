"""deep_1018, post hoc (10-20, user: the abstract shows the three levels together): every extent-instructed run of L3 · L4 · L5
(lv_analysis.json rows, H90g, L4's design at each level). Share of the removed volume past the level's midline, contra- vs
ipsi-instructed: means, ranges, and a stratified permutation test (relabelling within each level × start × wording stratum;
Monte Carlo, 50,000 draws per test (the calls below), seed 1020, one-sided). Also the novel wording alone and the neutral runs. Not pre-registered:
PREREG v4 judged L3 and L5 separately.   python pooled_analysis.py OUT_DIR"""
import json, sys
import numpy as np

OUT = sys.argv[1]; R = [w for w in json.load(open(f"{OUT}/lv_analysis.json"))["rows"] if w.get("past") is not None and not w.get("abort")]
A = {"L4": "r0", "L3": "rL3_0", "L5": "rL5_2"}
rng = np.random.default_rng(1020)


def test(rows, groups_c, groups_i, n=200000):
    strata = {}
    for w in rows:
        side = "c" if w["group"] in groups_c else "i" if w["group"] in groups_i else None
        if side is None:
            continue
        kind = "trained" if w["group"] in ("contra", "ipsi") else "novel"
        strata.setdefault((w["level"], w["start"], kind), {"c": [], "i": []})[side].append(w["past"])
    S = [(np.array(v["c"] + v["i"]), len(v["c"])) for v in strata.values() if v["c"] and v["i"]]
    def stat(perms):
        return np.mean([x[p[:k]].mean() - x[p[k:]].mean() for (x, k), p in zip(S, perms)])
    obs = stat([np.arange(len(x)) for x, _ in S])
    cnt = 0
    for _ in range(n):
        cnt += stat([rng.permutation(len(x)) for x, _ in S]) >= obs - 1e-12
    c = [v for w in rows if w["group"] in groups_c for v in [w["past"]]]; i = [v for w in rows if w["group"] in groups_i for v in [w["past"]]]
    return {"n_c": len(c), "n_i": len(i), "mean_c": float(np.mean(c)), "mean_i": float(np.mean(i)), "range_c": [min(c), max(c)], "range_i": [min(i), max(i)],
            "strata": len(S), "diff": float(obs), "p": (cnt + 1) / (n + 1), "c_gt_all_i": float(np.mean([x > max(i) for x in c]))}


res = {"all": test(R, ("contra", "ncontra"), ("ipsi", "nipsi"), 50000), "novel": test(R, ("ncontra",), ("nipsi",), 50000),
       "trained": test(R, ("contra",), ("ipsi",), 50000)}
for lv in ("L3", "L4", "L5"):   # per level on the pooled basis (all extent-instructed runs, strata start × wording) — user 10-20
    res[f"all_{lv}"] = test([w for w in R if w["level"] == lv], ("contra", "ncontra"), ("ipsi", "nipsi"), 50000)
nu = [w["past"] for w in R if w["group"] == "neutral"]
res["neutral"] = {"n": len(nu), "mean": float(np.mean(nu)), "range": [min(nu), max(nu)]}
per_level = {lv: {"c": float(np.mean([w["past"] for w in R if w["level"] == lv and w["group"] in ("contra", "ncontra")])),
                  "i": float(np.mean([w["past"] for w in R if w["level"] == lv and w["group"] in ("ipsi", "nipsi")]))} for lv in ("L3", "L4", "L5")}
res["per_level_means"] = per_level
json.dump(res, open(f"{OUT}/pooled_analysis.json", "w"), indent=1)
L = ["POST HOC — L3 · L4 · L5 together (H90g, 105 runs; PREREG v4 judged levels separately)"]
for k in ("all", "novel", "trained", "all_L3", "all_L4", "all_L5"):
    r = res[k]
    L.append(f"  {k:<8} contra n {r['n_c']} mean {r['mean_c']:.1f} % (range {r['range_c'][0]:.1f}–{r['range_c'][1]:.1f}) | ipsi n {r['n_i']} mean {r['mean_i']:.1f} % "
             f"(range {r['range_i'][0]:.1f}–{r['range_i'][1]:.1f}) | stratified diff {r['diff']:+.1f} points over {r['strata']} strata, p {r['p']:.6f} (Monte Carlo) | contra runs above every ipsi run {r['c_gt_all_i']:.0%}")
L.append(f"  neutral  n {res['neutral']['n']} mean {res['neutral']['mean']:.1f} % (range {res['neutral']['range'][0]:.1f}–{res['neutral']['range'][1]:.1f})")
L.append("  per level means contra / ipsi: " + " · ".join(f"{lv} {v['c']:.1f} / {v['i']:.1f} %" for lv, v in per_level.items()))
open(f"{OUT}/pooled_analysis.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
