"""deep_1018 stage 2d (PREREG.md v2): the end level 0.965 runs (R_H90g_e965_*) against the 0.85 series (R_H90g_pf_contra* / ipsi*)
and the L4 contra demos — reach, volume, who ended, reach at the same volume as the demos; the v2 predictions H-end / H-path
and the ipsi specificity, one-sided permutation tests contra e965 > pf.

  python e965_analysis.py OUT_DIR    (after stage1.py / stage2.py / stage2_progress.py on OUT_DIR)"""
import json, os, re, sys
from itertools import combinations
import numpy as np

OUT = sys.argv[1]
S2 = json.load(open(f"{OUT}/stage2.json"))
S1 = json.load(open(f"{OUT}/stage1.json"))
cov = {w["run"]: w.get("target_cov") for w in S1["runs"]}


def perm(a, b):
    al = list(a) + list(b); obs = np.mean(a) - np.mean(b); k = t = 0
    for c in combinations(range(len(al)), len(a)):
        s = set(c); k += np.mean([al[i] for i in c]) - np.mean([al[i] for i in range(len(al)) if i not in s]) >= obs - 1e-9; t += 1
    return obs, k / t


sel = lambda tag, g: [r for r in S2 if r["model"] == "H90g" and r["level"] == "L4" and f"_{tag}_{g}" in r["run"] and re.search(rf"_{g}\d", r["run"])]
E = {g: sel("e965", g) for g in ("contra", "ipsi")}; P = {g: sel("pf", g) for g in ("contra", "ipsi")}
L = ["end level 0.965 (e965) vs 0.85 (pf), H90g, L4, C1006 r0, trained sentence; L4 contra demos: reach +10.3 mm, 1,804 mm3 (medians +10.4 / 1,804)", ""]
for tag, D in (("e965", E), ("pf", P)):
    for g, rs in D.items():
        for r in rs:
            f = lambda v: "  —  " if v is None else f"{v:+5.1f}"
            L.append(f"  {r['run']:<30} ended by {str(r['end']):<7} reach {r['reach']:+5.1f}  {r['mm3']:5.0f} mm3 | contra demos at that volume {f(r['demo_contra_at_V'])}"
                     f" | target cov {cov.get(r['run']) or 0:.2f} | tip meas {f(r.get('tip_meas'))} | progress end {r['progress_end']} | clamped {r['clamped']}")
L.append("")
ec, ei = E["contra"], E["ipsi"]
if ec:
    k5 = sum(r["reach"] > 5 for r in ec); kv = sum(r["mm3"] >= 1400 and r["reach"] > 5 for r in ec); mv = np.mean([r["mm3"] for r in ec])
    L.append(f"contra e965: n {len(ec)}, reach mean {np.mean([r['reach'] for r in ec]):+.1f} mm, volume mean {mv:.0f} mm3, reach > 5 mm {k5}/{len(ec)},"
             f" volume ≥ 1,400 and reach > 5 {kv}/{len(ec)}, ended by the harness (time) {sum(r['end'] == 'time' for r in ec)}")
    verdict = ("H-end confirmed" if kv >= 4 else "H-path confirmed" if (mv >= 1300 and k5 <= 1) else
               "neither (volume did not rise to 1,300 mm3)" if mv < 1300 and k5 <= 1 else "in between — neither confirmed")
    L.append(f"  → {verdict}   (v2: H-end = ≥ 4/5 with ≥ 1,400 mm3 and > 5 mm; H-path = mean volume ≥ 1,300 and > 5 mm in ≤ 1/5)")
    L.append(f"  criterion ① (10-16): contra > 5 mm in ≥ 4 of 5 and ipsi past midline − 2 mm in 0 → "
             f"{'met' if k5 >= 4 and len(ec) >= 5 and ei and sum(r['reach'] > 2 for r in ei) == 0 else 'not met'}")
if ei:
    k2 = sum(r["reach"] > 2 for r in ei)
    L.append(f"ipsi e965: n {len(ei)}, reach mean {np.mean([r['reach'] for r in ei]):+.1f} mm, past +2 mm {k2}/{len(ei)} → "
             f"{'the sentence keeps ipsi on its side' if k2 == 0 else 'the end level alone cannot carry the extent' if k2 >= 2 else 'one ipsi run past +2 mm'}")
for g in ("contra", "ipsi"):
    if len(E[g]) >= 2 and len(P[g]) >= 2:
        d, p = perm([r["reach"] for r in E[g]], [r["reach"] for r in P[g]]); dv, pv = perm([r["mm3"] for r in E[g]], [r["mm3"] for r in P[g]])
        L.append(f"{g}: e965 − pf reach {d:+.1f} mm (p {p:.3f}), volume {dv:+.0f} mm3 (p {pv:.3f}), one-sided")
if len(ec) >= 2 and len(ei) >= 2:
    d, p = perm([r["reach"] for r in ec], [r["reach"] for r in ei]); L.append(f"e965 contra − ipsi reach {d:+.1f} mm, p {p:.3f}")
print("\n".join(L))
