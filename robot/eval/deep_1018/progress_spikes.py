"""deep_1018, post hoc (after PREREG v2 runs): how the policy's progress output reaches the end level — a gradual rise or a
one-answer jump. Per run: answers (done_log), answers >= 0.85 / >= 0.965, jumps (previous answer < 0.8, this one >= 0.85),
the last two answers.

  python progress_spikes.py OUT_DIR"""
import glob, json, os, sys
import numpy as np

LV = os.path.expanduser("~/il-robot/runs/act_smoke/live"); OUT = sys.argv[1]
rows = []
for r in sorted(glob.glob(f"{LV}/R_H90g_e965_*") + glob.glob(f"{LV}/R_H90g_pf_*")):
    if not os.path.isdir(r):
        continue
    m = json.load(open(f"{r}/live.json"))["meta"]; mx = [max(e[1]) for e in m.get("done_log") or [] if e[1]]
    if len(mx) < 2:
        continue
    rows.append({"run": os.path.basename(r), "end": m.get("drill_end"), "answers": len(mx),
                 "ge85": sum(v >= 0.85 for v in mx), "ge965": sum(v >= 0.965 for v in mx),
                 "jumps": sum(1 for i in range(1, len(mx)) if mx[i] >= 0.85 and mx[i - 1] < 0.8),
                 "last2": [round(mx[-2], 2), round(mx[-1], 2)], "median": float(np.median(mx))})
json.dump(rows, open(f"{OUT}/progress_spikes.json", "w"), indent=1)
L = ["progress output per run (post hoc): answers · ≥ 0.85 · ≥ 0.965 · jumps < 0.8 → ≥ 0.85 · median · last two answers", ""]
for w in rows:
    L.append(f"  {w['run']:<32} {str(w['end']):<7} {w['answers']:4} {w['ge85']:3} {w['ge965']:3} {w['jumps']:3}  median {w['median']:.2f}  last {w['last2'][0]:.2f} → {w['last2'][1]:.2f}")
pe = [w for w in rows if w["end"] == "policy"]
jl = sum(1 for w in pe if w["last2"][0] < 0.8)
L += ["", f"policy-ended runs: {len(pe)}, last answer before the end < 0.8 (a one-answer jump) in {jl}"]
open(f"{OUT}/progress_spikes.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L[-1:]))
