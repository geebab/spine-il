"""deep_1018 stage 2d: the policy's progress output against the removed volume on the robot (done_log × live.json removed_mm3),
next to what the progress label of a contra demo would be at that volume (removed / its own final volume, median of the
level's lang2-contra demos). Volume at which each run's output first reached the end level 0.85.

  python stage2_progress.py OUT_DIR"""
import json, os, re, sys
import numpy as np
H = os.path.expanduser("~"); LV = f"{H}/il-robot/runs/act_smoke/live"; OUT = sys.argv[1]
S1 = json.load(open(f"{OUT}/stage1.json"))
fin = {}
for d in S1["demos"]:
    fin.setdefault((d["level"], d["label"]), []).append(d["mm3"])
L = ["run · end · volume when the progress output first ≥ 0.85 · its progress there vs a contra demo's label at that volume (V / final, median)"
     " · same for an ipsi demo · progress at 600 / 900 mm3"]
agg = {}
for w in S1["runs"]:
    m_ = re.search(r"_(L[35])?(ncontra|nipsi|contra|ipsi|neutral)\d", w["run"])
    if not m_:
        continue
    lv, g = m_.group(1) or "L4", m_.group(2)
    m = json.load(open(f"{LV}/{w['run']}/live.json"))["meta"]; vol = np.asarray(m.get("removed_mm3") or [], float); dl = m.get("done_log") or []
    pts = [(vol[min(e[0], len(vol) - 1)], max(e[1])) for e in dl if e[1] and len(vol)]
    if not pts:
        continue
    P = np.array(pts); k = np.nonzero(P[:, 1] >= 0.85)[0]
    v85 = float(P[k[0], 0]) if len(k) else None
    lab = lambda lab_, v: float(np.median([min(1.0, v / f) for f in fin[(lv, lab_)]])) if v is not None and (lv, lab_) in fin else None
    def p_at(v):
        j = np.nonzero(P[:, 0] >= v)[0]; return float(P[j[0], 1]) if len(j) else None
    r = {"run": w["run"], "level": lv, "group": g, "model": w["model"], "end": w["end"], "v85": v85, "c_label": lab("contralateral", v85), "i_label": lab("ipsilateral", v85),
         "p600": p_at(600), "p900": p_at(900)}
    agg.setdefault((w["model"], lv, g), []).append(r)
    f = lambda v, s="+.2f": "  —  " if v is None else format(v, s)
    L.append(f"  {w['run']:<32} {str(w['end']):<7} ≥0.85 at {f(v85, '5.0f')} mm3 | contra-demo label there {f(r['c_label'], '.2f')} · ipsi-demo {f(r['i_label'], '.2f')} | output @600 {f(r['p600'], '.2f')} @900 {f(r['p900'], '.2f')}")
L += ["", "means per model × level × group"]
for k in sorted(agg):
    rs = agg[k]; mv = lambda key: np.mean([r[key] for r in rs if r[key] is not None]) if any(r[key] is not None for r in rs) else float("nan")
    L.append(f"  {k[0]} {k[1]} {k[2]:<8} n {len(rs)}  reached 0.85 in {sum(r['v85'] is not None for r in rs)}  at {mv('v85'):5.0f} mm3 | contra-demo label there {mv('c_label'):.2f} · ipsi-demo {mv('i_label'):.2f} | output @600 {mv('p600'):.2f} @900 {mv('p900'):.2f}")
json.dump(agg and {" ".join(k): v for k, v in agg.items()}, open(f"{OUT}/stage2_progress.json", "w"), indent=1)
open(f"{OUT}/stage2_progress.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
