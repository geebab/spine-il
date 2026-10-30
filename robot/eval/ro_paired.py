"""Paired comparison on the recordings both settings have (mean over runs per recording): ΔIoU, Δvol ratio, wins."""
import sys, statistics as st, ro_summary_data as D
a, b = sys.argv[1], sys.argv[2]
common = [r for r in D.recs if D.by.get((a, r)) and D.by.get((b, r))]
def m(s, r, f): return st.mean(f(d) for d in D.by[(s, r)])
iou = lambda d: d["rr_hr"]["iou"]; vol = lambda d: d["robot_rec_mm3"] / d["hand_rec_mm3"]
di = [m(a, r, iou) - m(b, r, iou) for r in common]; dv = [m(a, r, vol) - m(b, r, vol) for r in common]
print(f"{a} vs {b} on {len(common)} recs: ΔIoU {st.mean(di):+.3f} (wins {sum(x > 0 for x in di)}/{len(di)}), Δvol× {st.mean(dv):+.2f}")
