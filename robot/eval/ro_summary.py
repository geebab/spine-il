"""Robot-origin sessions: per setting × recording means of the recorded-vs-recorded metrics (contaminated excluded)."""
import json, glob, re, collections, statistics as st
from pathlib import Path
J = Path.home() / "il-robot/runs/jitter_iter"
bad = {w for w in (J / "work/ro_contaminated.txt").read_text().split() if w.startswith("pr_")}
bad |= {"pr_3868s20"}
recs = ["3868", "91cb", "bef5", "d340", "bad1", "0338", "919d", "6639"]
by = collections.defaultdict(list)
for f in glob.glob(str(J / "ro_s_*_iou.json")):
    d = json.loads(open(f).read()); t = d.get("tag", "")
    m = re.match(r"pr_(\w{4})(\w)(\d)$", t) or re.match(r"pr_(\w{4})0(\d)(\d)$", t)
    if not m or t in bad: continue
    s = m[2] if m[2].isalpha() else "d"
    by[(s, m[1])].append(d)
def g(d, k, m): return (d.get(k) or {}).get(m)
rows = []
for s in sorted({k[0] for k in by}):
    per = {r: by.get((s, r), []) for r in recs}
    have = [r for r in recs if per[r]]
    def mean(fn): return st.mean(st.mean(fn(d) for d in per[r]) for r in have)
    rows.append((s, sum(len(v) for v in per.values()), len(have),
                 mean(lambda d: g(d, "rr_hr", "iou")), mean(lambda d: d["robot_rec_mm3"] / d["hand_rec_mm3"]),
                 mean(lambda d: g(d, "rr_hr", "tol1_dice")), mean(lambda d: g(d, "rr_hr", "assd_mm")),
                 st.mean([g(d, "rs_hr", "iou") for r in have for d in per[r] if g(d, "rs_hr", "iou")] or [float("nan")]),
                 " ".join(f"{r}:{'/'.join(f'{g(d, 'rr_hr', 'iou'):.2f}' for d in per[r]) or '-'}" for r in recs)))
print("set  n recs  IoU   vol×  tol1  ASSD  simIoU(tick-matched, runs that have it)  per recording")
for r in rows: print(f"{r[0]:3s} {r[1]:2d} {r[2]}/8  {r[3]:.3f} {r[4]:.2f}  {r[5]:.3f} {r[6]:.2f}  {r[7]:.3f}  {r[8]}")
