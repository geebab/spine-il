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
