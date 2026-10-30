"""Pair per-arm candidates at the same P and run the two-arm strict check."""
import sys, time, pickle, collections, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import verify_two_arm as V
from lerobot_robot_spinearm import safezone as sz
A = pickle.load(open(P / "stageA.pkl", "rb")); res = A["res"]; d = A["d"]
arms, cam, info = V.load_models(); z = sz.Zone(arms, cam); z.set_start(NEUTRAL)
TOP = int(sys.argv[1]) if len(sys.argv) > 1 else 5

def pick(lst, a):
    lst = sorted(lst, key=lambda r: (r["oerr"] + 0.2 * np.abs(r["q"] - CUR[a])[:5].sum() / 5 - 0.5 * min(r["marg"], 20)))
    out = []
    for r in lst:
        if all(np.abs(r["q"] - o["q"]).max() > 3 for o in out):
            out.append(r)
        if len(out) >= TOP: break
    return out

by = {a: collections.defaultdict(list) for a in res}
for a in res:
    for r in res[a]:
        if r["ok"]: by[a][r["ip"]].append(r)
pairs, fails = [], collections.Counter(); t0 = time.time()
for ip in range(len(A["Ps"])):
    L, R = pick(by["left"][ip], "left"), pick(by["right"][ip], "right")
    for l in L:
        for r in R:
            ok, why, m = z.check({"left": l["q"], "right": r["q"]}, strict=True, see=("left", "right"))
            for w in why: fails[w.split(" ")[0] + " " + " ".join(x for x in w.split(" ")[1:] if not any(ch.isdigit() for ch in x))] += 1
            pairs.append(dict(ip=ip, l=l, r=r, ok=ok, why=why, m=m, vec=(r["tip"] - l["tip"]) * 1000))
print(f"{len(pairs)} pairs {sum(p['ok'] for p in pairs)} ok in {time.time()-t0:.0f}s")
print(fails.most_common(12))
pickle.dump(pairs, open(P / "stageB.pkl", "wb"))
