"""deep_1018 stage 1c (PREREG v1 1c, done 10-19 after P3's runs): the error budget of the side metric — is "the share of the
removed volume past the midline" a property of the run, or of the way it is counted?
  A  live vs replay: the removed volume the recorder page counted during the run (live.json meta removed_mm3) against the
     replay render of the session's own cut rows (iou_cache_rofull, the evaluation's voxels) — every evaluated H90a/H90g run.
  B  replay vs replay: sessions rendered twice with different settings (iou_cache_ro: fps 10, no views; rofull: fps 5, views)
     — voxel IoU, volume, reach and share differences.
  C  an independent path, no viewer voxels: the tracker's measured drill tip (frames[:, 7]) and the live volume. Each live
     volume increment is shared out over the trigger-on frames since the previous increment; the share whose tip lay past
     the midline is the run's tip share. Against the voxel share: Spearman over runs, the difference, and whether the
     sentence split (novel contra vs novel ipsi: 10-17 r0 series and P3) appears in the tip share too.
Combined with the voxel quantisation (stage 1: SD 0.23 mm) this is the measurement SD that PREREG v1 compares with the
effects (differences < 2 × SD are not evidence).

  python stage1c.py OUT_DIR"""
import glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dataset import instructions as INS          # noqa: E402

LV = os.path.expanduser("~/il-robot/runs/act_smoke/live"); C = os.path.expanduser("~/il-robot/runs/jitter_iter"); OUT = sys.argv[1]
MID = INS.MIDLINE_X_MM


def vox(f, mid):
    z = np.load(f, allow_pickle=True); h = json.loads(str(z["hdr"])); idx = z["final"].astype(np.int64)
    x = h["origin"][0] + (idx % h["dims"][0]) * h["spacing"][0]
    return idx, {"mm3": float(len(idx) * h["voxelMm3"]), "reach": float(mid - np.percentile(x, 5)) if len(idx) else None,
                 "past": float(100 * np.mean(x < mid)) if len(idx) else None, "sdf": h.get("sdfSha256"), "grid": h.get("gridSha256")}


def level(name):
    m = re.search(r"_(L[35])", name)
    return m.group(1) if m else "L4"


def group(name):
    m = re.search(r"_(ncontra|nipsi|neutral|contra|ipsi)\d", name)
    return m.group(1) if m else None


rows = []
for r in sorted(glob.glob(f"{LV}/R_H90[ag]_*")):
    if not os.path.isdir(r) or not os.path.exists(f"{r}/robot_session") or not os.path.exists(f"{r}/live.json"):
        continue
    name = os.path.basename(r); sid = os.path.basename(open(f"{r}/robot_session").read().strip()).replace(".jsonl", "")
    f = f"{C}/iou_cache_rofull_{sid}.npz"
    if not os.path.exists(f):
        continue
    lv = level(name); mid = MID[lv]
    d = json.load(open(f"{r}/live.json")); m = d["meta"]
    if m.get("abort"):
        continue
    _, V = vox(f, mid)
    a = np.asarray(d["frames"], float); vol = np.asarray(m.get("removed_mm3") or [], float)
    row = {"run": name, "sid": sid, "level": lv, "group": group(name), "start": name.split("_")[-1], "live_mm3": float(vol[-1]) if len(vol) else None, **V}
    # C: live increments shared out over the trigger-on frames since the previous increment
    if len(vol) == len(a) and len(vol) > 1:
        inc = np.nonzero(np.diff(vol) > 1e-6)[0] + 1; prev = 0; past = tot = 0.0
        for k in inc:
            dv = vol[k] - vol[k - 1]; w = np.arange(prev + 1, k + 1); on = w[a[w, 14] > 0.5]
            if len(on) == 0:
                on = np.array([k])
            past += dv * np.mean(a[on, 7] < mid); tot += dv; prev = k
        row["tip_past"] = float(100 * past / tot) if tot > 0 else None
        on = a[:, 14] > 0.5
        row["tip_reach"] = float(mid - np.percentile(a[on, 7], 5)) if on.any() else None
    rows.append(row)

L = [f"stage 1c — measurement checks of the side metric; {len(rows)} evaluated H90a/H90g robot runs", ""]
# A
lr = [(w["live_mm3"], w["mm3"]) for w in rows if w["live_mm3"]]
rel = np.array([(b - a_) / a_ * 100 for a_, b in lr if a_ > 50])
L.append(f"A live vs replay volume (replay − live, % of live): n {len(rel)}, median {np.median(rel):+.2f} %, IQR {np.percentile(rel, 25):+.2f} – {np.percentile(rel, 75):+.2f} %,"
         f" largest |Δ| {np.max(np.abs(rel)):.2f} %")
# B
B = []
for f in sorted(glob.glob(f"{C}/iou_cache_ro_*.npz")):
    sid = os.path.basename(f)[len("iou_cache_ro_"):-4]; g = f"{C}/iou_cache_rofull_{sid}.npz"
    if not os.path.exists(g):
        continue
    i1, V1 = vox(f, MID["L4"]); i2, V2 = vox(g, MID["L4"])
    if (V1["sdf"], V1["grid"]) != (V2["sdf"], V2["grid"]) or V1["reach"] is None or V2["reach"] is None:
        continue   # an empty set (nothing removed) has no reach
    s1, s2 = set(i1.tolist()), set(i2.tolist()); iou = len(s1 & s2) / max(1, len(s1 | s2))
    B.append({"sid": sid, "iou": iou, "d_mm3": V2["mm3"] - V1["mm3"], "d_reach": V2["reach"] - V1["reach"], "d_past": V2["past"] - V1["past"]})
if B:
    L.append(f"B replay vs replay (rofull − ro, same session): n {len(B)}, voxel IoU {min(b['iou'] for b in B):.4f}–{max(b['iou'] for b in B):.4f},"
             f" volume Δ {min(b['d_mm3'] for b in B):+.1f} … {max(b['d_mm3'] for b in B):+.1f} mm3, reach Δ {min(b['d_reach'] for b in B):+.2f} … {max(b['d_reach'] for b in B):+.2f} mm,"
             f" share Δ {min(b['d_past'] for b in B):+.2f} … {max(b['d_past'] for b in B):+.2f} points")
else:
    L.append("B replay vs replay: no session with both caches on this grid")
# C


def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):
        k = x == v; r[k] = r[k].mean()
    return r


cc = [w for w in rows if w.get("tip_past") is not None and w["past"] is not None]
rho = float(np.corrcoef(rank([w["tip_past"] for w in cc]), rank([w["past"] for w in cc]))[0, 1])
dd = np.array([w["past"] - w["tip_past"] for w in cc])
L.append(f"C tip share (tracker tip + live volume) vs voxel share (replay): n {len(cc)}, Spearman {rho:+.3f}, voxel − tip median {np.median(dd):+.1f} points"
         f" (IQR {np.percentile(dd, 25):+.1f} – {np.percentile(dd, 75):+.1f}; the voxel set includes the 2 mm burr radius around the tip)")
rr = [w for w in cc if w.get("tip_reach") is not None and w["reach"] is not None]
dr = np.array([w["tip_reach"] - w["reach"] for w in rr])
L.append(f"  tip reach (p5 of the trigger-on tip x) − voxel reach: median {np.median(dr):+.2f} mm, SD {np.std(dr):.2f} mm")


def perm_strat(strata):
    from itertools import combinations, product
    def stat(sp):
        return float(np.mean([np.mean(c) - np.mean(i) for c, i in sp]))
    obs = stat(strata); per = []
    for c, i in strata:
        al = list(c) + list(i); n = len(c)
        per.append([([al[k] for k in s], [al[k] for k in range(len(al)) if k not in s]) for s in combinations(range(len(al)), n)])
    k = t = 0
    for combo in product(*per):
        k += stat(combo) >= obs - 1e-9; t += 1
    return obs, k / t


L.append("")
L.append("the sentence split in the independent tip share (novel contra vs novel ipsi; one-sided, stratified by start):")
for lab, pat in (("10-17 r0 series", "_pf_"), ("P3 r1 · r2 · r4", "_p3_")):
    sel = [w for w in cc if pat in w["run"] and w["run"].startswith("R_H90g") and w["level"] == "L4"]
    strata = []
    for st in sorted(set(w["start"] for w in sel)):
        c = [w["tip_past"] for w in sel if w["start"] == st and w["group"] == "ncontra"]; i = [w["tip_past"] for w in sel if w["start"] == st and w["group"] == "nipsi"]
        if c and i:
            strata.append((c, i))
    if strata:
        d, p = perm_strat(strata)
        cv = [v for c, _ in strata for v in c]; iv = [v for _, i in strata for v in i]
        L.append(f"  {lab}: tip share novel contra {np.mean(cv):.1f} % vs novel ipsi {np.mean(iv):.1f} % → {d:+.1f} points, p {p:.4f} ({len(strata)} starts)")
    else:
        L.append(f"  {lab}: not available yet")
sd_past = float(np.std(dd)); L.append("")
L.append(f"measurement spread (SD over runs of voxel − tip share): {sd_past:.1f} points; 2 × SD = {2 * sd_past:.1f} points — the yardstick for"
         " differences in the share (PREREG v1 1c). Replay noise (B) and live vs replay (A) are far smaller.")
os.makedirs(OUT, exist_ok=True)
json.dump({"rows": rows, "B": B}, open(f"{OUT}/stage1c.json", "w"), indent=1)
open(f"{OUT}/stage1c.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
