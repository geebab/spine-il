"""Stall rule basis: per demo, the least bone removed in any 25 s (and 15 s) of its drilling (phase 4), from the dataset's
per-frame removed volume (action drill.progress × removed_mm3_final, render removedMm3, 15 fps)."""
import json, sys, glob
import numpy as np, pandas as pd
DS = sys.argv[1]; W = {25: 375, 15: 225}
eps = [json.loads(l) for l in open(f"{DS}/meta/spine_il/episodes.jsonl") if l.strip()]
df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action", "phase.index"]) for f in sorted(glob.glob(f"{DS}/data/*/*.parquet"))])
seen, rows = set(), []
for e in eps:
    if e.get("source") != "hand" or e["recording"] in seen:
        continue
    seen.add(e["recording"])
    d = df[df.episode_index == e["episode_index"]].sort_values("frame_index")
    ph = np.asarray([np.ravel(x)[0] for x in d["phase.index"]]); a = np.stack(d["action"].to_numpy())
    vol = a[:, 15] * e["removed_mm3_final"]; v4 = vol[ph == 4]
    r = {"session": e["session_id"], "level": (e.get("goal") or {}).get("level"), "drill_s": round(len(v4) / 15, 1),
         "rate_mm3_s": round(float((v4[-1] - v4[0]) / (len(v4) / 15)), 1)}
    for s, n in W.items():
        r[f"min_{s}s"] = round(float(np.min(v4[n:] - v4[:-n])), 1) if len(v4) > n else None
    rows.append(r)
for s in W:
    v = [r for r in rows if r[f"min_{s}s"] is not None]; m = min(v, key=lambda r: r[f"min_{s}s"])
    print(f"{DS.split('/')[-1]}: {len(rows)} demos, least in any {s} s = {m[f'min_{s}s']} mm3 ({m['session'][-4:]}, {m['level']}); "
          f"no window {len(rows) - len(v)}")
print("rate over the phase", min(r["rate_mm3_s"] for r in rows), "-", max(r["rate_mm3_s"] for r in rows), "mm3/s")
json.dump(rows, open(sys.argv[2], "w"), indent=1) if len(sys.argv) > 2 else None


def longest_flat(DS):
    """Longest stretch of phase-4 frames without any removal (s) and the same with the trigger off as well."""
    out = []
    seen = set()
    for e in eps:
        if e.get("source") != "hand" or e["recording"] in seen:
            continue
        seen.add(e["recording"])
        d = df[df.episode_index == e["episode_index"]].sort_values("frame_index")
        ph = np.asarray([np.ravel(x)[0] for x in d["phase.index"]]); a = np.stack(d["action"].to_numpy())
        a4 = a[ph == 4]; vol = a4[:, 15]; trig = a4[:, 14] > 0.5
        flat = np.r_[False, np.diff(vol) <= 1e-9]
        best = cur = bt = ct = 0
        for f, t in zip(flat, trig):
            cur = cur + 1 if f else 0; best = max(best, cur)
            ct = ct + 1 if (f and not t) else 0; bt = max(bt, ct)
        out.append((e["session_id"][-4:], round(best / 15, 1), round(bt / 15, 1)))
    return out


if len(sys.argv) > 3:
    o = longest_flat(DS)
    print("longest no-removal stretch:", sorted(o, key=lambda x: -x[1])[:4], "| no removal and trigger off:", sorted(o, key=lambda x: -x[2])[:4])
