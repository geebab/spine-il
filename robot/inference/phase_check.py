import asyncio, json, sys, numpy as np, glob
from pathlib import Path
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import rollout as R
import pyarrow.parquet as pq
EP = int(sys.argv[1])
async def main():
    ds = Path.home() / "il-robot/datasets/spine_rev4_20260929"
    meta, S = R.episode(ds, EP)
    lab, fi = [], []
    for f in sorted(glob.glob(str(ds / "data/*/*.parquet"))):
        t = pq.read_table(f); e = np.asarray(t.column("episode_index").to_pylist()); sel = np.nonzero(e == EP)[0]
        if len(sel):
            pi = t.column("phase.index").to_pylist(); k = t.column("frame_index").to_pylist()
            lab += [int(np.ravel(pi[i])[0]) for i in sel]; fi += [k[i] for i in sel]
    lab = np.asarray(lab)[np.argsort(fi)]
    hdr = next(d for d in map(json.loads, open(meta["recording"])) if d.get("k") == "hdr")
    fe = hdr["fe"]; flags = [f"{k}={fe[k]}" for k in ("bite", "meltk", "meltv", "touchoff", "meltmax", "scopecal") if fe.get(k) is not None]
    ws, r = await R.session("ws://127.0.0.1:18780/v1/step", hdr, S[0], 256, flags)
    tr = R.PhaseTracker(R.demo_portal(ds), 1 / 15); got = []
    for k in range(len(S)):
        img, m = await R.step(ws, S[k], 1000 / 15)
        got.append(tr.update(m, S[k]))
    await ws.close()
    got = np.asarray(got); n = min(len(got), len(lab))
    print(json.dumps({"episode": EP, "agree": round(float((got[:n] == lab[:n]).mean()), 3), "removed_mm3": tr.vol,
                      "starts_label": {p: round(float(np.argmax(lab == p)) / 15, 1) for p in range(1, 7) if (lab == p).any()},
                      "starts_tracker": {p: round(float(np.argmax(got == p)) / 15, 1) for p in range(1, 7) if (got == p).any()}}))
asyncio.run(main())
