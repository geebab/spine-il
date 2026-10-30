import asyncio, json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import rollout as R
async def main():
    ds = Path.home() / "il-robot/datasets/spine_rev4_step"; meta, S = R.episode(ds, 6)
    hdr = next(d for d in map(json.loads, open(meta["recording"])) if d.get("k") == "hdr")
    fe = hdr["fe"]; flags = [f"{k}={fe[k]}" for k in ("bite", "meltk", "meltv", "touchoff", "meltmax", "scopecal") if fe.get(k) is not None]
    ws, r = await R.session("ws://127.0.0.1:18780/v1/step", hdr, S[0], 256, flags, allow_outside=True)
    print("reset ok, setupS", r.get("setupS"))
    armed_at = None; vis4 = []; ph4 = (meta["phases"]["4"][0] * 15, meta["phases"]["4"][1] * 15)
    for k in range(len(S)):
        img, m = await R.step(ws, S[k], 1000 / 15)
        mm = m.get("meta") or {}
        if armed_at is None and mm.get("drillArmed"): armed_at = k
        if ph4[0] <= k <= ph4[1]:
            iv = ((mm.get("bridge") or {}).get("drill") or {}).get("inView") or {}; vis4.append(bool(iv.get("inFov") and iv.get("visible")))
    b = (m.get("meta") or {}).get("bridge") or {}
    print(json.dumps({"drill_armed_frame": armed_at, "armed_s": None if armed_at is None else round(armed_at / 15, 1),
                      "phase3_start_s": meta["phases"]["3"][0], "removed_mm3": b.get("removedMm3"), "drill_visible_phase4": round(float(np.mean(vis4)), 3)}))
    await ws.close()
asyncio.run(main())
