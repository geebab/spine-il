#!/usr/bin/env python3
"""Does feeding the viewer every n-th measured frame cut the same bone as feeding every frame?

    python robot/inference/live/feed_test.py LIVE.json --stride 5 [--api ws://127.0.0.1:18780/v1/step]

Replays the measured trajectory of a closed-loop run (live_loop.py output: frames at 15 fps with the trigger) through a
step session, stepping only every n-th frame with the session clock advanced by n frames, and prints the removed volume
over time. The run itself fed every frame (stride 1), so its meta.removed_mm3 is the reference.
"""
import argparse, asyncio, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import rollout as RO   # noqa: E402


async def main(a):
    d = json.loads(Path(a.live).read_text()); F = np.asarray(d["frames"], float); m = d["meta"]
    with open(d["ref_recording"]) as fh:
        hdr = next(x for x in map(json.loads, fh) if x.get("k") == "hdr")
    fe = hdr.get("fe") or {}
    flags = [f"{k}={fe[k]}" for k in ("bite", "meltk", "meltv", "touchoff", "meltmax", "scopecal") if fe.get(k) is not None]
    dt = 1000.0 / d["fps"]
    ws, r = await RO.session(a.api, hdr, F[0], 256, flags, allow_outside=True)
    vols, t0, last = [], time.time(), -1
    try:
        img, mm = await RO.step(ws, F[0], dt)
        for j in range(len(F)):
            if (j + 1) % a.stride == 0 or j == len(F) - 1:
                img, mm = await RO.step(ws, F[j], dt * (j - last), images=not a.no_images); last = j
                vols.append([j, float(((mm.get("meta") or {}).get("bridge") or {}).get("removedMm3") or 0.0)])
    finally:
        try:
            await ws.send(json.dumps({"op": "close"}))
        except Exception:
            pass
        await ws.close()
    ref = m.get("removed_mm3") or []
    out = {"stride": a.stride, "steps": len(vols), "seconds": round(time.time() - t0, 1), "removed_mm3": round(vols[-1][1], 1),
           "reference_mm3": round(ref[-1], 1) if ref else None,
           "curve": [[j, round(v, 1), round(ref[j], 1) if j < len(ref) else None] for j, v in vols[::max(1, len(vols) // 12)]]}
    print(json.dumps(out))


ap = argparse.ArgumentParser(); ap.add_argument("live"); ap.add_argument("--stride", type=int, default=5)
ap.add_argument("--api", default="ws://127.0.0.1:18780/v1/step")
ap.add_argument("--no-images", action="store_true", help="image-free steps (step op images: false)")
asyncio.run(main(ap.parse_args()))
