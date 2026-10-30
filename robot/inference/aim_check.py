import asyncio, json, sys, numpy as np
from pathlib import Path
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import rollout as R
def angle(m):
    b = (m.get("meta") or {}).get("bridge") or {}; c = b["endoCamera"]; B = np.asarray(b["drill"]["B"], float)
    p = np.asarray(c["position"], float); d = np.asarray(c["focalPoint"], float) - p; t = B - p
    iv = b["drill"].get("inView") or {}
    return round(float(np.degrees(np.arccos(np.dot(d, t) / np.linalg.norm(d) / np.linalg.norm(t)))), 2), bool(iv.get("inFov") and iv.get("visible"))
async def main():
    ds = Path.home() / "il-robot/datasets/spine_rev4_step"; ep = 6
    meta, S = R.episode(ds, ep)
    hdr = next(d for d in map(json.loads, open(meta["recording"])) if d.get("k") == "hdr")
    fe = hdr["fe"]; flags = [f"{k}={fe[k]}" for k in ("bite", "meltk", "meltv", "touchoff", "meltmax", "scopecal") if fe.get(k) is not None]
    k = int(round((meta["phases"]["4"][0] + 5.0) * 15))       # 5 s into drilling
    v = S[k].copy(); v[14] = 0.0
    ws, _ = await R.session("ws://127.0.0.1:18780/v1/step", hdr, v, 256, flags)
    for _ in range(5): img, m = await R.step(ws, v, 66.7)
    print("demo pose: angle to drill", angle(m))
    # push the endoscope 12° away from the drill, pivoting about the skin entry
    b = m["meta"]["bridge"]; P = np.asarray(b["endoPose"]["entry"], float)
    w = np.cross(np.asarray(b["endoCamera"]["focalPoint"]) - np.asarray(b["endoCamera"]["position"]), [0, 0, 1.0]); w /= np.linalg.norm(w)
    Rw = R._rotvec_R(w * np.radians(12)); v2 = v.copy(); v2[0:3] = P + Rw @ (v[0:3] - P)
    q = v[3:7]; import numpy.linalg as L
    # rotate the quaternion by Rw (via aim_endo helper on a fake meta would be circular; use scipy-free conversion)
    from safety import _slerp  # noqa
    def R_to_q(Rm):
        import math
        t = np.trace(Rm); S_ = math.sqrt(t + 1) * 2; return np.array([0.25 * S_, (Rm[2, 1] - Rm[1, 2]) / S_, (Rm[0, 2] - Rm[2, 0]) / S_, (Rm[1, 0] - Rm[0, 1]) / S_])
    w_, x_, y_, z_ = q / L.norm(q)
    Rt = np.array([[1-2*(y_*y_+z_*z_), 2*(x_*y_-w_*z_), 2*(x_*z_+w_*y_)], [2*(x_*y_+w_*z_), 1-2*(x_*x_+z_*z_), 2*(y_*z_-w_*x_)], [2*(x_*z_-w_*y_), 2*(y_*z_+w_*x_), 1-2*(x_*x_+y_*y_)]])
    v2[3:7] = R_to_q(Rw @ Rt)
    for _ in range(5): img, m = await R.step(ws, v2, 66.7)
    print("pushed 12°: angle", angle(m))
    for i in range(40):
        v2, deg = R.aim_endo(v2, m, 0.3, 1.5, 6.0)
        img, m = await R.step(ws, v2, 66.7)
        if i % 5 == 4: print(f"aim step {i+1}: corrected {deg:.2f}°, angle to drill {angle(m)}")
    await ws.close()
asyncio.run(main())
