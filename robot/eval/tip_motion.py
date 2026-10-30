"""Drill tip motion while the trigger is on: robot-origin session vs the hand recording window, in RECORDING time.
Path length (raw), path length after a 0.3 s (recording-time) moving average, and jitter RMS = raw − smoothed."""
import json, sys, numpy as np
def load(f, win=None, speed=1.0):
    rows = [json.loads(l) for l in open(f)]
    t0 = min(r["tExp"] for r in rows if r.get("k") == "f" and r.get("tExp"))
    fr = [r for r in rows if r.get("k") == "f" and r.get("tool") == "drill" and not r.get("lost") and r.get("scene")]
    if win: fr = [r for r in fr if win[0] * 1000 <= r["tExp"] - t0 <= win[1] * 1000]
    t = np.array([(r["tExp"] - t0) / 1000 * speed for r in fr]); p = np.array([r["scene"]["tip"] if isinstance(r["scene"], dict) else r["tip"] for r in fr], float)
    on = np.array([bool(r.get("btn")) for r in fr])
    return t, p, on
def stats(t, p, on, w=0.3):
    g = np.arange(t[0], t[-1], 0.01); P = np.stack([np.interp(g, t, p[:, i]) for i in range(3)], 1)
    O = np.interp(g, t, on.astype(float)) > 0.5
    k = int(w / 0.01); ker = np.ones(k) / k
    S = np.stack([np.convolve(np.pad(P[:, i], (k // 2, k - 1 - k // 2), mode="edge"), ker, "valid") for i in range(3)], 1)
    seg = O[1:] & O[:-1]
    raw = np.linalg.norm(np.diff(P, axis=0), axis=1)[seg].sum(); smp = np.linalg.norm(np.diff(S, axis=0), axis=1)[seg].sum()
    jit = np.sqrt((np.linalg.norm(P - S, axis=1)[O] ** 2).mean())
    return {"on_s": round(float(O.sum()) * 0.01, 1), "path_raw_mm": round(float(raw), 1), "path_smooth_mm": round(float(smp), 1),
            "jitter_rms_mm": round(float(jit), 3)}
if __name__ == "__main__":
  rob, speed, hand, a, b = sys.argv[1], float(sys.argv[2]), sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
  print("robot", stats(*load(rob, None, speed)))
  print("hand ", stats(*load(hand, (a, b))))
