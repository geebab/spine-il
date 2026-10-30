#!/usr/bin/env python3
"""Learn a general feed-forward from ILC corrections (the idea of training a model on ILC results so it applies to
new trajectories, e.g. arXiv:1903.00082): correction (camera mm) = W · [v, a, x_abs, 1], features from the same
TargetShaper state the follower uses (x relative + engage tip, v, causal a). Linear least squares with ridge,
leave-one-trajectory-out validation.

  python robot/ff_fit.py --pair ILC.npz RUN_WITH_THAT_ILC … --out ~/il-robot/calib/ff_left.json
(RUN = one replay run made with that correction file: gives the recording, shaper settings and engage tip.)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("endo_follow", HERE / "endo_follow.py")
ef = importlib.util.module_from_spec(spec)
_argv, sys.argv = sys.argv, ["x"]
spec.loader.exec_module(ef)
sys.argv = _argv


def features(run: Path, t_grid: np.ndarray):
    cfg = json.loads((run / "config.json").read_text())
    a = cfg["args"]
    ev = [json.loads(line) for line in (run / "events.jsonl").read_text().splitlines()]
    eng = next(e for e in ev if e["k"] == "engage")
    R0 = np.asarray(eng["R0_tip_mm"]) / 1000.0
    tip = np.asarray(cfg["tip_local_m"])
    rp = ef.load_replay(a["replay_run"])
    sh = ef.TargetShaper(a["filter"], a["shape_hz"], a["shape_vmax"], a["shape_amax"], a["frame_jump_mm"],
                         a["deadband_mm"], a["deadband_deg"], np.asarray(cfg["T_base_cam"])[:3, :3], a.get("z_hz"),
                         a.get("z_deadband_mm"), a["lead_ms"] / 1000.0)
    F = []
    for k in range(len(rp["t"])):
        dp = rp["R"][k] @ tip + rp["pos"][k] - rp["H0_tip"]
        sh(dp, np.eye(3), int(rp["vis"][k]), rp["t"][k])
        F.append(np.concatenate([sh.v * 1000, sh.acc * 1000, (R0 + sh.x) * 1000, [1.0]]))
    F = np.array(F)
    return np.array([np.interp(t_grid, rp["t"], F[:, i]) for i in range(F.shape[1])]).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, action="append", metavar=("ILC", "RUN"), required=True)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--use", default="v,a,x", help="feature groups to fit (others get zero weight): v, a, x")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = []
    for ilc_f, run in a.pair:
        il = np.load(ilc_f)
        t, corr = il["t"], il["corr"] * 1000
        m = t > 1.0
        F = features(Path(run), t[m])
        keep = np.zeros(10, bool)
        keep[-1] = True
        for g, sl in (("v", slice(0, 3)), ("a", slice(3, 6)), ("x", slice(6, 9))):
            if g in a.use.split(","):
                keep[sl] = True
        data.append((F * keep, corr[m], Path(ilc_f).name))

    def fit(ds):
        A = np.vstack([d[0] for d in ds])
        Y = np.vstack([d[1] for d in ds])
        s = A.std(0) + 1e-9
        s[-1] = 1.0
        An = A / s
        W = np.linalg.solve(An.T @ An + a.ridge * len(A) * np.eye(An.shape[1]), An.T @ Y).T / s
        return W

    def score(W, d):
        r = d[1] - d[0] @ W.T
        return np.sqrt((r ** 2).sum(1).mean()), np.sqrt((d[1] ** 2).sum(1).mean())

    for i, d in enumerate(data):
        W = fit(data[:i] + data[i + 1:]) if len(data) > 1 else fit(data)
        res, tot = score(W, d)
        print(f"[ff] held-out {d[2]}: correction RMS {tot:.2f} → residual {res:.2f} mm "
              f"({100 * (1 - res ** 2 / tot ** 2):.0f}% of its energy explained)")
    W = fit(data)
    Path(a.out).write_text(json.dumps({"kind": "spinearm-ff", "features": "v(3) mm/s, a(3) mm/s², x_abs(3) mm, 1",
                                       "frame": "camera", "W": W.tolist(), "sources": [p[0] for p in a.pair]}, indent=1))
    print(f"[ff] saved {a.out}")


if __name__ == "__main__":
    main()
