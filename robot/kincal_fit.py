#!/usr/bin/env python3
"""Kinematic calibration v2: hand-eye X, Z + joint offsets + joint-origin (link) corrections from logged runs.

Samples: quasi-static server ticks (encoder moved < --still-deg per tick over ±3 ticks) paired with the arm marker's
raw pose captured within ±15 ms of the encoder read. Fit = Levenberg–Marquardt on the constellation's ball positions
with Tikhonov regularisation on the link corrections; runs are split into fit / hold-out.

  python robot/kincal_fit.py --runs ~/il-robot/runs/latency/2026092[34]T*_{steps,replay} --out ~/il-robot/calib/kincal_left.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot_robot_spinearm import handeye as he
from lerobot_robot_spinearm import latency as L
from lerobot_robot_spinearm.ee import URDF
from lerobot_robot_spinearm.kincal import Chain
from lerobot_robot_spinearm.protocol import JOINTS

TOOL = Path.home() / "il-robot/tracking/tool_calib.json"


def samples(run: Path, still_deg: float):
    cfg = json.loads((run / "config.json").read_text())
    ev = L.load(run / "events.jsonl")
    offs = [c["off"] for c in ev["clock"]] or [cfg["clock_wall_minus_mono_ms"]]
    off = float(np.median(offs))
    role = {f["fn"]: f["robot"] for f in ev["frame"]}
    rob = sorted((m for m in ev["tap"] if role.get(m.get("fn")) == m["tool"] and not m.get("lost")),
                 key=lambda m: m["ts"])
    if not rob:
        return []
    ts = np.array([m["ts"] for m in rob])
    st = sorted((s for s in ev["state"] if s.get("tr") and s.get("pos")), key=lambda s: s["tick"])
    out = []
    for k in range(3, len(st) - 3):
        if st[k + 3]["tick"] - st[k - 3]["tick"] != 6:
            continue
        P = np.array([[st[i]["pos"][j] for j in JOINTS] for i in range(k - 3, k + 4)])
        if np.abs(np.diff(P, axis=0)).max() > still_deg:
            continue
        t_read = ((st[k]["t"] + st[k]["tr"]) / 2) * 1000 + off
        i = int(np.searchsorted(ts, t_read))
        near = [m for m in rob[max(0, i - 3):i + 3] if abs(m["ts"] - t_read) <= 15]
        if not near:
            continue
        m = min(near, key=lambda m: abs(m["ts"] - t_read))
        out.append((P[3], he.compose(he.quat_to_R(m["raw_quat"]), np.asarray(m["raw_pos"]) / 1000.0), run.name))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--handeye", default=str(Path.home() / "il-robot/calib/handeye_left.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--still-deg", type=float, default=0.15)
    ap.add_argument("--max-per-run", type=int, default=150)
    ap.add_argument("--reg-mm", type=float, default=2.0, help="prior σ of each link correction (mm)")
    ap.add_argument("--links", action="store_true", help="also fit joint-origin corrections (else offsets + X, Z only)")
    ap.add_argument("--holdout", type=float, default=0.25)
    a = ap.parse_args()
    pts_m, tip_m = he.tool_calib(TOOL)
    X0, Z0, d = he.load(a.handeye)
    off0 = he.joint_offsets(d)
    rng = np.random.default_rng(0)
    runs = sorted({Path(r) for r in a.runs if (Path(r) / "events.jsonl").exists()})
    S = []
    for r in runs:
        s = samples(r, a.still_deg)
        if len(s) > a.max_per_run:
            s = [s[i] for i in sorted(rng.choice(len(s), a.max_per_run, replace=False))]
        S += s
    names = sorted({s[2] for s in S})
    hold = set(rng.choice(names, max(1, int(len(names) * a.holdout)), replace=False))
    fit = [s for s in S if s[2] not in hold]
    val = [s for s in S if s[2] in hold]
    print(f"[kincal] {len(S)} quasi-static samples from {len(names)} runs → fit {len(fit)}, hold-out {len(val)}")
    ch = Chain(URDF)
    nj = 6

    def unpack(p):
        dq = off0 + p[:nj]
        X = X0 @ he._exp_se3(p[nj:nj + 6])
        Z = Z0 @ he._exp_se3(p[nj + 6:nj + 12])
        dxyz = p[nj + 12:].reshape(6, 3) if a.links else None
        return dq, X, Z, dxyz

    def resid(p, data, with_prior=True):
        dq, X, Z, dxyz = unpack(p)
        Zi = he.inv(Z)
        r = []
        for q, C, _ in data:
            P = Zi @ ch.fk(q + dq, dxyz) @ X
            r.append(((pts_m @ P[:3, :3].T + P[:3, 3]) - (pts_m @ C[:3, :3].T + C[:3, 3])).ravel())
        r = np.concatenate(r)
        if with_prior:
            pri = [p[0] / np.deg2rad(1) * 1e-4, p[5] / np.deg2rad(1) * 1e-4]  # pan / joint-6 offsets: unobservable
            if a.links:
                pri += list(p[nj + 12:] / (a.reg_mm / 1000.0) * 1e-3)
            r = np.concatenate([r, pri])
        return r

    def stats(p, data):
        dq, X, Z, dxyz = unpack(p)
        Zi = he.inv(Z)
        tip, org = [], []
        for q, C, _ in data:
            P = Zi @ ch.fk(q + dq, dxyz) @ X
            org.append(np.linalg.norm(P[:3, 3] - C[:3, 3]) * 1000)
            tip.append(np.linalg.norm((P[:3, :3] @ tip_m + P[:3, 3]) - (C[:3, :3] @ tip_m + C[:3, 3])) * 1000)
        tip, org = np.array(tip), np.array(org)
        return f"tip RMS {np.sqrt((tip ** 2).mean()):.2f} (p95 {np.percentile(tip, 95):.2f}) · origin RMS {np.sqrt((org ** 2).mean()):.2f} mm"

    npar = nj + 12 + (18 if a.links else 0)
    p = np.zeros(npar)
    print("[kincal] current model : fit", stats(p, fit), "| hold-out", stats(p, val))
    lam = 1e-3
    r = resid(p, fit)
    cost = float(r @ r)
    for it in range(30):
        J = np.empty((r.size, npar))
        for k in range(npar):
            e = np.zeros(npar)
            e[k] = 1e-3 if k < nj else 1e-6
            J[:, k] = (resid(p + e, fit) - r) / e[k]
        H, g = J.T @ J, J.T @ r
        while True:
            step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
            rn = resid(p + step, fit)
            cn = float(rn @ rn)
            if cn < cost:
                p, r, cost, lam = p + step, rn, cn, max(lam / 3, 1e-9)
                break
            lam *= 5
            if lam > 1e8:
                break
        if lam > 1e8 or float(np.abs(step).max()) < 1e-8:
            break
    print("[kincal] calibrated    : fit", stats(p, fit), "| hold-out", stats(p, val))
    dq, X, Z, dxyz = unpack(p)
    print("[kincal] joint offsets", dict(zip(JOINTS, np.round(dq, 2).tolist())))
    if dxyz is not None:
        print("[kincal] link corrections mm", {j: np.round(dxyz[i] * 1000, 2).tolist() for i, j in enumerate(JOINTS)})
    Path(a.out).write_text(json.dumps({
        "kind": "spinearm-kincal", "version": 2, "T_base_cam": Z.tolist(), "T_link_const": X.tolist(),
        "joint_offset_deg": dict(zip(JOINTS, dq.tolist())),
        "joint_origin_delta_m": ({j: dxyz[i].tolist() for i, j in enumerate(JOINTS)} if dxyz is not None else None),
        "n_fit": len(fit), "n_holdout": len(val), "holdout_runs": sorted(hold)}, indent=1))
    print(f"[kincal] saved {a.out}")


if __name__ == "__main__":
    main()
