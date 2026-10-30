"""Eye-to-hand calibration: a marker constellation carried by the arm, seen by the fixed tracker camera.

Unknowns (both rigid, metres):
  Z = T_base←cam    — the tracker camera in the robot base frame
  X = T_link←const  — the constellation on the IK link (`marker_frame_link`); replaces the URDF's unmeasured
                      54 mm placeholder
Per pose i:  FK_i · X = Z · C_i   with FK_i = T_base←link(q_i) (encoder angles) and C_i = T_cam←const (tracker).

`solve()` = OpenCV's closed form (robot-world/hand-eye, Shah) + a Levenberg–Marquardt refinement on the
constellation's ball positions (what the camera actually measures), optionally with the joint zero offsets
(FK(q + dq)), and one outlier-rejection pass. Model angle = arm angle + dq everywhere FK/IK is used.
`urdf_with_tip()` appends a fixed `endo_tip_link` (constellation orientation, at the tool tip) so the IK can
put its position priority on the tip instead of the link origin.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

TIP_FRAME = "endo_tip_link"


# ---------------------------------------------------------------- SE(3) helpers
def inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, float).reshape(3)
    return T


def rotvec_to_R(v) -> np.ndarray:
    v = np.asarray(v, float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def R_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))))


def quat_to_R(q) -> np.ndarray:
    """[w, x, y, z] → 3×3 (tracker convention: tool → cam)."""
    w, x, y, z = (float(v) for v in q)
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _exp_se3(xi) -> np.ndarray:
    return compose(rotvec_to_R(xi[:3]), xi[3:])


# ---------------------------------------------------------------- solve
def closed_form(fk: list[np.ndarray], obs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """(X, Z) from OpenCV's robot-world/hand-eye solver.

    OpenCV's model is a camera on the gripper looking at a fixed world target: cTw = cTg · gTb · bTw. Here the
    constellation plays OpenCV's "camera" (it rides the gripper) and our tracker camera plays OpenCV's "world"
    (it is fixed): cTw := T_const←cam = C⁻¹, gTb := T_link←base = FK⁻¹ → bTw⁻¹ = Z, cTg⁻¹ = X.
    """
    import cv2

    A = [inv(C) for C in obs]
    B = [inv(F) for F in fk]
    R_b2w, t_b2w, R_g2c, t_g2c = cv2.calibrateRobotWorldHandEye(
        [a[:3, :3] for a in A], [a[:3, 3].reshape(3, 1) for a in A],
        [b[:3, :3] for b in B], [b[:3, 3].reshape(3, 1) for b in B],
        method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH)
    wTb = compose(R_b2w, t_b2w)  # T_cam←base
    cTg = compose(R_g2c, t_g2c)  # T_const←link
    return inv(cTg), inv(wTb)


def _ball_residual(X, Z, fk, obs, pts) -> np.ndarray:
    Zi = inv(Z)
    out = []
    for F, C in zip(fk, obs):
        P = Zi @ F @ X
        out.append((pts @ P[:3, :3].T + P[:3, 3]) - (pts @ C[:3, :3].T + C[:3, 3]))
    return np.concatenate(out).ravel()


def refine(X, Z, fk, obs, pts_m, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """Levenberg–Marquardt on the balls' 3-D positions; X, Z updated by right-multiplied exp(ξ)."""
    X, Z = X.copy(), Z.copy()
    lam = 1e-3
    r = _ball_residual(X, Z, fk, obs, pts_m)
    cost = float(r @ r)
    for _ in range(iters):
        J = np.zeros((r.size, 12))
        eps = 1e-6
        for k in range(12):
            d = np.zeros(12)
            d[k] = eps
            Xp, Zp = X @ _exp_se3(d[:6]), Z @ _exp_se3(d[6:])
            J[:, k] = (_ball_residual(Xp, Zp, fk, obs, pts_m) - r) / eps
        H = J.T @ J
        g = J.T @ r
        while True:
            step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
            Xn, Zn = X @ _exp_se3(step[:6]), Z @ _exp_se3(step[6:])
            rn = _ball_residual(Xn, Zn, fk, obs, pts_m)
            cn = float(rn @ rn)
            if cn < cost:
                X, Z, r, cost, lam = Xn, Zn, rn, cn, max(lam / 3, 1e-9)
                break
            lam *= 5
            if lam > 1e6:
                return X, Z
        if float(np.linalg.norm(step)) < 1e-10:
            break
    return X, Z


OFFSET_JOINTS = (1, 2, 3, 4)  # lift, elbow, wrist_flex, wrist_roll — a pan offset is a turn of Z about base z and a
# joint-6 offset a turn of X about the joint-6 axis: both unobservable, so they stay 0.


def refine_offsets(X, Z, q_deg, obs, pts_m, fk_fn, dq0=None, iters: int = 25):
    """Joint zero offsets (deg) together with X, Z: FK(q + dq) · X = Z · C. The arm's calibration puts joint zeros
    at mid-range, not at the URDF's zeros (elbow −7.9° measured 2026-09-23), which FK alone cannot absorb."""
    q_deg = np.asarray(q_deg, float)
    nj = len(OFFSET_JOINTS)
    dq = np.zeros(6) if dq0 is None else np.asarray(dq0, float).copy()

    def unpack(p):
        d = dq.copy()
        d[list(OFFSET_JOINTS)] += p[:nj]
        return d, X @ _exp_se3(p[nj:nj + 6]), Z @ _exp_se3(p[nj + 6:])

    def resid(p):
        d, Xp, Zp = unpack(p)
        return _ball_residual(Xp, Zp, [fk_fn(q + d) for q in q_deg], obs, pts_m)

    p = np.zeros(nj + 12)
    r = resid(p)
    cost, lam = float(r @ r), 1e-3
    eps = np.r_[np.full(nj, 1e-3), np.full(12, 1e-5)]
    for _ in range(iters):
        J = np.empty((r.size, p.size))
        for k in range(p.size):
            e = np.zeros(p.size)
            e[k] = eps[k]
            J[:, k] = (resid(p + e) - r) / eps[k]
        H, g = J.T @ J, J.T @ r
        while True:
            step = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-12), g)
            rn = resid(p + step)
            cn = float(rn @ rn)
            if cn < cost:
                p, r, cost, lam = p + step, rn, cn, max(lam / 3, 1e-9)
                break
            lam *= 5
            if lam > 1e8:
                return unpack(p)
        if float(np.abs(step).max()) < 1e-7:
            break
    return unpack(p)


def residuals(X, Z, fk, obs, pts_m, tip_m) -> list[dict]:
    """Per pose: constellation-origin / tip position error (mm), rotation error (deg), ball RMS (mm)."""
    Zi = inv(Z)
    out = []
    for F, C in zip(fk, obs):
        P = Zi @ F @ X
        d_org = (P[:3, 3] - C[:3, 3]) * 1000
        d_tip = ((P[:3, :3] @ tip_m + P[:3, 3]) - (C[:3, :3] @ tip_m + C[:3, 3])) * 1000
        balls = ((pts_m @ P[:3, :3].T + P[:3, 3]) - (pts_m @ C[:3, :3].T + C[:3, 3])) * 1000
        out.append({"org_mm": float(np.linalg.norm(d_org)), "tip_mm": float(np.linalg.norm(d_tip)),
                    "rot_deg": R_angle_deg(P[:3, :3].T @ C[:3, :3]),
                    "ball_rms_mm": float(np.sqrt((balls ** 2).sum(1).mean()))})
    return out


def summarize(res: list[dict]) -> dict:
    if not res:
        return {"n": 0}
    out = {"n": len(res)}
    for k in ("org_mm", "tip_mm", "rot_deg", "ball_rms_mm"):
        v = np.array([r[k] for r in res])
        out[k] = {"rms": round(float(np.sqrt((v ** 2).mean())), 3), "median": round(float(np.median(v)), 3),
                  "max": round(float(v.max()), 3)}
    return out


def solve(fk, obs, pts_m, tip_m, reject: bool = True, q_deg=None, fk_fn=None) -> dict:
    """Closed form → refine → (with q_deg + fk_fn: joint offsets too) → drop poses whose ball RMS >
    max(2.5·median, 2 mm) and solve again. Returns X, Z, dq (deg, 6; zeros without offsets), keep."""
    def once(idx):
        fk_k, obs_k = [fk[i] for i in idx], [obs[i] for i in idx]
        X, Z = refine(*closed_form(fk_k, obs_k), fk_k, obs_k, pts_m)
        dq = np.zeros(6)
        if q_deg is not None and fk_fn is not None:
            dq, X, Z = refine_offsets(X, Z, [q_deg[i] for i in idx], obs_k, pts_m, fk_fn)
        return X, Z, dq

    keep = list(range(len(fk)))
    X, Z, dq = once(keep)
    if reject and len(fk) >= 8:
        fk_now = fk if not dq.any() else [fk_fn(np.asarray(q, float) + dq) for q in q_deg]
        rms = np.array([r["ball_rms_mm"] for r in residuals(X, Z, fk_now, obs, pts_m, tip_m)])
        lim = max(2.5 * float(np.median(rms)), 2.0)
        keep = [i for i, v in enumerate(rms) if v <= lim]
        if len(fk) > len(keep) >= 6:
            X, Z, dq = once(keep)
    return {"X": X, "Z": Z, "dq": dq, "keep": keep}


# ---------------------------------------------------------------- IK frame at the tip
def _rpy(R: np.ndarray) -> tuple[float, float, float]:
    """URDF rpy (fixed-axis X-Y-Z, R = Rz(y) Ry(p) Rx(r))."""
    return (float(np.arctan2(R[2, 1], R[2, 2])), float(np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))),
            float(np.arctan2(R[1, 0], R[0, 0])))


def urdf_with_tip(urdf_in: Path | str, X: np.ndarray, tip_m, out: Path | str, parent: str = "marker_frame_link",
                  name: str = TIP_FRAME) -> Path:
    """Copy the URDF with a fixed `name` frame = X · (tip, constellation orientation) under `parent`."""
    T = X @ compose(np.eye(3), tip_m)
    xyz = " ".join(f"{v:.6f}" for v in T[:3, 3])
    rpy = " ".join(f"{v:.6f}" for v in _rpy(T[:3, :3]))
    add = (f'  <link name="{name}"/>\n  <joint name="{name}_joint" type="fixed">\n'
           f'    <origin xyz="{xyz}" rpy="{rpy}"/>\n    <parent link="{parent}"/>\n    <child link="{name}"/>\n'
           f'  </joint>\n</robot>')
    text = Path(urdf_in).read_text()
    i = text.rstrip().rfind("</robot>")
    out = Path(out)
    out.write_text(text[:i] + add + "\n")
    return out


# ---------------------------------------------------------------- io
def save(path: Path | str, X, Z, meta: dict) -> None:
    d = {"kind": "spinearm-handeye", "version": 1, "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
         "T_base_cam": np.round(Z, 8).tolist(), "T_link_const": np.round(X, 8).tolist(), "units": "m", **meta}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d, indent=1, ensure_ascii=False))


def load(path: Path | str) -> tuple[np.ndarray, np.ndarray, dict]:
    """→ (X = T_link←const, Z = T_base←cam, full dict). `joint_offsets(d)` gives the joint zero offsets."""
    d = json.loads(Path(path).read_text())
    return np.asarray(d["T_link_const"], float), np.asarray(d["T_base_cam"], float), d


def joint_offsets(d: dict) -> np.ndarray:
    """Model angle = arm angle + offset (deg, JOINTS order); zeros for files without offsets."""
    from .protocol import JOINTS

    off = d.get("joint_offset_deg") or {}
    return np.array([float(off.get(j, 0.0)) for j in JOINTS])


def tool_calib(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Tracker tool model (tool_calib.json) → (ball positions, tip) in metres, constellation frame."""
    d = json.loads(Path(path).read_text())
    pts = np.asarray(d["points_local"], float) / 1000.0
    # A robot-mounted marker block has no tool tip (a fresh `calibrate` writes none): use the ball centroid, so the
    # "tip" residual reports the block centre.
    tip = d.get("tip_local")
    return pts, (np.asarray(tip, float) / 1000.0 if tip is not None else pts.mean(0))
