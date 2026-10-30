"""Shared setup: per-arm tip kinematics (as endo_follow), hand start, safe zone."""
import json, sys, numpy as np
from pathlib import Path
H = Path.home()
sys.path.insert(0, str(Path(__import__("os").environ.get("SPINE_IL_REPO", str(H / "spine-il"))) / "robot"))
from lerobot_robot_spinearm import handeye as he, recording as rec
from lerobot_robot_spinearm.ee import URDF, TipIK
from lerobot_robot_spinearm.protocol import JOINTS
P = Path(__file__).parent
TIPF = {"left": H / "il-robot/calib/tracker_models/tool_calib_endo_robotL_tip.json",
        "right": H / "il-robot/calib/tracker_models/tool_calib_drill_robotR_tip.json"}
BOUNDS = {"left": {'shoulder_pan': 81.978, 'shoulder_lift': 102.198, 'elbow_flex': 96.088, 'wrist_flex': 100.308, 'wrist_roll': 150.0, 'gripper': 64.879},
          "right": {'shoulder_pan': 77.626, 'shoulder_lift': 102.462, 'elbow_flex': 96.440, 'wrist_flex': 103.121, 'wrist_roll': 150.0, 'gripper': 65.275}}
NEUTRAL = {"left": np.array([3.56, -101.85, 59.43, 99.52, 1.89, -60.66]),
           "right": np.array([1.41, -101.41, 59.69, 102.24, -1.54, -64.31])}
CUR = {"left": np.array([0, -60, 50, 70, 0, -40.]), "right": np.array([3.5, -49, 23.8, 83.4, 8.4, -45.2])}
TOOL = {"left": "endo", "right": "drill"}

class Kin:
    def __init__(self, arm, margin=5.0, w_ori=1.0, w_post=1e-3):
        self.arm = arm
        self.X, self.Z, meta = he.load(H / f"il-robot/calib/handeye_{arm}.json")
        self.off = he.joint_offsets(meta)
        self.pts_m, self.tip_m = he.tool_calib(TIPF[arm])
        self.urdf = he.urdf_with_tip(URDF, self.X, self.tip_m, P / f"tip_{arm}.urdf")
        b = BOUNDS[arm]
        self.lo = np.array([-b[j] for j in JOINTS]); self.hi = np.array([b[j] for j in JOINTS])
        lim = {j: (-b[j] + self.off[i] + margin, b[j] + self.off[i] - margin) for i, j in enumerate(JOINTS)}
        self.ik = TipIK(self.urdf, he.TIP_FRAME, lim, w_ori=w_ori, w_post=w_post, vmax_dps=3000, rate_hz=90)
        self.Zi = he.inv(self.Z)
    def T_cam(self, q):            # tip frame in camera frame
        return self.Zi @ self.ik.fk(np.asarray(q, float) + self.off)
    def tip_cam(self, q):
        return self.T_cam(q)[:3, 3]
    def solve(self, q0, R_cam, p_cam, n=200, post=None):
        """IK: tip at p_cam (camera), orientation R_cam (camera, tip frame), from q0 (arm deg)."""
        T = self.Z @ he.compose(R_cam, p_cam)
        q = np.asarray(q0, float) + self.off
        self.ik.set_posture(q if post is None else np.asarray(post, float) + self.off)
        for _ in range(n):
            qn = self.ik.ik(q, T, iters=1)
            if np.abs(qn - q).max() < 1e-4: q = qn; break
            q = qn
        return q - self.off
    def margin(self, q):
        q = np.asarray(q); return np.minimum(q - self.lo, self.hi - q)

def hand_start():
    path = H / "il-robot/viewer-data/sessions/2026-09-28/s_20260928T165947_altician_lumbar1_trk_3868.jsonl"
    cal = H / "il-robot/calib/tracker_models/hand_20260928"
    out = {}
    for tool, f in (("endo", "tool_calib.json"), ("drill", "tool_calib_drill.json")):
        rp = rec.load_recording(str(path), tool, calib_path=str(cal / f), t_start_s=16)
        out[tool] = rp
    return out

def rot_angle(Ra, Rb):
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))

def _log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1); th = np.arccos(c)
    if th < 1e-9: return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * np.sin(th))
    return w * th

def _exp(w):
    th = np.linalg.norm(w)
    if th < 1e-12: return np.eye(3)
    k = w / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx

def slerp(Ra, Rb, f):
    return Ra @ _exp(f * _log(Ra.T @ Rb))
