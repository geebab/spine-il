"""Robot safety gate for inference: every proposed tool-space step must be one the two arms can play back.

The checks are the start-pose search's (robot/eval/pairstart/search_clear.py `replay`), run per step against a FIXED pair of
robot start poses: the tool's displacement from the start pose (camera frame, through the recording's T0) moves the arm's
tip from its start pose — endo_follow's relative replay — then tip IK (as endo_follow), and the step passes only if
  · the safe zone holds (zone.check, relaxed 1.5 mm self/floor margins, both robot markers seen by the camera)
  · gripper Clearance ≥ 90 mm (endo_follow skips goals below 85)
  · the two arms stay ≥ 10 mm apart
  · IK reaches the target (≤ 5 mm, ≤ 10°) inside the joint limits
  · the displacement stays inside the replay sphere (RADIUS_M; 80 mm until 10-15, then 130 mm for both tools — user: the
    endoscope and drill radius to the maximum, guard_pairs.py MAX_RUN_LIMIT_MM; endo_follow runs with --radius-mm 130)
A failing step is shrunk toward the last safe frame (halving, up to 4 times), else the tools hold (trigger off).
turn_wait (10-07, off by default): when a step fails ONLY because the arms' IK does not reach its orientation, the tip
goes on and the turn waits — the proposed position with as much of the turn as passes (half, a quarter, none). The arms
make about a tenth of a turn they are asked for, so a turn they will not make must not stop a move they can make: the
24-demo policy was held for 30 min by an endoscope turn of 12° (the aim rule asking again every chunk). Every frame that
is sent still passes every check above.

    gate = RobotGate(start_state_scene15, T0, q_left, q_right)
    v_safe, info = gate.filter(prev_safe, proposed)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PS = Path(__file__).resolve().parents[1] / "eval" / "pairstart"
if str(PS) not in sys.path:
    sys.path.insert(0, str(PS))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ARMS = {"left": ("endo", 0), "right": ("drill", 7)}
CR_MIN, ARMS_MIN_MM, PERR_MM, OERR_DEG, RADIUS_M = 90.0, 10.0, 5.0, 10.0, 0.130


def _q2R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _R2q(R: np.ndarray, ref=None) -> np.ndarray:
    """Rotation matrix → wxyz, on the side of `ref` when given."""
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z]); q /= np.linalg.norm(q)
    return -q if (ref is not None and float(np.dot(q, ref)) < 0) else q


def only_turn(why: str) -> bool:
    """True when every reason of a failed check is an IK orientation miss with the tip in place ("left IK 0.0 mm / 11.5°")."""
    parts = [w.strip() for w in why.split(";") if w.strip()]
    if not parts:
        return False
    for w in parts:
        t = w.split()
        if len(t) != 6 or t[1] != "IK" or t[3] != "mm":
            return False
        try:
            if float(t[2]) > PERR_MM:
                return False
        except ValueError:
            return False
    return True


class RobotGate:
    def __init__(self, start: np.ndarray, T0: dict, q_left, q_right):
        import common as C                                   # pairstart: Kin, NEUTRAL, TOOL
        import verify_two_arm as V
        from lerobot_robot_spinearm import handeye as he, safezone as sz
        from lerobot_robot_spinearm.ee import Clearance
        self.C, self.he, self.sz = C, he, sz
        self.CL = Clearance()
        self.arms, cam, _ = V.load_models()
        self.zone = sz.Zone(self.arms, cam); self.zone.set_start(C.NEUTRAL)
        for a in self.zone.self_floor:                       # relaxed zone, as the replay check
            for k in list(self.zone.self_floor[a]):
                self.zone.self_floor[a][k] -= 1.5
        self.K = {a: C.Kin(a, margin=3.0, w_ori=0.1, w_post=0.1) for a in ARMS}     # replay IK (endo_follow)
        self.R0s, self.t0s = np.asarray(T0["R"], float).reshape(3, 3), np.asarray(T0["t"], float)
        self.H0 = {a: self._cam(start, off) for a, (_, off) in ARMS.items()}       # hand start pose, camera frame
        q0 = {"left": np.asarray(q_left, float), "right": np.asarray(q_right, float)}
        self.FK0, self.q = {}, {}
        for a in ARMS:
            k = self.K[a]; qi = q0[a] + k.off
            k.ik.set_posture(qi); self.q[a] = qi; self.FK0[a] = k.ik.fk(qi)
        self.n_checked = self.n_shrunk = self.n_held = self.n_turn_cut = 0
        self.turn_wait = False
        self.arms_cutoff_m = ARMS_MIN_MM / 1000.0

    def _cam(self, v: np.ndarray, off: int) -> tuple[np.ndarray, np.ndarray]:
        R = self.R0s.T @ _q2R(v[off + 3:off + 7])
        p = self.R0s.T @ (np.asarray(v[off:off + 3], float) - self.t0s) / 1000.0
        return R, p

    def check(self, v: np.ndarray) -> tuple[bool, str, dict]:
        """(passes, why, joint solution) for the frame v (dataset 15-vector, scene mm) — the robot is not touched."""
        he, q, why = self.he, {}, []
        for a, (tool, off) in ARMS.items():
            k = self.K[a]; Rbc = k.Z[:3, :3]
            Ri, pi = self._cam(v, off); R0, P0 = self.H0[a]
            dp = pi - P0
            if np.linalg.norm(dp) > RADIUS_M:
                return False, f"{tool} {np.linalg.norm(dp) * 1000:.0f} mm from its start (> {RADIUS_M * 1000:.0f} mm replay sphere)", {}
            dR = Ri @ R0.T
            T = he.compose(Rbc @ dR @ Rbc.T @ self.FK0[a][:3, :3], self.FK0[a][:3, 3] + Rbc @ dp)
            qa = k.ik.ik(self.q[a], T, iters=10)
            Ta = k.ik.fk(qa)
            perr = 1000 * np.linalg.norm(Ta[:3, 3] - T[:3, 3])
            c = np.clip((np.trace(Ta[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)
            oerr = float(np.degrees(np.arccos(c)))
            if perr > PERR_MM or oerr > OERR_DEG:
                why.append(f"{a} IK {perr:.1f} mm / {oerr:.1f}°")
            if self.CL.mm(qa) < CR_MIN:
                why.append(f"{a} clearance {self.CL.mm(qa):.0f} mm")
            if (k.margin(qa - k.off) < 0).any():
                why.append(f"{a} joint limit")
            q[a] = qa
        qz = {a: q[a] - self.K[a].off for a in ARMS}
        ok, zwhy, _ = self.zone.check(qz, strict=False, see=("left", "right"))
        if not ok:
            why += list(zwhy[:2])
        po = {a: self.arms[a].pose(qz[a])["world"] for a in ARMS}
        # only "closer than ARMS_MIN_MM" is asked: part pairs whose boxes are farther apart than that cannot decide it, and
        # parts_dist leaves them out. With the 70 mm used until 10-12 every pair within 70 mm was measured point by point —
        # 146 of a check's 217 ms — for a number that was then only compared with 10 mm (same verdicts: test_gate_cutoff).
        d = 1000 * self.sz.parts_dist(po["left"], po["right"], self.arms_cutoff_m)
        if d < ARMS_MIN_MM:
            why.append(f"arms {d:.0f} mm apart")
        return (not why), "; ".join(map(str, why)), q

    def achieved(self, v: np.ndarray, physical: bool = False) -> np.ndarray:
        """The frame the arms take for the last frame that passed (their joint solution self.q), in the dataset's 15-vector.

        The arms' IK puts the tip where it was asked (a hard task) and turns the tool only softly (w_ori 0.1 against a
        posture pull of 0.1): of a 10° turn asked at the anchor about 1° is made (10-07). A robot run measures this pose,
        not the commanded one — rollout.py --arm-model feeds it back to the policy as live_loop.py does.
        physical=True: the tools as the camera sees them on the arms, i.e. what the robot session's page draws — the arm's
        own tool rotation in place of "the hand's rotation at the anchor, turned as the arm turned": the two differ by the
        pair's orientation error (3868 pair: endoscope 10°, drill 1°)."""
        out = np.array(v, float)
        for a, (_, off) in ARMS.items():
            k = self.K[a]; Rbc = k.Z[:3, :3]; Ta = k.ik.fk(self.q[a])
            R0, P0 = self.H0[a]
            dp = Rbc.T @ (Ta[:3, 3] - self.FK0[a][:3, 3])
            Rc = Rbc.T @ Ta[:3, :3] if physical else (Rbc.T @ Ta[:3, :3] @ self.FK0[a][:3, :3].T @ Rbc) @ R0
            out[off:off + 3] = self.R0s @ ((P0 + dp) * 1000.0) + self.t0s
            out[off + 3:off + 7] = _R2q(self.R0s @ Rc, np.asarray(v[off + 3:off + 7], float))
        return out

    def filter(self, prev: np.ndarray, proposed: np.ndarray) -> tuple[np.ndarray, dict]:
        """The proposed frame if safe, else the largest safe step toward it (halving), else hold `prev` (trigger off)."""
        self.n_checked += 1
        cand, f = np.array(proposed, float), 1.0
        for i in range(5):
            ok, why, q = self.check(cand)
            if ok:
                self.q.update(q)
                if i:
                    self.n_shrunk += 1
                return cand, {"ok": True, "scale": f, "why": None if i == 0 else first_why}
            if i == 0:
                first_why = why
                if self.turn_wait and only_turn(why):   # the tip goes on, the turn waits — of the tool(s) whose arm missed it
                    from safety import _slerp
                    offs = [off + 3 for a, (_, off) in ARMS.items() if f"{a} IK" in why]
                    for g in (0.5, 0.25, 0.0):
                        c2 = np.array(proposed, float)
                        for off in offs:
                            c2[off:off + 4] = _slerp(np.asarray(prev[off:off + 4], float), np.asarray(proposed[off:off + 4], float), g)
                        ok2, _, q2 = self.check(c2)
                        if ok2:
                            self.q.update(q2); self.n_turn_cut += 1
                            return c2, {"ok": True, "scale": 1.0, "turn": g, "turn_off": offs, "why": first_why}
            f /= 2
            cand = np.array(prev, float) + f * (np.array(proposed, float) - np.array(prev, float))
            for off in (3, 10):
                cand[off:off + 4] /= np.linalg.norm(cand[off:off + 4])
            cand[14] = proposed[14]
        self.n_held += 1
        held = np.array(prev, float); held[14] = 0.0
        return held, {"ok": False, "scale": 0.0, "why": first_why}
