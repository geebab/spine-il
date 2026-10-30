"""End-effector helpers for the marker-carrying SO-101 — client side (the arm server only speaks joints).

- `MarkerKinematics`: FK/IK on `assets/so101_marker.urdf`, frame `marker_frame_link` (54 mm off the joint-6
  axis). All six joints take part: joint 6 swings the marker on an arc, so it is an IK axis, not a gripper.
- `ee_box()`: workspace box from `assets/ee_bounds.json` (table-top p1–p99 envelope, shrunk by a margin,
  z floored at `Z_FLOOR_M`).
- `EETeleop`: accumulates a Cartesian target from per-tick deltas (stick/keys) and returns joint goals for
  `SpineArm.send_action`. The measured pose is never fed back into the target — that made the arm creep.

Needs the `ee` extra (placo via lerobot[kinematics]): `uv pip install -e "robot/lerobot_robot_spinearm[ee]"`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .protocol import JOINTS

ASSETS = Path(__file__).with_name("assets")
URDF = ASSETS / "so101_marker.urdf"
BOUNDS_JSON = ASSETS / "ee_bounds.json"
TARGET_FRAME = "marker_frame_link"
Z_FLOOR_M = 0.04  # marker never below 4 cm over the base plane (bracket/table clearance)
# IK joint limits when no server bounds are given: the union of both arms' calibrated ranges (deg).
# The packaged URDF carries the same numbers. Tighter (CAD) limits push poses like the folded neutral
# (lift -101.9, wrist_flex 99.5) and corrupt the solution — see MarkerKinematics.
KIN_BOUNDS = {"shoulder_pan": (-110.0, 110.0), "shoulder_lift": (-103.0, 103.0), "elbow_flex": (-97.0, 97.0),
              "wrist_flex": (-104.0, 104.0), "wrist_roll": (-150.0, 150.0), "gripper": (-66.0, 66.0)}


def ee_box(path: Path | str = BOUNDS_JSON, margin_mm: float = 30.0, z_floor_m: float = Z_FLOOR_M) -> tuple[np.ndarray, np.ndarray]:
    """(lo, hi) in metres, base_link frame: the p1–p99 table-top envelope shrunk by `margin_mm`, z floored."""
    b = json.loads(Path(path).read_text())["above_table_p1_p99_mm"]
    lo = np.array([b["x"][0] + margin_mm, b["y"][0] + margin_mm, b["z"][0]]) / 1000.0
    hi = np.array([b["x"][1] - margin_mm, b["y"][1] - margin_mm, b["z"][1] - margin_mm]) / 1000.0
    lo[2] = max(lo[2], z_floor_m)
    return lo, hi


class MarkerKinematics:
    """FK/IK for the marker frame with all six joints. Angles in degrees (same as the server).

    The IK target is a full pose (position + orientation, soft, position weighted higher), so all six joints
    take part and nothing drifts: joint 6 and wrist_roll orient the marker instead of being left free.
    Joint limits: placo pushes any joint outside its limit back by up to ~10° per solve, which wrecks the
    solution (the CAD file's limits were tighter than the calibrated arm: lift ±100 vs ±102, wrist_flex ±95
    vs ±100, and joint 6 was modelled as a jaw, -10..100°). So the limits are set from the arm: `KIN_BOUNDS`
    by default, the server's calibration bounds via `set_limits(status["bounds"])`.
    Velocity limits bound every solve (`set_speed`): near singularities or out of reach the QP otherwise
    jumps 60–180° in one tick (elbow flips).
    """

    def __init__(self, urdf: Path | str = URDF, frame: str = TARGET_FRAME, bounds: dict | None = KIN_BOUNDS,
                 vmax_dps: float = 60.0, rate_hz: float = 30.0):
        from lerobot.model.kinematics import RobotKinematics  # placo — `ee` extra

        self.kin = RobotKinematics(str(urdf), frame, list(JOINTS))
        self.limits: dict[str, tuple[float, float]] = {}
        self.set_limits(bounds or {})
        self.set_speed(vmax_dps, rate_hz)

    def set_limits(self, bounds: dict) -> None:
        """bounds: {joint: (lo_deg, hi_deg)} — e.g. the `bounds` field of the server's `status` reply."""
        for j in JOINTS:
            if j in bounds:
                lo, hi = (float(x) for x in bounds[j])
                self.limits[j] = (lo, hi)
                self.kin.robot.set_joint_limits(j, float(np.deg2rad(lo)), float(np.deg2rad(hi)))

    def set_speed(self, vmax_dps: float, rate_hz: float = 30.0) -> None:
        """Cap each IK solve at vmax_dps/rate_hz degrees per joint (4 solves/tick → 8°/tick at 60°/s, 30 Hz)."""
        self.kin.solver.dt = 1.0 / rate_hz
        self.kin.solver.enable_velocity_limits(True)
        self.kin.robot.set_velocity_limits(float(np.deg2rad(vmax_dps)))

    def clamp(self, q_deg) -> np.ndarray:
        q = np.array(q_deg, dtype=float)
        for i, j in enumerate(JOINTS):
            if j in self.limits:
                q[i] = np.clip(q[i], *self.limits[j])
        return q

    def fk(self, q_deg) -> np.ndarray:
        """4×4 pose of the marker frame in base_link."""
        return self.kin.forward_kinematics(np.asarray(q_deg, dtype=float))

    def xyz(self, q_deg) -> np.ndarray:
        return self.fk(q_deg)[:3, 3]

    def ik(self, q_deg, target: np.ndarray, iters: int = 4, orientation_weight: float = 0.01) -> np.ndarray:
        """placo runs one QP step per call — iterate a few times so a per-tick step converges.
        Weights: position 1 per metre, orientation `orientation_weight` per radian (0.01 → 1 rad ≈ 1 cm)."""
        q = self.clamp(q_deg)
        for _ in range(iters):
            q = self.kin.inverse_kinematics(q, target, position_weight=1.0, orientation_weight=orientation_weight)
        return q


class TipIK:
    """IK for a tool point far from the wrist (the endoscope tip, 166 mm out): position first, orientation soft,
    and stay near a reference posture.

    Why not `MarkerKinematics.ik`: its single soft frame task converges to the exact 6-D pose whatever the weights,
    and holding the full orientation while moving the tip sideways takes ≈ 28° of joint motion per 2 cm on this
    arm (pan / wrist_roll / joint 6 counter-rotating) — the joints then arrive at different speeds and the tip
    swung 5–7 cm (2026-09-23 steps test). Here the position is a hard task, the orientation a soft one and a posture
    task pulls toward the engage posture: the same 2 cm takes ≈ 4.6° with the tip exact and ≈ 2° of tilt.
    Angles in degrees (model angles), limits as {joint: (lo, hi)}.
    """

    def __init__(self, urdf: Path | str, frame: str, limits: dict, w_ori: float = 0.1, w_post: float = 0.01,
                 vmax_dps: float = 240.0, rate_hz: float = 90.0, axis_local=None, w_axis: float = 1.0,
                 lock: dict | None = None):
        import placo

        self.robot = placo.RobotWrapper(str(urdf))
        self.solver = placo.KinematicsSolver(self.robot)
        self.solver.mask_fbase(True)
        for j, (lo, hi) in limits.items():
            self.robot.set_joint_limits(j, float(np.deg2rad(lo)), float(np.deg2rad(hi)))
        self.solver.enable_joint_limits(True)
        self.solver.dt = 1.0 / rate_hz
        self.solver.enable_velocity_limits(True)
        self.robot.set_velocity_limits(float(np.deg2rad(vmax_dps)))
        self.frame = frame
        self.pos = self.solver.add_position_task(frame, np.zeros(3))
        # axis_local (tool frame): align only that axis (a drill's shaft) instead of the full orientation — with the
        # tip as a hard task that is 3 + 2 constraints, which a 5-DOF arm meets exactly; the roll about the shaft
        # is left to the posture task (the drill's cut does not depend on it)
        self.axis_local = None if axis_local is None else np.asarray(axis_local, float) / np.linalg.norm(axis_local)
        self.w_axis = w_axis
        if self.axis_local is None:
            self.ori = self.solver.add_orientation_task(frame, np.eye(3))
        else:
            self.ori = self.solver.add_axisalign_task(frame, self.axis_local, self.axis_local)
        self.post = self.solver.add_joints_task()
        # lock {joint: weight}: an extra, stronger pull of those joints toward the engage posture (e.g. wrist_roll,
        # which the axis task leaves free and which otherwise wanders)
        self.lock = dict(lock or {})
        self.lock_tasks = {j: self.solver.add_joints_task() for j in self.lock}
        self.w_ori, self.w_post = w_ori, w_post
        self.hard = True
        self.fallbacks = 0

    def _set(self, q_deg) -> None:
        for j, v in zip(JOINTS, np.deg2rad(np.asarray(q_deg, float))):
            self.robot.set_joint(j, float(v))
        self.robot.update_kinematics()

    def fk(self, q_deg) -> np.ndarray:
        self._set(q_deg)
        return np.array(self.robot.get_T_world_frame(self.frame))

    def set_posture(self, q_deg) -> None:
        q = {j: float(v) for j, v in zip(JOINTS, np.deg2rad(np.asarray(q_deg, float)))}
        self.post.set_joints(q)
        for j, t in self.lock_tasks.items():
            t.set_joints({j: q[j]})

    def ik(self, q_deg, target: np.ndarray, iters: int = 3) -> np.ndarray:
        """A few QP steps from q_deg toward target (4×4). A hard position task that turns infeasible (out of reach /
        limits) falls back to a heavily weighted soft one for that call."""
        self._set(q_deg)
        self.pos.target_world = np.asarray(target[:3, 3], float)
        if self.axis_local is None:
            self.ori.R_world_frame = np.asarray(target[:3, :3], float)
            self.ori.configure("ori", "soft", self.w_ori)
        else:
            self.ori.targetAxis_world = np.asarray(target[:3, :3], float) @ self.axis_local
            self.ori.configure("axis", "soft", self.w_axis)
        self.post.configure("post", "soft", self.w_post)
        for j, t in self.lock_tasks.items():
            t.configure(f"lock_{j}", "soft", self.lock[j])
        for _ in range(iters):
            try:
                self.pos.configure("pos", "hard", 1.0)
                self.solver.solve(True)
            except RuntimeError:
                self.fallbacks += 1
                self.pos.configure("pos", "soft", 100.0)
                self.solver.solve(True)
            self.robot.update_kinematics()
        return np.rad2deg([self.robot.get_joint(j) for j in JOINTS])


CLEAR_FRAMES = ("gripper_link", "moving_jaw_so101_v1_link", "gripper_frame_link")
MIN_CLEARANCE_MM = 110.0


class Clearance:
    """Self-collision proxy: the smallest distance from the gripper motor / jaw / jaw tip frames to the shoulder_lift
    joint (motor 2, `upper_arm_link` origin). The folded rest pose sits at ≈53 mm and there wrist/joint-6 motions
    drive the jaw into motor 2 (2026-09-23, hand-eye run); a moderately open arm (lift −60, elbow 40) is ≈130 mm."""

    def __init__(self, urdf: Path | str = URDF):
        self.ref = MarkerKinematics(urdf, "upper_arm_link", bounds=None)
        self.kins = [MarkerKinematics(urdf, f, bounds=None) for f in CLEAR_FRAMES]

    def mm(self, q_deg) -> float:
        p0 = self.ref.fk(q_deg)[:3, 3]
        return 1000.0 * min(float(np.linalg.norm(k.fk(q_deg)[:3, 3] - p0)) for k in self.kins)


class EETeleop:
    """Per-tick Cartesian teleop → joint goals (degrees) for `SpineArm.send_action`.

    `step(q_obs, dxyz, jog)`: dxyz in [-1, 1]³ (× `step_m` per tick) translates the target pose; `jog`
    {joint: ±1} (× `jog_deg`) moves those joints directly and re-bases the target on the resulting pose, so
    a wrist/joint-6 jog reorients the marker exactly like the old joint jog — and the next translation holds
    that orientation with all six joints. The target starts at the FK of the observed pose on the first step
    after `reset()`, then only accumulates commands, clipped to the box. Measured positions are never fed back.
    Leashes: the target is kept within `leash_m` of the pose the IK reached (an unreachable command cannot
    run away and snap the arm later) and within `track_leash_m` of the *measured* marker pose. The second
    one keeps a diagonal command diagonal: the heavy joints (lift/elbow → z) lag the light ones (pan → x/y),
    and without it x/y finish first and z arrives afterwards (2026-09-22 field report). Measured pose is used
    only as a leash, never as the target itself (that fed servo error back into the command and crept).
    """

    def __init__(self, kin: MarkerKinematics | None = None, box: tuple[np.ndarray, np.ndarray] | None = None,
                 step_m: float = 0.006, jog_deg: float = 1.0, ik_iters: int = 4, orientation_weight: float = 0.01,
                 leash_m: float = 0.02, track_leash_m: float = 0.02, closed_loop: bool = True, jog_lead_deg: float = 8.0):
        self.kin = kin or MarkerKinematics()
        self.lo, self.hi = box if box is not None else ee_box()
        self.step_m, self.jog_deg = step_m, jog_deg
        self.ik_iters, self.orientation_weight = ik_iters, orientation_weight
        self.leash_m, self.track_leash_m, self.closed_loop = leash_m, track_leash_m, closed_loop
        self.jog_lead_deg = jog_lead_deg
        self._goals: dict[str, float] | None = None
        self.target: np.ndarray | None = None
        self.q_cmd: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self.target is not None

    def reset(self) -> None:
        self.target = None
        self.q_cmd = None
        self._goals = None

    def step(self, q_obs: dict[str, float], dxyz=(0.0, 0.0, 0.0), jog: dict[str, float] | None = None) -> dict[str, float]:
        q_meas = self.kin.clamp([float(q_obs[j]) for j in JOINTS])
        if self.target is None:
            self.q_cmd = q_meas.copy()
            self.target = self.kin.fk(self.q_cmd).copy()
            self._goals = None
        moved = False
        if jog and any(jog.values()):
            # Jogged joints integrate (goal runs up to jog_lead_deg ahead of the measured joint — a goal only
            # 1° ahead makes the servo creep); the other joints are anchored to the measured pose.
            base = q_meas.copy() if self.closed_loop else self.q_cmd.copy()
            for j, v in jog.items():
                if v:
                    i = JOINTS.index(j)
                    goal = self.q_cmd[i] + float(v) * self.jog_deg
                    base[i] = float(np.clip(goal, q_meas[i] - self.jog_lead_deg, q_meas[i] + self.jog_lead_deg))
            self.q_cmd = self.kin.clamp(base)
            self.target = self.kin.fk(self.q_cmd).copy()  # re-base: the jog is exact, orientation follows
            if not np.any(np.asarray(dxyz, dtype=float)):
                self._goals = {j: float(v) for j, v in zip(JOINTS, self.q_cmd)}
                return dict(self._goals)  # pure jog: exact joint goal, no IK round trip
            moved = True
        before = self.target[:3, 3].copy()
        self.target[:3, 3] = np.clip(self.target[:3, 3] + np.asarray(dxyz, dtype=float) * self.step_m, self.lo, self.hi)
        moved = moved or bool(np.any(self.target[:3, 3] != before))
        if not moved and self._goals is not None:
            return dict(self._goals)  # target unchanged (idle stick / box wall): repeat the goal, never re-solve
        # Closed loop: solve from where the arm *is*, so each tick's goal is a step from the measured pose
        # toward the target — the path stays straight even when the loaded joints lag the light ones.
        start = q_meas if self.closed_loop else self.q_cmd
        try:
            self.q_cmd = self.kin.ik(start, self.target, self.ik_iters, self.orientation_weight)
        except RuntimeError:  # placo QPError: keep the last solution
            pass
        self._leash(self.kin.xyz(self.q_cmd), self.leash_m)
        self._leash(self.kin.xyz(q_meas), self.track_leash_m)
        self._goals = {j: float(v) for j, v in zip(JOINTS, self.q_cmd)}
        return dict(self._goals)

    def _leash(self, anchor: np.ndarray, length: float) -> None:
        gap = self.target[:3, 3] - anchor
        if np.linalg.norm(gap) > length:
            self.target[:3, 3] = anchor + gap / np.linalg.norm(gap) * length

    def jog(self, q_obs: dict[str, float], axes: dict[str, float]) -> dict[str, float]:
        """Direct joint jog through the same pose target (all six joint goals)."""
        return self.step(q_obs, (0.0, 0.0, 0.0), axes)
