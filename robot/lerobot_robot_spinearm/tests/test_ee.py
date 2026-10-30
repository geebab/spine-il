"""EE helpers: workspace box, marker kinematics, teleop accumulation — and one run against a --sim server."""

import random
import subprocess
import sys
import time

import numpy as np
import pytest

pytest.importorskip("placo")

from lerobot_robot_spinearm import SpineArm, SpineArmConfig  # noqa: E402
from lerobot_robot_spinearm.client import ArmRpc, ArmTimeout  # noqa: E402
from lerobot_robot_spinearm.ee import Z_FLOOR_M, EETeleop, MarkerKinematics, ee_box  # noqa: E402
from lerobot_robot_spinearm.protocol import JOINTS  # noqa: E402

NEUTRAL = dict(zip(JOINTS, [3.6, -101.9, 59.4, 99.5, 1.9, -60.7]))
MID = dict(zip(JOINTS, [20.0, -40.0, 30.0, 10.0, -30.0, -20.0]))


@pytest.fixture(scope="module")
def kin():
    return MarkerKinematics()


def test_box_margin_and_z_floor():
    lo, hi = ee_box()
    assert lo[2] == pytest.approx(Z_FLOOR_M)  # file says 4.3 mm — floored to 4 cm
    assert lo[0] == pytest.approx(-0.2257) and hi[0] == pytest.approx(0.3943)  # ±30 mm margin
    assert lo[1] < 0 < hi[1] and hi[2] < 0.5


def test_joint6_swings_marker(kin):
    q = np.array(list(NEUTRAL.values()))
    q2 = q.copy()
    q2[5] += 10
    chord_mm = np.linalg.norm(kin.xyz(q2) - kin.xyz(q)) * 1000
    assert chord_mm == pytest.approx(2 * 54 * np.sin(np.deg2rad(5)), abs=0.5)  # 54 mm arm


@pytest.mark.parametrize("pose", [MID, NEUTRAL])
def test_ik_tracks_a_tick_step(kin, pose):
    q = np.array(list(pose.values()))
    target = kin.fk(q).copy()
    target[:3, 3] += [0.006, -0.006, 0.006]  # one full-stick tick
    q2 = kin.ik(q, target)
    assert np.linalg.norm(kin.xyz(q2) - target[:3, 3]) < 0.003  # folded neutral converges over 2 ticks
    assert np.abs(q2 - q).max() < 8.1  # velocity limit: 4 solves × 60°/s / 30 Hz


def _ori_deg(Ta, Tb):
    r = Ta[:3, :3].T @ Tb[:3, :3]
    return np.degrees(np.arccos(np.clip((np.trace(r) - 1) / 2, -1, 1)))


def test_translation_holds_orientation_with_all_six_joints(kin):
    t = EETeleop(kin)
    T0 = None
    goals = dict(MID)
    for _ in range(16):  # 5 cm forward, 5 cm up — reachable from MID; the (ideal) arm follows each goal
        goals = t.step(goals, (1, 0, 0) if _ < 8 else (0, 0, 1))
        T0 = T0 if T0 is not None else t.target.copy()
    qf = np.array([goals[j] for j in JOINTS])
    assert np.linalg.norm(kin.xyz(qf) - t.target[:3, 3]) < 0.001
    assert _ori_deg(T0, kin.fk(qf)) < 1.0  # no wrist drift: orientation is part of the target
    assert abs(goals["wrist_roll"] - MID["wrist_roll"]) > 0.1 and abs(goals["gripper"] - MID["gripper"]) > 0.1


def test_unreachable_target_is_leashed(kin):
    t = EETeleop(kin)
    q = dict(MID)
    for _ in range(80):  # 48 cm requested forward: box clips at ~39 cm, reach ends earlier
        goals = t.step(q, (1, 0, 0))
    qf = np.array([goals[j] for j in JOINTS])
    assert np.linalg.norm(t.target[:3, 3] - kin.xyz(qf)) <= t.leash_m + 1e-6
    assert all(abs(goals[j] - MID[j]) < 180 for j in JOINTS)


@pytest.mark.parametrize("bounds", ["default", None])
def test_idle_ik_is_a_noop_at_neutral(bounds):
    """The limits (KIN_BOUNDS / the URDF file) must contain the folded neutral pose, else placo drags the
    arm inside them the moment the stick is touched."""
    k = MarkerKinematics() if bounds == "default" else MarkerKinematics(bounds=None)
    q = np.array(list(NEUTRAL.values()))
    assert np.abs(k.ik(q, k.fk(q).copy()) - q).max() < 0.05


def test_set_limits_applies_server_bounds(kin):
    lo, hi = kin.kin.robot.get_joint_limits("shoulder_pan")
    kin.set_limits({"shoulder_pan": (-82.0, 82.0)})
    lo2, hi2 = kin.kin.robot.get_joint_limits("shoulder_pan")
    assert (lo2, hi2) == pytest.approx((np.deg2rad(-82), np.deg2rad(82))) and lo2 != lo
    kin.set_limits({"shoulder_pan": (np.rad2deg(lo), np.rad2deg(hi))})


def test_teleop_accumulates_and_clips_to_box(kin):
    t = EETeleop(kin, step_m=0.01)
    x0 = kin.xyz(list(MID.values()))[0]
    goals = dict(MID)
    for _ in range(3):
        goals = t.step(goals, (1, 0, 0))  # ideal arm: follows each goal
    assert set(goals) == set(JOINTS)
    assert t.target[0, 3] == pytest.approx(min(x0 + 0.03, t.hi[0]), abs=1e-4)
    for _ in range(100):
        goals = t.step(goals, (0, 0, -1))
    assert t.target[2, 3] == pytest.approx(t.lo[2])  # z floor holds


def test_jog_is_exact_and_rebases_target(kin):
    t = EETeleop(kin, jog_deg=2.0)
    t.step(MID)
    before = t.target[:3, 3].copy()
    goals = t.step(MID, (0, 0, 0), {"gripper": 1.0})
    assert goals["gripper"] == pytest.approx(MID["gripper"] + 2.0, abs=1e-3)
    assert {j: goals[j] for j in JOINTS[:5]} == pytest.approx({j: MID[j] for j in JOINTS[:5]}, abs=1e-3)
    swing_mm = np.linalg.norm(t.target[:3, 3] - before) * 1000
    assert swing_mm == pytest.approx(2 * 54 * np.sin(np.deg2rad(1.0)), abs=0.3)  # marker moved on its arc
    goals = t.jog(MID, {"wrist_flex": -1.0, "wrist_roll": 1.0})
    assert goals["wrist_flex"] == pytest.approx(MID["wrist_flex"] - 2.0, abs=1e-3)
    assert goals["wrist_roll"] == pytest.approx(MID["wrist_roll"] + 2.0, abs=1e-3)


@pytest.fixture(scope="module")
def sim_port(tmp_path_factory):
    port = random.randint(20000, 40000)
    neutral = tmp_path_factory.mktemp("n") / "neutral.json"
    neutral.write_text('{"_units": "deg", "spine_left": %s}' % str(NEUTRAL).replace("'", '"'))
    proc = subprocess.Popen([sys.executable, "-m", "lerobot_robot_spinearm.server.main", "--arm", "left", "--sim",
                             "--cmd-port", str(port), "--pub-port", str(port + 1), "--neutral", str(neutral)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t_end = time.time() + 10
    while time.time() < t_end:
        try:
            ArmRpc(port=port, timeout_ms=300).must(cmd="state")
            break
        except ArmTimeout:
            time.sleep(0.1)
    else:
        raise RuntimeError("sim server did not come up")
    yield port
    proc.terminate()
    proc.wait(5)


def test_teleop_against_sim_server(sim_port, kin):
    arm = SpineArm(SpineArmConfig(arm="left", cmd_port=sim_port, pub_port=sim_port + 1, owner="test"))
    arm.connect()
    tele = EETeleop(kin)
    obs = arm.get_observation()
    q0 = {j: obs[f"{j}.pos"] for j in JOINTS}
    x0 = kin.xyz(list(q0.values()))[0]
    for _ in range(30):  # 1 s of full stick forward
        obs = arm.get_observation()
        goals = tele.step({j: obs[f"{j}.pos"] for j in JOINTS}, (1, 0, 0))
        arm.send_action({f"{j}.pos": v for j, v in goals.items()})
        time.sleep(1 / 30)
    time.sleep(0.5)  # let the sim servo settle
    obs = arm.get_observation()
    x1 = kin.xyz([obs[f"{j}.pos"] for j in JOINTS])[0]
    assert x1 - x0 > 0.03, f"marker did not move forward: {x0:.3f} -> {x1:.3f}"
    assert arm.home(wait=True)["state"] == "HOLD"
    assert arm.torque_off()["state"] == "OFF"
    arm.disconnect()


def test_jog_beyond_a_bound_is_clamped(kin):
    """Right arm, 2026-09-22: joint 6 jogged past its calibration bound → placo 'Infeasible QP' crashed the teleop."""
    kin.set_limits({"gripper": (-65.3, 65.3)})
    t = EETeleop(kin, jog_deg=1.0)
    q = dict(MID, gripper=-64.4)
    for _ in range(5):
        goals = t.step(q, (0, 0, 0), {"gripper": -1.0})
    assert goals["gripper"] == pytest.approx(-65.3, abs=1e-3)
    t.reset()
    g = t.step(dict(MID, gripper=-66.0), (1, 0, 0))["gripper"]  # measured outside → clamped, no exception
    assert -65.3 - 1e-3 <= g <= 65.3
    kin.set_limits({"gripper": (-66.0, 66.0)})


def test_target_waits_for_a_lagging_arm(kin):
    """Diagonal stays diagonal: with the arm not moving at all, the target may lead by one leash only."""
    t = EETeleop(kin, track_leash_m=0.02)
    p0 = kin.xyz(list(MID.values()))
    for _ in range(30):
        t.step(MID, (1, 0, 1))  # q_obs never changes: a stalled/lagging arm
    lead = t.target[:3, 3] - p0
    assert np.linalg.norm(lead) <= 0.02 + 1e-6
    assert lead[0] == pytest.approx(lead[2], abs=1e-3)  # x and z lead equally
