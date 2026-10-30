"""The proxy Robot as LeRobot sees it: plugin discovery, connect/observe/act against a --sim server."""

import random
import subprocess
import sys
import time

import pytest

from lerobot_robot_spinearm import SpineArm, SpineArmConfig, SpineBiArm, SpineBiArmConfig
from lerobot_robot_spinearm.client import ArmRpc, ArmTimeout
from lerobot_robot_spinearm.protocol import JOINTS


def _wait(port, s=10.0):
    t = time.time() + s
    while time.time() < t:
        try:
            ArmRpc(port=port, timeout_ms=300).must(cmd="state")
            return
        except ArmTimeout:
            time.sleep(0.1)
    raise RuntimeError("sim server did not come up")


@pytest.fixture(scope="module")
def sim_servers():
    base = random.randint(20000, 40000)
    procs = []
    for arm, i in (("left", 0), ("right", 10)):
        cmd = [sys.executable, "-m", "lerobot_robot_spinearm.server.main", "--arm", arm, "--sim",
               "--cmd-port", str(base + i), "--pub-port", str(base + i + 1)]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    try:
        _wait(base)
        _wait(base + 10)
        yield base
    finally:
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait(5)


def test_registered_with_lerobot():
    from lerobot.robots.config import RobotConfig
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    choices = RobotConfig.get_known_choices()
    assert "spine_arm" in choices and "spine_bi_arm" in choices


def test_single_arm_roundtrip(sim_servers):
    from lerobot.robots import make_robot_from_config

    robot = make_robot_from_config(SpineArmConfig(arm="left", cmd_port=sim_servers, pub_port=sim_servers + 1, owner="test"))
    assert isinstance(robot, SpineArm)
    robot.connect()
    obs = robot.get_observation()
    assert set(obs) == {f"{j}.pos" for j in JOINTS}
    sent = robot.send_action({"shoulder_pan.pos": 15.0})
    assert sent["shoulder_pan.pos"] == 15.0
    for _ in range(30):
        robot.send_action({"shoulder_pan.pos": 15.0})
        time.sleep(1 / 30)
    assert abs(robot.get_observation()["shoulder_pan.pos"] - 15.0) < 2.0
    lease = robot.status()["lease"]
    assert lease["tool"] == "test" and lease["verified"]  # the ident socket vouches for this account
    robot.disconnect()
    assert ArmRpc(port=sim_servers).must(cmd="state")["lease"] is None  # released


def test_second_client_refused_while_leased(sim_servers):
    a = SpineArm(SpineArmConfig(arm="left", cmd_port=sim_servers, pub_port=sim_servers + 1, owner="a"))
    b = SpineArm(SpineArmConfig(arm="left", cmd_port=sim_servers, pub_port=sim_servers + 1, owner="b"))
    a.connect()
    with pytest.raises(Exception, match="lease_held"):
        b.connect()
    a.disconnect()


def test_bi_arm_prefixes(sim_servers):
    cfg = SpineBiArmConfig(id="spine",
                           left=SpineArmConfig(arm="left", cmd_port=sim_servers, pub_port=sim_servers + 1, owner="bi"),
                           right=SpineArmConfig(arm="right", cmd_port=sim_servers + 10, pub_port=sim_servers + 11, owner="bi"))
    robot = SpineBiArm(cfg)
    assert list(robot.action_features)[:2] == ["left_shoulder_pan.pos", "left_shoulder_lift.pos"]
    robot.connect()
    obs = robot.get_observation()
    assert len(obs) == 12 and "right_gripper.pos" in obs
    sent = robot.send_action({"left_elbow_flex.pos": 5.0, "right_wrist_roll.pos": -5.0})
    assert sent == {"left_elbow_flex.pos": 5.0, "right_wrist_roll.pos": -5.0}
    robot.disconnect()
