"""LeRobot robot configs for the arm-server proxy: `spine_arm` (one arm) and `spine_bi_arm` (both).

Registered with `RobotConfig.register_subclass`, so `--robot.type=spine_arm` works in every LeRobot
CLI once this distribution is installed (LeRobot auto-imports `lerobot_robot_*` packages).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("spine_arm")
@dataclass
class SpineArmConfig(RobotConfig):
    arm: str = "left"  # left = endoscope arm, right = drill arm
    host: str = "127.0.0.1"
    cmd_port: int | None = None  # default 5601 (left) / 5611 (right)
    pub_port: int | None = None  # default 5602 / 5612
    owner: str | None = None  # lease owner shown to others; default = your login name
    auto_enable: bool = True  # torque on when connecting (server starts torque-off)
    release_on_disconnect: bool = True
    step_deg: float | None = None  # ask the server for a smaller per-tick cap (≤ 10)
    timeout_ms: int = 1000
    lease_force: bool = False  # take over a lease whose holder has been idle ≥ 5 s
    # "zmq" = arm server directly; "ros2" = through spinearm_bridge (/<arm>/joint_command, services) — needs the ROS
    # environment sourced and ros-bridge running. host/ports/lease_force are ZMQ-only.
    transport: str = "zmq"


@RobotConfig.register_subclass("spine_bi_arm")
@dataclass
class SpineBiArmConfig(RobotConfig):
    left: SpineArmConfig = field(default_factory=lambda: SpineArmConfig(arm="left"))
    right: SpineArmConfig = field(default_factory=lambda: SpineArmConfig(arm="right"))
