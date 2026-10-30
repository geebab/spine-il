"""Stage 1 (observe-only): arm bridge(s), tracker bridge, calibration TF, robot_state_publisher per arm, Foxglove.

  ros2 launch spinearm_bridge bridge.launch.py arms:=left [calib:=kincal_v2]

calib picks one fit for both the camera/marker TF and the joint offsets (they must come from the same fit):
handeye (default, hand-eye + joint offsets) or kincal_v2 (refit on ~15k follower samples; 1.4 mm in the follow
workspace, but no better at the folded home pose — 4.7 mm / 1.8° vs 4.9 mm / 1.0°, 2026-09-26).
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CALIB = {"kincal_v2": "~/il-robot/calib/kincal_{arm}_v2.json", "handeye": "~/il-robot/calib/handeye_{arm}.json",
         # the hand-eye joint zeros with the camera / base placement fitted to the as-built model (fit_asbuilt.py)
         "asbuilt": "~/il-robot/calib/asbuilt_{arm}.json"}
URDF = os.path.expanduser(
    "~/spine-il/robot/lerobot_robot_spinearm/lerobot_robot_spinearm/assets/so101_marker.urdf")


def _nodes(context):
    from spinearm_bridge.visual_urdf import build_for_arm

    arms = [a for a in LaunchConfiguration("arms").perform(context).split(",") if a]
    kin = LaunchConfiguration("urdf").perform(context)
    kind = LaunchConfiguration("calib").perform(context)
    calib = {arm: os.path.expanduser(CALIB[kind].format(arm=arm)) for arm in arms}
    # Robot model with visuals: SO-101 meshes + the arm's marker block placed by its calibration (visual_urdf.py).
    share = get_package_share_directory("spinearm_description")
    descr = {}
    for arm in arms:
        # link names carry the arm prefix (= TF frame names), so viewers need no frame-prefix setting
        descr[arm], rep = build_for_arm(arm, kin, f"{share}/urdf/so101_new_calib.urdf", calib[arm],
                                        LaunchConfiguration("markers").perform(context), f"{share}/meshes/markers",
                                        link_prefix=f"{arm}_")
        print(f"[bridge.launch] {arm} robot model: {rep}", flush=True)
    # Foxglove is reachable from the whole tailnet (Tailscale Serve :8445). Since stage 2 the arm takes
    # /<arm>/joint_command and ~/acquire·~/enable, so a client that may publish or call services can move the arm —
    # view-only by default; foxglove_control:=true restores the bridge's full capability set.
    fox = {"address": "127.0.0.1", "port": 8765}
    if LaunchConfiguration("foxglove_control").perform(context) != "true":
        fox["capabilities"] = ["connectionGraph", "assets"]
    out = [Node(package="spinearm_bridge", executable="tracker_bridge", name="tracker_bridge"),
           Node(package="spinearm_bridge", executable="calib_tf", name="calib_tf", parameters=[{"arms": arms, **{f"calib_{a}": p for a, p in calib.items()}}]),
           Node(package="foxglove_bridge", executable="foxglove_bridge", name="foxglove_bridge",
                parameters=[fox])]
    for arm in arms:
        out += [Node(package="spinearm_bridge", executable="arm_bridge", namespace=arm, name="arm_bridge",
                     parameters=[{"arm": arm, "calib": calib[arm],
                                  "allow_acquire": LaunchConfiguration("allow_acquire").perform(context) == "true"}]),
                Node(package="robot_state_publisher", executable="robot_state_publisher", namespace=arm,
                     parameters=[{"robot_description": descr[arm], "frame_prefix": ""}]),
                Node(package="spinearm_bridge", executable="description_pub", namespace=arm,
                     parameters=[{"robot_description": descr[arm]}])]
    return out


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("arms", default_value="left"),
                              # taking the arm over ROS (~/acquire): off by default — ROS callers are not identified
                              DeclareLaunchArgument("allow_acquire", default_value="false"),
                              DeclareLaunchArgument("urdf", default_value=URDF),
                              DeclareLaunchArgument("calib", default_value="handeye"),
                              DeclareLaunchArgument("markers", default_value="~/il-robot/calib/markers.json"),
                              DeclareLaunchArgument("foxglove_control", default_value="false"),
                              OpaqueFunction(function=_nodes)])
