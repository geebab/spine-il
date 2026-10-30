"""Recording replay (visualisation only): a viewer session recording (recorder v2 .jsonl) → /replay/… next to the live
graph, so Foxglove shows it with the robot model and the live tracker.

  ros2 launch spinearm_bridge replay.launch.py file:=~/il-robot/sessions/s_….jsonl [rate:=0.5] [loop:=true]
      [start_s:=10.0] [end_s:=40.0] [clock_mode:=wall|sim] [namespace:=/replay] [tf_parent:=camera|scene|none]
      [camera_frame:=tracker_cam] [scene_frame:=scene] [drawn:=true] [foxglove:=false]

`file` is required. namespace:=/ publishes on the live topic names (/tracker/…) — only without tracker_bridge.

Next to bridge.launch.py (its foxglove_bridge, calib_tf's <arm>_base_link → tracker_cam): keep the defaults — wall
clock, registration published as tracker_cam → scene. Standalone (no live graph): foxglove:=true starts a view-only
foxglove_bridge here; with clock_mode:=sim it runs on the recording's /clock.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

ARGS = {"rate": "1.0", "loop": "false", "start_s": "0.0", "end_s": "0.0", "namespace": "/replay",
        "clock_mode": "wall", "tf_parent": "camera", "camera_frame": "tracker_cam", "scene_frame": "scene",
        "drawn": "true", "foxglove": "false"}


def _nodes(context):
    a = {k: LaunchConfiguration(k).perform(context) for k in ("file", *ARGS)}
    on = {k: a[k].strip().lower() in ("1", "true", "yes") for k in ("loop", "drawn", "foxglove")}
    a["file"] = os.path.expanduser(a["file"])
    a["namespace"] = a["namespace"] or "/"   # '' and '/' both mean the live names; '/' never reaches rcl as ''
    # strings stay strings (launch_ros would YAML-parse them)
    text = {k: ParameterValue(a[k], value_type=str)
            for k in ("file", "namespace", "clock_mode", "tf_parent", "camera_frame", "scene_frame")}
    params = {**text, "rate": float(a["rate"]), "loop": on["loop"], "start_s": float(a["start_s"]),
              "end_s": float(a["end_s"]), "drawn": on["drawn"],
              "use_sim_time": False}   # the player is the /clock source; it paces itself on the wall clock
    out = [Node(package="spinearm_bridge", executable="recording_player", name="recording_player", output="screen",
                parameters=[params])]
    if on["foxglove"]:   # view-only, like bridge.launch.py's default
        fox = {"address": "127.0.0.1", "port": 8765, "capabilities": ["connectionGraph", "assets"],
               "use_sim_time": a["clock_mode"] == "sim"}
        out.append(Node(package="foxglove_bridge", executable="foxglove_bridge", name="foxglove_bridge",
                        parameters=[fox]))
    return out


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument("file", description="recorder v2 .jsonl to play")]
                             + [DeclareLaunchArgument(k, default_value=v) for k, v in ARGS.items()]
                             + [OpaqueFunction(function=_nodes)])
