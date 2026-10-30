"""calib_tf — calibration files → /tf_static:
  <arm>_base_link → tracker_cam            (hand-eye Z = T_base←cam)
  <arm>_marker_frame_link → <arm>_endo_marker (X = T_link←const, constellation on the jaw)
  <arm>_endo_marker → <arm>_endo_tip        (tool tip, tracker tool calibration)
  scene → tracker_cam                        (optional: viewer registration T0 = T_scene←cam, JSON {"T0": 4×4 mm})
Re-run the node after re-calibrating."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster

from .convert import R_to_xyzw, load_calib


def _tf(node, parent, child, T):
    t = TransformStamped()
    t.header.stamp = node.get_clock().now().to_msg()
    t.header.frame_id, t.child_frame_id = parent, child
    t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = (float(v) for v in T[:3, 3])
    (t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z,
     t.transform.rotation.w) = R_to_xyzw(T[:3, :3])
    return t


class CalibTf(Node):
    def __init__(self):
        super().__init__("calib_tf")
        arms = self.declare_parameter("arms", ["left"]).value
        tool = self.declare_parameter(
            "tool_calib", "~/il-robot/tracking/tool_calib.json").value
        reg = self.declare_parameter("registration", "").value
        cam = self.declare_parameter("camera_frame", "tracker_cam").value
        tfs = []
        tip = np.asarray(json.loads(Path(tool).expanduser().read_text())["tip_local"], float) / 1000.0
        for i, arm in enumerate(arms):
            path = self.declare_parameter(f"calib_{arm}", f"~/il-robot/calib/handeye_{arm}.json").value
            c = load_calib(path)
            if i == 0:  # one camera: its pose comes from the first arm's hand-eye; others are consistent by calibration
                tfs.append(_tf(self, f"{arm}_base_link", cam, c["Z"]))
            else:  # express the other base in the camera frame so the tree stays a tree
                tfs.append(_tf(self, cam, f"{arm}_base_link", np.linalg.inv(c["Z"])))
            tfs.append(_tf(self, f"{arm}_marker_frame_link", f"{arm}_endo_marker", c["X"]))
            T_tip = np.eye(4)
            T_tip[:3, 3] = tip
            tfs.append(_tf(self, f"{arm}_endo_marker", f"{arm}_endo_tip", T_tip))
        if reg:
            T0 = np.asarray(json.loads(Path(reg).expanduser().read_text())["T0"], float)
            T0[:3, 3] /= 1000.0
            tfs.append(_tf(self, "scene", cam, T0))
        self.br = StaticTransformBroadcaster(self)
        self.br.sendTransform(tfs)
        self.get_logger().info(f"published {len(tfs)} static transforms")


def main():
    rclpy.init()
    rclpy.spin(CalibTf())


if __name__ == "__main__":
    main()
