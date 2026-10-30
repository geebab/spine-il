"""tracker_bridge — tracker tap (UDP, tracker_service.py --tap) → /tracker/<tool>/tracked (TrackedPose) and
/tracker/<tool>/pose (PoseStamped, filtered), stamped with the camera frame's mid-exposure time. Tool names are the
tracker's (endo/endo2 follow screen side); identity by geometry comes with the stage-2 follow node.

/tracker/balls (MarkerArray, ≤ 30 Hz): the balls each tool's raw pose implies, drawn at their real size, so a viewer
can overlay what the camera sees on the robot model without Foxglove's metre-long default pose arrows."""

from __future__ import annotations

import json
import os
import socket
import time

import numpy as np

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from spinearm_msgs.msg import TrackedPose
from visualization_msgs.msg import Marker, MarkerArray

from .convert import split_stamp_ms, tap_to_fields


TRACKING = "~/il-robot/tracking"
TOOL_MODELS = {"endo": f"{TRACKING}/tool_calib.json", "endo2": f"{TRACKING}/tool_calib.json",
               "drill": f"{TRACKING}/tool_calib_drill_robotR.json"}
TOOL_RGBA = {"endo": (0.1, 0.9, 0.3, 0.9), "endo2": (1.0, 0.6, 0.1, 0.9), "drill": (0.95, 0.2, 0.9, 0.9)}


def _R_xyzw(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _fill_pose(p, pos, q):
    p.position.x, p.position.y, p.position.z = pos
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q


class TrackerBridge(Node):
    def __init__(self):
        super().__init__("tracker_bridge")
        port = self.declare_parameter("port", 9885).value
        self.frame = self.declare_parameter("frame_id", "tracker_cam").value
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.pubs: dict[str, tuple] = {}
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.last_fn, self.frames, self.gaps = None, 0, 0
        # ball geometry per tool (metres, tool frame) for /tracker/balls
        models = json.loads(self.declare_parameter("tool_models_json", json.dumps(TOOL_MODELS)).value)
        self.balls_local: dict[str, tuple[np.ndarray, float]] = {}
        for tool, path in models.items():
            try:
                d = json.load(open(os.path.expanduser(path)))
                self.balls_local[tool] = (np.asarray(d["points_local"], float) / 1000.0,
                                          float(d.get("sphere_radius_mm", 7.0)) / 1000.0)
            except (OSError, KeyError, ValueError) as e:
                self.get_logger().warn(f"no ball model for {tool}: {e}")
        self.pub_balls = self.create_publisher(MarkerArray, "/tracker/balls", 10)
        self.latest_balls: dict[str, tuple] = {}   # tool → (stamp msg, points (n,3) m, wall s)
        self.create_timer(0.002, self.drain)
        self.create_timer(1.0, self.diag)
        self.create_timer(1 / 30, self.publish_balls)

    def _pub(self, tool):
        if tool not in self.pubs:
            self.pubs[tool] = (self.create_publisher(TrackedPose, f"/tracker/{tool}/tracked", qos_profile_sensor_data),
                               self.create_publisher(PoseStamped, f"/tracker/{tool}/pose", qos_profile_sensor_data))
        return self.pubs[tool]

    def drain(self):
        while True:
            try:
                data, _ = self.sock.recvfrom(65536)
            except BlockingIOError:
                return
            f = tap_to_fields(json.loads(data))
            if not f["tool"]:
                continue
            if f["frame"] != self.last_fn:
                if self.last_fn is not None and f["frame"] > self.last_fn + 1:
                    self.gaps += f["frame"] - self.last_fn - 1
                self.last_fn, self.frames = f["frame"], self.frames + 1
            tp, ps = self._pub(f["tool"])
            m = TrackedPose()
            m.header.stamp.sec, m.header.stamp.nanosec = split_stamp_ms(f["stamp_ms"])
            m.header.frame_id = self.frame
            for k in ("tool", "frame", "lost", "visible", "rmse", "still", "held", "t_arrival", "t_pose", "t_send"):
                setattr(m, k, f[k])
            if "pos" in f:
                _fill_pose(m.pose, f["pos"], f["quat"])
                _fill_pose(m.pose_raw, f["pos_raw"], f["quat_raw"])
                if f["tip"]:
                    m.tip.x, m.tip.y, m.tip.z = f["tip"]
                if f["tip_raw"]:
                    m.tip_raw.x, m.tip_raw.y, m.tip_raw.z = f["tip_raw"]
                s = PoseStamped()
                s.header = m.header
                s.pose = m.pose
                ps.publish(s)
                if f["tool"] in self.balls_local:
                    P, _ = self.balls_local[f["tool"]]
                    pts = P @ _R_xyzw(f["quat_raw"]).T + np.asarray(f["pos_raw"])
                    self.latest_balls[f["tool"]] = (m.header.stamp, pts, time.time())
            tp.publish(m)

    def publish_balls(self):
        now = time.time()
        arr = MarkerArray()
        for i, tool in enumerate(sorted(self.balls_local)):
            mk = Marker()
            mk.header.frame_id = self.frame
            mk.ns, mk.id, mk.type = "tracker_balls", i, Marker.SPHERE_LIST
            lb = self.latest_balls.get(tool)
            if lb is None or now - lb[2] > 0.3:     # lost: remove the tool's balls
                mk.action = Marker.DELETE
                arr.markers.append(mk)
                continue
            mk.header.stamp = lb[0]
            mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0
            d = 2 * self.balls_local[tool][1]
            mk.scale.x = mk.scale.y = mk.scale.z = d
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = TOOL_RGBA.get(tool, (0.9, 0.9, 0.9, 0.9))
            mk.points = [Point(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in lb[1]]
            arr.markers.append(mk)
        self.pub_balls.publish(arr)

    def diag(self):
        d = DiagnosticArray()
        d.header.stamp = self.get_clock().now().to_msg()
        total = self.frames + self.gaps
        st = DiagnosticStatus(name="spinearm/tracker", hardware_id="d435",
                              level=DiagnosticStatus.OK if self.frames else DiagnosticStatus.STALE,
                              message=f"{self.frames} frames/s")
        st.values = [KeyValue(key="frames_per_s", value=str(self.frames)),
                     KeyValue(key="missed_frames", value=str(self.gaps)),
                     KeyValue(key="miss_pct", value=f"{100 * self.gaps / total:.2f}" if total else "0")]
        self.frames = self.gaps = 0
        d.status = [st]
        self.pub_diag.publish(d)


def main():
    rclpy.init()
    rclpy.spin(TrackerBridge())


if __name__ == "__main__":
    main()
