"""recording_player — plays a viewer session recording (recorder v2 JSONL, rec-2.3.x) into the ROS graph, so Foxglove
shows it next to the robot model and the live tracker. Visualisation only: not a timing tool (clock choice, replay
determinism and arm execution are solved in their own code — fid decision B4). Parsing: recording.py, pacing and cut
bookkeeping: replay_timeline.py (both ROS-free).

  ros2 launch spinearm_bridge replay.launch.py file:=/path/s_….jsonl [rate:=0.5] [loop:=true] [clock_mode:=sim]
  ros2 run spinearm_bridge recording_player --ros-args -p file:=/path/s_….jsonl -p rate:=0.5

Published (<ns> = parameter `namespace`, default /replay; '/' or '' = the live names — never next to tracker_bridge):
  <ns>/tracker/<tool>/tracked  TrackedPose, camera frame, stamp = the row's camera-frame time (tExp)
  <ns>/tracker/<tool>/pose     PoseStamped (filtered pose)
  <ns>/tracker/balls           MarkerArray: raw pose × the tool's ball model (like tracker_bridge, other colours)
  <ns>/drill/cuts              MarkerArray SPHERE_LIST, scene frame, cumulative, sphere radius = cut rMm (latched,
                               frame-locked, chunks of ≤ 2000 stamps, re-sent ≤ 5 Hz + everything every 2 s)
  <ns>/drawn                   MarkerArray, scene frame: each tool as the viewer drew it (at the viewer's clock)
  /tf_static (or /tf)          the recording's registration: camera_frame → scene_frame (tf_parent:=camera, default —
                               the camera frame keeps calib_tf's <arm>_base_link parent) or scene_frame → camera_frame
  /clock                       clock_mode:=sim only: recording time (consumers run with use_sim_time:=true)
  /diagnostics                 spinearm/replay: progress, pacing lag, screen staleness
Clock: wall (default) stamps = the wall time playback started + (t − window start) / rate, live-like, so the replay sits
next to live data; sim stamps = the recorded time and publishes /clock (the rosbag convention, for a standalone replay).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.logging import get_logger
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from spinearm_msgs.msg import TrackedPose
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from . import recording as rec
from . import replay_timeline as tl
from .convert import split_stamp_ms

TICK_S = 0.005            # playback tick; stamps come from the schedule, so the tick only bounds the publish delay
MAX_PER_TICK = 500        # events per tick; a late player catches up over the next ticks (nothing is dropped)
BALL_STALE_MS = 300.0     # recording ms without a pose → that tool's balls disappear (tracker_bridge: 0.3 s)
CUTS_MIN_PERIOD_S = 0.2   # cuts: the chunks that grew are re-sent at most 5 Hz …
CUTS_REPUBLISH_S = 2.0    # … and every chunk every 2 s, for a subscriber that missed the latched copy
SHAFT_M = 0.08            # drawn tool: shaft line drawn behind the tip / ball
REPLAY_RGBA = {"endo": (0.2, 0.75, 1.0, 0.85), "drill": (1.0, 0.85, 0.15, 0.85)}   # ≠ tracker_bridge's live colours
CUT_RGBA = {"bone": (1.0, 0.45, 0.15, 0.9), "soft": (0.9, 0.25, 0.45, 0.7)}
DRILL_ON_RGBA = (1.0, 0.1, 0.1, 0.95)


def _fill_pose(p, pos, q):
    p.position.x, p.position.y, p.position.z = pos
    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q


def _pct(a, q):
    return f"{np.percentile(a, q):.1f}" if a else "-"


def _point(p) -> Point:
    return Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))


class RecordingPlayer(Node):
    def __init__(self):
        super().__init__("recording_player")
        self.path = os.path.expanduser(self.declare_parameter("file", "").value)
        self.rate = float(self.declare_parameter("rate", 1.0).value)
        self.loop = bool(self.declare_parameter("loop", False).value)
        start_s = float(self.declare_parameter("start_s", 0.0).value)
        end_s = float(self.declare_parameter("end_s", 0.0).value)          # ≤ 0: to the end
        self.ns = rec.topic_namespace(self.declare_parameter("namespace", "/replay").value)
        self.clock_mode = self.declare_parameter("clock_mode", "wall").value
        self.scene_frame = self.declare_parameter("scene_frame", "scene").value
        self.cam_frame = self.declare_parameter("camera_frame", "tracker_cam").value
        self.tf_parent = self.declare_parameter("tf_parent", "camera").value
        show_drawn = bool(self.declare_parameter("drawn", True).value)
        clock_hz = float(self.declare_parameter("clock_hz", 100.0).value)
        models = json.loads(self.declare_parameter("tool_models_json", json.dumps(rec.REPLAY_TOOL_MODELS)).value)
        if not self.path:
            raise ValueError("parameter 'file' (a recorder v2 .jsonl) is required")
        if self.clock_mode not in ("wall", "sim"):
            raise ValueError(f"clock_mode must be 'wall' or 'sim', got {self.clock_mode!r}")
        if self.tf_parent not in ("camera", "scene", "none"):
            raise ValueError(f"tf_parent must be 'camera', 'scene' or 'none', got {self.tf_parent!r}")
        if self.scene_frame == self.cam_frame:
            raise ValueError("scene_frame and camera_frame must differ")
        if self.get_parameter("use_sim_time").value:
            # this node paces itself on the wall clock and is the /clock source; on sim time its timers would wait
            # for a /clock that only they publish
            self.get_logger().warning("use_sim_time turned off for recording_player (it publishes /clock itself)")
            self.set_parameters([Parameter("use_sim_time", Parameter.Type.BOOL, False)])
        if not self.ns:
            self.get_logger().warning("namespace '/' — publishing on the live topic names; do not run tracker_bridge")

        self.rec = rec.load(self.path, drawn=show_drawn)
        for line in rec.summary(self.rec):
            self.get_logger().info(line)
        for w in self.rec.warnings:
            self.get_logger().warning(w)
        self.pre, self.events, t_a, t_b = tl.window(self.rec, start_s, end_s)
        self.sched = tl.Schedule([e.t for e in self.events], t_a, t_b, self.rate)
        if self.loop and t_b - t_a < 1.0:
            self.get_logger().warning("window shorter than 1 ms — loop off")
            self.loop = False

        self.pubs: dict[str, tuple] = {}
        for tool in self.rec.tools:
            self._pub(tool)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_balls = self.create_publisher(MarkerArray, f"{self.ns}/tracker/balls", 10)
        self.pub_cuts = self.create_publisher(MarkerArray, f"{self.ns}/drill/cuts", latched)
        self.pub_drawn = self.create_publisher(MarkerArray, f"{self.ns}/drawn", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.pub_clock = self.create_publisher(Clock, "/clock", 10) if self.clock_mode == "sim" else None
        tf_on = self.tf_parent != "none"
        self.tf_static = StaticTransformBroadcaster(self) if tf_on and self.rec.static_T is not None else None
        self.tf_dynamic = TransformBroadcaster(self) if tf_on and any(e.kind == "tf" for e in self.events) else None

        # ball geometry per tool (metres, tool frame), checked against the calibration the recording was made with
        self.balls_local: dict[str, tuple[np.ndarray, float]] = {}
        for tool, path in models.items():
            try:
                pts, radius, sha = rec.load_ball_model(path)
            except (OSError, KeyError, ValueError) as e:
                self.get_logger().warning(f"no ball model for {tool}: {e}")
                continue
            want = self.rec.tool_calib_sha(tool)
            if want and sha != want:
                self.get_logger().warning(f"{tool} ball model {path} is sha {sha}, the recording used {want} — "
                                          "balls may show another tool's geometry")
            self.balls_local[tool] = (pts, radius)
        self.latest_balls: dict[str, tuple] = {}   # tool → (stamp msg, points (n, 3) m, recording ms)

        self.cuts = tl.CutAccumulator()
        self.cut_pts: dict[int, list] = {}         # marker id → its Points, each stamp converted once
        self.cut_style: dict[int, tuple] = {}      # marker id → (target, radius m)
        self.cuts_dirty, self.last_cut_pub, self.last_cut_full = True, -1e9, -1e9
        self.last_tf_t: float | None = None        # recording ms of the last per-row TF sent
        self.n_pub_s, self.loops, self.done, self.last_ev = 0, 0, False, "-"
        self.lag_ms: list[float] = []                               # publish time − scheduled time
        self.screen_lag: dict[str, list[float]] = {}                # row tExp − drawn.tExp, per tool
        self.clock_period, self.last_clock = 1.0 / max(clock_hz, 1.0), -1.0
        self._start(time.monotonic())
        edge = {"none": "off", "camera": f"{self.cam_frame} → {self.scene_frame}",
                "scene": f"{self.scene_frame} → {self.cam_frame}"}[self.tf_parent]
        self.get_logger().info(
            f"playing {len(self.events)} events, {(t_b - t_a) / 1000:.2f} s at ×{self.rate:g} ({self.clock_mode} clock"
            f"{', loop' if self.loop else ''}); topics under '{self.ns or '/'}'; registration TF {edge}"
            f"{' (per row)' if self.tf_dynamic else ''}")
        self.create_timer(TICK_S, self.tick)
        self.create_timer(1 / 30, self.publish_markers)
        self.create_timer(1.0, self.diag)

    # ── time ──────────────────────────────────────────────────────────────────────────────────────────────────────
    def _ms(self, t_ms: float) -> float:
        """Recording ms → the ms this node stamps with (sim: unchanged; wall: this run's wall clock)."""
        return t_ms if self.clock_mode == "sim" else self.sched.wall_ms(t_ms)

    def _stamp(self, stamp, t_ms: float):
        stamp.sec, stamp.nanosec = split_stamp_ms(self._ms(t_ms))

    def _scene_t(self, t_ms: float) -> float:
        """Stamp time for a scene-frame marker: with per-row registration TF, never after the last TF sent (a viewer
        looks the marker's frame up at its stamp)."""
        return t_ms if self.last_tf_t is None else min(t_ms, self.last_tf_t)

    def _publish_clock(self, t_ms: float):
        c = Clock()
        c.clock.sec, c.clock.nanosec = split_stamp_ms(t_ms)
        self.pub_clock.publish(c)

    def _start(self, now: float):
        self.sched.start(now, time.time() * 1000.0)
        self.cuts.clear()
        for e in self.pre:                 # cuts before the window: the bone as it was when playback starts
            self.cuts.apply(e.data)
        self.cuts_dirty, self.last_tf_t = True, None
        self.latest_balls.clear()
        if self.pub_clock is not None:
            self._publish_clock(self.sched.t_start)
            self.last_clock = now
        if self.tf_static is not None:     # again on every loop: a viewer that cleared its TF on the jump gets it back
            self.tf_static.sendTransform([self._tf(self.rec.static_T, self.sched.t_start)])

    # ── playback ──────────────────────────────────────────────────────────────────────────────────────────────────
    def tick(self):
        now = time.monotonic()
        if self.sched.finished(now):
            if not self.loop:
                if not self.done:
                    self.done = True
                    self.get_logger().info(f"replay finished ({len(self.events)} events)")
                return
            self.loops += 1
            self._start(now)               # sim clock jumps back: sim-time consumers clear their TF buffers
        due = self.sched.due(now, MAX_PER_TICK)
        if self.pub_clock is not None and (len(due) or now - self.last_clock >= self.clock_period):
            self._publish_clock(self.sched.position_ms(now))   # before the messages it covers
            self.last_clock = now
        for i in due:
            e = self.events[i]
            self.lag_ms.append((now - self.sched.due_mono_s(e.t)) * 1000.0)
            if e.kind == "pose":
                self._pose(e)
            elif e.kind == "tf":
                if self.tf_dynamic is not None:
                    self.tf_dynamic.sendTransform(self._tf(e.data["T"], e.t))
                    self.last_tf_t = e.t
            elif e.kind == "cut":
                self.cuts.apply(e.data)    # numpy only here; messages are built at ≤ 5 Hz in publish_markers
                self.cuts_dirty = True
            elif e.kind == "drawn":
                self._drawn(e)
            elif e.kind == "ev":
                self.last_ev = str(e.data.get("ev"))
        self.n_pub_s += len(due)

    def _pub(self, tool):
        if tool not in self.pubs:
            base = f"{self.ns}/tracker/{tool}"
            self.pubs[tool] = (self.create_publisher(TrackedPose, f"{base}/tracked", qos_profile_sensor_data),
                               self.create_publisher(PoseStamped, f"{base}/pose", qos_profile_sensor_data))
        return self.pubs[tool]

    def _pose(self, e):
        f = e.data["fields"]
        tp, ps = self._pub(f["tool"])
        m = TrackedPose()
        self._stamp(m.header.stamp, f["stamp_ms"])
        m.header.frame_id = self.cam_frame
        for k in ("tool", "frame", "lost", "visible", "rmse", "still", "held"):
            setattr(m, k, f[k])
        if f["t_send_ms"]:                 # t_arrival / t_pose are not in the recording (0)
            m.t_send = self._ms(f["t_send_ms"]) / 1000.0
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
                pts = rec.balls_in_camera(self.balls_local[f["tool"]][0], f["pos_raw"], f["quat_raw"])
                self.latest_balls[f["tool"]] = (m.header.stamp, pts, e.t)
        tp.publish(m)
        if e.data.get("screen_lag_ms") is not None:
            self.screen_lag.setdefault(f["tool"], []).append(e.data["screen_lag_ms"])

    def _tf(self, T_scene_cam: np.ndarray, t_ms: float) -> TransformStamped:
        scene_parent, trans, quat = rec.registration_tf(T_scene_cam, self.tf_parent)
        t = TransformStamped()
        self._stamp(t.header.stamp, t_ms)
        if scene_parent:
            t.header.frame_id, t.child_frame_id = self.scene_frame, self.cam_frame
        else:
            t.header.frame_id, t.child_frame_id = self.cam_frame, self.scene_frame
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = trans
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat
        return t

    # ── markers ───────────────────────────────────────────────────────────────────────────────────────────────────
    def _marker(self, ns: str, mid: int, mtype: int, t_ms: float, rgba, frame: str) -> Marker:
        mk = Marker()
        mk.header.frame_id = frame
        self._stamp(mk.header.stamp, t_ms)
        mk.ns, mk.id, mk.type, mk.action = ns, mid, mtype, Marker.ADD
        mk.pose.orientation.w = 1.0
        mk.color.r, mk.color.g, mk.color.b, mk.color.a = (float(v) for v in rgba)
        return mk

    def _drawn(self, e):
        d = e.data
        tool = d["tool"]
        rgba = DRILL_ON_RGBA if tool == "drill" and d["drill_on"] else REPLAY_RGBA.get(tool, (0.9, 0.9, 0.9, 0.85))
        if d["lost"]:                      # the viewer keeps drawing the last pose after pose_lost
            rgba = (*rgba[:3], 0.35)
        base = 10 * (sorted(REPLAY_RGBA).index(tool) if tool in REPLAY_RGBA else 9)
        t = self._scene_t(e.t)
        arr = MarkerArray()
        sph = self._marker("replay_drawn", base, Marker.SPHERE, t, rgba, self.scene_frame)
        sph.frame_locked = True            # like the cut markers: the scene TF may be dynamic
        size = 2 * d["radius_m"] if tool == "drill" and d.get("radius_m") else 0.004   # radius as of that row
        sph.scale.x = sph.scale.y = sph.scale.z = float(size)
        sph.pose.position.x, sph.pose.position.y, sph.pose.position.z = d["p"]
        arr.markers.append(sph)
        if d["dir"] is not None:           # the shaft behind the scope tip / drill ball
            ln = self._marker("replay_drawn", base + 1, Marker.LINE_STRIP, t, rgba, self.scene_frame)
            ln.frame_locked = True
            ln.scale.x = 0.0015
            p, u = np.asarray(d["p"]), np.asarray(d["dir"])
            ln.points = [_point(v) for v in (p - SHAFT_M * u, p)]
            arr.markers.append(ln)
        self.pub_drawn.publish(arr)

    def publish_markers(self):
        now = time.monotonic()
        pos = self.sched.position_ms(now)
        arr = MarkerArray()
        for i, tool in enumerate(sorted(self.balls_local)):
            mk = Marker()
            mk.header.frame_id = self.cam_frame
            mk.ns, mk.id, mk.type = "replay_balls", i, Marker.SPHERE_LIST
            lb = self.latest_balls.get(tool)
            if lb is None or pos - lb[2] > BALL_STALE_MS:   # lost / not started: remove the tool's balls
                mk.action = Marker.DELETE
                arr.markers.append(mk)
                continue
            mk.header.stamp = lb[0]
            mk.action = Marker.ADD
            mk.pose.orientation.w = 1.0
            mk.scale.x = mk.scale.y = mk.scale.z = 2 * self.balls_local[tool][1]
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = REPLAY_RGBA.get(tool, (0.9, 0.9, 0.9, 0.85))
            mk.points = [_point(p) for p in lb[1]]
            arr.markers.append(mk)
        if arr.markers:
            self.pub_balls.publish(arr)
        if now - self.last_cut_full >= CUTS_REPUBLISH_S:
            self._publish_cuts(now, pos, full=True)
        elif self.cuts_dirty and now - self.last_cut_pub >= CUTS_MIN_PERIOD_S:
            self._publish_cuts(now, pos, full=False)

    def _publish_cuts(self, now: float, pos: float, full: bool):
        """Only the stamps added since the last call become Points; only the chunks they touched are re-sent (all of
        them when `full`). Stamp = where playback is, frame-locked: the stamps are fixed in the scene, so the latest
        registration is the right one (a per-row TF has no history back to an old cut)."""
        arr = MarkerArray()
        new = self.cuts.take_new()
        grown = {mid for mid, _, _ in new}
        for mid in self.cuts.take_removed():
            self.cut_pts.pop(mid, None)
            self.cut_style.pop(mid, None)
            if mid not in grown:
                mk = Marker()
                mk.header.frame_id = self.scene_frame
                mk.ns, mk.id, mk.action = "replay_cuts", mid, Marker.DELETE
                arr.markers.append(mk)
        for mid, (target, _label, radius), pts in new:
            self.cut_pts.setdefault(mid, []).extend(_point(p) for p in pts)
            self.cut_style[mid] = (target, radius)
        t = self._scene_t(pos)
        for mid in sorted(self.cut_pts if full else grown):
            target, radius = self.cut_style[mid]
            mk = self._marker("replay_cuts", mid, Marker.SPHERE_LIST, t, CUT_RGBA.get(target, (0.8, 0.8, 0.8, 0.8)),
                              self.scene_frame)
            mk.frame_locked = True
            mk.scale.x = mk.scale.y = mk.scale.z = 2 * radius
            mk.points = list(self.cut_pts[mid])
            arr.markers.append(mk)
        self.pub_cuts.publish(arr)
        self.cuts_dirty, self.last_cut_pub = False, now
        if full:
            self.last_cut_full = now

    # ── diagnostics ───────────────────────────────────────────────────────────────────────────────────────────────
    def _live_clash(self) -> str:
        """With the live topic names, anyone else publishing them (tracker_bridge) makes replay and live data mix."""
        if self.ns:
            return ""
        busy = [t for t in (f"/tracker/{tool}/tracked" for tool in self.rec.tools) if self.count_publishers(t) > 1]
        return f"another publisher on {', '.join(busy)}" if busy else ""

    def diag(self):
        now = time.monotonic()
        d = DiagnosticArray()
        if self.clock_mode == "sim":
            d.header.stamp.sec, d.header.stamp.nanosec = split_stamp_ms(self.sched.position_ms(now))
        else:
            d.header.stamp = self.get_clock().now().to_msg()
        prog = self.sched.progress(now)
        t_s = (self.sched.position_ms(now) - self.sched.t_start) / 1000.0
        span_s = (self.sched.t_end - self.sched.t_start) / 1000.0
        late = bool(self.lag_ms) and max(self.lag_ms) > 100.0
        clash = self._live_clash()
        st = DiagnosticStatus(name="spinearm/replay", hardware_id=self.rec.session_id or "recording")
        st.level = (DiagnosticStatus.ERROR if clash else DiagnosticStatus.STALE if self.done else
                    DiagnosticStatus.WARN if late or self.rec.warnings else DiagnosticStatus.OK)
        st.message = (f"{'done' if self.done else 'playing'} {100 * prog:.0f}% {t_s:.1f}/{span_s:.1f} s "
                      f"×{self.rate:g}" + (f" loop {self.loops}" if self.loop else "") + (" LATE" if late else "")
                      + (f" — {clash}" if clash else ""))
        st.values = [KeyValue(key="progress_pct", value=f"{100 * prog:.1f}"),
                     KeyValue(key="recording_s", value=f"{t_s:.3f}"),
                     KeyValue(key="window_s", value=f"{span_s:.3f}"),
                     KeyValue(key="events_published", value=f"{self.sched.cursor}/{len(self.events)}"),
                     KeyValue(key="events_per_s", value=str(self.n_pub_s)),
                     KeyValue(key="lag_ms_p50", value=_pct(self.lag_ms, 50)),
                     KeyValue(key="lag_ms_max", value=f"{max(self.lag_ms):.1f}" if self.lag_ms else "-")]
        st.values += [KeyValue(key=f"screen_lag_ms_p50_{tool}", value=_pct(v, 50))
                      for tool, v in sorted(self.screen_lag.items())]
        st.values += [KeyValue(key="cut_rows", value=str(self.cuts.n_rows)),
                      KeyValue(key="cut_stamps", value=str(self.cuts.n_points)),
                      KeyValue(key="cut_mod_voxels", value=str(self.cuts.mod)),
                      KeyValue(key="last_event", value=self.last_ev),
                      KeyValue(key="clock_mode", value=self.clock_mode),
                      KeyValue(key="loops", value=str(self.loops)),
                      KeyValue(key="warnings", value=" | ".join(self.rec.warnings) or "-"),
                      KeyValue(key="file", value=self.path)]
        self.lag_ms, self.screen_lag, self.n_pub_s = [], {}, 0
        d.status = [st]
        self.pub_diag.publish(d)


def main():
    rclpy.init()
    try:
        node = RecordingPlayer()
    except (OSError, ValueError) as e:     # missing file, bad parameter: one line, not a traceback
        get_logger("recording_player").fatal(f"cannot start: {e}")
        rclpy.try_shutdown()
        return 1
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
