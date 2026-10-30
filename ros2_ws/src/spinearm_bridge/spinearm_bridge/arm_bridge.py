"""arm_bridge — arm server ↔ ROS 2 (ROS2-1-design.md §4).

Observe (stage 1): ZMQ PUB → /<arm>/arm_state (ArmState), /<arm>/joint_states (JointState, model angles), /diagnostics.
Command (stage 2):
  /<arm>/joint_command  JointState, model angles in rad (same convention as joint_states), reliable depth 1 → `goal`
  services (std_srvs/Trigger) ~/acquire ~/release ~/enable ~/hold ~/home ~/torque_off ~/clear_faults ~/estop
  parameters step_deg, accel → `set`
Nothing moves until a client calls ~/acquire (lease) and ~/enable. estop needs no lease. If commands stop, the server's
watchdog holds the arm after 250 ms; the bridge renews the lease while idle so an ~/acquire'd session stays owned.
"""

from __future__ import annotations

import json
import threading
import time

import numpy as np
import rclpy
import zmq
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rcl_interfaces.msg import SetParametersResult
from rclpy.experimental import EventsExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import JointState
from spinearm_msgs.msg import ArmState
from std_srvs.srv import Trigger

from .convert import JOINTS, PORTS, command_to_goal, load_calib, mono_offset_s, split_stamp, state_to_fields
from .rpc import Rpc

CMD_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
SIMPLE = ("enable", "hold", "home", "torque_off", "clear_faults")


class ArmBridge(Node):
    def __init__(self):
        super().__init__("arm_bridge")
        self.arm = self.declare_parameter("arm", "left").value
        host = self.declare_parameter("host", "127.0.0.1").value
        pub_port = self.declare_parameter("pub_port", PORTS[self.arm][1]).value
        calib = self.declare_parameter("calib", f"~/il-robot/calib/handeye_{self.arm}.json").value
        try:
            self.joint_offset = load_calib(calib)["joint_offset_deg"]
        except (OSError, KeyError, ValueError) as e:
            self.get_logger().warn(f"no calibration ({e}) — joint_states in raw arm angles")
            self.joint_offset = [0.0] * 6
        self.sub = zmq.Context.instance().socket(zmq.SUB)
        self.sub.setsockopt(zmq.RCVHWM, 1000)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "state ")
        self.sub.connect(f"tcp://{host}:{pub_port}")
        self.pub_state = self.create_publisher(ArmState, "arm_state", qos_profile_sensor_data)
        self.pub_js = self.create_publisher(JointState, "joint_states", 10)
        self.pub_diag = self.create_publisher(DiagnosticArray, "/diagnostics", 10)
        self.off = mono_offset_s()
        self.last = None
        self.n = 0
        # State stream in its own thread (blocking poll), not an executor timer: keeps the executor free for
        # joint_command so a command is not queued behind a drain pass. rclpy publishers are thread-safe.
        threading.Thread(target=self._drain_loop, daemon=True, name="arm-state").start()
        self.create_timer(1.0, self.diag)

        # ── command side — single-threaded EventsExecutor (see main), so the REQ socket has one user
        self.rpc = Rpc(f"tcp://{host}:{self.declare_parameter('rpc_port', PORTS[self.arm][0]).value}")
        st = self.rpc.call(cmd="status")   # read-only, no lease: joint bounds/neutral for ROS-side clients
        self.declare_parameter("bounds_json", json.dumps(st.get("bounds") or {}))
        self.declare_parameter("neutral_json", json.dumps(st.get("neutral") or {}))
        self.declare_parameter("joint_offset_deg", [float(v) for v in self.joint_offset])
        self.owner = f"ros-{self.arm}"
        self.token: str | None = None
        self.seq = 0
        self.last_cmd = 0.0
        self.cmd_n = self.cmd_err = 0
        self.lat_ms: list[float] = []   # joint_command stamp → bridge callback
        self.rtt_ms: list[float] = []   # goal RPC round trip
        self.create_subscription(JointState, "joint_command", self.on_command, CMD_QOS)
        # ROS 2 has no caller identity (every local account can join the DDS graph), and the bridge leases as its own
        # account (owner1) — so taking the arm over ROS is off unless the owner starts the bridge with allow_acquire:=true.
        self.allow_acquire = bool(self.declare_parameter("allow_acquire", False).value)
        self.create_service(Trigger, "~/acquire", self.srv_acquire)
        self.create_service(Trigger, "~/release", self.srv_release)
        self.create_service(Trigger, "~/estop", self.srv_estop)
        for c in SIMPLE:
            self.create_service(Trigger, f"~/{c}", lambda req, res, c=c: self._simple(c, res))
        self.declare_parameter("step_deg", 0.0)   # 0 = leave the server's value
        self.declare_parameter("accel", 0)
        self.add_on_set_parameters_callback(self.on_params)
        self.create_timer(2.0, self.renew)

    def _drain_loop(self):
        while rclpy.ok():
            if self.sub.poll(200):
                self.drain()

    def drain(self):
        while True:
            try:
                raw = self.sub.recv_string(zmq.NOBLOCK)
            except zmq.Again:
                return
            f = state_to_fields(json.loads(raw[len("state "):]), self.off, self.joint_offset)
            sec, ns = split_stamp(f["stamp"])
            m = ArmState()
            m.header.stamp.sec, m.header.stamp.nanosec = sec, ns
            m.header.frame_id = f"{self.arm}_base_link"
            for k in ("tick", "state", "torque", "bus_ok", "faults", "position", "goal", "command", "load",
                      "temperature", "step_deg", "accel", "t_read", "t_write", "goal_seq", "goal_rx"):
                setattr(m, k, f[k])
            m.name = JOINTS
            self.pub_state.publish(m)
            js = JointState()
            js.header = m.header
            js.name = JOINTS
            js.position = f["joint_rad"]
            self.pub_js.publish(js)
            self.last, self.n = f, self.n + 1

    # ── command side ──────────────────────────────────────────────────────────────────────────────────────────
    def _call(self, **msg) -> dict:
        r = self.rpc.call(**msg)
        if r.get("error") in ("no_lease", "bad_token"):
            self.token = None
        return r

    def on_command(self, m: JointState):
        now = time.time()
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if stamp > 0:
            self.lat_ms.append((now - stamp) * 1000)
        if self.token is None:
            self.cmd_err += 1
            self.get_logger().warn("joint_command without a lease — call ~/acquire and ~/enable",
                                   throttle_duration_sec=2.0)
            return
        try:
            pos = command_to_goal(list(m.name), list(m.position), self.joint_offset)
        except ValueError as e:
            self.cmd_err += 1
            self.get_logger().error(f"bad joint_command: {e}", throttle_duration_sec=2.0)
            return
        self.seq += 1
        t0 = time.perf_counter()
        r = self._call(cmd="goal", token=self.token, pos=pos, seq=self.seq)
        self.rtt_ms.append((time.perf_counter() - t0) * 1000)
        self.last_cmd = now
        if r.get("ok"):
            self.cmd_n += 1
        else:
            self.cmd_err += 1
            self.get_logger().warn(f"goal rejected: {r.get('error')} {r.get('detail', '')}", throttle_duration_sec=2.0)

    def srv_acquire(self, req, res):
        if not self.allow_acquire:
            res.success = False
            res.message = ("acquire is disabled on this bridge (observe-only): ROS callers are not identified. "
                           "Use the ZMQ plugin/spine-arm-ctl, or the owner restarts ros-bridge with allow_acquire:=true")
            self.get_logger().warn("refused ~/acquire (allow_acquire is false)")
            return res
        r = self._call(cmd="lease", owner=self.owner)
        self.token = r.get("token") if r.get("ok") else None
        res.success = bool(self.token)
        res.message = f"lease {'ok' if self.token else 'failed'}: state {r.get('state', r.get('error'))}" + (
            f" (held by {r['holder']})" if r.get("holder") else "")
        return res

    def srv_release(self, req, res):
        if self.token is None:
            res.success, res.message = True, "no lease"
            return res
        r = self._call(cmd="release", token=self.token)
        self.token = None
        res.success, res.message = bool(r.get("ok")), json.dumps(r)[:200]
        return res

    def srv_estop(self, req, res):
        r = self.rpc.call(cmd="estop")   # no token — anyone may stop the arm
        res.success, res.message = bool(r.get("ok")), json.dumps(r)[:200]
        return res

    def _simple(self, cmd: str, res):
        if self.token is None:
            res.success, res.message = False, "no lease — call ~/acquire first"
            return res
        r = self._call(cmd=cmd, token=self.token)
        res.success = bool(r.get("ok"))
        res.message = (r.get("state") or r.get("error") or "") + (f" {r['detail']}" if r.get("detail") else "")
        return res

    def renew(self):
        if self.token is not None and time.time() - self.last_cmd > 2.0:
            self._call(cmd="renew", token=self.token)

    def on_params(self, params):
        msg = {}
        for p in params:
            if p.name == "step_deg" and p.value:
                msg["step_deg"] = float(p.value)
            elif p.name == "accel" and p.value:
                msg["accel"] = int(p.value)
        if not msg:
            return SetParametersResult(successful=True)
        if self.token is None:
            return SetParametersResult(successful=False, reason="no lease — call ~/acquire first")
        r = self._call(cmd="set", token=self.token, **msg)
        return SetParametersResult(successful=bool(r.get("ok")), reason=str(r.get("error", "")))

    def diag(self):
        d = DiagnosticArray()
        d.header.stamp = self.get_clock().now().to_msg()
        st = DiagnosticStatus(name=f"spinearm/{self.arm}", hardware_id=f"so101_{self.arm}")
        if self.last is None:
            st.level, st.message = DiagnosticStatus.STALE, "no arm-server state"
        else:
            st.level = DiagnosticStatus.ERROR if self.last["faults"] else DiagnosticStatus.OK
            st.message = self.last["state"] + (" " + ",".join(self.last["faults"]) if self.last["faults"] else "")
            st.values = [KeyValue(key="rate_hz", value=str(self.n)),
                         KeyValue(key="temp_max_c", value=str(max(self.last["temperature"])))]
        pct = lambda a, q: f"{np.percentile(a, q):.2f}" if a else "-"  # noqa: E731
        st.values += [KeyValue(key="lease", value="held" if self.token else "none"),
                      KeyValue(key="cmd_per_s", value=str(self.cmd_n)),
                      KeyValue(key="cmd_rejected", value=str(self.cmd_err)),
                      KeyValue(key="cmd_latency_ms_p50", value=pct(self.lat_ms, 50)),
                      KeyValue(key="cmd_latency_ms_p99", value=pct(self.lat_ms, 99)),
                      KeyValue(key="goal_rtt_ms_p50", value=pct(self.rtt_ms, 50)),
                      KeyValue(key="goal_rtt_ms_p99", value=pct(self.rtt_ms, 99))]
        self.cmd_n = self.cmd_err = 0
        self.lat_ms, self.rtt_ms = [], []
        self.n = 0
        d.status = [st]
        self.pub_diag.publish(d)


def main():
    rclpy.init()
    # EventsExecutor (rclpy ≥ Jazzy, experimental): callback dispatch without rebuilding a wait set each spin —
    # the default executors added ~3 ms p50 / ~11 ms p99 between joint_command publish and this callback.
    ex = EventsExecutor()
    ex.add_node(ArmBridge())
    ex.spin()


if __name__ == "__main__":
    main()
