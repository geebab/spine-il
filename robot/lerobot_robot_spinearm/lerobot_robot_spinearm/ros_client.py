"""`RosArmRpc` — the `ArmRpc` surface (call / must / state / status / lease) over ROS 2 instead of ZMQ.

Transport layer of the LeRobot RFC split (ROS2-1-design.md §4 stage 2): the control code keeps talking "arm-server
commands" and does not care whether they travel over ZMQ or through `spinearm_bridge/arm_bridge`:

    goal                  → /<arm>/joint_command (JointState, model rad = arm deg + calibration joint offset)
    lease / enable / hold / home / torque_off / clear_faults / release / estop → /<arm>/arm_bridge/~<name> (Trigger)
    set step_deg / accel  → arm_bridge parameters
    state / status        ← latest /<arm>/arm_state (+ bridge parameters bounds_json, neutral_json)

The bridge owns the server lease, so `lease()` returns a placeholder token and `renew` is a no-op (the bridge renews).
A goal is fire-and-forget: its reply is synthesised from the last arm_state (ESTOP / FAULT / OFF → the same errors
the server would give), so a follower still disengages when the arm stops accepting goals.

Needs the ROS environment (`source /opt/ros/jazzy/setup.bash` and the ros2_ws overlay) in the calling process.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time

from .client import ArmError
from .protocol import JOINTS

TRIGGERS = ("acquire", "release", "enable", "hold", "home", "torque_off", "clear_faults", "estop")
REFUSING = {"ESTOP": "estop", "FAULT": "fault", "OFF": "off"}


class RosArmRpc:
    def __init__(self, arm: str = "left", timeout_s: float = 3.0):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.parameter_client import AsyncParameterClient
        from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import JointState
        from spinearm_msgs.msg import ArmState
        from std_srvs.srv import Trigger

        self.arm, self.timeout_s = arm, timeout_s
        self._JointState, self._Trigger = JointState, Trigger
        if not rclpy.ok():
            rclpy.init()
        self.node = rclpy.create_node(f"spinearm_client_{arm}_{os.getpid()}")
        self.pub = self.node.create_publisher(JointState, f"/{arm}/joint_command",
                                              QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        self._st = None
        self._st_evt = threading.Event()
        self.node.create_subscription(ArmState, f"/{arm}/arm_state", self._on_state, qos_profile_sensor_data)
        self.cli = {k: self.node.create_client(Trigger, f"/{arm}/arm_bridge/{k}") for k in TRIGGERS}
        self.params = AsyncParameterClient(self.node, f"/{arm}/arm_bridge")
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        self._spin = threading.Thread(target=self.ex.spin, daemon=True, name="ros-arm-rpc")
        self._spin.start()

        if not self.params.wait_for_services(timeout_s):
            raise ArmError({"error": "timeout", "detail": f"/{arm}/arm_bridge parameters — is ros-bridge running?"})
        p = self._get_params(["joint_offset_deg", "bounds_json", "neutral_json"])
        self.offset = dict(zip(JOINTS, p["joint_offset_deg"]))
        self.bounds = json.loads(p["bounds_json"] or "{}")
        self.neutral = json.loads(p["neutral_json"] or "{}")
        if not self._st_evt.wait(timeout_s):
            raise ArmError({"error": "timeout", "detail": f"no /{arm}/arm_state"})

    # ── plumbing ────────────────────────────────────────────────────────────────────────────────────────────────
    def _on_state(self, m) -> None:
        self._st = m
        self._st_evt.set()

    def _wait(self, fut):
        t0 = time.monotonic()
        while not fut.done():
            if time.monotonic() - t0 > self.timeout_s:
                raise ArmError({"error": "timeout", "detail": "ROS call"})
            time.sleep(0.002)
        return fut.result()

    def _get_params(self, names: list[str]) -> dict:
        from rclpy.parameter import parameter_value_to_python
        r = self._wait(self.params.get_parameters(names))
        return {n: parameter_value_to_python(v) for n, v in zip(names, r.values)}

    def _trigger(self, name: str) -> dict:
        c = self.cli[name]
        if not c.wait_for_service(self.timeout_s):
            return {"ok": False, "error": "timeout", "detail": f"/{self.arm}/arm_bridge/{name}"}
        r = self._wait(c.call_async(self._Trigger.Request()))
        out = {"ok": bool(r.success), "detail": r.message}
        if not r.success:
            out["error"] = "no_lease" if "no lease" in r.message else r.message.split(" ")[0] or name
        return out

    # ── ArmRpc surface ──────────────────────────────────────────────────────────────────────────────────────────
    def call(self, **msg) -> dict:
        cmd = msg.get("cmd")
        if cmd == "goal":
            st = self._st
            if st is not None and st.state in REFUSING:
                return {"ok": False, "error": REFUSING[st.state], "state": st.state}
            names = [j for j in JOINTS if j in msg["pos"]]
            # the server clamps to its bounds; do the same here so `sent` matches what it will receive
            sent = {j: min(max(float(msg["pos"][j]), self.bounds[j][0]), self.bounds[j][1]) if j in self.bounds
                    else float(msg["pos"][j]) for j in names}
            m = self._JointState()
            now = time.time()
            m.header.stamp.sec, m.header.stamp.nanosec = int(now), int((now % 1.0) * 1e9)
            m.header.frame_id = str(msg.get("seq", ""))
            m.name = names
            m.position = [math.radians(sent[j] + self.offset[j]) for j in names]
            self.pub.publish(m)
            return {"ok": True, "state": st.state if st else None, "rx": None, "sent": sent,
                    "pos": dict(zip(st.name, st.position)) if st else {}}
        if cmd in ("state", "status"):
            return {"ok": True, **self.state()} if cmd == "state" else {"ok": True, **self.status()}
        if cmd == "lease":
            return {**self._trigger("acquire"), "token": "ros", "state": self._st.state if self._st else None}
        if cmd == "renew":
            return {"ok": True}
        if cmd == "set":
            from rclpy.parameter import Parameter
            ps = []
            if "step_deg" in msg:
                ps.append(Parameter("step_deg", value=float(msg["step_deg"])))
            if "accel" in msg:
                ps.append(Parameter("accel", value=int(msg["accel"])))
            r = self._wait(self.params.set_parameters(ps))
            bad = [x.reason for x in r.results if not x.successful]
            return {"ok": not bad, **({"error": bad[0]} if bad else {})}
        if cmd in TRIGGERS:
            return self._trigger(cmd)
        return {"ok": False, "error": "bad_request", "detail": f"{cmd!r} not supported over ROS"}

    def must(self, **msg) -> dict:
        r = self.call(**msg)
        if not r.get("ok"):
            raise ArmError(r)
        return r

    def state(self) -> dict:
        m = self._st
        return {"state": m.state, "torque": m.torque, "faults": list(m.faults), "bus_ok": m.bus_ok, "tick": m.tick,
                "pos": dict(zip(m.name, m.position)), "step_deg": m.step_deg, "accel": m.accel,
                "temp": dict(zip(m.name, m.temperature)), "load": dict(zip(m.name, m.load))}

    def status(self) -> dict:
        return {**self.state(), "bounds": self.bounds, "neutral": self.neutral}

    def lease(self, owner: str, force: bool = False) -> str:
        self.must(cmd="lease", owner=owner)
        return "ros"

    def close(self) -> None:
        """Stop the spin thread before the interpreter exits — a daemon thread still inside rclpy at exit aborts the
        process ("terminate called without an active exception", rc −6)."""
        import rclpy
        self.ex.shutdown(timeout_sec=1.0)
        self._spin.join(timeout=2.0)
        self.node.destroy_node()
        rclpy.try_shutdown()
