"""Client side: `ArmRpc` (thin JSON/ZMQ call helper) and the LeRobot `Robot` proxies.

The proxies never touch a serial port; they hold a lease on the arm server and stream joint goals
in degrees. Modelled on lerobot/robots/lekiwi/lekiwi_client.py.
"""

from __future__ import annotations

import getpass
import json
import sys
import time
from functools import cached_property
from typing import Iterator

from lerobot.robots.robot import Robot
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from .config_spinearm import SpineArmConfig, SpineBiArmConfig
from .ident import fetch_ticket
from .protocol import DEFAULT_PORTS, JOINTS, cmd_endpoint


class ArmError(RuntimeError):
    def __init__(self, reply: dict):
        self.code = reply.get("error", "unknown")
        self.reply = reply
        super().__init__(f"{self.code}: {reply.get('detail', '')} {({k: v for k, v in reply.items() if k not in ('ok', 'error', 'detail')} or '')}".strip())


class ArmTimeout(ArmError):
    def __init__(self, endpoint: str):
        super().__init__({"error": "timeout", "detail": f"no reply from {endpoint} — is spine-arm-server running?"})


class ArmRpc:
    """One request → one reply. Lazy-pirate REQ: on timeout the socket is recreated (a REQ socket
    that missed its reply is otherwise stuck)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5601, timeout_ms: int = 1000, context=None):
        import zmq

        self._zmq = zmq
        self.host, self.port = host, port
        self.endpoint = cmd_endpoint(host, port)
        self.timeout_ms = timeout_ms
        self.ctx = context or zmq.Context.instance()
        self.sock = None
        self._open()

    def _open(self):
        zmq = self._zmq
        if self.sock is not None:
            self.sock.close()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(self.endpoint)

    def call(self, **msg) -> dict:
        """Send a command; returns the raw reply dict (check `ok`)."""
        try:
            self.sock.send_string(json.dumps(msg))
            reply = json.loads(self.sock.recv())
        except self._zmq.Again:
            self._open()
            raise ArmTimeout(self.endpoint) from None
        if reply.get("warn"):  # the server tells the lease holder about contention and changed servo gains
            print(f"[spine-arm {self.endpoint}] ⚠ {reply['warn']}", file=sys.stderr, flush=True)
        return reply

    def lease_msg(self, owner: str, force: bool = False) -> dict:
        """The `lease` request, with an ident ticket when the server is local (it proves which account asks)."""
        msg = {"cmd": "lease", "owner": owner, "force": force}
        ticket = fetch_ticket(self.port, getattr(self, "host", "127.0.0.1")) if getattr(self, "port", None) else None
        if ticket:
            msg["ident"] = ticket
        return msg

    def must(self, **msg) -> dict:
        """Like call() but raises ArmError on {"ok": false}."""
        reply = self.call(**msg)
        if not reply.get("ok"):
            raise ArmError(reply)
        return reply

    def close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    # convenience
    def state(self) -> dict:
        return self.must(cmd="state")

    def status(self) -> dict:
        return self.must(cmd="status")

    def lease(self, owner: str, force: bool = False) -> str:
        return self.must(**self.lease_msg(owner, force))["token"]


def watch(host: str = "127.0.0.1", pub_port: int = 5602, timeout_ms: int = 2000, context=None) -> Iterator[dict]:
    """Yield state snapshots from the PUB stream (observers; no lease needed)."""
    import zmq

    ctx = context or zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    # No CONFLATE: it bypasses topic filtering on SUB sockets. Keep a tiny queue and drain to the newest.
    sock.setsockopt(zmq.RCVHWM, 4)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt_string(zmq.SUBSCRIBE, "state ")
    sock.connect(cmd_endpoint(host, pub_port))
    try:
        while True:
            try:
                msg = sock.recv_string()
                while True:  # drop older queued snapshots
                    try:
                        msg = sock.recv_string(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                yield json.loads(msg[len("state "):])
            except zmq.Again:
                yield None
    finally:
        sock.close()


class SpineArm(Robot):
    """One SO-101 behind the arm server. Actions/observations: `<joint>.pos` in degrees (6 joints)."""

    config_class = SpineArmConfig
    name = "spine_arm"

    def __init__(self, config: SpineArmConfig):
        super().__init__(config)
        self.config = config
        cmd_port, pub_port = DEFAULT_PORTS[config.arm]
        self.cmd_port = config.cmd_port or cmd_port
        self.pub_port = config.pub_port or pub_port
        self.owner = config.owner or getpass.getuser()
        self.rpc: ArmRpc | None = None
        self.token: str | None = None
        self._connected = False
        self._last: dict | None = None

    @cached_property
    def _ft(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in JOINTS}

    @cached_property
    def observation_features(self) -> dict[str, type]:
        return dict(self._ft)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict(self._ft)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True  # calibration lives in the server

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if self.config.transport == "ros2":
            from .ros_client import RosArmRpc
            try:
                self.rpc = RosArmRpc(self.config.arm, timeout_s=max(1.0, 3 * self.config.timeout_ms / 1000))
            except ArmError as e:
                raise DeviceNotConnectedError(f"arm {self.config.arm} over ROS 2: {e}") from e
        elif self.config.transport == "zmq":
            self.rpc = ArmRpc(self.config.host, self.cmd_port, self.config.timeout_ms)
        else:
            raise ValueError(f"transport must be 'zmq' or 'ros2', not {self.config.transport!r}")
        try:
            msg = self.rpc.lease_msg(self.owner, self.config.lease_force) if isinstance(self.rpc, ArmRpc) else \
                {"cmd": "lease", "owner": self.owner, "force": self.config.lease_force}
            reply = self.rpc.call(**msg)
        except ArmTimeout as e:
            raise DeviceNotConnectedError(str(e)) from e
        if not reply.get("ok"):
            raise DeviceNotConnectedError(f"arm {self.config.arm}: {ArmError(reply)}")
        self.token = reply["token"]
        if self.config.step_deg is not None:
            self.rpc.must(cmd="set", token=self.token, step_deg=self.config.step_deg)
        if self.config.auto_enable and reply.get("state") == "OFF":
            self.rpc.must(cmd="enable", token=self.token)
        self._connected = True

    @check_if_not_connected
    def get_observation(self) -> dict:
        try:
            self._last = self.rpc.state()
        except ArmTimeout:
            if self._last is None:
                raise
        pos = self._last["pos"]
        return {f"{j}.pos": float(pos.get(j, 0.0)) for j in JOINTS}

    @check_if_not_connected
    def send_action(self, action: dict) -> dict:
        pos = {k[: -len(".pos")]: float(v) for k, v in action.items() if k.endswith(".pos")}
        reply = self.rpc.call(cmd="goal", token=self.token, pos=pos)
        if not reply.get("ok"):
            raise ArmError(reply)
        self._last = {**(self._last or {}), "pos": reply.get("pos", {}), "state": reply.get("state")}
        return {f"{j}.pos": float(v) for j, v in reply["sent"].items()}

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            self.rpc.call(cmd="hold", token=self.token)
            if self.config.release_on_disconnect:
                self.rpc.call(cmd="release", token=self.token)
        except ArmTimeout:
            pass
        finally:
            self.rpc.close()
            self._connected = False

    # --- extras beyond the Robot ABC ---
    def home(self, wait: bool = True, timeout_s: float = 40.0) -> dict:
        reply = self.rpc.must(cmd="home", token=self.token)
        if not wait:
            return reply
        t_end = time.monotonic() + timeout_s
        while time.monotonic() < t_end:
            s = self.rpc.state()
            self.rpc.call(cmd="renew", token=self.token)
            if s["state"] != "HOMING":
                return s
            time.sleep(0.2)
        return self.rpc.state()

    def estop(self) -> dict:
        return self.rpc.must(cmd="estop")

    def hold(self) -> dict:
        return self.rpc.must(cmd="hold", token=self.token)

    def torque_off(self, force: bool = False) -> dict:
        return self.rpc.must(cmd="torque_off", token=self.token, force=force)

    def status(self) -> dict:
        return self.rpc.status()

    def renew(self) -> dict:
        """Keep the lease alive while idle (any token command renews; observations don't)."""
        return self.rpc.must(cmd="renew", token=self.token)

    def clear_faults(self) -> dict:
        return self.rpc.must(cmd="clear_faults", token=self.token)

    @property
    def last(self) -> dict | None:
        """Last server snapshot seen by get_observation/send_action (state, pos, faults, ...)."""
        return self._last


class SpineBiArm(Robot):
    """Both arms; keys prefixed `left_`/`right_` exactly like LeRobot's bi_so_follower."""

    config_class = SpineBiArmConfig
    name = "spine_bi_arm"

    def __init__(self, config: SpineBiArmConfig):
        super().__init__(config)
        self.config = config
        self.left = SpineArm(config.left)
        self.right = SpineArm(config.right)

    @cached_property
    def observation_features(self) -> dict[str, type]:
        return {**{f"left_{k}": v for k, v in self.left.observation_features.items()},
                **{f"right_{k}": v for k, v in self.right.observation_features.items()}}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict(self.observation_features)

    @property
    def is_connected(self) -> bool:
        return self.left.is_connected and self.right.is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def connect(self, calibrate: bool = True) -> None:
        self.left.connect(calibrate)
        try:
            self.right.connect(calibrate)
        except Exception:
            self.left.disconnect()
            raise

    def get_observation(self) -> dict:
        return {**{f"left_{k}": v for k, v in self.left.get_observation().items()},
                **{f"right_{k}": v for k, v in self.right.get_observation().items()}}

    def send_action(self, action: dict) -> dict:
        left = {k[len("left_"):]: v for k, v in action.items() if k.startswith("left_")}
        right = {k[len("right_"):]: v for k, v in action.items() if k.startswith("right_")}
        out = {}
        if left:
            out.update({f"left_{k}": v for k, v in self.left.send_action(left).items()})
        if right:
            out.update({f"right_{k}": v for k, v in self.right.send_action(right).items()})
        return out

    def disconnect(self) -> None:
        for arm in (self.left, self.right):
            if arm.is_connected:
                arm.disconnect()

    def home(self, wait: bool = True) -> dict:
        return {"left": self.left.home(wait), "right": self.right.home(wait)}

    def estop(self) -> dict:
        return {"left": self.left.estop(), "right": self.right.estop()}
