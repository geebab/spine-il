"""Client side of the tracker's `--tap` stream (UDP, JSON per tool per camera frame) and the arm's PUB stream.

Tap message (tracker_service.py `_ToolRig._tap_send`), all times wall-clock ms (time.time()·1000):
  tool, fn (camera frame number), ts (mid-exposure), ta (frame arrived in the tracker), tp (this tool's pose
  done), tsend, tsrc ('global-mid' = hardware mid-exposure; 'global' / 'arrival' = fallbacks),
  pos / quat (wxyz, tool→cam) / tip = the output the viewer gets (filtered), raw_pos / raw_quat / raw_tip =
  before the filter, or lost=1.
Both endoscope constellations are the same model; the tracker names them `endo` / `endo2` by screen side,
which flips when they cross. Consumers must not trust the names — `pick_robot()` tells them apart by
geometry (the arm's FK prediction, else continuity).
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque

import numpy as np

from .handeye import quat_to_R

ENDO_TOOLS = ("endo", "endo2")


def wall_ms() -> float:
    return time.time() * 1000.0


def mono_offset_ms() -> float:
    """wall − monotonic in ms (both CLOCK_REALTIME / CLOCK_MONOTONIC of this host; constant unless the wall
    clock is stepped). Median of a few reads to dodge a preemption between the two calls."""
    v = []
    for _ in range(5):
        a = time.time()
        m = time.monotonic()
        b = time.time()
        v.append(((a + b) / 2 - m) * 1000.0)
    return float(np.median(v))


class TapReceiver:
    """Blocking UDP receiver; stamps `trecv` (wall ms) on every message."""

    def __init__(self, port: int = 9880, host: str = "127.0.0.1", rcvbuf: int = 1 << 20):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        self.sock.bind((host, port))

    def recv(self, timeout_s: float = 0.5) -> dict | None:
        self.sock.settimeout(timeout_s)
        try:
            data, _ = self.sock.recvfrom(65536)
        except socket.timeout:
            return None
        m = json.loads(data)
        m["trecv"] = round(wall_ms(), 3)
        return m

    def flush(self) -> int:
        """Drop everything queued (e.g. while the caller was busy moving the arm); returns how many."""
        self.sock.setblocking(False)
        n = 0
        try:
            while True:
                self.sock.recvfrom(65536)
                n += 1
        except BlockingIOError:
            pass
        finally:
            self.sock.setblocking(True)
        return n

    def close(self) -> None:
        self.sock.close()


class FrameAssembler:
    """Groups the per-tool messages of one camera frame; `add()` returns {tool: msg} once every tool in
    `tools` has reported for that frame number (the tracker sends a message per rig per frame, lost or not)."""

    def __init__(self, tools=ENDO_TOOLS):
        self.tools = set(tools)
        self.fn = None
        self.msgs: dict[str, dict] = {}

    def add(self, m: dict) -> dict[str, dict] | None:
        if m.get("tool") not in self.tools:
            return None
        if m.get("fn") != self.fn:
            self.fn, self.msgs = m.get("fn"), {}
        self.msgs[m["tool"]] = m
        if set(self.msgs) >= self.tools:
            out, self.msgs = self.msgs, {}
            return out
        return None


def pose_of(m: dict | None, tip_local_m: np.ndarray, raw: bool = False) -> dict | None:
    """Tap message → {R, pos_m, tip_m, ts} in the camera frame (metres); None if lost."""
    if m is None or m.get("lost") or m.get("pos") is None:
        return None
    pk, qk = ("raw_pos", "raw_quat") if raw and m.get("raw_pos") is not None else ("pos", "quat")
    R = quat_to_R(m[qk])
    p = np.asarray(m[pk], float) / 1000.0
    return {"R": R, "pos": p, "tip": R @ tip_local_m + p, "ts": m.get("ts"), "fn": m.get("fn"), "tool": m["tool"]}


def pick_robot(poses: dict[str, dict | None], ref_tip_m: np.ndarray | None, gate_m: float = 0.06,
               margin_m: float = 0.05):
    """Which visible endoscope constellation rides the arm.

    ref_tip_m: where the arm's tip should be (its last fix a frame ago, else the FK prediction). The nearer
    constellation is the arm's if it is within gate_m + margin_m and the other one is at least margin_m farther;
    a lone constellation is the arm's within gate_m, the hand's beyond gate_m + margin_m. Returns
    (robot_pose | None, hand_pose | None, status) with status 'ok', 'robot_lost', 'hand_lost', 'none' or
    'ambiguous'. A strict absolute gate (40 mm) stopped a run when the arm's marker was re-acquired 47 mm off the
    prediction while the hand was 185 mm away (2026-09-23) — the relative test keeps that case.
    """
    vis = [p for p in poses.values() if p is not None]
    if not vis:
        return None, None, "none"
    if ref_tip_m is None:
        return None, None, "ambiguous"
    d = [float(np.linalg.norm(p["tip"] - ref_tip_m)) for p in vis]
    if len(vis) == 1:
        if d[0] <= gate_m:
            return vis[0], None, "hand_lost"
        if d[0] >= gate_m + margin_m:
            return None, vis[0], "robot_lost"
        return None, None, "ambiguous"
    i = int(np.argmin(d))
    j = 1 - i
    if d[i] <= gate_m + margin_m and d[j] - d[i] >= margin_m:
        return vis[i], vis[j], "ok"
    return None, None, "ambiguous"


class StateLog:
    """Background SUB on the arm's PUB port: keeps every state snapshot (no draining) with `trecv` (wall ms),
    and the latest one for the control loop."""

    def __init__(self, host: str = "127.0.0.1", pub_port: int = 5602, sink=None):
        import zmq

        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.RCVHWM, 1000)
        self.sock.setsockopt(zmq.RCVTIMEO, 200)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "state ")
        self.sock.connect(f"tcp://{host}:{pub_port}")
        self.sink = sink
        self.latest: dict | None = None
        self.hist: deque = deque(maxlen=64)
        self._stop = threading.Event()
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        import zmq

        while not self._stop.is_set():
            try:
                raw = self.sock.recv_string()
            except zmq.Again:
                continue
            s = json.loads(raw[len("state "):])
            s["trecv"] = round(wall_ms(), 3)
            self.latest = s
            self.hist.append(s)
            if self.sink is not None:
                self.sink(s)

    def stop(self):
        self._stop.set()
        self.th.join(1.0)
        self.sock.close()
