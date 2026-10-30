"""The robot side of a closed-loop policy run (live_loop.py): measured tool poses in, pose targets and the trigger out.

Two endo_follow replays (--replay-stream-udp, started by robot/eval/robot_origin_run.sh with RO_LIVE) carry the tools; this
link talks to them and to the tracker:

  measured   the tracker tap (a copy on --tap-port) → each robot tool's tip and orientation in the camera frame, turned
             into the dataset's 15-vector (scene mm, hand tool → scene quaternions) — what the policy and the viewer see
  commanded  a 15-vector frame + the wall time the arms should be there → one UDP message per arm (a recorder row's pose:
             marker origin + quaternion in the recording's camera frame, the same maths as policy2rec.py)
  trigger    the drill trigger on a time schedule → the robot relay's --trig-udp (10 Hz heartbeat, as endo_follow sends it)

Mapping. The arms replay relative to their start: arm tip target = R0_tip + (tip_c − tip_c0), marker rotation target =
(R_c·R_c0ᵀ)·R0_R, with tip_c/R_c the commanded hand pose in the recording's camera frame (through T0), tip_c0/R_c0 that
of the start (ready) frame and R0_tip/R0_R the arm's own marker at engage (its events.jsonl `engage` row). The measured
pose is the inverse: tip_c = tip_c0 + (tip_meas − R0_tip) + c, R_c = (R_meas·R0_Rᵀ)·R_c0.
c (--scene-map anchor, the default) is the paired start's residual on the endoscope: the FE robot session registers the
scene on the drill tip (robot_origin_rec.py anchor), so the endoscope sits c away from its nominal start — the policy sees
the tools where the recorded session has them, and commands are shifted back by c. `relative`: c = 0 (each tool on its own).
"""
from __future__ import annotations

import json
import socket
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from lerobot_robot_spinearm import handeye as he
from lerobot_robot_spinearm.tap import ENDO_TOOLS

M = Path.home() / "il-robot/calib/tracker_models"
ROBOT_CALIB = {"endo": M / "tool_calib_endo_robotL_tip.json", "drill": M / "tool_calib_drill_robotR_tip.json"}
TOOLS = (("endo", 0), ("drill", 7))


def q2R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R2q(R: np.ndarray, ref=None) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1) * 2; q = [0.25 * S, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; q = [(R[2, 1] - R[1, 2]) / S, 0.25 * S, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, 0.25 * S, (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, 0.25 * S]
    q = np.asarray(q); q /= np.linalg.norm(q)
    return -q if (ref is not None and np.dot(q, ref) < 0) else q


class RobotLink:
    def __init__(self, hdr: dict, start: np.ndarray, run: str, tap_port: int = 9884, stream_ports=None,
                 trig_addr=("127.0.0.1", 9872), scene_map: str = "anchor", latency_dir=None):
        T0 = hdr["registration"]["T0"]
        self.R0s, self.t0s = np.asarray(T0["R"], float).reshape(3, 3), np.asarray(T0["t"], float)
        self.tl = {t: np.asarray(hdr["toolCalib"][t]["tipLocal"], float) / 1000.0 for t, _ in TOOLS}   # recorder rows' tip model
        self.tip_r = {t: he.tool_calib(ROBOT_CALIB[t])[1] for t, _ in TOOLS}                           # the arms' marker blocks
        self.start = np.asarray(start, float).copy()
        self.Rc0 = {t: self.R0s.T @ q2R(start[o + 3:o + 7]) for t, o in TOOLS}
        self.tc0 = {t: self.R0s.T @ (start[o:o + 3] - self.t0s) / 1000.0 for t, o in TOOLS}
        self.run, self.scene_map = run, scene_map
        self.ldir = Path(latency_dir or Path.home() / "il-robot/runs/latency")
        self.ports = stream_ports or {"endo": 9891, "drill": 9892}
        self.out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.trig_addr = trig_addr
        self.eng: dict = {}
        self.t_engage = time.time()
        self.c = {t: np.zeros(3) for t, _ in TOOLS}
        self.hist = {t: deque(maxlen=90 * 180) for t, _ in TOOLS}     # (ts wall ms, tip m, R) camera frame
        self.sched: list = []                                           # (t wall ms, trigger on)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_endo = None
        self._ages: deque = deque(maxlen=400)   # (arrival wall ms, arrival − frame stamp ms) of the last tracker samples
        self._ev_pos: dict = {}                 # how far each follower's events.jsonl has been read (fault())
        self._fault: str | None = None
        self.tap = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.tap.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self.tap.bind(("127.0.0.1", tap_port)); self.tap.settimeout(0.2)
        self.threads = [threading.Thread(target=self._tap_loop, daemon=True), threading.Thread(target=self._trig_loop, daemon=True)]
        for th in self.threads:
            th.start()

    # ------------------------------------------------------------------ tracker
    def _tap_loop(self) -> None:
        while not self._stop.is_set():
            try:
                m = json.loads(self.tap.recv(65536))
            except (socket.timeout, ValueError):
                continue
            tool = m.get("tool")
            now = time.time() * 1000.0
            ts = float(m.get("ts") or now)
            if m.get("ts"):
                with self._lock:
                    self._ages.append((now, now - ts))
            if not -50.0 <= now - ts <= 120.0:
                # the stamp is not the frame's time (10-05: the D435 global-time stamps drifted up to 0.5 s ahead of and 9.8 s
                # behind the host clock while the frames arrived on time) — file the pose under its arrival instead
                # (camera → here is about 35 ms), or every measured frame would be taken from the wrong moment
                ts = now - 35.0
            if m.get("lost") or m.get("pos") is None or m.get("quat") is None or int(m.get("vis", 4)) < 4:
                continue
            if tool in ENDO_TOOLS:
                R = he.quat_to_R(m["quat"]); tip = R @ self.tip_r["endo"] + np.asarray(m["pos"], float) / 1000.0
                ref = self._last_endo if self._last_endo is not None else (self.eng.get("endo") or {}).get("tip")
                if ref is not None and np.linalg.norm(tip - ref) > 0.06:   # another endoscope block in view (a hand tool)
                    continue
                self._last_endo = tip; tool = "endo"
            elif tool == "drill":
                R = he.quat_to_R(m["quat"]); tip = R @ self.tip_r["drill"] + np.asarray(m["pos"], float) / 1000.0
            else:
                continue
            with self._lock:
                self.hist[tool].append((ts, tip, R))

    def engaged(self) -> bool:
        """True once both arms' engage rows (their own marker at the start: R0_tip, R0_R — written when the run's GO file
        appears) have been read; the endoscope's start residual c is then set."""
        if len(self.eng) == 2:
            return True
        found = {}
        for tool, _ in TOOLS:
            f = self.ldir / f"{self.run}_{tool}" / "events.jsonl"
            if not f.exists():
                return False
            for line in open(f):
                if '"engage"' in line[:40]:
                    e = json.loads(line)
                    found[tool] = {"tip": np.asarray(e["R0_tip_mm"], float) / 1000.0, "R": np.asarray(e["R0_R"], float), "t": e["t"]}
                    break
        if len(found) < 2:
            return False
        if self.scene_map == "anchor":   # the FE session's scene is registered on the drill tip
            self.c["endo"] = (found["endo"]["tip"] - self.tc0["endo"]) - (found["drill"]["tip"] - self.tc0["drill"])
        self.eng = found
        self.t_engage = max(e["t"] for e in found.values()) / 1000.0   # wall s
        return True

    def stamp_age_ms(self, window_s: float = 1.0) -> float | None:
        """Median of (arrival − frame stamp) over the tracker samples of the last `window_s`, None without samples: about
        35 ms normally (camera → tracker 16, pose 14, send 3)."""
        now = time.time() * 1000.0
        with self._lock:
            a = [age for (t, age) in self._ages if now - t <= window_s * 1000.0]
        return float(np.median(a)) if a else None

    def fault(self, stale_ms: float = 120.0) -> str | None:
        """Why the arms cannot be following, or None.
        A follower left its replay: a `disengage` row in its events.jsonl (a refused goal — the arm server's lease is gone
        after 15 s without goals — or a lost tool); that stays for the run. Or the tracker's frame stamps are older than
        the followers accept (endo_follow --stale-ms 150: every frame is skipped): on 10-05 the D435 global-time stamps
        drifted from 0.1 s to 9.8 s behind the host clock within four minutes while the frames arrived on time; that can
        pass, so it is reported only while it lasts."""
        for tool, _ in TOOLS:
            try:
                with open(self.ldir / f"{self.run}_{tool}" / "events.jsonl", "rb") as fh:
                    fh.seek(self._ev_pos.get(tool, 0)); data = fh.read()
            except OSError:
                continue
            end = data.rfind(b"\n") + 1                    # whole rows only: the follower is still writing
            self._ev_pos[tool] = self._ev_pos.get(tool, 0) + end
            i = data.find(b'{"k":"disengage"', 0, end)
            if i >= 0:
                try:
                    why = json.loads(data[i:data.index(b"\n", i)]).get("reason")
                except ValueError:
                    why = "?"
                self._fault = f"the {tool} arm's follower stopped ({why})"
        if self._fault:
            return self._fault
        age = self.stamp_age_ms()
        if age is not None and age > stale_ms:
            return f"the tracker's frame stamps are {age:.0f} ms behind the clock (the followers skip frames over 150 ms)"
        return None   # stamps ahead of the clock (10-05 14:32, 16:55: up to +0.5 s for a minute) are not skipped by the
        # followers, and _tap_loop files those poses under their arrival time — nothing to wait for

    def _stat(self, tool: str, t0: float, t1: float):
        """Median tip (m) and mean rotation of the samples in [t0, t1] wall ms, their count, the tip's std (mm) and the gap
        (ms) from the window's middle to the nearest sample (which is used alone when the window holds none)."""
        mid = (t0 + t1) / 2.0
        with self._lock:
            h = self.hist[tool]
            sel, best = [], None
            for s_ in reversed(h):
                if best is None or abs(s_[0] - mid) < abs(best[0] - mid):
                    best = s_
                if t0 <= s_[0] <= t1:
                    sel.append(s_)
                elif s_[0] < t0:
                    break
        if best is None:
            return None
        if not sel:
            return best[1], best[2], 0, np.zeros(3), abs(best[0] - mid)
        tips = np.asarray([x[1] for x in sel])
        qs = np.asarray([R2q(x[2], R2q(sel[0][2])) for x in sel])
        q = qs.mean(0); q /= np.linalg.norm(q)
        return np.median(tips, axis=0), q2R(q), len(sel), tips.std(0) * 1000.0, min(abs(x[0] - mid) for x in sel)

    def state_at(self, t_ms: float | None = None, half_ms: float = 0.0, window=None) -> tuple[np.ndarray, float]:
        """The measured 15-vector (trigger 0) at wall time t_ms (default: now) — the median over ±half_ms, or over
        `window` = (t0, t1) — and how far the nearest sample is from that time (ms). The robot tip wanders about 1 mm
        around its target while holding (the visual servo follows the tracker's slow noise: 10-04, std 1.4-1.7 mm, mean
        error 0.07 mm), so a single sample is a poor anchor for the next chunk; the hold's median is not.
        self.last_std: the tip's std (mm, camera axes) per tool over the window used."""
        t_ms = time.time() * 1000.0 if t_ms is None else t_ms
        t0, t1 = window if window is not None else (t_ms - half_ms, t_ms + half_ms)
        v, worst = self.start.copy(), 0.0
        v[14] = 0.0
        self.last_std = {}
        for tool, o in TOOLS:
            st = self._stat(tool, t0, t1)
            if st is None:
                return v, float("inf")
            tip, R, n, sd, gap = st
            worst = max(worst, gap)
            self.last_std[tool] = sd
            e = self.eng[tool]
            tc = self.tc0[tool] + (tip - e["tip"]) + self.c[tool]
            Rc = R @ e["R"].T @ self.Rc0[tool]
            v[o:o + 3] = self.R0s @ tc * 1000.0 + self.t0s
            v[o + 3:o + 7] = R2q(self.R0s @ Rc, self.start[o + 3:o + 7])
        return v, worst

    # ------------------------------------------------------------------ commands
    def arm_frame(self, v: np.ndarray) -> np.ndarray:
        """The frame as the arms take it (the endoscope's start residual removed) — what the safety gate must check."""
        out = np.array(v, float)
        for tool, o in TOOLS:
            out[o:o + 3] = out[o:o + 3] - self.R0s @ self.c[tool] * 1000.0
        return out

    def scene_frame(self, v: np.ndarray) -> np.ndarray:
        out = np.array(v, float)
        for tool, o in TOOLS:
            out[o:o + 3] = out[o:o + 3] + self.R0s @ self.c[tool] * 1000.0
        return out

    def send_chunk(self, frames, times_ms) -> None:
        for v, t in zip(frames, times_ms):
            a = self.arm_frame(v)
            for tool, o in TOOLS:
                Rc = self.R0s.T @ q2R(a[o + 3:o + 7])
                tc = self.R0s.T @ (a[o:o + 3] - self.t0s) / 1000.0
                msg = {"tool": tool, "t_wall": float(t), "quat": R2q(Rc).round(7).tolist(),
                       "pos": (tc - Rc @ self.tl[tool]).round(7).tolist()}
                self.out.sendto(json.dumps(msg).encode(), ("127.0.0.1", self.ports[tool]))
            with self._lock:
                self.sched.append((float(t), bool(v[14] > 0.5)))

    def trigger_off_at(self, t_ms: float) -> None:
        with self._lock:
            self.sched.append((float(t_ms), False))

    def _trig_loop(self) -> None:
        last = None
        while not self._stop.is_set():
            now = time.time() * 1000.0
            with self._lock:
                on = False
                for t, v in self.sched:
                    if t <= now:
                        on = v
                self.sched = self.sched[-400:]
            self._trig(on)
            if on != last:
                last = on
            time.sleep(0.05)
        self._trig(False, end=True)

    def _trig(self, on: bool, end: bool = False) -> None:
        msg = {"t": time.time() * 1000.0, "tool": "drill", "btn_trig": int(on), "src": "robot", "run": self.run}
        if end:
            msg["end"] = 1
        try:
            self.out.sendto(json.dumps(msg).encode(), self.trig_addr)
        except OSError:
            pass

    def close(self) -> None:
        """Trigger off (and its channel released), the arms told the stream is over (they hold and release)."""
        if self._stop.is_set():
            return
        self._stop.set()
        for th in self.threads:
            th.join(1.0)
        for tool, _ in TOOLS:
            for _ in range(3):
                self.out.sendto(json.dumps({"end": 1}).encode(), ("127.0.0.1", self.ports[tool]))
        self.tap.close()
