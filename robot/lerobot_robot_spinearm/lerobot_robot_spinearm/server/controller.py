"""Arm controller: lease, safety state machine and the 30 Hz tick. Pure with respect to time and I/O
(clock and bus are injected) so it is unit-tested with SimBus and a fake clock.

Threading contract: `handle()` is called from the RPC thread, `tick()` from the control thread.
Only the control thread touches the bus; `handle()` either answers from the last snapshot or hands a
request to the control thread and waits for its result (≤ 2 s).
"""

from __future__ import annotations

import logging
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..protocol import JOINTS, POWER_CYCLE_FAULTS, error, ok
from .bus import BusLost

log = logging.getLogger("spine-arm")


@dataclass
class Params:
    rate_hz: float = 30.0
    step_deg: float = 10.0  # per-tick joint step cap (mirrors SOFollower max_relative_target)
    watchdog_s: float = 0.25  # MOVING without a goal for this long → hold
    lease_ttl_s: float = 15.0
    lease_idle_force_s: float = 5.0
    temp_max: int = 55
    temp_clear: int = 50
    overload_err_deg: float = 8.0  # tracking error vs the last per-tick command …
    overload_load: int = 100  # … with |load| below this → Feetech protection has tripped
    overload_ticks: int = 21  # ≈ 0.7 s
    stall_ticks: int = 20
    stall_min_move: float = 0.15
    home_step: float = 1.5
    home_tol: float = 3.0
    home_phase_s: tuple[float, float] = (8.0, 25.0)
    torque_off_tol: float = 3.0
    slow_read_every: int = 30  # temperature / torque-enable reads
    accel: int = 32  # servo Acceleration register (lower = gentler starts, more lag); live via set{accel}


@dataclass
class Lease:
    owner: str  # OS account (verified by the ident socket) or "<typed name>?" when unverified
    token: str
    expires: float
    last_cmd: float
    tool: str = ""  # what the client called itself (--owner / script name), shown next to the account
    verified: bool = False
    contest: dict | None = None  # the latest refused attempt by someone else: {"by", "at", "force"}
    contest_told: bool = True  # whether the holder has been warned about `contest` yet


@dataclass
class Request:
    kind: str
    args: dict
    done: threading.Event = field(default_factory=threading.Event)
    result: dict | None = None


REG_READ = ("P_Coefficient", "D_Coefficient", "I_Coefficient", "Acceleration", "Torque_Limit", "Max_Torque_Limit",
            "Protection_Current", "Overload_Torque", "Minimum_Startup_Force", "CW_Dead_Zone", "CCW_Dead_Zone")
REG_WRITE = {"P_Coefficient": (4, 64), "D_Coefficient": (0, 64), "I_Coefficient": (0, 16),
             # joint 6 carries the endoscope now, not a gripper: LeRobot caps it at 50 % torque / 25 % overload
             "Max_Torque_Limit": (100, 1000), "Overload_Torque": (20, 90)}


def _r4(t: float | None) -> float | None:
    return None if t is None else round(t, 4)


class ArmController:
    def __init__(self, arm: str, bus, bounds: dict[str, tuple[float, float]], neutral: dict[str, float] | None,
                 params: Params | None = None, clock=time.monotonic, sim: bool = False, verify_owner: bool = False):
        self.arm, self.bus, self.bounds = arm, bus, bounds
        # Lease owners: with verify_owner (real arms) only a kernel-vouched account (ident ticket) may lease.
        self.verify_owner = verify_owner
        self.ident = None  # IdentServer, attached by main
        self._revoked: dict[str, str] = {}  # token → why it stopped working (taken over / replaced)
        # Servo gains live in the servos' EEPROM and outlive a lease: remember the values found at startup and
        # every change made since, so the next person is warned and the baseline can be written back.
        self.gain_base: dict[str, dict[str, int]] | None = None
        self.gain_mods: dict[str, dict] = {}
        self.neutral = self._clamp_neutral(neutral) if neutral else None
        self.p = params or Params()
        self.clock, self.sim = clock, sim
        self.lock = threading.Lock()
        self.pending: deque[Request] = deque()
        self.state = "OFF"
        self.faults: set[str] = set()
        self.lease: Lease | None = None
        self.bus_ok = False
        self.tick_n = 0
        self.pos: dict[str, float] = {}
        self.load: dict[str, int] = {}
        self.temp: dict[str, int] = {}
        self.torque = False
        self.target: dict[str, float] | None = None  # where the client wants the joints
        self.cmd: dict[str, float] | None = None  # last per-tick command actually written
        self.last_goal_t = 0.0
        # latency instrumentation (clock seconds): the latest goal's (client seq, receive time), and per tick the
        # time the position read finished / the goal write finished plus the goal stamp that write carried.
        self._goal_stamp: tuple = (None, None)
        self._written_stamp: tuple = (None, None)
        self.t_read: float | None = None
        self.t_write: float | None = None
        self._overload_ticks = 0
        self._stall_ticks = 0
        self._hot_reads = 0
        self._prev_pos: dict[str, float] | None = None
        self._home: dict | None = None
        self._last_reconnect = 0.0
        self.last_error = ""
        self.snapshot: dict = self._make_snapshot(self.clock())

    # ------------------------------------------------------------------ RPC side
    def handle(self, msg: dict) -> dict:
        cmd = msg.get("cmd")
        now = self.clock()
        if cmd in ("status", "state"):
            with self.lock:
                return ok(**self.snapshot, neutral=self.neutral, bounds=self.bounds) if cmd == "status" else ok(**self.snapshot)
        if cmd == "lease":
            return self._lease(msg, now)
        if cmd == "regs":  # read-only register dump (servo tuning), no token
            names = [n for n in (msg.get("names") or ["P_Coefficient", "D_Coefficient", "I_Coefficient"]) if n in REG_READ]
            return self._run(Request("regs", {"names": names}))
        if cmd == "estop":  # anyone, no token — safety first
            return self._run(Request("estop", {}))
        if cmd == "fake":
            if not self.sim:
                return error("sim_only")
            return self._run(Request("fake", msg))
        if cmd not in ("renew", "release", "enable", "goal", "hold", "home", "torque_off", "clear_faults", "set",
                       "set_regs", "restore_gains", "keep_gains"):
            return error("bad_request", detail=f"unknown cmd {cmd!r}")
        with self.lock:
            err = self._check_token(msg.get("token"), now)
            if err:
                return err
            self.lease.last_cmd = now
            self.lease.expires = now + self.p.lease_ttl_s
            warn = None
            if self.lease.contest and not self.lease.contest_told:  # tell the holder once
                c = self.lease.contest
                warn = f"{c['by']} tried to take this arm {now - c['at']:.0f} s ago" + (" (force)" if c["force"] else "")
                self.lease.contest_told = True
            if cmd == "renew":
                return self._warn(ok(expires_in=self.p.lease_ttl_s), warn)
        if cmd == "goal":
            return self._warn(self._goal(msg, now), warn)
        if cmd == "set":  # runtime tunables (step cap, servo acceleration) — bounded
            step = float(msg.get("step_deg", self.p.step_deg))
            self.p.step_deg = float(np.clip(step, 0.5, 10.0))
            if "accel" in msg:  # bus write → control thread
                r = self._run(Request("accel", {"accel": int(np.clip(int(msg["accel"]), 8, 254))}))
                if not r.get("ok"):
                    return r
            return self._warn(ok(step_deg=self.p.step_deg, accel=self.p.accel), warn)
        return self._warn(self._run(Request(cmd, msg)), warn)

    @staticmethod
    def _warn(reply: dict, *warns: str | None) -> dict:
        warns = [w for w in warns if w]
        if warns:
            reply = {**reply, "warn": "; ".join([reply["warn"]] + warns if reply.get("warn") else warns)}
        return reply

    def _check_token(self, token, now) -> dict | None:
        if self.lease is None:
            return error("no_lease")
        if now >= self.lease.expires:
            self.lease = None
            return error("no_lease", detail="lease expired")
        if token != self.lease.token:
            why = self._revoked.get(token) if isinstance(token, str) else None
            return error("bad_token", holder=self.lease.owner, **({"detail": why} if why else {}))
        return None

    def _revoke(self, token: str, why: str) -> None:
        self._revoked[token] = why
        while len(self._revoked) > 16:
            self._revoked.pop(next(iter(self._revoked)))

    def _lease(self, msg: dict, now: float) -> dict:
        tool = str(msg.get("owner") or "anonymous")[:32]
        user = self.ident.redeem(msg.get("ident")) if self.ident is not None else None
        if user is None and self.verify_owner:
            return error("unverified", detail="this arm only leases to a verified local account — use spine-arm-ctl "
                                              "or the SpineArm plugin on the robot server (they fetch an ident ticket)")
        owner = user or f"{tool}?"  # '?' = a typed-in name nobody checked (sim / remote clients)
        who = owner if tool == "anonymous" or owner in (tool, f"{tool}?") else f"{owner} ({tool})"
        with self.lock:
            held = self.lease if self.lease and now < self.lease.expires else None
            warns = []
            if held and held.owner != owner:
                idle = now - held.last_cmd
                force = bool(msg.get("force"))
                if not (force and idle >= self.p.lease_idle_force_s):
                    held.contest, held.contest_told = {"by": who, "at": now, "force": force}, False
                    log.warning("lease refused: %s wants the arm, %s holds it (idle %.1f s%s)", who, held.owner, idle,
                                ", force" if force else "")
                    return error("lease_held", holder=held.owner, idle_s=round(idle, 1),
                                 detail=f"{held.owner} is using this arm (idle {idle:.1f} s) — they have been told "
                                        f"you asked; --force works after {self.p.lease_idle_force_s:.0f} s idle")
                self._revoke(held.token, f"taken by {who} after {idle:.0f} s idle (force)")
                log.warning("lease taken: %s took the arm from %s (idle %.1f s)", who, held.owner, idle)
                warns.append(f"took the arm from {held.owner} (idle {idle:.0f} s)")
            elif held:
                idle = now - held.last_cmd
                if held.tool != tool and idle < self.p.lease_idle_force_s and not msg.get("force"):
                    return error("lease_held", holder=held.owner, idle_s=round(idle, 1),
                                 detail=f"your other client '{held.tool}' is using this arm — stop it, or --force")
                self._revoke(held.token, "replaced by a newer lease of the same account")
                warns.append("replaced your earlier lease (its token no longer works)")
            if self.gain_mods:
                warns.append("servo gains differ from the baseline: " + self._gain_summary())
            self.lease = Lease(owner, secrets.token_hex(8), now + self.p.lease_ttl_s, now, tool=tool,
                               verified=user is not None)
            log.info("lease → %s", who)
            return self._warn(ok(token=self.lease.token, ttl=self.p.lease_ttl_s, state=self.state, owner=owner),
                              *warns)

    def _gain_summary(self) -> str:
        return "; ".join(f"{n} {m['values']} by {m['by']}" for n, m in self.gain_mods.items())

    def _goal(self, msg: dict, now: float) -> dict:
        pos = msg.get("pos")
        if not isinstance(pos, dict) or not pos:
            return error("bad_request", detail="pos must be a non-empty {joint: deg} dict")
        try:
            wanted = {j: float(pos[j]) for j in JOINTS if j in pos}
        except (TypeError, ValueError):
            return error("bad_request", detail="joint values must be numbers")
        if set(pos) - set(JOINTS) or not wanted:
            return error("bad_request", detail=f"unknown joints {sorted(set(pos) - set(JOINTS))}")
        with self.lock:
            if self.state == "ESTOP":
                return error("estop")
            if self.state == "FAULT":
                return error("fault", faults=sorted(self.faults))
            if self.state == "OFF":
                return error("off", detail="call enable first")
            if self.state == "HOMING":
                return error("busy", detail="homing")
            clamped = {j: float(np.clip(v, *self.bounds[j])) for j, v in wanted.items()}
            base = dict(self.target) if self.target else dict(self.pos)
            base.update(clamped)
            self.target = base
            self.last_goal_t = now
            self._goal_stamp = (msg.get("seq"), now)
            if self.state == "HOLD":
                self.state = "MOVING"
            return ok(sent=clamped, state=self.state, pos=dict(self.pos), rx=round(now, 4))

    def _run(self, req: Request, timeout: float = 2.0) -> dict:
        with self.lock:
            self.pending.append(req)
        if not req.done.wait(timeout):
            return error("timeout", detail="control loop did not answer")
        return req.result or error("bad_request")

    # ------------------------------------------------------------------ control side
    def tick(self) -> dict:
        now = self.clock()
        dt = 1.0 / self.p.rate_hz
        self.tick_n += 1
        self.t_read = self.t_write = None
        with self.lock:
            reqs = list(self.pending)
            self.pending.clear()
            if self.lease and now >= self.lease.expires:
                log.info("lease of %s expired", self.lease.owner)
                self.lease = None
                if self.state == "MOVING":
                    self._hold_now()
        try:
            if not self.bus_ok:
                self._try_reconnect(now)
            if self.bus_ok:
                if self.gain_base is None:
                    self.gain_base = self.bus.read_regs(list(REG_WRITE))
                    log.info("servo gains at startup: %s", {n: v for n, v in self.gain_base.items() if n != "Max_Torque_Limit"})
                self.pos = self.bus.read_positions()
                self.t_read = self.clock()
                self.load = self.bus.read_loads()
                if self.tick_n % self.p.slow_read_every == 1:
                    self.temp = self.bus.read_temps()
                    torque = self.bus.read_torque()
                    self._check_power_cycle(torque)
                    self._check_temperature()
            for req in reqs:
                req.result = self._execute(req, now)
                req.done.set()
            if self.bus_ok:
                self._step_motion(now)
                self.bus.step(dt)
        except BusLost as e:
            self._on_bus_lost(e, now)
            for req in reqs:
                if not req.done.is_set():
                    req.result = error("bus_lost", detail=str(e))
                    req.done.set()
        with self.lock:
            self.snapshot = self._make_snapshot(now)
            return dict(self.snapshot)

    def _try_reconnect(self, now: float) -> None:
        if now - self._last_reconnect < 2.0:
            return
        self._last_reconnect = now
        try:
            self.bus.connect()
            self.bus_ok = True
            self.faults.discard("bus_lost")
            self.faults.discard("dropout")
            self.faults.discard("stall")  # stale: a stall right before a power drop (2026-09-22) — re-evaluated live
            self.faults.discard("home_stalled")
            self._stall_ticks = 0
            self._home = None
            self.state = "OFF"  # torque state unknown after a reconnect — require enable
            torque = self.bus.read_torque()
            if all(torque.values()):
                self.torque = True
                self.bus.write_goals(self.bus.read_positions())  # hold where we are
                self.state = "HOLD" if not self.faults else "FAULT"
            log.info("bus connected, state=%s", self.state)
        except BusLost as e:
            self.last_error = str(e)

    def _on_bus_lost(self, e: BusLost, now: float) -> None:
        self.bus_ok = False
        self.last_error = str(e)
        self.faults.add("bus_lost")
        if self.state not in ("OFF", "ESTOP"):
            self.state = "FAULT"
        self.target = None
        log.error("bus lost: %s", e)

    def _check_power_cycle(self, torque: dict[str, int]) -> None:
        on = all(torque.values())
        if self.state != "OFF" and not on:
            # RAM registers (Acceleration, Torque_Limit, Goal_Position) were reset by a 12 V power-cycle.
            # That same power-cycle is what clears Feetech overload protection, so drop that fault here.
            self.state, self.torque, self.target, self._home = "OFF", False, None, None
            self.faults.discard("overload_protection")
            self.faults.add("power_cycled")
            log.warning("torque dropped unexpectedly (12 V power-cycle?) → OFF")
        self.torque = on

    def _check_temperature(self) -> None:
        # Two consecutive hot reads (1 s apart): a browning-out bus returns garbage right before it dies
        # (2026-09-22: "overheat" at 31 °C, bus lost 7 s later), and one bad byte must not fault the arm.
        hot = bool(self.temp) and max(self.temp.values()) >= self.p.temp_max
        self._hot_reads = self._hot_reads + 1 if hot else 0
        if self._hot_reads >= 2:
            log.error("temperatures %s", self.temp)
            self._fault("overheat")
        elif "overheat" in self.faults and self.temp and max(self.temp.values()) <= self.p.temp_clear:
            self.faults.discard("overheat")
            if self.state == "FAULT" and not self.faults:
                self.state = "HOLD"

    def _fault(self, name: str) -> None:
        if name not in self.faults:
            log.error("FAULT %s (state was %s)", name, self.state)
        self.faults.add(name)
        if self.state not in ("OFF", "ESTOP"):
            self._hold_now()
            self.state = "FAULT"
        self.target, self._home = None, None

    def _hold_now(self) -> None:
        if self.bus_ok and self.torque and self.pos:
            try:
                self.bus.write_goals(dict(self.pos))
            except BusLost as e:
                self._on_bus_lost(e, self.clock())
        self.cmd = dict(self.pos) if self.pos else None
        self._overload_ticks = self._stall_ticks = 0

    def _execute(self, req: Request, now: float) -> dict:
        k, a = req.kind, req.args
        if k == "estop":
            self._hold_now()
            self.state, self.target, self._home = "ESTOP", None, None
            with self.lock:
                self.lease = None
            log.warning("ESTOP")
            return ok(state=self.state)
        if k == "fake":
            fault, joint = a.get("fault"), a.get("joint")
            if fault == "clear":
                self.bus.clear()
            else:
                self.bus.inject(fault, joint)
            return ok(faults=list(self.bus.faults))
        if not self.bus_ok:
            return error("bus_lost", detail=self.last_error)
        if k == "regs":
            return ok(regs=self.bus.read_regs(a["names"]))
        if k == "set_regs":  # servo gains (EEPROM): only with torque off, bounded, one register per call
            name, vals = a.get("name"), a.get("values") or {}
            if name not in REG_WRITE:
                return error("bad_request", detail=f"register must be one of {sorted(REG_WRITE)}")
            if self.state != "OFF" or self.torque:
                return error("busy", detail="torque must be off (EEPROM writes need Lock = 0)")
            lo, hi = REG_WRITE[name]
            vals = {j: int(np.clip(int(v), lo, hi)) for j, v in vals.items() if j in JOINTS}
            self.bus.write_regs(name, vals)
            now_regs = self.bus.read_regs([name])
            by = self.lease.owner if self.lease else "?"
            log.info("register %s ← %s (by %s)", name, vals, by)
            self._note_gains(name, now_regs[name], by)
            return self._warn(ok(regs=now_regs), "servo gains now differ from the baseline — "
                                                  "`servo_gains.py baseline` restores, `servo_gains.py keep` adopts"
                              if self.gain_mods else None)
        if k == "restore_gains":
            if self.state != "OFF" or self.torque:
                return error("busy", detail="torque must be off (EEPROM writes need Lock = 0)")
            return ok(restored=self._restore_gains())
        if k == "keep_gains":  # the current values become the baseline (a deliberate, lasting change)
            self.gain_base = self.bus.read_regs(list(REG_WRITE))
            log.warning("servo gain baseline replaced by %s: %s", self.lease.owner if self.lease else "?", self._gain_summary())
            self.gain_mods.clear()
            return ok(baseline=self.gain_base)
        if k == "accel":
            self.p.accel = a["accel"]
            self.bus.set_accel(self.p.accel)
            log.info("servo acceleration → %d", self.p.accel)
            return ok(accel=self.p.accel)
        if k == "enable":
            if self.state == "ESTOP":
                return error("estop", detail="clear_faults first")
            if self.faults - {"power_cycled"}:
                return error("fault", faults=sorted(self.faults))
            self.bus.torque_on_safe()
            if not all(self.bus.read_torque().values()):
                return error("bus_lost", detail="torque did not enable on all joints")
            self.torque, self.state = True, "HOLD"
            self.faults.discard("power_cycled")
            self.cmd, self.target = dict(self.pos), None
            log.info("torque on → HOLD")
            return ok(state=self.state)
        if k in ("hold", "release"):
            if self.state in ("MOVING", "HOMING"):
                self._hold_now()
                self.state, self.target, self._home = "HOLD", None, None
            if k == "release":
                with self.lock:
                    self.lease = None
                if self.gain_mods:  # not restored automatically: a gain A/B sets gains, releases, then moves
                    return ok(state=self.state, warn="servo gains still differ from the baseline ("
                              + self._gain_summary() + ") — the next person is told; `servo_gains.py baseline` "
                              "restores them, `servo_gains.py keep` makes them the new baseline")
            return ok(state=self.state)
        if k == "clear_faults":
            blocking = self.faults & POWER_CYCLE_FAULTS
            if blocking:
                return error("fault", faults=sorted(blocking), detail="power-cycle 12 V, then enable")
            if "overheat" in self.faults:
                return error("fault", faults=["overheat"], detail=f"wait until ≤ {self.p.temp_clear} °C")
            self.faults.clear()
            self.state = "HOLD" if self.torque else "OFF"
            self._home = None
            return ok(state=self.state)
        if k == "home":
            if self.state not in ("HOLD", "MOVING"):
                return error("estop" if self.state == "ESTOP" else "off" if self.state == "OFF" else "fault",
                             state=self.state)
            if not self.neutral:
                return error("no_neutral")
            phases = []
            if "gripper" in self.neutral:  # fold the marker joint first — the bracket hits the arm otherwise
                phases.append({"gripper": self.neutral["gripper"]})
            phases.append({j: v for j, v in self.neutral.items() if j != "gripper"})
            self._home = {"phases": phases, "i": 0, "started": now, "no_move": 0}
            self.target, self.state = None, "HOMING"
            log.info("homing: %d phases", len(phases))
            return ok(state=self.state)
        if k == "torque_off":
            if self.state in ("MOVING", "HOMING"):
                return error("busy", state=self.state)
            if not a.get("force") and self.neutral:
                far = {j: round(self.pos[j] - v, 1) for j, v in self.neutral.items()
                       if abs(self.pos[j] - v) > self.p.torque_off_tol}
                if far:
                    return error("not_at_neutral", far=far, detail="home first, or pass force")
            self.bus.torque_off()
            self.torque, self.state, self.target = False, "OFF", None
            log.info("torque off")
            return ok(state=self.state)
        return error("bad_request", detail=f"unknown request {k}")

    def _step_motion(self, now: float) -> None:
        if self.state == "HOMING":
            self._step_home(now)
            return
        if self.state != "MOVING" or not self.target:
            self._prev_pos = dict(self.pos)
            return
        if now - self.last_goal_t > self.p.watchdog_s:
            log.info("watchdog: no goal for %.0f ms → HOLD", (now - self.last_goal_t) * 1000)
            self._hold_now()
            self.state, self.target = "HOLD", None
            return
        cmd = {}
        for j, tgt in self.target.items():
            cmd[j] = self.pos[j] + float(np.clip(tgt - self.pos[j], -self.p.step_deg, self.p.step_deg))
        self._monitor_tracking()
        if self.state != "MOVING":
            return
        stamp = self._goal_stamp
        self.bus.write_goals(cmd)
        self.t_write = self.clock()
        self._written_stamp = stamp
        self.cmd = cmd
        self._prev_pos = dict(self.pos)

    def _monitor_tracking(self) -> None:
        """Tracking error is measured against the *last per-tick command*, not the far target:
        a healthy servo trails it by ~1°; one in Feetech overload protection trails by the full
        step while its load reads small."""
        if not self.cmd:
            return
        err = {j: abs(self.cmd[j] - self.pos[j]) for j in self.cmd}
        low_load_lag = [j for j, e in err.items() if e > self.p.overload_err_deg and abs(self.load.get(j, 0)) < self.p.overload_load]
        self._overload_ticks = self._overload_ticks + 1 if low_load_lag else 0
        if self._overload_ticks >= self.p.overload_ticks:
            log.error("overload protection suspected on %s (lag with low load) — power-cycle 12 V then enable", low_load_lag)
            self._fault("overload_protection")
            return
        moved = max(abs(self.pos[j] - self._prev_pos[j]) for j in JOINTS) if self._prev_pos else 1.0
        lagging = [j for j, e in err.items() if e > 3.0]
        self._stall_ticks = self._stall_ticks + 1 if (lagging and moved < self.p.stall_min_move) else 0
        if self._stall_ticks >= self.p.stall_ticks:
            log.error("stall on %s (no progress under load)", lagging)
            self._fault("stall")

    def _step_home(self, now: float) -> None:
        h = self._home
        if h is None:
            self.state = "HOLD"
            return
        targets = h["phases"][h["i"]]
        err = {j: t - self.pos[j] for j, t in targets.items()}
        if all(abs(e) <= self.p.home_tol for e in err.values()):
            h["i"] += 1
            h["started"], h["no_move"] = now, 0
            if h["i"] >= len(h["phases"]):
                log.info("home reached")
                self._hold_now()
                self.state, self._home = "HOLD", None
            return
        moved = max(abs(self.pos[j] - self._prev_pos[j]) for j in targets) if self._prev_pos else 1.0
        h["no_move"] = h["no_move"] + 1 if moved < self.p.stall_min_move else 0
        if h["no_move"] >= self.p.stall_ticks or now - h["started"] > self.p.home_phase_s[min(h["i"], 1)]:
            log.error("home stalled at phase %d: remaining %s", h["i"], {j: round(e, 1) for j, e in err.items()})
            self._fault("home_stalled")  # torque stays on — never drop a raised arm
            return
        cmd = {j: self.pos[j] + float(np.clip(e, -self.p.home_step, self.p.home_step)) for j, e in err.items()}
        self.bus.write_goals(cmd)
        self.cmd = {**(self.cmd or {}), **cmd}
        self._prev_pos = dict(self.pos)

    # ------------------------------------------------------------------ helpers
    def _note_gains(self, name: str, values: dict[str, int], by: str) -> None:
        base = (self.gain_base or {}).get(name, {})
        diff = {j: v for j, v in values.items() if j in base and v != base[j]}
        if diff:
            self.gain_mods[name] = {"values": diff, "by": by, "at": self.clock()}
        else:
            self.gain_mods.pop(name, None)

    def _restore_gains(self) -> dict:
        """Write the startup values back into every modified register (control thread, torque off)."""
        restored = {}
        for name in list(self.gain_mods):
            base = {j: v for j, v in self.gain_base[name].items() if j in self.gain_mods[name]["values"]}
            self.bus.write_regs(name, base)
            restored[name] = base
            self._note_gains(name, self.bus.read_regs([name])[name], "?")
        log.info("servo gains restored to the startup values: %s", restored)
        return restored

    def _clamp_neutral(self, neutral: dict[str, float]) -> dict[str, float]:
        out = {}
        for j, v in neutral.items():
            if j not in JOINTS:
                continue
            lo, hi = self.bounds[j]
            c = float(np.clip(v, lo, hi))
            if abs(c - v) > 1.0:
                log.warning("neutral %s=%.1f is outside bounds (%.1f..%.1f) — clamped to %.1f", j, v, lo, hi, c)
            out[j] = c
        return out

    def _make_snapshot(self, now: float) -> dict:
        lease = None
        if self.lease:
            lease = {"owner": self.lease.owner, "tool": self.lease.tool, "verified": self.lease.verified,
                     "expires_in": round(self.lease.expires - now, 1)}
            c = self.lease.contest
            if c and now - c["at"] < 60:
                lease["contest"] = {"by": c["by"], "ago_s": round(now - c["at"], 1), "force": c["force"]}
        return {
            "t": round(now, 3), "arm": self.arm, "state": self.state, "torque": self.torque,
            "pos": {j: round(v, 2) for j, v in self.pos.items()},
            "goal": {j: round(v, 2) for j, v in self.target.items()} if self.target else None,
            "load": dict(self.load), "temp": dict(self.temp),
            "lease": lease, "faults": sorted(self.faults), "bus_ok": self.bus_ok,
            "tick": self.tick_n, "sim": self.sim, "step_deg": self.p.step_deg, "accel": self.p.accel,
            "last_error": self.last_error,
            "gains_modified": {n: {"values": m["values"], "by": m["by"]} for n, m in self.gain_mods.items()} or None,
            # latency instrumentation — tr: position read done, tw: goal write done (None = no write this tick),
            # gseq/grx: client seq and receive time of the goal that write carried, cmd: the per-tick command.
            "tr": _r4(self.t_read), "tw": _r4(self.t_write),
            "gseq": self._written_stamp[0], "grx": _r4(self._written_stamp[1]),
            "cmd": {j: round(v, 2) for j, v in self.cmd.items()} if self.cmd else None,
        }
