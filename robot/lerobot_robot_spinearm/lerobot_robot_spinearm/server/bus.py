"""Bus backends: the real Feetech bus and a kinematic/servo-behaviour simulator with fault injection.

Both expose the same small surface so the controller is tested against `SimBus` and runs unchanged
on `RealBus`. Positions are degrees; loads are signed 0.1 % units (|load| ≤ 1000) as the servo
reports them; temperatures °C.
"""

from __future__ import annotations

import logging

import json
from pathlib import Path

import numpy as np

from ..protocol import JOINTS, WRIST_ROLL_SOFT_LIMIT


log = logging.getLogger("spine-arm.bus")


class BusLost(ConnectionError):
    """Raised by a backend when a read/write fails (missing ID, USB gone, timeout)."""


class RealBus:
    """FeetechMotorsBus built straight from a calibration JSON (no `Robot` object, no $HOME writes)."""

    def __init__(self, port: str, calibration_path: str | Path):
        from lerobot.motors import Motor, MotorCalibration, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus

        raw = json.loads(Path(calibration_path).read_text())
        self.calibration = {j: MotorCalibration(**raw[j]) for j in JOINTS}
        # Every joint in degrees — including joint 6, which no longer is a 0..100 gripper.
        motors = {j: Motor(i + 1, "sts3215", MotorNormMode.DEGREES) for i, j in enumerate(JOINTS)}
        self.bus = FeetechMotorsBus(port, motors, calibration=self.calibration)
        self.sim = False

    def bounds(self) -> dict[str, tuple[float, float]]:
        return bounds_from_calibration({j: (c.range_min, c.range_max) for j, c in self.calibration.items()})

    def connect(self) -> None:
        try:
            if self.bus.is_connected:  # reconnect after bus_lost: lerobot refuses connect() on an open port
                self.disconnect(disable_torque=False)
            self.bus.connect()  # pings all six IDs; a missing motor raises
            if not self.bus.is_calibrated:
                raise BusLost("motor EEPROM calibration differs from the calibration file (wrong arm/port?)")
        except BusLost:
            raise
        except Exception as e:  # ConnectionError, RuntimeError from lerobot
            raise BusLost(str(e)) from e

    def disconnect(self, disable_torque: bool = False) -> None:
        try:
            self.bus.disconnect(disable_torque=disable_torque)
        except Exception:
            pass

    def _read(self, name: str, normalize: bool):
        try:
            return self.bus.sync_read(name, normalize=normalize, num_retry=2)
        except Exception as e:
            raise BusLost(f"{name}: {e}") from e

    def read_positions(self) -> dict[str, float]:
        return {j: float(v) for j, v in self._read("Present_Position", True).items()}

    def read_loads(self) -> dict[str, int]:
        return {j: int(v) for j, v in self._read("Present_Load", False).items()}

    def read_temps(self) -> dict[str, int]:
        return {j: int(v) for j, v in self._read("Present_Temperature", False).items()}

    def read_torque(self) -> dict[str, int]:
        return {j: int(v) for j, v in self._read("Torque_Enable", False).items()}

    def read_regs(self, names: list[str]) -> dict[str, dict[str, int]]:
        """Raw register values (e.g. P/D/I_Coefficient) per joint."""
        return {n: {j: int(v) for j, v in self._read(n, False).items()} for n in names}

    def write_regs(self, name: str, values: dict[str, int]) -> None:
        """Raw register write. EEPROM registers (P/D/I_Coefficient …) only stick with torque off (Lock = 0)."""
        try:
            self.bus.sync_write(name, {j: int(v) for j, v in values.items()}, normalize=False)
        except Exception as e:
            raise BusLost(f"write {name}: {e}") from e

    def write_goals(self, goals: dict[str, float]) -> None:
        try:
            self.bus.sync_write("Goal_Position", {j: float(v) for j, v in goals.items()})
        except Exception as e:
            raise BusLost(f"Goal_Position: {e}") from e

    accel = 32  # Feetech Acceleration register (×100 steps/s² ≈ 8.8°/s² per unit); set_accel() changes it live

    def set_accel(self, value: int) -> None:
        self.accel = int(value)
        try:
            self.bus.sync_write("Acceleration", dict.fromkeys(JOINTS, self.accel), normalize=False)
        except Exception as e:
            raise BusLost(f"accel: {e}") from e

    def torque_on_safe(self) -> None:
        """Goal=present *before* enabling torque (no jump), gentle acceleration, full torque limit.

        Never uses SOFollower.configure(): its torque_disabled() context re-enables torque against a
        stale goal register. A lower Torque_Limit makes Feetech overload protection trip earlier.
        """
        for attempt in (1, 2):  # enable_torque writes 'Lock' per motor with no retry; one lost packet is not a lost bus
            try:
                raw = self.bus.sync_read("Present_Position", normalize=False, num_retry=2)
                self.bus.sync_write("Goal_Position", raw, normalize=False)
                self.bus.sync_write("Acceleration", dict.fromkeys(JOINTS, self.accel), normalize=False)
                self.bus.sync_write("Torque_Limit", dict.fromkeys(JOINTS, 1000), normalize=False)
                self.bus.enable_torque()
                return
            except Exception as e:
                if attempt == 2:
                    raise BusLost(f"torque_on: {e}") from e
                log.warning("torque_on retry after: %s", e)

    def torque_off(self) -> None:
        try:
            self.bus.disable_torque()
        except Exception as e:
            raise BusLost(f"torque_off: {e}") from e

    def step(self, dt: float) -> None:  # real servos step themselves
        pass


class SimBus:
    """Kinematic arm with a first-order servo model. Same surface as RealBus, plus fault injection.

    Servo model (per tick): move toward the goal by `lag`·error, capped at `vmax_dps`. Load reads
    proportional to the remaining error (sign-magnitude like the real bus). Faults:
      protection — output collapses to 10 % and load stays small (Feetech overload protection)
      dropout    — reads raise BusLost until cleared
      overheat   — temperatures read 60 °C
      obstacle   — the given joint cannot move past its current position (+ high load)
    """

    def __init__(self, bounds: dict[str, tuple[float, float]], start: dict[str, float] | None = None,
                 vmax_dps: float = 120.0, lag: float = 0.6):
        self.bounds = bounds
        self.pos = dict(start) if start else {j: 0.0 for j in JOINTS}
        self.goal = dict(self.pos)
        self.torque = False
        self.vmax_dps, self.lag = vmax_dps, lag
        self.faults: dict[str, str | None] = {}  # fault → joint or None
        self.connected = False
        self.sim = True
        self.temp = 30

    # --- fault injection (test/ctl only) ---
    def inject(self, fault: str, joint: str | None = None) -> None:
        if fault == "power_cycle":  # operator pulled the 12 V plug: torque drops, protection resets
            self.torque = False
            self.faults.pop("protection", None)
            return
        self.faults[fault] = joint

    def clear(self, fault: str | None = None) -> None:
        if fault is None:
            self.faults.clear()
        else:
            self.faults.pop(fault, None)

    # --- bus surface ---
    def connect(self) -> None:
        if "dropout" in self.faults:
            raise BusLost("sim dropout")
        self.connected = True

    def disconnect(self, disable_torque: bool = False) -> None:
        if disable_torque:
            self.torque = False
        self.connected = False

    def _check(self):
        if "dropout" in self.faults:
            raise BusLost("sim dropout")

    def read_positions(self) -> dict[str, float]:
        self._check()
        return dict(self.pos)

    def read_loads(self) -> dict[str, int]:
        self._check()
        out = {}
        for j in JOINTS:
            err = self.goal[j] - self.pos[j] if self.torque else 0.0
            gain = 2.0 if "protection" in self.faults else 20.0
            if "obstacle" in self.faults and self.faults["obstacle"] == j and abs(err) > 0.5:
                gain = 60.0
            out[j] = int(np.clip(err * gain, -1000, 1000))
        return out

    def read_temps(self) -> dict[str, int]:
        self._check()
        return dict.fromkeys(JOINTS, 60 if "overheat" in self.faults else self.temp)

    def read_torque(self) -> dict[str, int]:
        self._check()
        return dict.fromkeys(JOINTS, 1 if self.torque else 0)

    def write_goals(self, goals: dict[str, float]) -> None:
        self._check()
        if self.torque:
            for j, v in goals.items():
                lo, hi = self.bounds[j]
                self.goal[j] = float(np.clip(v, lo, hi))

    accel = 32

    def set_accel(self, value: int) -> None:
        self._check()
        self.accel = int(value)

    regs: dict | None = None

    def read_regs(self, names: list[str]) -> dict[str, dict[str, int]]:
        self._check()
        if self.regs is None:
            self.regs = {"P_Coefficient": dict.fromkeys(JOINTS, 16), "D_Coefficient": dict.fromkeys(JOINTS, 32),
                         "I_Coefficient": dict.fromkeys(JOINTS, 0)}
        return {n: dict(self.regs.get(n, {})) for n in names}

    def write_regs(self, name: str, values: dict[str, int]) -> None:
        self.read_regs([name])
        self.regs.setdefault(name, {}).update({j: int(v) for j, v in values.items()})

    def torque_on_safe(self) -> None:
        self._check()
        self.goal = dict(self.pos)
        self.torque = True

    def torque_off(self) -> None:
        self._check()
        self.torque = False

    def step(self, dt: float) -> None:
        if not self.torque:
            return
        gain = 0.1 if "protection" in self.faults else self.lag
        for j in JOINTS:
            if "obstacle" in self.faults and self.faults["obstacle"] == j:
                continue
            err = self.goal[j] - self.pos[j]
            self.pos[j] += float(np.clip(err * gain, -self.vmax_dps * dt, self.vmax_dps * dt))


def bounds_from_calibration(ranges: dict[str, tuple[int, int]]) -> dict[str, tuple[float, float]]:
    """DEGREES normalisation is symmetric about the range midpoint: deg = (raw − mid)·360/4095.

    So the calibrated range is exactly ±(max−min)/2·360/4095 and the neutral pose can sit 0.3° inside
    it — no extra margin (the servo EEPROM Min/Max_Position_Limit already clamps raw goals).
    wrist_roll is calibrated over the full turn; ±150° keeps it away from the wrap point.
    """
    out = {}
    for j, (lo, hi) in ranges.items():
        half = (hi - lo) * 360.0 / 4095.0 / 2.0
        out[j] = (-WRIST_ROLL_SOFT_LIMIT, WRIST_ROLL_SOFT_LIMIT) if j == "wrist_roll" else (-half, half)
    return out
