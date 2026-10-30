"""ArmController against SimBus with a fake clock — no hardware, no sockets."""

import pytest

from lerobot_robot_spinearm.protocol import DEFAULT_BOUNDS, JOINTS
from lerobot_robot_spinearm.server.bus import SimBus
from lerobot_robot_spinearm.server.controller import ArmController, Params

NEUTRAL = {"shoulder_pan": 0.0, "shoulder_lift": -100.0, "elbow_flex": 60.0, "wrist_flex": 90.0, "wrist_roll": 0.0, "gripper": -45.0}


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class Rig:
    """Controller + sim bus + clock; `run(n)` advances n ticks at 30 Hz."""

    def __init__(self, neutral=NEUTRAL, start=None, sim=True):
        self.clock = Clock()
        self.bus = SimBus(dict(DEFAULT_BOUNDS), start=start or dict(neutral))
        self.ctl = ArmController("left", self.bus, dict(DEFAULT_BOUNDS), neutral, Params(), clock=self.clock, sim=sim)
        self.run(1)  # first tick connects the bus

    def run(self, n=1):
        snap = None
        for _ in range(n):
            self.clock.t += 1 / 30
            snap = self.ctl.tick()
        return snap

    def do(self, **msg):
        """RPC-side handle; requests that need the control thread are completed by ticking."""
        import threading

        out = {}

        def call():
            out["r"] = self.ctl.handle(msg)

        th = threading.Thread(target=call)
        th.start()
        for _ in range(20):
            th.join(0.005)
            if not th.is_alive():
                break
            self.run(1)
        th.join(2)
        return out["r"]

    def lease(self, owner="alice"):
        r = self.do(cmd="lease", owner=owner)
        assert r["ok"], r
        return r["token"]

    def ready(self, owner="alice"):
        tok = self.lease(owner)
        assert self.do(cmd="enable", token=tok)["ok"]
        assert self.ctl.state == "HOLD"
        return tok


def test_starts_off_and_requires_lease_then_enable():
    rig = Rig()
    assert rig.ctl.state == "OFF" and rig.ctl.bus_ok
    assert rig.do(cmd="goal", pos={"shoulder_pan": 5})["error"] == "no_lease"
    tok = rig.lease()
    assert rig.do(cmd="goal", token=tok, pos={"shoulder_pan": 5})["error"] == "off"
    assert rig.do(cmd="enable", token=tok)["ok"]
    assert rig.ctl.state == "HOLD" and rig.bus.torque


def test_bounds_and_step_cap():
    rig = Rig()
    tok = rig.ready()
    r = rig.do(cmd="goal", token=tok, pos={"shoulder_pan": 500, "wrist_roll": -400, "elbow_flex": 60})
    assert r["ok"] and r["sent"]["shoulder_pan"] == 110 and r["sent"]["wrist_roll"] == -150
    assert rig.ctl.state == "MOVING"
    rig.run(1)
    assert abs(rig.ctl.cmd["shoulder_pan"] - 0.0) <= 10.0 + 1e-6  # one tick → at most step_deg from start
    assert rig.do(cmd="goal", token=tok, pos={"bogus": 1})["error"] == "bad_request"


def test_goal_tracks_target_over_ticks():
    rig = Rig()
    tok = rig.ready()
    for _ in range(40):
        rig.do(cmd="goal", token=tok, pos={"shoulder_pan": 30})
        rig.run(1)
    assert abs(rig.bus.pos["shoulder_pan"] - 30) < 2.0


def test_latency_stamps_follow_goal_seq_to_the_bus_write():
    rig = Rig()
    tok = rig.ready()
    snap = rig.run(1)
    assert snap["tr"] is not None and snap["tw"] is None and snap["gseq"] is None  # holding: read, no write
    r = rig.do(cmd="goal", token=tok, pos={"shoulder_pan": 20}, seq=7)
    assert r["ok"] and r["rx"] == round(rig.clock.t, 4)
    rx = r["rx"]
    snap = rig.run(1)
    assert snap["gseq"] == 7 and snap["grx"] == rx and snap["tw"] >= snap["tr"] >= rx
    assert snap["cmd"]["shoulder_pan"] == pytest.approx(10.0, abs=0.01)  # step cap from 0
    rig.do(cmd="goal", token=tok, pos={"shoulder_pan": 20}, seq=8)
    assert rig.run(1)["gseq"] == 8


def test_watchdog_holds_when_client_stops_streaming():
    rig = Rig()
    tok = rig.ready()
    rig.do(cmd="goal", token=tok, pos={"shoulder_pan": 60})
    rig.run(3)
    assert rig.ctl.state == "MOVING"
    rig.run(10)  # 333 ms without a goal
    assert rig.ctl.state == "HOLD" and rig.ctl.target is None
    p = rig.bus.pos["shoulder_pan"]
    rig.run(10)
    assert abs(rig.bus.pos["shoulder_pan"] - p) < 1e-6  # nothing written while idle


def test_lease_exclusive_force_and_expiry():
    rig = Rig()
    tok = rig.lease("alice")
    r = rig.do(cmd="lease", owner="bob")
    assert r["error"] == "lease_held" and r["holder"] == "alice?"  # unverified name (sim)
    assert rig.do(cmd="lease", owner="bob", force=True)["error"] == "lease_held"  # alice not idle yet
    rig.clock.t += 6
    assert rig.do(cmd="lease", owner="bob", force=True)["ok"]  # idle ≥ 5 s
    assert rig.do(cmd="renew", token=tok)["error"] == "bad_token"
    rig.clock.t += 20
    rig.run(1)
    assert rig.ctl.lease is None
    assert rig.do(cmd="hold", token="x")["error"] == "no_lease"


def test_estop_from_anyone_and_recovery():
    rig = Rig()
    tok = rig.ready()
    rig.do(cmd="goal", token=tok, pos={"shoulder_pan": 40})
    rig.run(2)
    assert rig.do(cmd="estop")["ok"]
    assert rig.ctl.state == "ESTOP" and rig.ctl.lease is None and rig.bus.torque
    tok2 = rig.lease("bob")
    assert rig.do(cmd="goal", token=tok2, pos={"shoulder_pan": 1})["error"] == "estop"
    assert rig.do(cmd="enable", token=tok2)["error"] == "estop"
    assert rig.do(cmd="clear_faults", token=tok2)["ok"] and rig.ctl.state == "HOLD"


def test_overload_protection_detected_and_needs_power_cycle():
    rig = Rig()
    tok = rig.ready()
    rig.bus.inject("protection")
    for _ in range(40):
        rig.do(cmd="goal", token=tok, pos={"shoulder_pan": 90})
        rig.run(1)
        if rig.ctl.state == "FAULT":
            break
    assert "overload_protection" in rig.ctl.faults and rig.ctl.state == "FAULT" and rig.bus.torque
    assert rig.do(cmd="clear_faults", token=tok)["error"] == "fault"
    assert rig.do(cmd="enable", token=tok)["error"] == "fault"
    rig.bus.inject("power_cycle")  # operator pulls 12 V: torque drops, protection resets
    rig.run(31)  # slow read notices Torque_Enable == 0
    assert rig.ctl.state == "OFF" and "power_cycled" in rig.ctl.faults and "overload_protection" not in rig.ctl.faults
    assert rig.do(cmd="enable", token=tok)["ok"] and rig.ctl.state == "HOLD" and not rig.ctl.faults


def test_stall_under_load():
    rig = Rig()
    tok = rig.ready()
    rig.bus.inject("obstacle", "elbow_flex")
    for _ in range(40):
        rig.do(cmd="goal", token=tok, pos={"elbow_flex": 20})
        rig.run(1)
        if rig.ctl.state == "FAULT":
            break
    assert "stall" in rig.ctl.faults
    rig.bus.clear("obstacle")
    assert rig.do(cmd="clear_faults", token=tok)["ok"] and rig.ctl.state == "HOLD"


def test_dropout_and_reconnect():
    rig = Rig()
    tok = rig.ready()
    rig.bus.inject("dropout")
    rig.run(2)
    assert rig.ctl.state == "FAULT" and "bus_lost" in rig.ctl.faults and not rig.ctl.bus_ok
    assert rig.do(cmd="enable", token=tok)["error"] == "bus_lost"
    rig.bus.clear("dropout")
    rig.clock.t += 3
    rig.run(2)
    assert rig.ctl.bus_ok and "bus_lost" not in rig.ctl.faults
    assert rig.ctl.state == "HOLD"  # torque was still on → holds in place


def test_overheat_blocks_until_cool():
    rig = Rig()
    tok = rig.ready()
    rig.bus.inject("overheat")
    rig.run(61)  # two consecutive slow reads
    assert "overheat" in rig.ctl.faults and rig.ctl.state == "FAULT"
    assert rig.do(cmd="clear_faults", token=tok)["error"] == "fault"
    rig.bus.clear("overheat")
    rig.run(31)
    assert "overheat" not in rig.ctl.faults and rig.ctl.state == "HOLD"


def test_home_folds_joint6_first_then_rest_and_torque_off_only_at_neutral():
    start = {**NEUTRAL, "shoulder_pan": 40.0, "elbow_flex": 10.0, "gripper": 30.0}
    rig = Rig(start=start)
    tok = rig.ready()
    assert rig.do(cmd="torque_off", token=tok)["error"] == "not_at_neutral"
    assert rig.do(cmd="home", token=tok)["ok"] and rig.ctl.state == "HOMING"
    # phase 1: only the gripper moves
    for _ in range(3):
        rig.run(1)
    assert abs(rig.bus.pos["shoulder_pan"] - 40.0) < 1e-6 and rig.bus.pos["gripper"] < 30.0
    rig.run(200)
    assert rig.ctl.state == "HOLD"
    assert all(abs(rig.bus.pos[j] - NEUTRAL[j]) <= 3.0 for j in JOINTS)
    assert rig.do(cmd="torque_off", token=tok)["ok"] and rig.ctl.state == "OFF" and not rig.bus.torque


def test_home_stall_keeps_torque():
    rig = Rig(start={**NEUTRAL, "shoulder_pan": 40.0})
    tok = rig.ready()
    rig.bus.inject("obstacle", "shoulder_pan")
    assert rig.do(cmd="home", token=tok)["ok"]
    rig.run(60)
    assert "home_stalled" in rig.ctl.faults and rig.bus.torque


def test_neutral_without_degrees_units_skips_gripper():
    rig = Rig(neutral={j: v for j, v in NEUTRAL.items() if j != "gripper"}, start={**NEUTRAL, "wrist_flex": 0.0})
    tok = rig.ready()
    assert rig.do(cmd="home", token=tok)["ok"]
    assert len(rig.ctl._home["phases"]) == 1


def test_release_drops_lease_and_holds():
    rig = Rig()
    tok = rig.ready()
    rig.do(cmd="goal", token=tok, pos={"shoulder_pan": 40})
    rig.run(2)
    assert rig.do(cmd="release", token=tok)["ok"]
    assert rig.ctl.lease is None and rig.ctl.state == "HOLD" and rig.bus.torque
    assert rig.do(cmd="lease", owner="bob")["ok"]  # free again immediately


def test_fake_only_in_sim():
    rig = Rig(sim=False)
    assert rig.do(cmd="fake", fault="dropout")["error"] == "sim_only"


@pytest.mark.parametrize("cmd", ["renew", "release", "enable", "hold", "home", "torque_off", "clear_faults"])
def test_token_required(cmd):
    rig = Rig()
    rig.lease("alice")
    assert rig.do(cmd=cmd, token="nope")["error"] == "bad_token"


def test_reconnect_clears_stale_stall():
    rig = Rig()
    rig.ready()
    rig.ctl.faults.add("stall"); rig.ctl.state = "FAULT"
    rig.bus.inject("dropout"); rig.run(3)
    assert not rig.ctl.bus_ok
    rig.bus.clear("dropout"); rig.run(80)  # > 2 s reconnect interval
    assert rig.ctl.bus_ok and "stall" not in rig.ctl.faults


def test_single_hot_read_does_not_fault():
    rig = Rig()
    rig.ready()
    rig.bus.inject("overheat"); rig.run(31)   # one slow read sees it
    rig.bus.clear("overheat"); rig.run(60)
    assert "overheat" not in rig.ctl.faults and rig.ctl.state == "HOLD"


def test_set_accel_goes_to_the_bus_and_is_published():
    rig = Rig()
    tok = rig.ready()
    r = rig.do(cmd="set", token=tok, accel=100)
    assert r["ok"] and r["accel"] == 100 and rig.bus.accel == 100
    assert rig.run(1)["accel"] == 100
    assert rig.do(cmd="set", token=tok, accel=1000)["accel"] == 254  # clamped
