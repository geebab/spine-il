"""RPC thread + REQ clients over inproc (shared ZMQ context), controller ticked by a control thread."""

import threading
import time

import pytest
import zmq

from lerobot_robot_spinearm.client import ArmRpc
from lerobot_robot_spinearm.protocol import DEFAULT_BOUNDS
from lerobot_robot_spinearm.server.bus import SimBus
from lerobot_robot_spinearm.server.controller import ArmController
from lerobot_robot_spinearm.server.rpc import serve_rpc


@pytest.fixture
def server():
    ctx = zmq.Context()
    bus = SimBus(dict(DEFAULT_BOUNDS))
    ctl = ArmController("left", bus, dict(DEFAULT_BOUNDS), None, sim=True)
    stop = threading.Event()
    ep = "inproc://arm-test"
    threading.Thread(target=serve_rpc, args=(ctl, ep, stop, ctx), daemon=True).start()

    def loop():
        while not stop.is_set():
            ctl.tick()
            time.sleep(1 / 60)

    threading.Thread(target=loop, daemon=True).start()
    time.sleep(0.05)
    yield ctx, ep, ctl, bus
    stop.set()
    time.sleep(0.2)
    ctx.term()


def rpc(ctx, ep):
    r = ArmRpc.__new__(ArmRpc)
    r._zmq, r.endpoint, r.timeout_ms, r.ctx, r.sock = zmq, ep, 1000, ctx, None
    r._open()
    return r


def test_lease_exclusive_and_estop_from_non_holder(server):
    ctx, ep, ctl, _ = server
    a, b = rpc(ctx, ep), rpc(ctx, ep)
    tok = a.must(cmd="lease", owner="alice")["token"]
    r = b.call(cmd="lease", owner="bob")
    assert r["error"] == "lease_held" and r["holder"] == "alice?"  # unverified name (sim)
    assert a.must(cmd="enable", token=tok)["state"] == "HOLD"
    assert a.must(cmd="goal", token=tok, pos={"shoulder_pan": 20})["state"] == "MOVING"
    assert b.must(cmd="estop")["state"] == "ESTOP"
    assert a.call(cmd="goal", token=tok, pos={"shoulder_pan": 20})["error"] == "no_lease"  # estop dropped it
    assert b.must(cmd="status")["state"] == "ESTOP"


def test_bad_json_and_bad_token_get_replies(server):
    ctx, ep, _, _ = server
    raw = ctx.socket(zmq.REQ)
    raw.setsockopt(zmq.RCVTIMEO, 1000)
    raw.connect(ep)
    raw.send(b"{not json")
    assert b"bad_request" in raw.recv()
    raw.close()
    c = rpc(ctx, ep)
    c.must(cmd="lease", owner="x")
    assert c.call(cmd="hold", token="wrong")["error"] == "bad_token"


def test_state_stream_shape(server):
    ctx, ep, _, _ = server
    s = rpc(ctx, ep).must(cmd="state")
    assert {"state", "pos", "load", "temp", "lease", "faults", "bus_ok", "tick"} <= set(s)
