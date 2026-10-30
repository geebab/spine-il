"""Lease identity (ident tickets), contention warnings and servo-gain tracking."""

import getpass
import random
import threading
import time

from lerobot_robot_spinearm.ident import IdentServer, fetch_ticket
from test_controller import Rig


class FakeIdent:
    """Stands in for the ident socket: tickets are just 'ticket-<user>'."""

    def redeem(self, ticket):
        return ticket[len("ticket-"):] if isinstance(ticket, str) and ticket.startswith("ticket-") else None


def rig(verify=True):
    r = Rig()
    r.ctl.verify_owner = verify
    r.ctl.ident = FakeIdent()
    return r


def lease(r, user, **kw):
    return r.do(cmd="lease", owner=kw.pop("tool", user), ident=f"ticket-{user}", **kw)


def test_verified_arm_refuses_typed_names():
    r = rig()
    assert r.do(cmd="lease", owner="owner1")["error"] == "unverified"  # no ticket: `--owner owner1` is not enough
    got = lease(r, "alice")
    assert got["ok"] and got["owner"] == "alice"
    r.run(1)
    assert r.ctl.snapshot["lease"]["verified"] and r.ctl.snapshot["lease"]["owner"] == "alice"


def test_claiming_someone_elses_name_does_not_take_the_arm():
    r = rig()
    lease(r, "owner1")
    got = r.do(cmd="lease", owner="owner1", ident="ticket-bob")  # bob types owner1's name
    assert got["error"] == "lease_held" and got["holder"] == "owner1"


def test_holder_is_warned_once_and_force_leaves_a_reason():
    r = rig()
    tok = lease(r, "alice")["token"]
    assert lease(r, "bob")["error"] == "lease_held"
    r.run(1)
    assert r.ctl.snapshot["lease"]["contest"]["by"] == "bob"
    assert "bob tried to take this arm" in r.do(cmd="renew", token=tok)["warn"]
    assert "warn" not in r.do(cmd="renew", token=tok)  # told once
    r.clock.t += 6
    took = lease(r, "bob", force=True)
    assert took["ok"] and "took the arm from alice" in took["warn"]
    bad = r.do(cmd="renew", token=tok)
    assert bad["error"] == "bad_token" and "taken by bob" in bad["detail"]


def test_same_account_second_client_needs_force_while_first_is_active():
    r = rig()
    tok = lease(r, "alice", tool="teleop")["token"]
    assert lease(r, "alice", tool="bench")["error"] == "lease_held"
    again = lease(r, "alice", tool="teleop")  # the same tool restarting may replace its own lease
    assert again["ok"] and "replaced" in again["warn"]
    assert "same account" in r.do(cmd="renew", token=tok)["detail"]


def test_sim_keeps_unverified_names_marked():
    r = rig(verify=False)
    assert r.do(cmd="lease", owner="carol")["owner"] == "carol?"


def test_gain_changes_are_tracked_restored_and_kept():
    r = rig()
    tok = lease(r, "alice")["token"]  # torque is off (state OFF) — EEPROM writes allowed
    base = r.ctl.gain_base["P_Coefficient"]["elbow_flex"]
    out = r.do(cmd="set_regs", token=tok, name="P_Coefficient", values={"elbow_flex": base + 8})
    assert out["ok"] and "differ from the baseline" in out["warn"]
    r.run(1)
    assert r.ctl.snapshot["gains_modified"]["P_Coefficient"] == {"values": {"elbow_flex": base + 8}, "by": "alice"}
    rel = r.do(cmd="release", token=tok)
    assert "still differ" in rel["warn"] and r.bus.regs["P_Coefficient"]["elbow_flex"] == base + 8  # not auto-undone
    got = lease(r, "bob")
    assert "servo gains differ" in got["warn"] and "alice" in got["warn"]  # the next person is told
    assert r.do(cmd="restore_gains", token=got["token"])["restored"] == {"P_Coefficient": {"elbow_flex": base}}
    assert r.bus.regs["P_Coefficient"]["elbow_flex"] == base and not r.ctl.gain_mods
    r.do(cmd="set_regs", token=got["token"], name="P_Coefficient", values={"elbow_flex": base + 4})
    assert r.do(cmd="keep_gains", token=got["token"])["ok"] and not r.ctl.gain_mods
    assert r.ctl.gain_base["P_Coefficient"]["elbow_flex"] == base + 4


def test_ident_socket_names_the_calling_account():
    port = random.randint(40000, 60000)
    srv = IdentServer(port)
    stop = threading.Event()
    threading.Thread(target=srv.serve, args=(stop,), daemon=True).start()
    try:
        for _ in range(50):
            ticket = fetch_ticket(port)
            if ticket:
                break
            time.sleep(0.02)
        assert srv.redeem(ticket) == getpass.getuser()
        assert srv.redeem(ticket) is None  # one-time
        assert fetch_ticket(port, host="10.0.0.1") is None  # remote: no ticket
    finally:
        stop.set()
