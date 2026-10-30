"""Arm-server command client (ZMQ REQ, JSON). One request in flight; a lost reply would wedge a REQ socket, so the
socket is rebuilt after a timeout (the server's REP side only loses that reply — see server/rpc.py)."""

from __future__ import annotations

import json
import socket

import zmq


def ident_ticket(endpoint: str) -> str | None:
    """One-time ticket from the local arm server's ident socket (abstract `\\0spine-arm-<port>`); it proves to the
    server which OS account is asking for the lease (see lerobot_robot_spinearm/ident.py)."""
    host, _, port = endpoint.rpartition(":")
    if not host.endswith(("127.0.0.1", "localhost")):
        return None
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(1.0)
    try:
        s.connect(f"\0spine-arm-{port}")
        return s.recv(256).decode().strip() or None
    except OSError:
        return None
    finally:
        s.close()


class Rpc:
    def __init__(self, endpoint: str, timeout_ms: int = 500):
        self.endpoint, self.timeout_ms = endpoint, timeout_ms
        self.ctx = zmq.Context.instance()
        self.sock = None
        self._open()

    def _open(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(self.endpoint)

    def call(self, **msg) -> dict:
        if msg.get("cmd") == "lease":
            ticket = ident_ticket(self.endpoint)
            if ticket:
                msg["ident"] = ticket
        self.sock.send_string(json.dumps(msg))
        if not self.sock.poll(self.timeout_ms):
            self._open()
            return {"ok": False, "error": "timeout"}
        return json.loads(self.sock.recv())
