"""Who is asking for the arm? A one-time ticket from a local abstract Unix socket, bound to the caller's OS account.

The lease `owner` used to be whatever the client typed, so `lease --owner owner1` could take owner1's arm without a
warning. Now the server runs a tiny ident socket (Linux abstract namespace, `\\0spine-arm-<cmd port>`); a client
connects, the kernel reports its uid (SO_PEERCRED), and the server answers with a random ticket tied to that
account. The client sends the ticket with `lease`; the server redeems it once (10 s) and uses the account name as
the owner. The ZMQ wire format is unchanged — a remote client (another machine, a sim on a laptop) simply has no
ticket, and a server that verifies owners (the real arms) refuses its lease.
"""

from __future__ import annotations

import logging
import pwd
import secrets
import socket
import struct
import threading
import time

log = logging.getLogger("spine-arm.ident")
TICKET_TTL_S = 10.0


def ident_address(cmd_port: int) -> str:
    return f"\0spine-arm-{cmd_port}"


def fetch_ticket(cmd_port: int, host: str = "127.0.0.1", timeout: float = 1.0) -> str | None:
    """Ask the local arm server for an identity ticket; None when there is no local ident socket."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        return None
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect(ident_address(cmd_port))
        data = b""
        while not data.endswith(b"\n") and len(data) < 256:
            chunk = s.recv(256)
            if not chunk:
                break
            data += chunk
        return data.decode().strip() or None
    except OSError:
        return None
    finally:
        s.close()


def _username(uid: int) -> str:
    try:
        return pwd.getpwuid(uid).pw_name
    except KeyError:
        return f"uid{uid}"


class IdentServer:
    """Issues tickets on the abstract socket and redeems them for the controller (thread-safe)."""

    def __init__(self, cmd_port: int, clock=time.monotonic):
        self.address = ident_address(cmd_port)
        self.clock = clock
        self.lock = threading.Lock()
        self.tickets: dict[str, tuple[str, float]] = {}

    def issue(self, user: str) -> str:
        now = self.clock()
        ticket = secrets.token_hex(16)
        with self.lock:
            self.tickets = {t: v for t, v in self.tickets.items() if v[1] > now}
            self.tickets[ticket] = (user, now + TICKET_TTL_S)
        return ticket

    def redeem(self, ticket) -> str | None:
        if not isinstance(ticket, str):
            return None
        with self.lock:
            user, expires = self.tickets.pop(ticket, (None, 0.0))
        return user if user and self.clock() < expires else None

    def serve(self, stop: threading.Event) -> None:
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(self.address)
        srv.listen(8)
        srv.settimeout(0.2)
        log.info("ident socket @%s", self.address[1:])
        try:
            while not stop.is_set():
                try:
                    conn, _ = srv.accept()
                except TimeoutError:
                    continue
                with conn:
                    try:
                        cred = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
                        _pid, uid, _gid = struct.unpack("3i", cred)
                        conn.sendall((self.issue(_username(uid)) + "\n").encode())
                    except OSError as e:
                        log.warning("ident: %s", e)
        finally:
            srv.close()
