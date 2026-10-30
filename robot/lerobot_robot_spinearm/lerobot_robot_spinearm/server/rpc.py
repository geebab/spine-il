"""ZMQ REP loop for the arm server. Runs in its own thread; owns its socket.

A REP socket cannot deadlock on a client that dies mid-request: only the reply is lost. Handlers
never block on bus I/O — `ArmController.handle()` either answers from the last snapshot or waits
(≤ 2 s) for the control thread.
"""

from __future__ import annotations

import json
import logging
import threading

from ..protocol import error

log = logging.getLogger("spine-arm.rpc")


def serve_rpc(controller, endpoint: str, stop: threading.Event, context=None) -> None:
    import zmq

    ctx = context or zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(endpoint)
    log.info("rpc listening on %s", endpoint)
    try:
        while not stop.is_set():
            if not sock.poll(100):
                continue
            raw = sock.recv()
            try:
                msg = json.loads(raw)
                if not isinstance(msg, dict):
                    raise ValueError("message must be a JSON object")
                reply = controller.handle(msg)
            except Exception as e:  # never leave the REP socket without a reply
                log.exception("bad request")
                reply = error("bad_request", detail=str(e))
            sock.send_string(json.dumps(reply))
    finally:
        sock.close()
