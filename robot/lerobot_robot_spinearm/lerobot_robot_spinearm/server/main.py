"""spine-arm-server — owns one SO-101 arm (or a simulated one) and serves it over ZMQ.

  spine-arm-server --arm left                          # real arm on /dev/so101_left, ports 5601/5602
  spine-arm-server --arm left --sim --cmd-port 6601    # virtual arm (any user, any machine)

Startup is torque-OFF; a client must `lease` then `enable`. Ctrl+C / SIGTERM: hold in place, keep torque.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

from ..protocol import DEFAULT_BOUNDS, DEFAULT_PORTS, JOINTS, cmd_endpoint
from .bus import RealBus, SimBus, bounds_from_calibration
from .controller import ArmController, Params
from .rpc import serve_rpc
from ..ident import IdentServer

log = logging.getLogger("spine-arm")
REPO_ROBOT_DIR = Path(__file__).resolve().parents[3]  # …/spine-il/robot


def find_calibration(arm: str, rid: str, explicit: str | None) -> Path | None:
    candidates = [explicit] if explicit else [
        f"/srv/il-robot/calibration/{rid}.json",
        REPO_ROBOT_DIR / "calibration" / f"{rid}.json",
        Path.home() / f".cache/huggingface/lerobot/calibration/robots/so_follower/{rid}.json",
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return Path(c)
    return None


def load_neutral(path: Path | None, rid: str) -> dict[str, float] | None:
    if not path or not path.is_file():
        return None
    data = json.loads(path.read_text())
    pose = data.get(rid)
    if not pose:
        return None
    pose = {j: float(v) for j, v in pose.items() if j in JOINTS}
    if data.get("_units") != "deg" and "gripper" in pose:
        # Recorded while joint 6 was still normalised 0..100 (gripper) — not degrees. Skip phase 1.
        log.warning("%s has no '_units': 'deg' — ignoring its gripper value (record it again with spine-arm-ctl record-neutral)", path)
        pose.pop("gripper")
    return pose


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["left", "right"], required=True)
    ap.add_argument("--port", help="serial port (default /dev/so101_<arm>)")
    ap.add_argument("--id", help="calibration id (default spine_<arm>)")
    ap.add_argument("--calibration", help="calibration JSON (default: /srv/il-robot/calibration, repo, ~/.cache)")
    ap.add_argument("--neutral", help="neutral_pose.json (default: repo robot/neutral_pose.json)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cmd-port", type=int)
    ap.add_argument("--pub-port", type=int)
    ap.add_argument("--sim", action="store_true", help="virtual arm instead of the serial bus")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--step-deg", type=float, default=10.0, help="per-tick joint step cap (°)")
    ap.add_argument("--accel", type=int, default=32, help="servo Acceleration register (8–254; runtime: set{accel})")
    ap.add_argument("--log", help="also append log lines to this file")
    ap.add_argument("--verify-owner", choices=["auto", "on", "off"], default="auto",
                    help="lease only to a kernel-verified local account (auto: on for a real arm, off for --sim)")
    args = ap.parse_args(argv)

    rid = args.id or f"spine_{args.arm}"
    cmd_port, pub_port = DEFAULT_PORTS[args.arm]
    cmd_port, pub_port = args.cmd_port or cmd_port, args.pub_port or pub_port
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if args.log:
        handlers.append(logging.FileHandler(args.log))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers, force=True)

    calib = find_calibration(args.arm, rid, args.calibration)
    if args.sim:
        ranges = None
        if calib:
            raw = json.loads(calib.read_text())
            ranges = {j: (raw[j]["range_min"], raw[j]["range_max"]) for j in JOINTS}
        bounds = bounds_from_calibration(ranges) if ranges else dict(DEFAULT_BOUNDS)
        neutral = load_neutral(Path(args.neutral) if args.neutral else REPO_ROBOT_DIR / "neutral_pose.json", rid)
        start = dict(neutral) if neutral and len(neutral) == 6 else {j: 0.0 for j in JOINTS}
        bus = SimBus(bounds, start=start)
        log.info("SIM arm %s (bounds from %s)", args.arm, calib or "defaults")
    else:
        if not calib:
            log.error("no calibration file for %s", rid)
            return 2
        bus = RealBus(args.port or f"/dev/so101_{args.arm}", calib)
        bounds = bus.bounds()
        neutral = load_neutral(Path(args.neutral) if args.neutral else REPO_ROBOT_DIR / "neutral_pose.json", rid)
        log.info("arm %s on %s, calibration %s", args.arm, args.port or f"/dev/so101_{args.arm}", calib)

    verify = args.verify_owner == "on" or (args.verify_owner == "auto" and not args.sim)
    ctl = ArmController(args.arm, bus, bounds, neutral, Params(rate_hz=args.rate, step_deg=args.step_deg, accel=args.accel),
                        sim=args.sim, verify_owner=verify)
    stop = threading.Event()
    ctl.ident = IdentServer(cmd_port)
    threading.Thread(target=ctl.ident.serve, args=(stop,), daemon=True).start()
    log.info("lease owners %s", "verified (ident ticket required)" if verify else "not verified (typed names shown with '?')")
    rpc = threading.Thread(target=serve_rpc, args=(ctl, cmd_endpoint(args.host, cmd_port), stop), daemon=True)
    rpc.start()

    import zmq

    pub = zmq.Context.instance().socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 10)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(cmd_endpoint(args.host, pub_port))
    log.info("pub on %s:%d · pid %d", args.host, pub_port, os.getpid())

    def on_signal(signum, _frame):
        log.info("signal %s → hold + keep torque", signum)
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    period = 1.0 / args.rate
    last_state = None
    try:
        while not stop.is_set():
            t0 = time.perf_counter()
            snap = ctl.tick()
            pub.send_string("state " + json.dumps(snap))
            lease = snap["lease"] or {}
            key = (snap["state"], tuple(snap["faults"]), lease.get("owner"), (lease.get("contest") or {}).get("by"),
                   bool(snap.get("gains_modified")))
            if key != last_state:
                pub.send_string("event " + json.dumps({"t": snap["t"], "state": snap["state"], "faults": snap["faults"],
                                                       "lease": snap["lease"], "gains_modified": snap.get("gains_modified")}))
                last_state = key
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    finally:
        try:
            ctl.handle({"cmd": "estop"}) if ctl.state in ("MOVING", "HOMING") else None
        except Exception:
            pass
        ctl.tick()
        bus.disconnect(disable_torque=False)
        pub.close()
        log.info("stopped (torque kept)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
