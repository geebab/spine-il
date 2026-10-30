#!/usr/bin/env python3
"""Read / set the arm's servo gains (EEPROM: P/D/I_Coefficient, joint-6 torque caps) through the arm server.

EEPROM writes need torque off, so `set` homes the arm (folded rest pose), switches torque off, writes, reads back
and enables torque again (the arm stays at rest, holding). LeRobot's SO-101 setup: P 16, D 32, I 0 (the Feetech
default is P 32); joint 6 capped at 50 % torque / 25 % overload.

  python robot/servo_gains.py show
  python robot/servo_gains.py set --P shoulder_lift=24,elbow_flex=24 --I elbow_flex=2
  python robot/servo_gains.py restore            # LeRobot values
  python robot/servo_gains.py baseline           # back to the values the arm server found at startup (or `keep`)
  python robot/servo_gains.py keep               # adopt the current values as the server's baseline

Gains outlive your lease: the server remembers who changed what and warns the next person until `baseline` or
`keep`. Each run is logged to ~/il-robot/calib/servo_gains_last.json (your own home).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from lerobot_robot_spinearm.client import ArmRpc
from lerobot_robot_spinearm.protocol import DEFAULT_PORTS, JOINTS

REGS = ["P_Coefficient", "D_Coefficient", "I_Coefficient", "Max_Torque_Limit", "Overload_Torque"]
LEROBOT = {"P_Coefficient": dict.fromkeys(JOINTS, 16), "D_Coefficient": dict.fromkeys(JOINTS, 32),
           "I_Coefficient": dict.fromkeys(JOINTS, 0), "Max_Torque_Limit": {"gripper": 500},
           "Overload_Torque": {"gripper": 25}}


def kv(spec: str) -> dict[str, int]:
    out = {}
    for item in filter(None, (spec or "").split(",")):
        j, v = item.split("=")
        j = j.strip()
        if j not in JOINTS and j != "all":
            sys.exit(f"unknown joint {j}")
        for jj in (JOINTS if j == "all" else [j]):
            out[jj] = int(v)
    return out


def show(rpc) -> dict:
    r = rpc.must(cmd="regs", names=REGS)["regs"]
    for n in REGS:
        print(f"{n:17s} " + " ".join(f"{j[:6]}={r[n][j]}" for j in JOINTS))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["show", "set", "restore", "baseline", "keep"])
    ap.add_argument("--arm", default="left")
    for n in ("P", "D", "I"):
        ap.add_argument(f"--{n}", default="", help=f"{n}_Coefficient, e.g. shoulder_lift=24,elbow_flex=24 or all=24")
    ap.add_argument("--max-torque", default="", help="Max_Torque_Limit (0–1000), e.g. gripper=1000")
    ap.add_argument("--overload", default="", help="Overload_Torque (%%), e.g. gripper=80")
    a = ap.parse_args()
    rpc = ArmRpc("127.0.0.1", DEFAULT_PORTS[a.arm][0], timeout_ms=5000)
    if a.action == "show":
        show(rpc)
        mods = rpc.state().get("gains_modified")
        print("baseline: " + (f"modified {mods}" if mods else "unchanged"))
        return
    if a.action == "keep":
        tok = rpc.lease("servo_gains")
        try:
            print("new baseline:", rpc.must(cmd="keep_gains", token=tok)["baseline"])
        finally:
            rpc.call(cmd="release", token=tok)
        return
    plan = LEROBOT if a.action == "restore" else {
        "P_Coefficient": kv(a.P), "D_Coefficient": kv(a.D), "I_Coefficient": kv(a.I),
        "Max_Torque_Limit": kv(a.max_torque), "Overload_Torque": kv(a.overload)}
    plan = {k: v for k, v in plan.items() if v}
    if not plan and a.action != "baseline":
        sys.exit("nothing to set")
    tok = rpc.lease("servo_gains", force=True)
    try:
        if rpc.state()["state"] != "OFF":
            rpc.must(cmd="home", token=tok)
            t0 = time.time()
            while rpc.state()["state"] == "HOMING" and time.time() - t0 < 30:
                time.sleep(0.2)
            r = rpc.call(cmd="torque_off", token=tok)
            if not r.get("ok"):
                sys.exit(f"torque_off refused: {r}")
        if a.action == "baseline":
            plan = rpc.must(cmd="restore_gains", token=tok)["restored"]
            print(f"restored {plan or 'nothing (already at baseline)'}")
        for name, vals in (plan.items() if a.action != "baseline" else ()):
            r = rpc.must(cmd="set_regs", token=tok, name=name, values=vals)
            print(f"set {name} {vals} → {r['regs'][name]}")
        rpc.must(cmd="enable", token=tok)
    finally:
        rpc.call(cmd="release", token=tok)
    show(rpc)
    out = Path.home() / "il-robot/calib/servo_gains_last.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)  # a teammate's copy may be a link to the shared (read-only) file
    out.write_text(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%S"), "arm": a.arm, "action": a.action,
                               "plan": plan}, indent=1))


if __name__ == "__main__":
    main()
