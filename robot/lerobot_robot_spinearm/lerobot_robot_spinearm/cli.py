"""spine-arm-ctl — talk to an arm server from the shell.

  spine-arm-ctl --arm left status
  spine-arm-ctl --arm left watch                # 30 Hz state stream (no lease needed)
  spine-arm-ctl --arm left lease [--force]      # token is cached in ~/.cache/spine-arm/
  spine-arm-ctl --arm left enable | hold | home [--wait] | release | torque-off [--force] | clear
  spine-arm-ctl --arm left goal shoulder_pan=10 elbow_flex=-20
  spine-arm-ctl --arm left estop               # anyone, anytime
  spine-arm-ctl --arm left fake protection     # --sim servers only: protection|dropout|overheat|obstacle|power_cycle|clear
  spine-arm-ctl --arm left record-neutral --out robot/neutral_pose.json   # current pose → neutral (degrees)
Use --cmd-port/--pub-port for a --sim server on custom ports.
"""

from __future__ import annotations

import argparse
import getpass
import json
import sys
import time
from pathlib import Path

from .client import ArmError, ArmRpc, watch
from .protocol import DEFAULT_PORTS, JOINTS

TOKEN_DIR = Path.home() / ".cache" / "spine-arm"


def token_path(host: str, port: int) -> Path:
    return TOKEN_DIR / f"token-{host}-{port}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["left", "right"], default="left")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--cmd-port", type=int)
    ap.add_argument("--pub-port", type=int)
    ap.add_argument("--token", help="lease token (default: cached from the last `lease`)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("status", "state", "enable", "hold", "release", "clear", "estop", "renew"):
        sub.add_parser(name)
    p = sub.add_parser("watch"); p.add_argument("--n", type=int, default=0, help="stop after n snapshots (0 = forever)")
    p = sub.add_parser("lease"); p.add_argument("--owner", default=getpass.getuser()); p.add_argument("--force", action="store_true")
    p = sub.add_parser("home"); p.add_argument("--wait", action="store_true")
    p = sub.add_parser("torque-off"); p.add_argument("--force", action="store_true")
    p = sub.add_parser("goal"); p.add_argument("pairs", nargs="+", help="joint=deg …")
    p = sub.add_parser("set"); p.add_argument("--step-deg", type=float, required=True)
    p = sub.add_parser("fake"); p.add_argument("fault"); p.add_argument("--joint")
    p = sub.add_parser("record-neutral"); p.add_argument("--out", required=True); p.add_argument("--id", help="default spine_<arm>")
    args = ap.parse_args(argv)

    cmd_port, pub_port = DEFAULT_PORTS[args.arm]
    cmd_port, pub_port = args.cmd_port or cmd_port, args.pub_port or pub_port

    if args.cmd == "watch":
        n = 0
        for snap in watch(args.host, pub_port):
            if snap is None:
                print("(no state for 2 s — server down?)", file=sys.stderr)
                continue
            pos = " ".join(f"{j[:9]}{snap['pos'].get(j, float('nan')):7.1f}" for j in JOINTS)
            lease = snap["lease"]["owner"] if snap["lease"] else "-"
            if snap["lease"] and snap["lease"].get("contest"):
                lease += f" ⚠{snap['lease']['contest']['by']} wants it"
            if snap.get("gains_modified"):
                lease += " ⚠gains"
            print(f"\r{snap['state']:7} {pos}  lease={lease:8} faults={','.join(snap['faults']) or '-'}   ", end="", flush=True)
            n += 1
            if args.n and n >= args.n:
                print()
                break
        return 0

    rpc = ArmRpc(args.host, cmd_port)
    tpath = token_path(args.host, cmd_port)
    token = args.token or (tpath.read_text().strip() if tpath.is_file() else None)

    try:
        if args.cmd == "lease":
            reply = rpc.must(**rpc.lease_msg(args.owner, args.force))
            TOKEN_DIR.mkdir(parents=True, exist_ok=True)
            tpath.write_text(reply["token"])
            tpath.chmod(0o600)
            print(f"lease OK as {reply.get('owner', args.owner)} (ttl {reply['ttl']} s, state {reply['state']}) — token cached in {tpath}")
            return 0
        if args.cmd in ("status", "state"):
            print(json.dumps(rpc.must(cmd=args.cmd), indent=2, ensure_ascii=False))
            return 0
        if args.cmd == "estop":
            print(json.dumps(rpc.must(cmd="estop")))
            return 0
        if args.cmd == "fake":
            print(json.dumps(rpc.must(cmd="fake", fault=args.fault, joint=args.joint)))
            return 0
        if args.cmd == "record-neutral":
            snap = rpc.must(cmd="state")
            rid = args.id or f"spine_{args.arm}"
            out = Path(args.out)
            data = json.loads(out.read_text()) if out.is_file() else {}
            data["_note"] = "중립 자세(관절 °, 6번 포함). 팔 서버의 home 명령이 여기로 복귀한다."
            data["_units"] = "deg"
            data[rid] = {j: round(float(v), 2) for j, v in snap["pos"].items()}
            out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
            print(f"{rid} → {out}: {data[rid]}  (restart the arm server to reload)")
            return 0
        if token is None:
            print("no token — run `lease` first (or pass --token)", file=sys.stderr)
            return 2
        if args.cmd == "goal":
            pos = {}
            for pair in args.pairs:
                j, v = pair.split("=")
                pos[j] = float(v)
            print(json.dumps(rpc.must(cmd="goal", token=token, pos=pos)))
            return 0
        if args.cmd == "home":
            print(json.dumps(rpc.must(cmd="home", token=token)))
            if args.wait:
                while True:
                    time.sleep(0.3)
                    s = rpc.must(cmd="state")
                    rpc.call(cmd="renew", token=token)
                    print(f"\r{s['state']:7} " + " ".join(f"{j[:9]}{s['pos'].get(j, 0):7.1f}" for j in JOINTS), end="", flush=True)
                    if s["state"] != "HOMING":
                        print()
                        break
            return 0
        if args.cmd == "torque-off":
            print(json.dumps(rpc.must(cmd="torque_off", token=token, force=args.force)))
            return 0
        if args.cmd == "set":
            print(json.dumps(rpc.must(cmd="set", token=token, step_deg=args.step_deg)))
            return 0
        server_cmd = {"clear": "clear_faults"}.get(args.cmd, args.cmd)
        print(json.dumps(rpc.must(cmd=server_cmd, token=token)))
        return 0
    except ArmError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        rpc.close()


if __name__ == "__main__":
    raise SystemExit(main())
