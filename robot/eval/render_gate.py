#!/usr/bin/env python3
"""Render work gives way to the robot (user, 10-09: "외부에서 로봇 런 시도가 들어오면 렌더 pause하도록").

Until 10-09 it was the other way round: a robot run waited until the render API was idle (ro_queue2.sh "waiting for
renders to finish") — a 5-panel video could hold a run back for an hour. Now a watcher (`watch`, the user service
ube-render-gate) looks once a second for signs that the robot is in use, and while there is one

  * the render service's running job waits between two frames (bridge spine_render/pause.py: the pause file, which this
    watcher keeps fresh);
  * the processes that do render-type work on this machine — the clients of the render API and the CPU work around them
    (RENDER_SCRIPTS: video composing and encoding, evaluations, dataset exports, policy rollouts), with all their
    children, and whatever was started through `run` — are stopped (SIGSTOP);
  * work registered with `run --on-pause kill` is ended (a dry run: its step session cannot wait, and it would stand in
    a real run's way).

LINGER_S seconds after the last sign is gone everything goes on where it stopped (SIGCONT, pause file removed).

Signs that the robot is in use
  * a hold — `hold NAME`, taken by a script that is about to use the robot (the queue, when it takes a job) and lifted
    with `release NAME`; it counts only while the process that took it lives;
  * a robot process — an arm follower (endo_follow.py), a run's recorder page (robot_origin_rec.py), the two-arm
    collector (collect_both.py), the policy loop on the arms (live_loop.py --live-run; not a dry run's FAKE_ run);
  * an arm that is leased or not OFF — whoever drives it, from whichever account.
A robot job that is merely listed is not a sign: the queue takes it within 30 s and holds then (a sign for a listed job
would stop the queue's own evaluation of the session before, and the queue would never get to the job).

    render_gate.py watch                      the watcher (one at a time)
    render_gate.py status [--json]            exit 0 clear, 1 paused, 2 no watcher
    render_gate.py hold NAME [--pid N] [--reason TEXT] [--wait S]
                                              exit 0 render work is at rest, 3 not within S seconds, 4 no watcher
    render_gate.py release NAME
    render_gate.py settled                    exit 0 when render work is at rest under a pause
    render_gate.py run [--on-pause stop|kill] [--low] -- CMD …
                                              CMD as registered render work; not started while the robot is in use
                                              (stop: waits, kill: exit 75); --low: nice 15, cores 12-15, 2 math threads,
                                              a memory ceiling of 2 GB (--mem-gb) — past it the work is killed, alone
    render_gate.py resume                     by hand: continue every stopped render process (a watcher that died hard)

Files: $XDG_RUNTIME_DIR/ube-render/{pause, gate.json, holds/, work/} (tmpfs — nothing survives a reboot, and then no
process does either); log ~/il-robot/runs/robot_origin/render_gate.log.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RUNTIME = Path(os.environ.get("UBE_RENDER_GATE_DIR") or (Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}") / "ube-render"))
PAUSE = Path(os.environ.get("UBE_RENDER_PAUSE_FILE") or RUNTIME / "pause")    # = spine_render.pause.pause_file()
STATE, HOLDS, WORK = RUNTIME / "gate.json", RUNTIME / "holds", RUNTIME / "work"
LOG = Path(os.environ.get("UBE_RENDER_GATE_LOG") or Path.home() / "il-robot/runs/robot_origin/render_gate.log")
FRESH_S = 15.0          # as spine_render.pause.FRESH_S: a pause file older than this is a dead watcher's
ALIVE_S = 6.0           # the watcher writes its state every cycle; older than this = not running
LINGER_S = 10.0         # after the last sign: render work goes on this much later (an arm that blinks OFF between two moves)
KILL_GRACE_S = 5.0
APIS = [u for u in os.environ.get("RENDER_API_URLS", "http://127.0.0.1:18780,http://127.0.0.1:18781").split(",") if u]

# python scripts whose running means: the robot is in use
ROBOT_SCRIPTS = {"endo_follow.py": "an arm follower runs", "robot_origin_rec.py": "a robot run's recorder page is open",
                 "collect_both.py": "the two-arm collector runs"}
# python scripts that do render-type work: clients of the render API and the CPU work around them
RENDER_SCRIPTS = {"render_video.py", "policy_eval.py", "ro_iou.py", "iou_eval.py", "export_lerobot.py", "phase_video.py",
                  "rollout.py", "done_eval.py", "openloop_eval.py", "end_level.py", "vol_eval.py", "equiv_check.py"}


# ---------------------------------------------------------------- the process table
def procs() -> dict[int, dict]:
    """pid → {argv, state, ppid, pgid, start}; kernel threads and processes that vanish while read are left out."""
    out = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        try:
            raw = Path(f"/proc/{d}/cmdline").read_bytes()
            if not raw:
                continue
            r = Path(f"/proc/{d}/stat").read_text().rsplit(")", 1)[1].split()
        except (OSError, IndexError):
            continue
        out[int(d)] = {"argv": raw.rstrip(b"\0").decode("utf-8", "replace").split("\0"), "state": r[0], "ppid": int(r[1]),
                       "pgid": int(r[2]), "start": int(r[19])}
    return out


def script_of(argv: list[str]) -> str | None:
    """The file a python process runs (its basename), or its -m module. None for everything else — a shell whose command
    line merely mentions a script (`bash -c "… endo_follow.py …"`, a grep, an editor) is not that script running."""
    if not argv or not os.path.basename(argv[0]).startswith("python"):
        return None
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "-m":
            return argv[i + 1] if i + 1 < len(argv) else None
        if a == "-c":
            return None
        if a in ("-X", "-W"):
            i += 2
        elif a.startswith("-"):
            i += 1
        else:
            return os.path.basename(a)
    return None


def alive(pid: int, start: int | None, table: dict | None = None) -> bool:
    """The process with this pid is the one that was recorded (same start time) — pids are reused."""
    if table is not None:
        p = table.get(pid)
        return p is not None and (start is None or p["start"] == start)
    try:
        r = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return False
    return r[0] != "Z" and (start is None or int(r[19]) == start)


def start_of(pid: int) -> int | None:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def descendants(table: dict, roots: set[int]) -> set[int]:
    kids: dict[int, list[int]] = {}
    for pid, p in table.items():
        kids.setdefault(p["ppid"], []).append(pid)
    out, todo = set(), list(roots)
    while todo:
        x = todo.pop()
        if x in out:
            continue
        out.add(x)
        todo += kids.get(x, [])
    return out


# ---------------------------------------------------------------- small files
def read_json(p: Path) -> dict | None:
    try:
        d = json.loads(p.read_text())
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def write_json(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(f".{p.name}.{os.getpid()}")
    t.write_text(json.dumps(d, ensure_ascii=False))
    os.replace(t, p)


def log(msg: str) -> None:
    line = f"{time.strftime('%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass


def pause_reason(now: float | None = None) -> str | None:
    """As spine_render.pause.pause_reason (this file stands alone: the watcher runs without the bridge installed)."""
    try:
        age = (time.time() if now is None else now) - PAUSE.stat().st_mtime
    except OSError:
        return None
    if age > FRESH_S:
        return None
    return str((read_json(PAUSE) or {}).get("reason") or "pause file")


def watcher(now: float | None = None) -> dict | None:
    """The running watcher's last state, None when none runs."""
    st = read_json(STATE)
    if not st or (time.time() if now is None else now) - float(st.get("t") or 0) > ALIVE_S:
        return None
    return st if alive(int(st.get("pid") or 0), st.get("start")) else None


# ---------------------------------------------------------------- signs that the robot is in use
def hold_signs(table: dict | None = None, clean: bool = True) -> list[str]:
    out = []
    for f in sorted(HOLDS.glob("*.json")) if HOLDS.is_dir() else []:
        h = read_json(f)
        if h and alive(int(h.get("pid") or 0), h.get("start"), table):
            out.append(f"hold {f.stem}: {h.get('reason') or 'robot'}")
        elif clean:
            f.unlink(missing_ok=True)     # its owner is gone: a hold never outlives the script that took it
    return out


def process_signs(table: dict) -> list[str]:
    out = []
    for pid, p in table.items():
        s = script_of(p["argv"])
        if s in ROBOT_SCRIPTS:
            out.append(f"{ROBOT_SCRIPTS[s]} (pid {pid})")
        elif s == "live_loop.py" and "--live-run" in p["argv"]:
            i = p["argv"].index("--live-run")
            if i + 1 < len(p["argv"]) and not p["argv"][i + 1].startswith("FAKE_"):
                out.append(f"the policy loop drives the arms (pid {pid})")
    return sorted(set(out))


_RPC: dict[str, object] = {}


def arm_signs(timeout_ms: int = 300) -> list[str]:
    """An arm that is leased or not OFF. An arm server that does not answer is no sign (nothing can move without it)."""
    try:
        from lerobot_robot_spinearm.client import ArmRpc
        from lerobot_robot_spinearm.protocol import DEFAULT_PORTS
    except Exception:  # noqa: BLE001 — the watcher also runs where the arm package is not installed
        return []
    out = []
    for arm in ("left", "right"):
        try:
            c = _RPC.get(arm)
            if c is None:
                c = _RPC[arm] = ArmRpc("127.0.0.1", DEFAULT_PORTS[arm][0], timeout_ms=timeout_ms)
            s = c.call(cmd="state")
        except Exception:  # noqa: BLE001
            continue
        if s.get("lease") is not None:
            out.append(f"the {arm} arm is leased ({s.get('lease')})")
        elif s.get("state") not in (None, "OFF"):
            out.append(f"the {arm} arm is {s.get('state')}")
    return out


def robot_signs(table: dict, arms=arm_signs) -> list[str]:
    return hold_signs(table) + process_signs(table) + arms()


# ---------------------------------------------------------------- render work
def work_entries(table: dict | None = None, clean: bool = True) -> list[dict]:
    out = []
    for f in sorted(WORK.glob("*.json")) if WORK.is_dir() else []:
        w = read_json(f)
        if w and alive(int(w.get("pid") or 0), w.get("start"), table):
            out.append({**w, "file": str(f)})
        elif clean:
            f.unlink(missing_ok=True)
    return out


def render_pids(table: dict, on_pause: str = "stop") -> set[int]:
    """The processes of render-type work: RENDER_SCRIPTS and their children (stop), and registered work of that kind."""
    roots = {pid for pid, p in table.items() if script_of(p["argv"]) in RENDER_SCRIPTS} if on_pause == "stop" else set()
    for w in work_entries(table):
        if w.get("on_pause", "stop") == on_pause:
            roots |= {pid for pid, p in table.items() if p["pgid"] == w["pgid"]} | {int(w["pid"])}
    kill_roots = set()
    if on_pause == "stop":      # work that is to be ended is not also stopped (a stopped process does not see SIGTERM)
        kill_roots = render_pids(table, "kill")
    return {p for p in descendants(table, roots) if p in table and p != os.getpid()} - kill_roots


def service_health(url: str, timeout: float = 2.0) -> dict | None:
    tok = os.environ.get("RENDER_API_TOKEN") or ""
    f = Path.home() / ".config/spinearm-render-api/token"
    if not tok and f.exists():
        tok = f.read_text().strip()
    try:
        r = urllib.request.Request(url + "/v1/health", headers={"Authorization": f"Bearer {tok}"} if tok else {})
        return json.loads(urllib.request.urlopen(r, timeout=timeout).read())
    except Exception:  # noqa: BLE001 — a service that is not there renders nothing
        return None


def settled(health=service_health) -> tuple[bool, str]:
    """Render work is at rest under a pause: the watcher runs and has the pause in force, no render service renders (idle,
    or its job waits at the switch) and no render process runs."""
    st = watcher()
    if st is None:
        return False, "the watcher is not running"
    if not st.get("paused_since") or pause_reason() is None:
        return False, "no pause in force"
    for u in APIS:
        h = health(u)
        if h and h.get("busy") and not h.get("paused"):
            return False, f"{u} is rendering (a job's setup or packaging, or a step session)"
    table = procs()
    going = sorted(p for p in render_pids(table) | render_pids(table, "kill") if table[p]["state"] not in "TtZ")
    if going:
        return False, "render work still runs: " + ", ".join(f"{p} {script_of(table[p]['argv']) or os.path.basename(table[p]['argv'][0])}" for p in going[:4])
    return True, "render work is at rest"


# ---------------------------------------------------------------- the watcher
class Watch:
    def __init__(self, linger_s: float = LINGER_S, arms=arm_signs, sig=os.kill, sigpg=os.killpg):
        self.linger_s, self.arms, self.sig, self.sigpg = linger_s, arms, sig, sigpg
        self.since: float | None = None        # the pause in force began then
        self.clear_since: float | None = None  # the last sign went then
        self.signs: list[str] = []
        self.stopped: dict[int, int] = {}      # pid → start time, of what this watcher stopped
        self.killing: dict[int, float] = {}    # pgid → when SIGTERM was sent
        prev = read_json(STATE) or {}          # a watcher that died hard left processes stopped: they are this one's now
        self.stopped = {int(p): int(s) for p, s in (prev.get("stopped") or [])}
        if self.stopped or pause_reason() is not None:
            self.since = float(prev.get("paused_since") or time.time())

    def cycle(self, now: float | None = None, table: dict | None = None) -> None:
        now = time.time() if now is None else now
        table = procs() if table is None else table
        signs = robot_signs(table, self.arms)
        if signs:
            if self.since is None:
                self.since = now
                log(f"pause: {'; '.join(signs)}")
            elif self.clear_since is not None:
                log(f"pause goes on: {'; '.join(signs)}")
            self.clear_since, self.signs = None, signs
        elif self.since is not None:
            if self.clear_since is None:
                self.clear_since = now
            if now - self.clear_since >= self.linger_s:
                self.resume(table, now)
        if self.since is not None:
            write_json(PAUSE, {"reason": (signs or self.signs or ["robot"])[0], "signs": signs or self.signs, "since": round(self.since, 1), "by": os.getpid()})
            self.stop(table, now)
        write_json(STATE, {"pid": os.getpid(), "start": start_of(os.getpid()), "t": round(now, 2), "paused_since": self.since,
                           "signs": signs, "clear_since": self.clear_since, "stopped": sorted(self.stopped.items())})

    def stop(self, table: dict, now: float) -> None:
        self.stopped = {p: s for p, s in self.stopped.items() if alive(p, s, table)}
        for pid in sorted(render_pids(table)):
            if pid in self.stopped or table[pid]["state"] in "TtZ":
                continue
            try:
                self.sig(pid, signal.SIGSTOP)
            except (ProcessLookupError, PermissionError):
                continue
            self.stopped[pid] = table[pid]["start"]
            log(f"  stopped {pid} {script_of(table[pid]['argv']) or ' '.join(table[pid]['argv'][:2])[:60]}")
        for w in work_entries(table):
            if w.get("on_pause") != "kill":
                continue
            g = int(w["pgid"])
            if g not in self.killing:
                Path(w["file"]).with_suffix(".killed").write_text(self.signs[0] if self.signs else "robot")
                log(f"  ending {w.get('name') or w['pid']} (pid {w['pid']}): it cannot wait")
                self.killing[g] = now
                self._pg(g, signal.SIGTERM)
            elif now - self.killing[g] > KILL_GRACE_S:
                self._pg(g, signal.SIGKILL)
        self.killing = {g: t for g, t in self.killing.items() if any(p["pgid"] == g for p in table.values())}

    def _pg(self, pgid: int, sig: int) -> None:
        try:
            self.sigpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def resume(self, table: dict | None = None, now: float | None = None) -> None:
        now = time.time() if now is None else now
        n = 0
        for pid, start in sorted(self.stopped.items()):
            if alive(pid, start, table):
                try:
                    self.sig(pid, signal.SIGCONT)
                    n += 1
                except (ProcessLookupError, PermissionError):
                    pass
        PAUSE.unlink(missing_ok=True)
        if self.since is not None:
            log(f"resume after {now - self.since:.0f} s ({n} processes continued)")
        self.since = self.clear_since = None
        self.stopped, self.signs = {}, []


def cmd_watch(a) -> int:
    if watcher() is not None:
        print("a watcher is already running"); return 0
    for d in (HOLDS, WORK):
        d.mkdir(parents=True, exist_ok=True)
    w = Watch(a.linger)
    done = []
    signal.signal(signal.SIGTERM, lambda *_: done.append(1))
    signal.signal(signal.SIGINT, lambda *_: done.append(1))
    log(f"watcher started (pid {os.getpid()}, period {a.period} s, linger {a.linger} s)")
    while not done:
        t = time.time()
        try:
            w.cycle()
        except Exception as e:  # noqa: BLE001 — one bad look must not end the watch
            log(f"cycle failed: {type(e).__name__}: {e}")
        time.sleep(max(0.05, a.period - (time.time() - t)))
    w.resume()                  # nothing stays stopped behind a watcher that ends
    STATE.unlink(missing_ok=True)
    log("watcher ended")
    return 0


# ---------------------------------------------------------------- commands
def status() -> dict:
    st, table = watcher(), procs()
    stopped = sorted(p for p in render_pids(table) if table[p]["state"] in "Tt")
    return {"watcher": bool(st), "paused": pause_reason(), "paused_since": (st or {}).get("paused_since"),
            "signs": (st or {}).get("signs") if st else hold_signs(table, clean=False) + process_signs(table),
            "render_processes": len(render_pids(table)), "stopped": stopped,
            "work": [{k: w.get(k) for k in ("name", "pid", "on_pause")} for w in work_entries(table, clean=False)]}


def cmd_status(a) -> int:
    s = status()
    if a.json:
        print(json.dumps(s, ensure_ascii=False))
    else:
        if not s["watcher"]:
            print("the watcher is NOT running — render work does not give way by itself (systemctl --user start ube-render-gate)")
        if s["paused"]:
            mins = (time.time() - s["paused_since"]) / 60 if s["paused_since"] else 0
            print(f"render work is paused for {mins:.1f} min: {'; '.join(s['signs'] or [s['paused']])}  ({len(s['stopped'])} processes stopped)")
        else:
            print(f"clear — no sign of robot use; {s['render_processes']} render processes" + (f", {len(s['stopped'])} of them STOPPED (render_gate.py resume)" if s["stopped"] else ""))
    return 2 if not s["watcher"] else (1 if s["paused"] else 0)


def cmd_hold(a) -> int:
    pid = a.pid or os.getppid()
    write_json(HOLDS / f"{a.name}.json", {"pid": pid, "start": start_of(pid), "reason": a.reason or a.name, "t": round(time.time(), 1)})
    if watcher() is None:
        print("hold written, but the watcher is not running — nothing pauses by itself"); return 4
    t0 = time.time()
    while True:
        ok, why = settled()
        if ok:
            print(f"render work is at rest ({time.time() - t0:.1f} s)"); return 0
        if time.time() - t0 >= a.wait:
            print(f"not at rest after {a.wait:.0f} s: {why}"); return 3
        time.sleep(0.3)


def cmd_release(a) -> int:
    (HOLDS / f"{a.name}.json").unlink(missing_ok=True)
    return 0


def cmd_settled(a) -> int:
    ok, why = settled()
    print(why)
    return 0 if ok else 1


def cmd_resume(a) -> int:
    table = procs()
    for pid in sorted(p for p in render_pids(table) if table[p]["state"] in "Tt"):
        os.kill(pid, signal.SIGCONT); print("continued", pid, script_of(table[pid]["argv"]) or "")
    return 0


def cmd_run(a) -> int:
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        print("run: no command"); return 2
    if watcher() is None:
        print("[gate] the watcher is not running — started without protection", file=sys.stderr)
    if pause_reason() is not None:
        if a.on_pause == "kill":
            print(f"[gate] the robot is in use ({pause_reason()}) — not started", file=sys.stderr); return 75
        print(f"[gate] the robot is in use ({pause_reason()}) — waiting", file=sys.stderr)
        while pause_reason() is not None:
            time.sleep(2)
    env = dict(os.environ)
    pre = None
    if a.low:
        env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
        cores = {c for c in range(12, 16) if c in os.sched_getaffinity(0)}

        def pre():  # in the child, before exec
            os.nice(max(0, 15 - os.nice(0)))      # to niceness 15, wherever it starts from
            if cores:
                os.sched_setaffinity(0, cores)
    # a memory ceiling of its own (10-09 12:38: an analysis script of mine grew to 4 GB on this 5.8 GB machine — the kernel
    # killed it, but the camera pipeline had stalled by then and the tracker restarted twice). In a scope with MemoryMax
    # the work is killed alone when it passes its ceiling, long before the machine is short.
    cap = a.mem_gb if a.mem_gb is not None else (2.0 if a.low else 0.0)
    if cap > 0:
        cmd = ["systemd-run", "--user", "--scope", "--quiet", "--collect", "-p", f"MemoryMax={int(cap * 1024)}M", "-p", "MemorySwapMax=0", "--"] + cmd
    p = subprocess.Popen(cmd, start_new_session=True, env=env, preexec_fn=pre)
    f = WORK / f"{p.pid}.json"
    f.with_suffix(".killed").unlink(missing_ok=True)      # a pid comes round again
    write_json(f, {"pid": p.pid, "pgid": p.pid, "start": start_of(p.pid), "on_pause": a.on_pause, "name": a.name or os.path.basename(cmd[0]), "cmd": " ".join(cmd)[:300]})
    def forward(n, _):
        try:
            os.killpg(p.pid, n)
        except ProcessLookupError:
            pass
    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, forward)
    rc = p.wait()
    killed = f.with_suffix(".killed")
    gone = killed.exists()
    why = killed.read_text() if gone else ""
    f.unlink(missing_ok=True); killed.unlink(missing_ok=True)
    if gone:
        print(f"[gate] ended for the robot ({why}) — run it again later", file=sys.stderr); return 75
    return rc if rc >= 0 else 128 - rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="c", required=True)
    w = sub.add_parser("watch"); w.add_argument("--period", type=float, default=1.0); w.add_argument("--linger", type=float, default=LINGER_S)
    s = sub.add_parser("status"); s.add_argument("--json", action="store_true")
    h = sub.add_parser("hold"); h.add_argument("name"); h.add_argument("--pid", type=int); h.add_argument("--reason"); h.add_argument("--wait", type=float, default=0.0)
    r = sub.add_parser("release"); r.add_argument("name")
    sub.add_parser("settled"); sub.add_parser("resume")
    x = sub.add_parser("run"); x.add_argument("--on-pause", choices=("stop", "kill"), default="stop"); x.add_argument("--low", action="store_true")
    x.add_argument("--mem-gb", type=float, default=None, help="memory ceiling of the work (default: 2 with --low, none otherwise; 0 = none)")
    x.add_argument("--name"); x.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    return {"watch": cmd_watch, "status": cmd_status, "hold": cmd_hold, "release": cmd_release, "settled": cmd_settled,
            "resume": cmd_resume, "run": cmd_run}[a.c](a)


if __name__ == "__main__":
    sys.exit(main())
