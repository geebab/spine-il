"""Live guard for the paired-start runs (p?_<rec>): stop BOTH followers of a run if either arm's tool tip moves more than
--max-move-mm from where it engaged, or the two robot tips come closer than --min-sep-mm. Uses the tap rows' own tip
(the tracker model carries tip_local since 09-28). Reads the newest *_replay_p*_endo / _drill run dirs.
A run whose trajectory was verified over a longer travel (full cycle from the ready pose, robot/inference/verify_traj.py)
carries its own limits in runs/robot_origin/<run>/guard.json {"endo_mm", "drill_mm"} (robot_origin_run.sh RO_GUARD_MM),
capped at MAX_RUN_LIMIT_MM. The tip-separation stop is MIN_SEP_MM = 5 mm for every run (user, 10-07 22:35: "5mm로 완전히
변경"). It was 10 mm, lowered to 5 per run on the 10-06 sitting from 11:00 that day ("팁 간 10 mm 감시는 5mm로 하향": the
demos of that sitting work with the two tips 17 mm apart (median) and come within 10 mm in 12 of 13 episodes — a run at
10 mm would be stopped for doing what the demonstrations do). A run may still LOWER it: guard.json "min_sep_mm"
(RO_GUARD_SEP_MM), never below MIN_SEP_FLOOR_MM and never above --min-sep-mm.

  python guard_pairs.py [--max-move-mm 30] [--min-sep-mm 5]
"""
import argparse, glob, json, os, signal, subprocess, time
from pathlib import Path

L = Path.home() / "il-robot/runs/latency"
RO = Path.home() / "il-robot/runs/robot_origin"
MAX_RUN_LIMIT_MM = 130.0
MIN_SEP_MM = 5.0
MIN_SEP_FLOOR_MM = 3.0


def run_limits(stamp_tag: str, default: float, sep_default: float = MIN_SEP_MM) -> dict:
    """Travel limits per tool (mm from the engage tip) and the tip-separation stop (mm) of this run."""
    lim = {"endo": default, "drill": default, "sep": sep_default}
    try:
        g = json.loads((RO / stamp_tag / "guard.json").read_text())
    except (OSError, ValueError):
        return lim
    if not isinstance(g, dict):
        return lim
    try:
        lim.update({t: min(MAX_RUN_LIMIT_MM, max(default, float(g[f"{t}_mm"]))) for t in ("endo", "drill")})
    except (ValueError, KeyError, TypeError):
        pass
    try:   # only ever lower, and not below the floor
        lim["sep"] = min(sep_default, max(MIN_SEP_FLOOR_MM, float(g["min_sep_mm"])))
    except (ValueError, KeyError, TypeError):
        pass
    return lim


STATE: dict = {}   # path -> [offset, engaged, first_tip, last_tip]


def last_rows(path: Path, tool: str):
    """engage tip (first tap after engage) and the latest tap tip for this tool — reads only the new bytes."""
    st = STATE.setdefault(str(path), [0, False, None, None])
    with open(path, "rb") as f:
        f.seek(st[0])
        data = f.read()
    cut = data.rfind(b"\n") + 1
    st[0] += cut
    for raw in data[:cut].splitlines():
        s = raw.decode(errors="ignore")
        if '"engage"' in s[:30]:
            st[1] = True
            continue
        if st[1] and '"tap"' in s[:20] and '"tip"' in s:
            try:
                m = json.loads(s)
            except ValueError:
                continue
            if m.get("tool") == tool and not m.get("lost") and m.get("tip"):
                st[2] = st[2] or m["tip"]
                st[3] = m["tip"]
    return st[2], st[3]


def followers(run_prefix: str):
    out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True).stdout.splitlines()
    return [int(l.split()[0]) for l in out if "endo_follow.py" in l and run_prefix in l]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-move-mm", type=float, default=30.0)
    ap.add_argument("--min-sep-mm", type=float, default=MIN_SEP_MM)
    a = ap.parse_args()
    tripped = set()
    while True:
        runs = sorted(glob.glob(str(L / "*_replay_p?_*_drill")))
        if runs:
            rd = Path(runs[-1]); stamp_tag = rd.name[: -len("_drill")]
            re_ = L / f"{stamp_tag}_endo"
            pids = followers(stamp_tag)
            if pids and stamp_tag not in tripped and (rd / "events.jsonl").exists() and (re_ / "events.jsonl").exists():
                d0, d1 = last_rows(rd / "events.jsonl", "drill")
                e0, e1 = last_rows(re_ / "events.jsonl", "endo")
                why = []; lim = run_limits(stamp_tag, a.max_move_mm, a.min_sep_mm)
                dist = lambda p, q: sum((x - y) ** 2 for x, y in zip(p, q)) ** 0.5
                if d0 and d1 and dist(d0, d1) > lim["drill"]:
                    why.append(f"drill tip moved {dist(d0, d1):.0f} mm (limit {lim['drill']:.0f})")
                if e0 and e1 and dist(e0, e1) > lim["endo"]:
                    why.append(f"endo tip moved {dist(e0, e1):.0f} mm (limit {lim['endo']:.0f})")
                if d1 and e1 and dist(d1, e1) < lim["sep"]:
                    why.append(f"tips {dist(d1, e1):.1f} mm apart (limit {lim['sep']:g})")
                if why:
                    for p in pids:
                        try:
                            os.kill(p, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                    tripped.add(stamp_tag)
                    print(f"[guard] {time.strftime('%T')} stopped {stamp_tag}: {'; '.join(why)}", flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
