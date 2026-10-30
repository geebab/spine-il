"""What goes into the dataset: FE recorder v2 sessions (hand and robot-origin) and, per robot session, its run.

Hand session   <sessions>/<YYYY-MM-DD>/*.jsonl, hdr.origin.source = 'hand' (rev.4 set = 2026-09-28).
Robot session  <sessions>/robot/<YYYY-MM-DD>/*.jsonl, hdr.origin.source = 'robot', origin.robotRun = the run id,
               origin.replayOf = the hand session it replays (registration method 'anchor').
Robot run      ~/il-robot/runs/robot_origin/<run>/ (settings.json speed + overrides, result.json last line)
               + ~/il-robot/runs/latency/<run>_endo|_drill/ (events.jsonl + config.json of each arm)
               + ~/il-robot/runs/jitter_iter/ro_<sessionId>_iou.json (cut agreement robot vs hand recording).

Clocks (checked 2026-09-29 on 20260929T163714_replay_pr_3868j1):
  f.tExp            camera frame stamp, epoch ms, tsDomain global_time (host clock; tracker and arms share the host)
  tap.ts            = f.tExp − 1.25 ms for the same camera frame (tip identical to 0.0 mm) — same clock
  state.tr / t      arm server time.monotonic() s; epoch ms = tr·1000 + clock.off (off = wall − mono of the host,
                    logged every second by endo_follow; spread over a run < 0.001 ms)
  cmd.tik / tack    epoch ms (client wall clock)
  engage.t          epoch ms
Robot joints are therefore put on the session's tExp axis as tr·1000 + off: error = the tracker's global-time mapping
(≈1 ms) + where in the servo bus read `tr` falls (read of 6 servos, a few ms) — no network clock involved.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from lerobot_robot_spinearm.recording import read_header

HOME = Path.home()
SESSIONS_ROOT = HOME / "il-robot/viewer-data/sessions"
RUNS_ROOT = HOME / "il-robot/runs"
# rev.4 replay windows (s after the recording's first valid pose = round(first bone cut − 2 s)), as the robot runs used
HAND_WINDOWS_FROM_S = {"3868": 17.0, "91cb": 15.0, "bef5": 17.0, "d340": 27.0, "bad1": 12.0, "0338": 18.0,
                       "919d": 10.0, "6639": 14.0}
WINDOW_LEN_S = 30.0
CONTAMINATED_EXTRA = {"pr_3868s20": "speed test (0.20x, not the 0.33x series)"}
TASK_EPISODE = "L4-5 후궁 드릴링 / drill the L4 L5 lamina under endoscope view"


def short_id(session_id: str) -> str:
    """s_20260928T165947_altician_lumbar1_trk_3868 → 3868."""
    return str(session_id).rsplit("_", 1)[-1]


def last_json_line(path: Path) -> dict | None:
    """The last line of a file that parses as a JSON object (result.json has log lines before it)."""
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if line.startswith("{"):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                return d
    return None


@dataclass
class Session:
    path: Path
    hdr: dict
    source: str                      # hand | robot
    session_id: str
    replay_of: str | None = None     # robot: the hand session id
    run: str | None = None           # robot: run id
    skip: str | None = None          # why it is not used (None = used)
    meta: dict = field(default_factory=dict)

    @property
    def sid(self) -> str:
        return short_id(self.session_id)


def _footer(path: Path) -> dict | None:
    """The ftr row (read from the end: it is the last row the recorder writes)."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 65536))
        tail = fh.read().decode("utf-8", "replace").splitlines()
    for line in reversed(tail):
        if '"ftr"' in line:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if d.get("k") == "ftr":
                return d
    return None


def _has_bone_cut(path: Path) -> bool:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"k":"cut"' in line[:20] or '"k": "cut"' in line[:20]:
                if '"target":"bone"' in line or '"target": "bone"' in line:
                    return True
    return False


def _matches(hdr: dict, only: set | None) -> bool:
    if not only:
        return True
    run = (hdr.get("origin") or {}).get("robotRun") or ""
    tag = run.split("_replay_", 1)[-1] if "_replay_" in run else run
    return short_id(hdr.get("sessionId") or "") in only or tag in only


def check_session(path: Path, only: set | None = None) -> Session | None:
    """Header-level compatibility (no full parse). None = not a recorder file at all (or not in `only`)."""
    hdr = read_header(path)
    if hdr is None or not _matches(hdr, only):
        return None
    origin = hdr.get("origin") or {}
    src = origin.get("source")
    s = Session(path=Path(path), hdr=hdr, source=str(src), session_id=str(hdr.get("sessionId") or Path(path).stem),
                replay_of=origin.get("replayOf"), run=origin.get("robotRun"))
    reg = hdr.get("registration") or {}
    ftr = _footer(path)
    s.meta["verdict"] = (ftr or {}).get("operatorVerdict")
    s.meta["endReason"] = (ftr or {}).get("endReason")
    if src not in ("hand", "robot"):
        s.skip = "hdr.origin 없음(rev.4 이전 레코더 — 손/로봇 출처·도구 캘리브 대조가 없는 녹화)"
    elif hdr.get("timingSpec") != "v2-t0":
        s.skip = f"timingSpec={hdr.get('timingSpec')!r}(v2-t0 아님)"
    elif not reg.get("T0"):
        s.skip = "registration.T0 없음"
    elif reg.get("dynamic"):
        s.skip = "동적 정합(dynamic) — 행별 T 처리 미구현"
    elif ftr is None:
        s.skip = "ftr(푸터) 없음 — 미완결 세션"
    elif ftr.get("operatorVerdict") == "bad":
        s.skip = "operatorVerdict=bad"
    elif not _has_bone_cut(path):
        s.skip = "뼈 절삭(cut target=bone) 행 없음"
    return s


def discover(root: Path | None = None, only: set | None = None) -> list[Session]:
    """Every recorder file under root (hand dirs and robot/…), with skip reasons for the unusable ones. only = session
    short ids / run tags to keep (None = all)."""
    root = Path(root or SESSIONS_ROOT)
    out = []
    for p in sorted(root.rglob("*.jsonl")):
        s = check_session(p, only)
        if s is None:
            continue
        in_robot_dir = "robot" in p.relative_to(root).parts[:-1]
        if s.skip is None and (s.source == "robot") != in_robot_dir:
            s.skip = f"origin.source={s.source} 인데 {'robot/' if in_robot_dir else '손'} 폴더에 있다"
        out.append(s)
    return out


# ---------------------------------------------------------------- robot runs
def contaminated_tags(runs_root: Path | None = None) -> dict[str, str]:
    """{run tag: reason}: ro_contaminated.txt (render GPU contention) + CONTAMINATED_EXTRA."""
    runs_root = Path(runs_root or RUNS_ROOT)
    out = {}
    f = runs_root / "jitter_iter/work/ro_contaminated.txt"
    if f.exists():
        text = f.read_text(errors="replace")
        why = text.split(":", 1)[0].strip() if ":" in text else "ro_contaminated.txt"
        for tag in re.findall(r"\bpr_[0-9a-z]+\b", text):
            out[tag] = why
    out.update(CONTAMINATED_EXTRA)
    return out


def _events(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if isinstance(d, dict):
                rows.append(d)
    return rows


def run_meta(s: Session, runs_root: Path | None = None, contaminated: dict | None = None) -> None:
    """Fill s.meta for a robot session from its run; sets s.skip when the run or an arm log is missing."""
    runs_root = Path(runs_root or RUNS_ROOT)
    run = s.run
    if not run:
        s.skip = "origin.robotRun 없음"
        return
    rd = runs_root / "robot_origin" / run
    tag = run.split("_replay_", 1)[-1] if "_replay_" in run else run
    m = s.meta
    m.update(run=run, tag=tag, run_dir=str(rd))
    settings = {}
    if (rd / "settings.json").exists():
        try:
            settings = json.loads((rd / "settings.json").read_text())
        except ValueError:
            settings = {}
    m["settings"] = settings
    res = last_json_line(rd / "result.json") if (rd / "result.json").exists() else None
    m["result_session"] = (res or {}).get("sessionId")
    m["anchor"] = {k: (res or {}).get(k) for k in ("anchor", "tip_vec_err_mm", "hand_sep_mm", "robot_sep_mm")}
    arms = {}
    for arm in ("endo", "drill"):
        d = runs_root / "latency" / f"{run}_{arm}"
        if not (d / "events.jsonl").exists() or not (d / "config.json").exists():
            s.skip = f"팔 로그 없음: {d.name}"
            return
        arms[arm] = d
    m["arm_dirs"] = {k: str(v) for k, v in arms.items()}
    cfg = json.loads((arms["endo"] / "config.json").read_text()).get("args", {})
    speed = cfg.get("replay_speed")
    try:
        speed = float(settings.get("speed", speed))
    except (TypeError, ValueError):
        pass
    m["speed"] = float(speed) if speed else None
    m["from_s"] = cfg.get("replay_from_s")
    m["max_s"] = cfg.get("replay_max_s")
    m["overrides"] = {"left": settings.get("left_x", ""), "right": settings.get("right_x", ""),
                      "cfg": settings.get("cfg")}
    if not m["speed"]:
        s.skip = "replay_speed 없음"
        return
    iou = runs_root / "jitter_iter" / f"ro_{s.session_id}_iou.json"
    if iou.exists():
        try:
            d = json.loads(iou.read_text())
            rr = d.get("rr_hr") or {}
            m["iou"] = {"iou": rr.get("iou"), "dice": rr.get("dice"), "tol1_dice": rr.get("tol1_dice"),
                        "assd_mm": rr.get("assd_mm"), "hd95_mm": rr.get("hd95_mm"),
                        "robot_rec_mm3": d.get("robot_rec_mm3"), "hand_rec_mm3": d.get("hand_rec_mm3"),
                        "volume_ratio": (round(d["robot_rec_mm3"] / d["hand_rec_mm3"], 4)
                                         if d.get("robot_rec_mm3") and d.get("hand_rec_mm3") else None),
                        "window": d.get("window"), "drill": d.get("drill")}
        except (ValueError, KeyError, TypeError):
            m["iou"] = None
    else:
        m["iou"] = None
    cont = contaminated if contaminated is not None else contaminated_tags(runs_root)
    m["contaminated"] = tag in cont
    m["contaminated_reason"] = cont.get(tag)


def replay_span(arm_dirs: dict) -> tuple[float, float]:
    """(t_go, t_end) epoch ms of the replay both arms executed: the later engage → the earlier last goal."""
    starts, ends = [], []
    for arm, d in arm_dirs.items():
        ev = [r for r in _events(Path(d) / "events.jsonl") if r.get("k") in ("engage", "cmd")]
        eng = [r["t"] for r in ev if r.get("k") == "engage" and isinstance(r.get("t"), (int, float))]
        cmd = [r["tik"] for r in ev if r.get("k") == "cmd" and isinstance(r.get("tik"), (int, float))]
        if not eng or not cmd:
            raise ValueError(f"{d}: no engage / cmd rows")
        starts.append(min(eng))
        ends.append(max(cmd))
    return max(starts), min(ends)


def arm_timeline(arm_dir) -> dict:
    """One arm log → {t_state (epoch ms), q (N, 6) encoder deg, t_goal (epoch ms), goal (M, 6) deg, off_ms,
    off_spread_ms, joints}. Encoder q at the server's position read (tr); goals at the server's receive time (grx) of
    each new goal (gseq), else the state time."""
    rows = _events(Path(arm_dir) / "events.jsonl")
    offs = [r["off"] for r in rows if r.get("k") == "clock" and isinstance(r.get("off"), (int, float))]
    if not offs:
        raise ValueError(f"{arm_dir}: no clock rows (wall − monotonic offset)")
    offs.sort()
    off = offs[len(offs) // 2]
    st = [r for r in rows if r.get("k") == "state" and isinstance(r.get("pos"), dict)]
    if not st:
        raise ValueError(f"{arm_dir}: no server state rows")
    joints = list(st[0]["pos"].keys())
    ts, q, tg, g, seen = [], [], [], [], set()
    for r in st:
        tr = r.get("tr") if isinstance(r.get("tr"), (int, float)) else r.get("t")
        if not isinstance(tr, (int, float)):
            continue
        t_ms = tr * 1000.0 + off
        if ts and t_ms <= ts[-1]:
            continue
        ts.append(t_ms)
        q.append([float(r["pos"][j]) for j in joints])
        goal = r.get("goal")
        if isinstance(goal, dict) and all(isinstance(goal.get(j), (int, float)) for j in joints):
            key = r.get("gseq")
            if key is None or key not in seen:
                seen.add(key)
                grx = r.get("grx")
                tgm = grx * 1000.0 + off if isinstance(grx, (int, float)) else t_ms
                if not tg or tgm > tg[-1]:
                    tg.append(tgm)
                    g.append([float(goal[j]) for j in joints])
    return {"t_state": ts, "q": q, "t_goal": tg, "goal": g, "off_ms": off,
            "off_spread_ms": (offs[-1] - offs[0]) if offs else math.nan, "joints": joints}
