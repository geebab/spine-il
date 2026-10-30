"""Tiny synthetic FE recorder v2 sessions (hand + robot-origin) and a robot run, with known answers.

Scene = camera (T0 identity). Hand, 20 s at 30 Hz:
  0–1 s ready (tips still, resting "in" the flank: depth +12 mm with the entry 50 mm off the portal)
  1–2 s approach (outside: depth −5 at the portal), 2–18.5 s inside (depth 10 at the portal E)
  bone cuts 5.0–12.0 s every 0.1 s except a 8.0–9.2 s pause; trigger on 5–12 s
  18.5–19 s out to the ready pose, 19–20 s still.
Robot, 30 s at 30 Hz: tools inside from the start, cuts 4–20 s, trigger on 4–20 s; replay span 1–21 s at speed 0.5;
arm encoder q_j(t) = j + 0.001·(t − BASE) deg (t epoch ms), goals = q + 1.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

BASE = 1_790_000_000_000.0
HZ = 30.0
READY = np.array([100.0, 0.0, 0.0])
TARGET = np.array([50.0, -10.0, 0.0])
PORTAL = np.array([50.0, 0.0, 0.0])
FAR_ENTRY = np.array([100.0, 10.0, 0.0])
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def quat_z(a: float) -> list[float]:
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def hand_tip(t: float) -> np.ndarray:
    if t < 1.0:
        return READY.copy()
    if t < 2.0:
        return READY + (TARGET - READY) * (t - 1.0)
    if t < 18.5:
        return TARGET + np.array([0.0, 0.0, 0.5 * math.sin(t)])
    if t < 19.0:
        return TARGET + (READY - TARGET) * (t - 18.5) / 0.5
    return READY.copy()


def hand_skin(t: float) -> tuple[np.ndarray, float]:
    """(entry, depth)."""
    if t < 1.5 or t >= 18.75:
        return FAR_ENTRY, 12.0
    if t < 2.0:
        return PORTAL, -5.0
    if t < 18.5:
        return PORTAL, 10.0
    return PORTAL, -5.0


def _hdr(sid: str, source: str, run: str | None = None, replay_of: str | None = None) -> dict:
    return {"k": "hdr", "r": 0, "schemaVersion": 2, "recorderVersion": "rec-2.3.1-t0", "timingSpec": "v2-t0",
            "sessionId": sid, "operator": "synth",
            "registration": {"method": "fid3" if source == "hand" else "anchor",
                             "T0": {"R": [1, 0, 0, 0, 1, 0, 0, 0, 1], "t": [0, 0, 0]}, "dynamic": False},
            "drill": {"radiusMm": 2, "sizeKey": "medium"},
            "origin": {"v": 1, "source": source, "robotRun": run, "replayOf": replay_of}}


def _f_rows(t: float, tip: np.ndarray, entry, depth, btn: int, r: int, lost: bool = False) -> list[dict]:
    te = BASE + t * 1000.0
    rows = []
    for tool, off in (("endo", np.zeros(3)), ("drill", np.array([0.0, 0.0, 5.0]))):
        p = tip + off
        q = quat_z(0.05 * t + (0.3 if tool == "drill" else 0.0))
        d = {"k": "f", "r": r + len(rows), "tool": tool, "tExp": te, "tsd": "g", "pos": p.tolist(), "quat": q,
             "tip": p.tolist(), "vis": 4, "hold": 0, "scene": {"tip": p.tolist(), "u": [0, -1, 0]}}
        if lost and tool == "endo":
            d = {"k": "f", "r": d["r"], "tool": tool, "tExp": te, "lost": 1}
        drawn = {"tExp": te - 30.0, "lost": 0, "entry": entry.tolist()}
        if tool == "endo":
            drawn.update(tip=p.tolist(), depth=depth)
        else:
            u = np.array([0.0, -1.0, 0.0])
            B = entry + u * depth + (np.array([0, 0, 5.0]) if False else 0)
            drawn.update(B=B.tolist(), u=u.tolist())
            d.update(btn=btn, btnStale=0)
        d["drawn"] = drawn
        rows.append(d)
    return rows


def write_hand(path: Path, sid: str) -> dict:
    rows = [_hdr(sid, "hand")]
    r = 1
    n = int(20 * HZ) + 1
    for i in range(n):
        t = i / HZ
        entry, depth = hand_skin(t)
        btn = int(5.0 <= t < 12.0)
        rows += _f_rows(t, hand_tip(t), entry, depth, btn, r, lost=(10.0 <= t < 10.1))
        r += 2
    cuts = [c for c in np.arange(5.0, 12.0001, 0.1) if not (8.0 < c < 9.2)]
    for c in cuts:
        rows.append({"k": "cut", "r": r, "tExp": BASE + c * 1000.0, "tool": "drill", "target": "bone"})
        r += 1
    rows.append({"k": "ftr", "r": r, "endReason": "completed", "operatorVerdict": "good"})
    rows.sort(key=lambda d: d["r"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(d) for d in rows) + "\n")
    return {"cuts": (5.0, 12.0), "trigger": (5.0, 12.0)}


def write_robot(sessions: Path, runs: Path, sid: str, hand_sid: str, run: str, speed: float = 0.5) -> dict:
    rows = [_hdr(sid, "robot", run, hand_sid)]
    r = 1
    for i in range(int(30 * HZ) + 1):
        t = i / HZ
        btn = int(4.0 <= t < 20.0)
        rows += _f_rows(t, TARGET + np.array([0.0, 0.0, 0.2 * t]), PORTAL, 10.0, btn, r)
        r += 2
    for c in np.arange(4.0, 20.0001, 0.1):
        rows.append({"k": "cut", "r": r, "tExp": BASE + c * 1000.0, "tool": "drill", "target": "bone"})
        r += 1
    rows.append({"k": "ftr", "r": r, "endReason": "completed", "operatorVerdict": "good"})
    p = sessions / "robot" / "2026-09-29" / f"{sid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(d) for d in rows) + "\n")

    rd = runs / "robot_origin" / run
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "settings.json").write_text(json.dumps({"speed": str(speed), "left_x": "--vis-ki=0.5", "right_x": "",
                                                  "cfg": "pd_test"}))
    (rd / "result.json").write_text("[ro] log line\n" + json.dumps({"run": run, "sessionId": sid}) + "\n")
    off = BASE - 5_000_000.0  # wall − monotonic (ms)
    for arm, j0 in (("endo", 0.0), ("drill", 100.0)):
        d = runs / "latency" / f"{run}_{arm}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({"args": {"replay_speed": speed, "replay_from_s": 17.0,
                                                            "replay_max_s": 10.0}}))
        ev = [{"k": "clock", "wall": BASE, "mono": BASE - off, "off": off}]
        for i in range(0, 30 * 30 + 1):  # server state at 30 Hz over the whole session
            t_ms = BASE + i * 1000.0 / 30.0 + 7.0
            q = {j: j0 + k + 0.001 * (t_ms - BASE) for k, j in enumerate(JOINTS)}
            g = {j: v + 1.0 for j, v in q.items()}
            ev.append({"k": "state", "t": (t_ms - off) / 1000.0, "tr": (t_ms - off) / 1000.0, "pos": q, "goal": g,
                       "gseq": i, "grx": (t_ms - off) / 1000.0})
        ev.append({"k": "engage", "t": BASE + 1000.0})
        for i in range(0, 201):
            ev.append({"k": "cmd", "tik": BASE + 1000.0 + i * 100.0})
        (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev) + "\n")
    iou = runs / "jitter_iter"
    (iou / "work").mkdir(parents=True, exist_ok=True)
    tag = run.split("_replay_", 1)[1]
    (iou / f"ro_{sid}_iou.json").write_text(json.dumps({"tag": tag, "rr_hr": {"iou": 0.7, "dice": 0.8,
                                            "tol1_dice": 0.95, "assd_mm": 0.3, "hd95_mm": 1.2},
                                            "robot_rec_mm3": 110.0, "hand_rec_mm3": 100.0, "window": [17, 30]}))
    (iou / "work" / "ro_contaminated.txt").write_text(f"contaminated (render GPU contention): {tag}\n")
    return {"span_ms": (BASE + 1000.0, BASE + 21000.0), "speed": speed}
