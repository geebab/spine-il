#!/usr/bin/env python3
"""A robot replay of an FE recording → a render-API trajectory, so the viewer plays what the arms actually did.

    python robot/replay_to_render.py REC.jsonl --from-s 10 --len-s 60 [--endo RUN] [--drill RUN] -o robot.json
    python robot/replay_to_render.py REC.jsonl --from-s 10 --len-s 60 --hand -o hand.json   # the same, from the hand

`--endo` / `--drill`: endo_follow replay run dirs (left arm endoscope, right arm drill) of that recording window.
The arm's marker, as the camera saw it, is moved back onto the recorded hand: the replay drove the arm by the hand's
camera-frame displacement from its first pose H0 (tip: R0 + (H − H0), rotation: (R_H·R_H0⁻¹)·R_R0), so here the
inverse puts the arm's displacement from its own engage pose R0 onto H0:

    tip_eq(t) = H0_tip + (robot_tip(t) − R0_tip)      R_eq(t) = R_robot(t)·R_R0⁻¹·R_H0

Time: arm time since engage × replay speed = recording time since H0, so the trajectory plays at the recording's
speed. The drill trigger is the recording's own (`btn`), taken at that recording time — the arm has no trigger.
A tool no arm replayed plays the recorded hand (unless --robot-only).
`--hand` writes the recorded raw hand poses of the same window through the same code: render both with the same
options and the difference is the arm alone (the viewer's own cut rows are not used — both are `simulate`).
Poses go to the scene with the recording's registration (hdr T0), tip + quat (tool → scene), with the recording's
tool models (hdr.toolCalib) so the viewer draws the tips where these were computed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from lerobot_robot_spinearm import recording as rec
from lerobot_robot_spinearm.handeye import quat_to_R

TRACKER = Path.home() / "il-robot/tracking"
# the arm-carried blocks' models (the tracker's --model / --model2) and the hand models whose tip they carry
ROBOT_MODEL = {"endo": ("tool_calib.json", "tool_calib_endo_robotL.json"),
               "drill": ("tool_calib_drill.json", "tool_calib_drill_robotR.json")}
ROBOT_TOOL = {"endo": ("endo", "endo2"), "drill": ("drill",)}


ROBOT_TIP_MODELS = Path.home() / "il-robot/calib/tracker_models"   # the tracker's own arm models (tip_local set)
ROBOT_TIP_FILE = {"endo": "tool_calib_endo_robotL_tip.json", "drill": "tool_calib_drill_robotR_tip.json"}


def robot_tip_local(tool: str) -> np.ndarray:
    """The arm block's tip in its model frame, mm: the tracker's arm model when it carries tip_local (2026-09-28 on:
    ~/il-robot/calib/tracker_models, the same file the tracker and endo_follow --robot-tool-calib use), else the
    hand model's tip carried into the arm block's frame (Kabsch on the four balls)."""
    f = ROBOT_TIP_MODELS / ROBOT_TIP_FILE[tool]
    if f.exists():
        tip = json.loads(f.read_text()).get("tip_local")
        if tip is not None and len(tip) == 3:
            return np.asarray(tip, float)
    a = json.loads((TRACKER / ROBOT_MODEL[tool][0]).read_text())
    b = json.loads((TRACKER / ROBOT_MODEL[tool][1]).read_text())
    A, B = np.asarray(a["points_local"], float), np.asarray(b["points_local"], float)
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R @ (np.asarray(a["tip_local"], float) - ca) + cb


def R_to_quat(R: np.ndarray) -> list[float]:
    return [float(x) for x in rec.R_to_quat(R)]


def hand_tips(rp: dict) -> tuple[np.ndarray, np.ndarray]:
    """The recorded hand's tip (mm, camera) at its own sample times (4-ball rows)."""
    ok = np.asarray(rp["vis"]) >= 4
    tip_l = np.asarray(rp["tip_local_m"], float)
    tip = np.array([Ri @ tip_l + np.asarray(p, float) for Ri, p in zip(rp["R"], rp["pos"])]) * 1000.0
    return np.asarray(rp["t"], float)[ok], tip[ok]


def hand_lag(rp: dict, t: np.ndarray, tip: np.ndarray, grid=np.arange(0.0, 0.6, 0.005)) -> float:
    """The shift (recording s) that best lays the arm's tip path over the hand's: argmin RMS distance."""
    th, ph = hand_tips(rp)
    best = (np.inf, 0.0)
    for d in grid:
        m = (th >= t[0] - d) & (th <= t[-1] - d)
        if m.sum() < 20:
            continue
        q = np.stack([np.interp(th[m] + d, t, tip[:, i]) for i in range(3)], 1)
        r = float(np.sqrt(((q - ph[m]) ** 2).sum(1).mean()))
        best = min(best, (r, float(d)))
    return best[1]


def match_to(ts: np.ndarray, t: np.ndarray, tip: np.ndarray, R: np.ndarray):
    """The arm's samples averaged over each target sample's window (half-way to its neighbours): the same sample
    times and roughly the same bandwidth as the recording. Rotation = the normalised mean of sign-aligned quats."""
    edges = np.r_[ts[0] - (ts[1] - ts[0]) / 2, (ts[1:] + ts[:-1]) / 2, ts[-1] + (ts[-1] - ts[-2]) / 2]
    idx = np.searchsorted(t, edges)
    q = np.array([rec.R_to_quat(Ri) for Ri in R])
    out_t, out_p, out_R = [], [], []
    for k in range(len(ts)):
        a_, b_ = idx[k], idx[k + 1]
        if b_ <= a_:
            continue
        qq = q[a_:b_] * np.sign(q[a_:b_] @ q[a_])[:, None]
        qm = qq.mean(0)
        out_t.append(ts[k])
        out_p.append(tip[a_:b_].mean(0))
        out_R.append(rec.quat_to_R(qm / np.linalg.norm(qm)))
    return np.array(out_t), np.array(out_p), np.array(out_R)


def robot_track(run: Path, tool: str, pose: str = "raw") -> dict:
    """The arm marker's raw poses after engage: t (s since engage), tip (mm, cam), R, plus the engage pose."""
    tip_l = robot_tip_local(tool)
    eng, rows, cmds = None, [], []
    for line in open(run / "events.jsonl"):
        if '"engage"' in line[:30]:
            eng = json.loads(line)
            continue
        if eng is not None and line.startswith('{"k":"cmd"'):
            c = json.loads(line)
            cmds.append((c["tik"], c["target_raw_tip_cam_mm"], c["target_tip_cam_mm"]))
            continue
        if eng is None or '"tap"' not in line[:20]:
            continue
        m = json.loads(line)
        if m.get("tool") not in ROBOT_TOOL[tool] or m.get("lost") or m.get("raw_pos") is None:
            continue
        if m["tool"] != eng.get("robot_tool", m["tool"]):
            continue
        pk, qk = ("raw_pos", "raw_quat") if pose == "raw" else ("pos", "quat")   # filtered = what the FE is sent
        R = quat_to_R(m[qk])
        rows.append((m["ts"], np.asarray(m[pk], float) + R @ tip_l, R))
    if eng is None:
        raise SystemExit(f"{run}: no engage row")
    t0 = eng["t"]
    rows = [r for r in rows if r[0] >= t0]
    R0 = np.asarray(eng["R0_R"], float)
    # R0_tip in the run was computed with endo_follow's tip model; recompute with ours from the first fix instead
    return {"t": np.array([(r[0] - t0) / 1000.0 for r in rows]), "tip": np.array([r[1] for r in rows]),
            "R": np.array([r[2] for r in rows]), "R0": R0, "R0_tip": rows[0][1], "cfg": json.loads(
                (run / "config.json").read_text())["args"],
            "cmd_t": np.array([(c[0] - t0) / 1000.0 for c in cmds]), "cmd_raw": np.array([c[1] for c in cmds]),
            "cmd_shaped": np.array([c[2] for c in cmds])}


def best_shift(ta, xa, tb, xb, grid, detrend=False) -> float:
    """argmin_d RMS(xb(t) − xa(t − d)) over the overlap (arm seconds); detrend drops the mean offset."""
    best = (np.inf, 0.0)
    for d in grid:
        m = (tb - d >= ta[0]) & (tb - d <= ta[-1])
        if m.sum() < 50:
            continue
        a_ = np.stack([np.interp(tb[m] - d, ta, xa[:, i]) for i in range(3)], 1)
        e = xb[m] - a_
        if detrend:
            e = e - e.mean(0)
        best = min(best, (float(np.sqrt((e ** 2).sum(1).mean())), float(d)))
    return best[1]


def cmd_lag(tr: dict) -> tuple[float, float]:
    """The arm's lag behind the recording from its own logs (arm seconds): raw target → shaped command (the
    shaper; deterministic) + shaped command → marker (servo + mechanics). Both searched both ways."""
    t1 = best_shift(tr["cmd_t"], tr["cmd_raw"], tr["cmd_t"], tr["cmd_shaped"], np.arange(-0.2, 1.5, 0.01))
    t2 = best_shift(tr["cmd_t"], tr["cmd_shaped"], tr["t"], tr["tip"], np.arange(-0.3, 0.8, 0.01), detrend=True)
    return t1, t2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("--from-s", type=float, required=True)
    ap.add_argument("--len-s", type=float, required=True)
    ap.add_argument("--endo", default=None, help="left-arm endo_follow replay run dir")
    ap.add_argument("--drill", default=None, help="right-arm endo_follow replay run dir")
    ap.add_argument("--hand", action="store_true", help="the recorded hand instead of the arms (the reference)")
    ap.add_argument("--lag-s", default=None, help="shift the arm's samples earlier by this much (recording s); "
                                                     "'cmd' = from the run's own logs (shaper + arm lag, both ways); "
                                                     "'auto' = the shift that best overlays the hand's tip path")
    ap.add_argument("--match-sampling", choices=["none", "point", "window"], default="none",
                    help="put the arm's samples on the hand's sample times: point = its pose at that instant (the "
                         "fair pair of the hand's point samples), window = the mean of its samples around it (an "
                         "optimistic bound: it low-passes the arm only; ~15 arm samples per row at 0.2x)")
    ap.add_argument("--pose", choices=["raw", "filtered"], default="raw",
                    help="raw optical fixes (default) or the tracker's filtered pose — what the FE viewer is sent live; "
                         "the same choice for the hand (recording) and the arms (tap)")
    ap.add_argument("--robot-only", action="store_true", help="leave out a tool no arm replayed (default: the hand's)")
    ap.add_argument("--feed-hz", type=float, default=90.0)
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    hdr, rows, _ = rec.read_rows(a.recording)
    reg = hdr["registration"]["T0"]
    Rs, ts = np.asarray(reg["R"], float).reshape(3, 3), np.asarray(reg["t"], float)
    tc = hdr.get("toolCalib") or {}
    tools_over = {t: {"tipLocal": tc[t]["tipLocal"]} for t in ("endo", "drill") if t in tc and "tipLocal" in tc[t]}

    # the drill trigger on the recording's tExp axis
    btn_t = np.array([r["tExp"] for r in rows if r.get("k") == "f" and r.get("tool") == "drill" and "btn" in r
                      and r.get("tExp") is not None])
    btn_v = np.array([bool(r["btn"]) for r in rows if r.get("k") == "f" and r.get("tool") == "drill" and "btn" in r
                      and r.get("tExp") is not None])

    frames: dict[float, dict] = {}
    info = {}
    for tool in ("endo", "drill"):
        run = getattr(a, tool)
        hand = a.hand or run is None   # a tool no arm replayed plays the recorded hand, so the scene stays complete
        if run is None and a.robot_only:
            continue
        rp = rec.load_recording(a.recording, tool, t_start_s=a.from_s, t_end_s=a.from_s + a.len_s, pose=a.pose)
        t0_exp = float(rp["t0_ms"])            # tExp of the window's H0 row
        H0_tip, H0_R = np.asarray(rp["H0_tip"], float) * 1000.0, np.asarray(rp["H0_R"], float)
        if hand:
            t = np.asarray(rp["t"], float)
            tip_l = np.asarray(rp["tip_local_m"], float) * 1000.0
            R = np.array(rp["R"], float)
            tip = np.array([Ri @ tip_l + np.asarray(p, float) * 1000.0 for Ri, p in zip(R, rp["pos"])])
            vis = np.asarray(rp["vis"])
            keep = vis >= 4
            t, tip, R = t[keep], tip[keep], R[keep]
            src = f"hand (recorded {a.pose})"
        else:
            tr = robot_track(Path(run), tool, a.pose)
            speed = float(tr["cfg"]["replay_speed"])
            dR = np.einsum("nij,kj->nik", tr["R"], tr["R0"])      # R_robot · R_R0ᵀ
            R = np.einsum("nij,jk->nik", dR, H0_R)
            tip = H0_tip + (tr["tip"] - tr["R0_tip"])
            t = tr["t"] * speed
            lag, lag_parts = 0.0, None
            if a.lag_s is not None:   # the arm follows late: take its pose from `lag` later (recording seconds)
                if a.lag_s == "cmd":
                    lag_parts = cmd_lag(tr)
                    lag = sum(lag_parts) * speed
                elif a.lag_s == "auto":
                    lag = hand_lag(rp, t, tip)
                else:
                    lag = float(a.lag_s)
                t = t - lag
            th = np.asarray(rp["t"], float)[np.asarray(rp["vis"]) >= 4]
            if a.match_sampling == "window":   # each hand sample time: the mean of the arm's samples around it
                t, tip, R = match_to(th, t, tip, R)
            elif a.match_sampling == "point":  # each hand sample time: the arm's pose at that instant
                th = th[(th >= t[0]) & (th <= t[-1])]
                j = np.clip(np.searchsorted(t, th), 1, len(t) - 1)
                j = np.where(np.abs(t[j - 1] - th) < np.abs(t[j] - th), j - 1, j)
                tip = np.stack([np.interp(th, t, tip[:, i]) for i in range(3)], 1)
                R, t = R[j], th
            keep = (t >= 0) & (t <= a.len_s)
            t, tip, R = t[keep], tip[keep], R[keep]
            src = (f"robot {run} (speed {speed:g}, lag {lag * 1000:.0f} ms"
                   + (f" = (shaper {lag_parts[0] * 1000:.0f} + arm {lag_parts[1] * 1000:.0f} ms)×{speed:g}" if lag_parts else "")
                   + (f", {a.match_sampling} sampling" if a.match_sampling != "none" else "") + ")")
        info[tool] = {"source": src, "n": int(len(t)), "span_s": round(float(t[-1]), 2) if len(t) else 0}
        for ti, p, Ri in zip(t, tip, R):
            k = round(float(ti), 4)
            pose = {"tip": (Rs @ p + ts).round(3).tolist(), "quat": R_to_quat(Rs @ Ri)}
            if tool == "drill":
                i = int(np.clip(np.searchsorted(btn_t, t0_exp + ti * 1000.0), 0, len(btn_t) - 1))
                pose["trig"] = bool(btn_v[i]) if len(btn_t) else False
            frames.setdefault(k, {"t": k})[tool] = pose

    out = {"format": "spine-render-trajectory", "version": 1, "case": hdr.get("meshId"),
           "registration": {"R": Rs.reshape(-1).tolist(), "t": ts.tolist()}, "tools": tools_over,
           "feed_hz": a.feed_hz, "_source": {"recording": str(a.recording), "from_s": a.from_s, "len_s": a.len_s,
                                             "tools": info},
           "frames": [frames[k] for k in sorted(frames)]}
    Path(a.out).write_text(json.dumps(out))
    print(json.dumps(out["_source"], ensure_ascii=False))


if __name__ == "__main__":
    main()
