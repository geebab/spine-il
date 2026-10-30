#!/usr/bin/env python3
"""Policy output (tool-space trajectory in the bone/scene frame) → an FE recorder-v2 recording the robot can replay.

    python robot/inference/policy2rec.py --dataset DIR --episode 7 [--use action|state]      # rehearsal: a demo as "policy output"
    python robot/inference/policy2rec.py --traj out.json                                      # a real policy rollout

Trajectory = frames at a fixed fps of the dataset's 15-vector (robot/dataset/README.md): endo tip xyz (scene mm), endo quat
wxyz (tool → scene), drill tip xyz, drill quat wxyz, drill trigger 0/1. `--traj` JSON: {"fps": 15, "frames": [[15 floats], …],
"ref_recording": "<hand recording path>" (optional), "meta": {…}}.

The recording borrows the reference hand recording's header (registration T0, tool models, drill, FE flags) so the pose
maths is the recorder's own: R_cam = T0.Rᵀ·R_scene, tip_cam = T0.Rᵀ·(tip_scene − T0.t), marker pos = tip_cam − R_cam·tipLocal.
Rows carry both the filtered and `raw` pose (identical), vis 4, the drill `btn` (btnSrc "policy"), `scene.tip`; no cut rows
(the robot session will cut). hdr.origin = {"source": "policy", …}. Written to viewer-data/sessions/policy/<date>/ as
s_<stamp>_<case>_trk_<4hex>.jsonl — `robot/eval/new_recording.py <4hex>` then plays it like a hand recording.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import secrets
import time
from pathlib import Path

import numpy as np

SESS = Path.home() / "il-robot/viewer-data/sessions"


def q_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_q(R: np.ndarray) -> list[float]:
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    q = np.array([w, x, y, z]); q /= np.linalg.norm(q)
    return q.round(6).tolist()


def from_dataset(d: Path, ep: int, use: str) -> tuple[np.ndarray, float, dict]:
    import pyarrow.parquet as pq
    info = json.loads((d / "meta/info.json").read_text())
    rows = [t for f in sorted(glob.glob(str(d / "data/*/*.parquet"))) for t in [pq.read_table(f)]]
    frames, idx = [], []
    for t in rows:
        e = np.asarray(t.column("episode_index").to_pylist())
        sel = np.nonzero(e == ep)[0]
        if len(sel):
            col = t.column("action" if use == "action" else "observation.state").to_pylist()
            fi = t.column("frame_index").to_pylist()
            frames += [col[i] for i in sel]; idx += [fi[i] for i in sel]
    if not frames:
        raise SystemExit(f"episode {ep} not in {d}")
    frames = np.asarray(frames, float)[np.argsort(idx)]
    meta = [json.loads(l) for l in (d / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()][ep]
    return frames, float(info.get("fps", 15)), meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset"); src.add_argument("--traj")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--use", choices=["action", "state"], default="action",
                    help="dataset rehearsal: 'action' = the next-state targets a policy would emit")
    ap.add_argument("--ref-recording", default=None, help="hand recording whose header (T0, tool models) is borrowed")
    ap.add_argument("--out-dir", default=str(SESS / "policy"))
    ap.add_argument("--max-step-mm", type=float, default=2.0)
    ap.add_argument("--max-step-deg", type=float, default=3.0)
    a = ap.parse_args()

    if a.dataset:
        frames, fps, meta = from_dataset(Path(a.dataset), a.episode, a.use)
        ref = a.ref_recording or meta.get("recording")
        origin_extra = {"rehearsal": True, "dataset": str(a.dataset), "episode": a.episode, "use": a.use,
                        "demo_session": meta.get("session_id")}
    else:
        tj = json.loads(Path(a.traj).read_text())
        frames, fps = np.asarray(tj["frames"], float), float(tj.get("fps", 15))
        ref = a.ref_recording or tj.get("ref_recording")
        origin_extra = {"rehearsal": False, "traj": str(a.traj), **(tj.get("meta") or {})}
    if frames.ndim != 2 or frames.shape[1] != 15:
        raise SystemExit(f"frames must be N×15, got {frames.shape}")
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent))
    from safety import limit_traj
    frames, n_clamped = limit_traj(frames, a.max_step_mm, a.max_step_deg)   # never hand a jump to the robot
    origin_extra["limit"] = {"max_step_mm": a.max_step_mm, "max_step_deg": a.max_step_deg, "clamped_frames": n_clamped}
    if not ref:
        raise SystemExit("--ref-recording needed (header for T0 / tool models)")
    with open(ref) as fh:
        hdr = next(d for d in map(json.loads, fh) if d.get("k") == "hdr")
    T0 = hdr["registration"]["T0"]
    R0, t0 = np.asarray(T0["R"], float).reshape(3, 3), np.asarray(T0["t"], float)
    # a trajectory planned for the robot with a registration fitted to the arms' tools (rollout.py --gate-registration: a
    # reference recording made on another rig) is written in THAT camera frame: the arms, the robot link and the robot
    # session's anchor registration all read it from this recording. Q = this rig's camera ← the reference recording's.
    reg, Q = (origin_extra.get("registration") if a.traj else None), np.eye(3)
    if reg and reg.get("T0"):
        R_ref = R0
        R0, t0 = np.asarray(reg["T0"]["R"], float).reshape(3, 3), np.asarray(reg["T0"]["t"], float)
        Q = R0.T @ R_ref
    tl = {t: np.asarray(hdr["toolCalib"][t]["tipLocal"], float) for t in ("endo", "drill")}
    # the tracker's per-row `shaft` (tool axis, camera frame) and `up` — the FE drives take the tool direction from
    # `shaft`; without it the viewer falls back to another axis and draws the endoscope lying sideways (10-01). Both
    # come from the reference recording: shaft_local = median(R_camᵀ·shaft) per tool (constant), up = median(up).
    shaft_local, up_cam = {}, {}
    with open(ref) as fh:
        acc = {"endo": [], "drill": []}; ups = {"endo": [], "drill": []}
        for line in fh:
            if '"k":"f"' not in line[:12]:
                continue
            r0 = json.loads(line); t = r0.get("tool") or "endo"
            if t in acc and r0.get("shaft") and r0.get("quat") and not r0.get("lost"):
                acc[t].append(q_to_R(r0["quat"]).T @ np.asarray(r0["shaft"], float))
                if r0.get("up"):
                    ups[t].append(r0["up"])
            if all(len(v) >= 400 for v in acc.values()):
                break
    for t in ("endo", "drill"):
        if not acc[t]:
            raise SystemExit(f"reference recording has no `shaft` rows for {t}")
        m = np.median(np.asarray(acc[t]), axis=0); shaft_local[t] = m / np.linalg.norm(m)
        up_cam[t] = (Q @ np.median(np.asarray(ups[t]), axis=0) if ups[t] else None)   # a camera-frame vector: turned with the frame

    now = dt.datetime.now()
    case = hdr.get("meshId") or hdr.get("caseGroupKey") or "altician_lumbar1"
    taken = {f.stem.rsplit("_trk_", 1)[-1] for f in SESS.glob("**/*_trk_*.jsonl")}   # the 4-hex id names the recording for
    rid = secrets.token_hex(2)                                                         # new_recording.py — it must be unique
    while rid in taken:                                                                # (10-03: a reused id replayed an old file)
        rid = secrets.token_hex(2)
    sid = f"s_{now:%Y%m%dT%H%M%S}_{case}_trk_{rid}"
    h = dict(hdr)
    h.update({"sessionId": sid, "createdAtIso": now.astimezone().isoformat(timespec="seconds"), "operator": "policy",
              "startBy": "policy2rec"})
    h["origin"] = {"v": 1, "source": "policy", "from": "policy2rec", "robotRun": None, "replayOf": hdr.get("sessionId"),
                   "calib": (hdr.get("origin") or {}).get("calib"), "registration": {"method": "copied", "from": hdr.get("sessionId")},
                   "policy": origin_extra}
    if reg and reg.get("T0"):
        h["registration"] = {**hdr["registration"], "T0": {"R": R0.reshape(-1).tolist(), "t": t0.tolist()}}
        h["origin"]["registration"] = {"method": "fitted", "from": hdr.get("sessionId"), "file": reg.get("file"),
                                       "arms_of": reg.get("arms_of"), "turn_deg": (reg.get("fit") or {}).get("turn_deg")}
    for k in ("replay", "baseline"):
        h.pop(k, None)
    t_start = time.time() * 1000.0
    h["t0Wall"] = t_start
    out_rows, r = [h], 1
    step = 1000.0 / fps
    for i, fr in enumerate(frames):
        tE = t_start + i * step
        for tool, off in (("endo", 0), ("drill", 7)):
            tip_s = fr[off:off + 3]; q_s = fr[off + 3:off + 7]
            Rc = R0.T @ q_to_R(q_s)
            tip_c = R0.T @ (tip_s - t0)
            pos_c = tip_c - Rc @ tl[tool]
            qc = R_to_q(Rc)
            row = {"k": "f", "r": r, "i": i, "seq": i, "tool": tool, "tSrc": tE, "tMono": i * step, "tExp": tE, "tsd": "g",
                   "tExpSrc": tE, "tEmit": tE, "pos": pos_c.round(3).tolist(), "quat": qc, "tip": tip_c.round(3).tolist(),
                   "rmse": 0.5, "vis": 4, "split": 0, "still": 0, "hold": 0,
                   "raw": {"pos": pos_c.round(3).tolist(), "quat": qc, "tip": tip_c.round(3).tolist()},
                   "shaft": (Rc @ shaft_local[tool]).round(5).tolist(),
                   "scene": {"tip": np.asarray(tip_s).round(3).tolist(),
                             "u": (q_to_R(q_s) @ shaft_local[tool]).round(6).tolist()}}
            if up_cam[tool] is not None:
                row["up"] = np.asarray(up_cam[tool]).round(6).tolist()
            if tool == "drill":
                row.update({"btn": int(fr[14] > 0.5), "btnStale": 0, "btnSrc": "policy", "tBtn": tE})
            out_rows.append(row); r += 1
    d = Path(a.out_dir) / f"{now:%Y-%m-%d}"; d.mkdir(parents=True, exist_ok=True)
    f = d / f"{sid}.jsonl"
    f.write_text("\n".join(json.dumps(x, separators=(",", ":")) for x in out_rows) + "\n")
    print(json.dumps({"recording": str(f), "rid": rid, "frames": len(frames), "fps": fps, "ref": ref, "clamped_frames": n_clamped,
                      "sha256_12": hashlib.sha256(f.read_bytes()).hexdigest()[:12]}))


if __name__ == "__main__":
    main()
