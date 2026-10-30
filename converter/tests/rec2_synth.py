"""Synthetic recorder v2 (rec-2.3.0-t0) sessions for tests — rows shaped like real rig recordings.

The layout copies the viewer's recorder (sessionRecordSchemaV2*.ts) as seen in the 0551 rig session and the viewer's
committed sample: hdr (r = 0) → f rows (endo and drill share each camera frame's tExp) with ev / cut rows interleaved
→ ftr. Poses are analytic in the scene (tip linear in time, fixed shaft, constant-rate twist about the shaft) and are
written the way the recorder does (camera-frame pos / quat / tip / shaft through T0⁻¹, scene{tip,u}, 3 / 6 dp), so a
converted frame can be compared with `truth()` to rounding precision.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spine_il_converter.geometry import matrix_to_quat

T_EXP0 = 1_790_175_900_146.187  # epoch ms of camera frame 0
FRAME_MS = 1000.0 / 90.0
MONO0 = 140_045.275
TSRC_LAG_MS = 24.5  # tSrc − tExp (0551 median)
BTN_LAG_MS = 16.5  # tBtn − tExp (0551 median)
DRAWN_LAG = 3  # the screen shows the message 3 frames back
GRAVITY_UP_CAM = [0.048748, -0.9785, -0.200402]  # the relay's constant `up` — never a tool axis
T0_R = np.array([-0.011759992042560341, -0.20597252250652204, 0.9784871090410233,
                 -0.07503610473261806, -0.9756138904185375, -0.20626953194536804,
                 0.9971114709869278, -0.07584758924781883, -0.003982164182907382]).reshape(3, 3)
T0_T = np.array([-458.4576997833809, 280.4620319541602, 986.9416977682616])
PORTAL = np.array([48.9, -1.0, 1031.5])
PRE_INSERT_ENTRY = PORTAL + [40.0, 0.0, 0.0]  # where the viewer's skin cast lands while the scope is outside
DRILL_R_MM = 4.0
Z_LOCAL = np.array([0.0, 0.0, 1.0])


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _perp(v, axis) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return _unit(v - np.dot(v, axis) * axis)


def _rot(axis, angle_rad) -> np.ndarray:
    k = _unit(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * K @ K


@dataclass(frozen=True)
class ToolModel:
    shaft_local: tuple  # handle → tip, tool frame (the marker-plane normal is local +z)
    tip_len: float
    u: tuple  # fixed shaft direction in the scene
    tip0: tuple  # scene tip at t = 0
    vel: tuple  # mm/s
    facing: int  # +1: local +z faces the camera, −1: faces away
    twist0_deg: float
    twist_deg_s: float


ENDO = ToolModel(shaft_local=(-1.0, 0.0, 0.0), tip_len=160.0, u=tuple(_unit([-0.3503, -0.9067, -0.2348])),
                 tip0=tuple(PORTAL + 10.0 * _unit([0.3503, 0.9067, 0.2348])), vel=(0.0, 0.0, 0.0),
                 facing=+1, twist0_deg=-30.0, twist_deg_s=15.0)
DRILL = ToolModel(shaft_local=tuple(_unit([-0.9332, -0.0742, 0.3515])), tip_len=180.0,
                  u=tuple(_unit([-0.2706, -0.960964, 0.057658])), tip0=(40.0, 5.0, 1000.0), vel=(3.0, -2.0, 4.0),
                  facing=-1, twist0_deg=0.0, twist_deg_s=10.0)
ENDO_INSERT_MM_S = 20.0  # the endoscope advances along its shaft: depth −10 → +70 mm over 4 s
ENDO_LATERAL_MM_S = 1.0


@dataclass(frozen=True)
class RecSpec:
    session_id: str = "s_20260926T120000_synth_trk_ab12"
    duration_s: float = 4.0
    timing_spec: str = "v2-t0"
    # rec-2.3.1-t0 (FE round F1) adds ftr.tools.<t>.rateHz, build.gitAt 'page-load', build.serverStartSha — additive
    recorder_version: str = "rec-2.3.1-t0"
    dynamic: bool = False  # film registration: per-row T drifting away from T0
    robot_tools: bool = False  # tracker.cfg.tools also has the robot-mounted endo2 block
    calib_mismatch: bool = False  # hdr.toolCalib.drill sha differs from the tracker's
    baseline: dict | None = None
    t0_override: bool = False
    cut_rows: bool = True  # rec ≥ 2.3 writes `cut` rows (else only proxy.stamped, like rec-2.2)
    verdict: str | None = "good"
    footer: bool = True
    drawn: bool = True
    real_entries: bool = True
    header_basal: bool = False  # PLAN-0 hdr.basal (fallback source)
    trig: tuple = (1.5, 2.5)  # ZR held (s)
    btn_unknown: tuple = (2.4, 2.7)  # rows without `btn` (BT stall)
    cuts: tuple = (1.6, 2.4)  # engine stamps every other drill frame in this window
    invalid: tuple = (  # (tool, kind, start s, end s)
        ("endo", "lost", 2.00, 2.05),
        ("drill", "hold", 2.80, 2.812),
        ("endo", "vis", 3.30, 3.312),
        ("drill", "noscene", 3.50, 3.512),
    )
    still: tuple = (("endo", 1.0, 1.5),)
    no_T: tuple = ()  # dynamic only: (tool, start s, end s) rows written without T
    # viewer entry points: every message's skin cast differs a little (sub-mm), like the rig's
    entry_noise_mm: float = 0.15
    entry_alternate_mm: float = 0.0  # inserted entries alternate ±this in x (no stable window, like a shaky portal)
    frozen_entry: tuple | None = None  # (start s, end s, offset xyz): inserted entries re-use one wrong value
    reinserted: tuple | None = None  # (start s, offset xyz): from then on the scope sits in another portal
    drill_calib_file: str = "tool_calib_drill.json"
    drill_sha12: str = "340b2ac02ad2"


def frame_times(spec: RecSpec) -> np.ndarray:
    """Camera frames the tracker processed: every third 90 Hz frame is skipped, plus a 5-frame hole (≈65 Hz like 0551)."""
    k = np.arange(int(spec.duration_s * 1000.0 / FRAME_MS) + 1)
    keep = (k % 3 != 2) & ~((k >= 200) & (k < 205))
    return T_EXP0 + k[keep] * FRAME_MS


def T_at(spec: RecSpec, t_ms) -> tuple[np.ndarray, np.ndarray]:
    """T_scene←camera in effect at t (drifts for a dynamic registration)."""
    if not spec.dynamic:
        return T0_R, T0_T
    s = (t_ms - T_EXP0) / 1000.0
    return _rot([0.2, 0.3, 0.93], np.radians(2.0 * s)) @ T0_R, T0_T + np.array([0.5, -0.3, 0.2]) * s


def _R0(model: ToolModel, cam: np.ndarray) -> np.ndarray:
    """Tool → scene at twist 0: shaft_local → u, marker normal (local +z) facing (or not) the camera."""
    u = np.asarray(model.u)
    pos = np.asarray(model.tip0) - model.tip_len * u
    cos_a = float(np.dot(model.shaft_local, Z_LOCAL))
    n0 = cos_a * u + np.sqrt(1 - cos_a**2) * model.facing * _perp(cam - pos, u)
    b1 = np.asarray(model.shaft_local)
    b2 = _perp(Z_LOCAL, b1)
    c1, c2 = u, _perp(n0, u)
    return np.column_stack([c1, c2, np.cross(c1, c2)]) @ np.column_stack([b1, b2, np.cross(b1, b2)]).T


def tool_pose(tool: str, t_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Scene tip and R_scene←tool (tracker tool frame) at time t."""
    model = ENDO if tool == "endo" else DRILL
    s = (t_ms - T_EXP0) / 1000.0
    u = np.asarray(model.u)
    tip = np.asarray(model.tip0) + np.asarray(model.vel) * s
    if tool == "endo":
        tip = tip + u * ENDO_INSERT_MM_S * s + _perp([1.0, 0.0, 0.0], u) * ENDO_LATERAL_MM_S * s
    R = _rot(u, np.radians(model.twist0_deg + model.twist_deg_s * s)) @ _R0(model, T0_T)
    return tip, R


def endo_depth(t_ms: float) -> float:
    return float(np.dot(tool_pose("endo", t_ms)[0] - PORTAL, ENDO.u))


def truth(tool: str, t_ms, spec: RecSpec | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(tip (N,3), R_scene←tool (N,3,3) in the converter's frame — z = shaft, x = marker normal ⟂ shaft —, and the
    viewer's endoscope roll (N,) deg: camera-facing marker normal ⟂ u, measured about u from scene +Z ⟂ u)."""
    model = ENDO if tool == "endo" else DRILL
    tips, Rs, rolls = [], [], []
    for t in np.atleast_1d(t_ms):
        tip, R = tool_pose(tool, t)
        u = R @ np.asarray(model.shaft_local)
        x = _perp(R @ Z_LOCAL, u)
        tips.append(tip)
        Rs.append(np.column_stack([x, np.cross(u, x), u]))
        pos = tip - R @ (np.asarray(model.shaft_local) * model.tip_len)
        cam = T_at(spec or RecSpec(), t)[1]
        n = R @ Z_LOCAL if np.dot(R @ Z_LOCAL, cam - pos) > 0 else -(R @ Z_LOCAL)
        up, a = _perp(n, u), _perp([0.0, 0.0, 1.0], u)
        rolls.append(np.degrees(np.arctan2(np.dot(np.cross(a, up), u), np.dot(a, up))) if tool == "endo" else np.nan)
    return np.array(tips), np.array(Rs), np.array(rolls)


def _in(windows, tool, kind, s) -> bool:
    return any(w[0] == tool and w[1] == kind and w[2] <= s < w[3] for w in windows)


def build_rows(spec: RecSpec) -> list[dict]:
    times = frame_times(spec)
    rows: list[dict] = [_header(spec)]
    seq = {"endo": 85066, "drill": 85066}
    history: dict[str, list[dict]] = {"endo": [], "drill": []}  # per tool: messages with a pose (for drawn)
    cut_frames: set[float] = set()
    i = 0
    trig_prev = False
    for fi, t in enumerate(times):
        s = (t - T_EXP0) / 1000.0
        for tool in ("endo", "drill"):
            seq[tool] += 1
            row = {"k": "f", "r": 0, "i": i, "seq": seq[tool], "tool": tool,
                   "tSrc": int(np.floor(t + TSRC_LAG_MS + (fi * 7 + (tool == "drill")) % 9)),
                   "tMono": round(MONO0 + (t - T_EXP0) + 20.0 + (fi % 5), 3)}
            i += 1
            if spec.timing_spec == "v2-t0":
                row.update({"tExp": t, "tsd": "g", "tEmit": round(t + TSRC_LAG_MS + 0.4, 3)})
            trig_now = spec.trig[0] <= s < spec.trig[1]
            if tool == "drill" and trig_now != trig_prev:
                rows.append({"k": "ev", "r": 0, "i": i, "tMono": row["tMono"] - 0.5, "ev": "zr_down" if trig_now else "zr_up"})
                trig_prev = trig_now
            lost = _in(spec.invalid, tool, "lost", s)
            if lost:
                row["lost"] = 1
            else:
                _pose_fields(spec, row, tool, t, s)
            if tool == "drill" and spec.timing_spec == "v2-t0":
                row["tBtn"] = round(t + BTN_LAG_MS, 3)
                if spec.btn_unknown[0] <= s < spec.btn_unknown[1]:
                    row["btnStale"] = 1
                else:
                    row["btn"] = 1 if trig_now else 0
                    row["btnStale"] = 0
            if not lost:
                history[tool].append(row)
            drawn_src = history[tool][-1 - DRAWN_LAG] if len(history[tool]) > DRAWN_LAG else None
            if spec.drawn and spec.timing_spec == "v2-t0" and drawn_src is not None:
                row["drawn"] = _drawn(tool, drawn_src, spec)
            if tool == "drill":
                stamped = drawn_src is not None and drawn_src["tExp" if "tExp" in drawn_src else "tSrc"] in cut_frames
                row["proxy"] = {"p": row.get("scene", {}).get("tip", [0, 0, 0]), "q": row.get("scene", {}).get("tip", [0, 0, 0]),
                                "depthMm": 0.5, "touching": int(trig_now), "cutting": int(trig_now),
                                "stamped": int(stamped), "ageMs": 14.1}
            if fi % 7 == 3:
                row["fooFuture"] = {"a": 1}  # an unknown key — must be ignored
            rows.append(row)
            if tool == "drill" and spec.timing_spec == "v2-t0" and spec.cuts[0] <= s < spec.cuts[1] and fi % 2 == 0 and not lost:
                cut_frames.add(t)
                if spec.cut_rows:
                    tip, _ = tool_pose("drill", t)
                    c = tip - DRILL_R_MM * np.asarray(DRILL.u) + np.array([1 / 3, 1 / 7, 1 / 11]) * 1e-3
                    rows.append({"k": "cut", "r": 0, "i": i, "tMono": row["tMono"] + 1.25, "tExp": t, "tool": "drill",
                                 "target": "bone", "label": "vertebrae_bone_t000", "c": [c.tolist()], "rMm": 4,
                                 "fn": "smooth", "smooth": 1, "fresh": 1, "dv": 0, "mod": 17 + fi % 5})
        if fi == 40:
            rows.append({"k": "ev", "r": 0, "i": i, "tMono": MONO0 + 500.0, "ev": "imu_rx_gap", "tool": "drill",
                         "att": 1, "t0": T_EXP0 + 480.0, "t1": T_EXP0 + 515.0, "gapMs": 35.0})
            rows.append({"k": "img", "r": 0, "cam": "endoscope", "frameIdx": 0})  # a future row kind — ignored
        if fi == 90:
            rows.append({"k": "ev", "r": 0, "i": i, "tMono": MONO0 + 1300.0, "ev": "view_state", "key": "softCarve", "value": False})
    if spec.footer:
        rows.append(_footer(spec, times))
    for r, row in enumerate(rows):
        row["r"] = r
    return rows


def _pose_fields(spec: RecSpec, row: dict, tool: str, t: float, s: float) -> None:
    model = ENDO if tool == "endo" else DRILL
    tip_s, R_s = tool_pose(tool, t)
    R_T, t_T = T_at(spec, t)
    R_cam = R_T.T @ R_s
    tip_local = np.asarray(model.shaft_local) * model.tip_len
    pos_s = tip_s - R_s @ tip_local
    pos_cam = R_T.T @ (pos_s - t_T)
    tip_cam = R_T.T @ (tip_s - t_T)
    shaft_cam = R_cam @ np.asarray(model.shaft_local)
    q = matrix_to_quat(R_cam)
    r3 = lambda v: [round(float(x), 3) for x in v]  # noqa: E731
    r6 = lambda v: [round(float(x), 6) for x in v]  # noqa: E731
    row.update({"pos": r3(pos_cam), "quat": r6(q), "tip": r3(tip_cam), "shaft": r6(shaft_cam), "up": GRAVITY_UP_CAM,
                "rmse": 0.398, "vis": 3 if _in(spec.invalid, tool, "vis", s) else 4, "split": 0,
                "still": int(any(w[0] == tool and w[1] <= s < w[2] for w in spec.still))})
    if spec.timing_spec == "v2-t0":
        row["tExpSrc"] = t
        row["hold"] = int(_in(spec.invalid, tool, "hold", s))
        row["raw"] = {"pos": row["pos"], "quat": row["quat"], "tip": row["tip"]}
    if not _in(spec.invalid, tool, "noscene", s):
        row["scene"] = {"tip": r3(tip_s), "u": r6(R_s @ np.asarray(model.shaft_local))}
    if spec.dynamic and not any(w[0] == tool and w[1] <= s < w[2] for w in spec.no_T):
        row["T"] = {"R": r6(R_T.ravel()), "t": r3(t_T)}
    if spec.real_entries and tool == "endo" and int(round((t - T_EXP0) / FRAME_MS)) % 10 == 0:
        row["real"] = _pose_view("endo", t, row, spec)
    elif spec.real_entries and tool == "drill" and int(round((t - T_EXP0) / FRAME_MS)) % 5 == 0:
        row["real"] = _pose_view("drill", t, row, spec)


def viewer_entry(spec: RecSpec, t: float) -> np.ndarray:
    """The entry point the viewer recorded for the endoscope message of camera frame t."""
    depth = endo_depth(t)
    s = (t - T_EXP0) / 1000.0
    noise = spec.entry_noise_mm * np.array([np.sin(t * 0.37), np.cos(t * 0.53), np.sin(t * 0.71)])
    if depth <= 0:
        return PRE_INSERT_ENTRY + noise
    if spec.frozen_entry and spec.frozen_entry[0] <= s < spec.frozen_entry[1]:
        return PORTAL + np.asarray(spec.frozen_entry[2])
    parity = 1.0 if int(round((t - T_EXP0) / FRAME_MS)) % 2 == 0 else -1.0
    moved = np.asarray(spec.reinserted[1]) if spec.reinserted and s >= spec.reinserted[0] else 0.0
    return PORTAL + noise + moved + [parity * spec.entry_alternate_mm, 0.0, 0.0]


def _pose_view(tool: str, t: float, row: dict, spec: RecSpec) -> dict:
    r3 = lambda v: [round(float(x), 3) for x in v]  # noqa: E731
    r6 = lambda v: [round(float(x), 6) for x in v]  # noqa: E731
    tip, _ = tool_pose(tool, t)
    if tool == "endo":
        return {"tip": r3(tip), "viewDir": r6(ENDO.u), "depth": round(endo_depth(t), 3),
                "roll": round(float(truth("endo", t)[2][0]), 3), "entry": r3(viewer_entry(spec, t))}
    return {"B": r3(tip - DRILL_R_MM * np.asarray(DRILL.u)), "u": r6(DRILL.u),
            "entry": r3(np.asarray(DRILL.tip0) - 30.0 * np.asarray(DRILL.u)), "entryFallback": 0}


def _drawn(tool: str, src: dict, spec: RecSpec) -> dict:
    t = src.get("tExp", src["tSrc"])
    base = {"seq": src["seq"], "tSrc": src["tSrc"], "tExp": t, "tExpSrc": t, "still": src.get("still", 0),
            "hold": src.get("hold", 0), "lost": 0}
    if tool == "drill":
        base.update({"btnStale": 0, "drillOn": int(src.get("btn", 0) == 1)})
    return {**base, **_pose_view(tool, t, src, spec)}


def _header(spec: RecSpec) -> dict:
    tools = {"endo": {"fused": False, "calib": {"file": "tool_calib.json", "sha12": "2d48a98bb499"}},
             "drill": {"fused": True, "calib": {"file": spec.drill_calib_file, "sha12": spec.drill_sha12}}}
    if spec.robot_tools:
        tools["endo2"] = {"fused": False, "calib": {"file": "tool_calib.json", "sha12": "2d48a98bb499"}}
    hdr = {
        "k": "hdr", "r": 0, "schemaVersion": 2,
        "recorderVersion": spec.recorder_version if spec.timing_spec == "v2-t0" else "rec-2.0.0-core",
        "timingSpec": spec.timing_spec, "coordConvention": "scene=LPS_mm; cam=D435_optical_mm(+Y down)",
        "sessionId": spec.session_id, "createdAtIso": "2026-09-26T03:00:00.000Z", "operator": "synth",
        "portalMode": "tracker", "meshId": "altician_lumbar1", "caseGroupKey": None,
        "registration": {"method": "film" if spec.dynamic else "fid3",
                         "T0": {"R": T0_R.ravel().tolist(), "t": T0_T.tolist()},
                         "dynamic": spec.dynamic, "refId": 3 if spec.dynamic else None, "sceneFromRef": None,
                         "quality": {"nPoints": 5}},
        "drill": {"radiusMm": 4, "sizeKey": "large", "proxy": {"rule": "surface", "enabled": True}},
        "fe": {"rawabs": False, "cfgHash": "a4c7b9adfc50"},
        "quatOrder": "wxyz",
        "toolCalib": {"endo": {"sha256_12": "2d48a98bb499", "tipLocal": [-160.0, 0.0, 0.0]},
                      "drill": {"sha256_12": "ffffffffffff" if spec.calib_mismatch else spec.drill_sha12,
                                "gravityUpCam": GRAVITY_UP_CAM}},
        "clock": {"tsDomain": "global_time", "sameHost": True},
        "tracker": {"boot": "50dce2", "cfgHash": "8d76f7834c32", "cfg": {"v": 1, "fps": 90, "tools": tools},
                    "tsDomain": "global_time"},
        "baseline": spec.baseline, "drawnPoint": "vtk-apply",
        "replay": {"cutRows": True, "drawn": True, "version": 1, "carvedBeforeStart": {"bone": False, "soft": False},
                   "drillVersionStart": 0},
        "stream": {"fps": 90, "tools": ["endo", "drill"]},
        "build": {"gitSha": "7d51bc0c11b1", "gitDirty": False, "gitAt": "page-load", "serverStartSha": "7d51bc0c11b1"},
        "t0Wall": round(T_EXP0 + 8.8), "t0Mono": MONO0 - 20.4, "rowsPerTool": "native",
        "futureHeaderField": {"x": 1},
    }
    if not spec.cut_rows:
        hdr["recorderVersion"] = "rec-2.2.0-t0"
        del hdr["replay"]
    if spec.t0_override:
        hdr["t0Override"] = True
    if spec.header_basal:
        hdr["basal"] = {"entryWorld": PORTAL.tolist(), "targetWorld": (PORTAL + 50.0 * np.asarray(ENDO.u)).tolist()}
    return hdr


def _footer(spec: RecSpec, times: np.ndarray) -> dict:
    ftr = {"k": "ftr", "r": 0, "endReason": "completed", "durationS": round((times[-1] - times[0]) / 1000.0, 2),
           "tools": {"endo": {"nFrames": len(times), "nLost": 0, "nSeqDropped": 0, "stillFrac": 0.1, "rateHz": 60.0},
                     "drill": {"nFrames": len(times), "nLost": 0, "nSeqDropped": 0, "stillFrac": 0.0, "rateHz": 60.0}},
           "nGapEvents": 0, "uploadFailures": 0, "drillVersionStart": 0, "drillVersionEnd": 1,
           "operatorVerdict": spec.verdict, "verdictReason": None}
    return ftr


def dump(row: dict) -> str:
    return json.dumps(row)  # the relay writes Python json.dumps lines


def write_session(root: Path, spec: RecSpec = RecSpec(), rows: list[dict] | None = None, shuffle: bool = False,
                  resend: bool = False, drop_r: tuple = ()) -> Path:
    """Write DATA_ROOT/sessions/<date>/<sessionId>.jsonl. shuffle: upload chunks arrive out of order and hdr is not
    line 1; resend: a chunk is re-sent (identical) and one row is re-sent with different content after the original;
    drop_r: rows lost in transit."""
    rows = build_rows(spec) if rows is None else rows
    lines = [dump(row) for row in rows if row["r"] not in drop_r]
    if shuffle or resend:
        chunks = [lines[i : i + 37] for i in range(0, len(lines), 37)]
        order = list(range(len(chunks)))
        if shuffle:
            random.Random(7).shuffle(order)
            if order[0] == 0:  # keep hdr off line 1
                order = order[1:] + order[:1]
        out = []
        for j, c in enumerate(order):
            out += chunks[c]
            if resend and j == 2:
                out += chunks[order[1]]  # the whole previous chunk again (pagehide beacon + original request)
        if resend:
            victim = json.loads(chunks[order[3]][5])
            victim["tMono"] = -1.0
            if "scene" in victim:
                victim["scene"] = {"tip": [0.0, 0.0, 0.0], "u": [0.0, 0.0, 1.0]}
            out.append(dump(victim))  # a later copy with other content — the first copy must win
        lines = out
    path = Path(root) / "sessions" / "2026-09-26" / f"{spec.session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path
