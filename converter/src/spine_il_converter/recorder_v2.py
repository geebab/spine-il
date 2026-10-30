"""Load a real recorder v2 session (rec-2.x JSONL from the viewer's tracker recorder) into per-tool arrays.

Schema authority: `app/src/shells/clinic/lib/sessionRecordSchemaV2*.ts` in the viewer repo. Only the fields
below are read; the schema grows by extra keys and row kinds only, so everything else is ignored.

- Row order is `r` (hdr = 0, then written order). A missing `r` is a lost upload chunk (counted), a repeated
  `r` is a re-sent row and the first copy wins. File line order is not trusted: `hdr` need not be line 1.
- `f` rows (one per tracker message and tool): `tExp` camera-frame stamp (epoch ms — the dataset time axis,
  shared by both tools of one frame), `tSrc` tracker send time (diagnostic only), `pos` + `quat`
  (w, x, y, z; tool → D435 camera), `scene{tip,u}` = the viewer's composeToolPose(T, …) in LPS mm,
  `lost` / `hold` / `vis` / `still`, drill `btn` bits (key absent = unknown), `drawn` (what the screen showed
  when the row arrived), `real` (what the drive applied with this very message), drill `proxy.stamped`,
  per-row `T` when the registration is dynamic.
- `cut` rows: stamp batches the drill engine actually applied — kept verbatim for labels and replay.
- `hdr`: `registration.T0` {R row-major 9, t mm} = T_scene←camera, `timingSpec`, `tracker.cfg.tools`,
  `toolCalib`, `baseline`, overrides, `replay`.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np

from . import schema
from .geometry import quat_to_matrix, tool_rotation

FORMAT_REC2 = "rec2"
FORMAT_PLAN0 = "plan0"  # PLAN-0 §5.3 synthetic fixture schema (fixtures.py) — handled by session.py
TIMING_T0 = "v2-t0"
TIMING_CORE = "v2-core"
BTN_TRIG = 1  # sessionRecordSchemaV2.BTN_TRIG (ZR)
MIN_VIS = 4  # fewer visible balls = 3-ball pose (endo_follow drops those frames too)
ROBOT_TOOLS = ("endo2",)  # robot-mounted marker blocks (tracker branch `tracker-endo2`)
# Tool models of robot-mounted blocks tracked under a plain tool name: the right arm's drill block B
# (tool_calib_drill_robotR.json) is geometrically ≤ 0.95 mm from the hand drill and is also called "drill".
ROBOT_MODEL_SHA12 = ("162a00282c56",)
PLANE_FACING_MIN_COS = 0.1  # trackerEndoMap.TRACKER_PLANE_FACING_MIN_COS
MIN_NORMAL_SHAFT_DEG = 10.0  # below this the marker normal ⟂ shaft (the tool frame's x axis, the roll) is ill-defined
SESSION_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,120}")  # sessionRecordSchema.SESSION_ID_RE (= relay safe_sid)
# per-tool row counts: every row, then why rows are not valid poses (a row counts once, first reason wins)
COUNT_KEYS = ("rows", "lost", "no_T", "no_scene", "bad_pose", "hold", "vis_lt4", "no_time", "dup_time")


def detect_format(path: str | Path) -> str:
    """`plan0` for the synthetic PLAN-0 fixtures (recorderVersion '*-fixture'), else `rec2` (the real recorder)."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and row.get("k") == "hdr":
                fixture = str(row.get("recorderVersion", "")).endswith("-fixture")
                return FORMAT_PLAN0 if fixture or ("r" not in row and "capture" in row) else FORMAT_REC2
    return FORMAT_REC2


@dataclass
class ToolTrack:
    """One tool's `f` rows in time order (a repeated time stamp keeps its first row), as arrays."""

    tool: str
    t: np.ndarray  # (N,) ms on the session time axis (tExp; tSrc only for v2-core files)
    t_src: np.ndarray  # (N,) ms tracker send time — diagnostic only
    r: np.ndarray  # (N,) row numbers
    valid: np.ndarray  # (N,) a fresh optical measurement with a scene pose
    tip: np.ndarray  # (N, 3) scene.tip, LPS mm (NaN where the row carries no pose)
    rot: np.ndarray  # (N, 3, 3) scene ← tool: z = scene.u, x = marker-plane normal ⟂ z (NaN where no pose)
    roll: np.ndarray  # (N,) endoscope viewer roll (deg), NaN for the drill and rows without a pose
    roll_viewer: np.ndarray  # (N,) roll is the viewer's own recorded value for this message
    roll_rule_err: np.ndarray  # (N,) |viewer − replicated rule| (deg) where both exist, else NaN
    trig: np.ndarray  # (N,) drill trigger 0/1; rows without `btn` hold the last known value
    trig_known: np.ndarray  # (N,) the row carried `btn`
    stamped_t: np.ndarray  # (M,) drill: times of engine snapshots with proxy.stamped = 1
    entry_t: np.ndarray  # (K,) endoscope: times of recorded viewer entry points
    entry: np.ndarray  # (K, 3) entry (skin portal) LPS mm
    entry_dir: np.ndarray  # (K, 3) view direction with that entry
    entry_depth: np.ndarray  # (K,) tip depth past the entry, mm (> 0 = scope inserted)
    entry_src: str  # "drawn" | "real" | ""
    normal_shaft_deg: float  # median angle between the marker-plane normal and the shaft (NaN without poses)
    counts: dict

    @property
    def n_rows(self) -> int:
        return self.counts["rows"]

    @property
    def lost_frac(self) -> float:
        return self.counts["lost"] / self.counts["rows"] if self.counts["rows"] else 1.0


@dataclass
class Recording:
    path: Path
    header: dict = field(default_factory=dict)
    header_line: str = ""  # verbatim hdr line
    footer: dict | None = None
    tools: dict[str, ToolTrack] = field(default_factory=dict)
    cuts: list[tuple[str, dict]] = field(default_factory=list)  # (verbatim line, row) in r order
    time_axis: str = "tExp"
    stats: dict = field(default_factory=dict)
    issues: list[schema.Issue] = field(default_factory=list)
    format: ClassVar[str] = FORMAT_REC2

    @property
    def session_id(self) -> str:
        sid = self.header.get("sessionId")
        return sid if isinstance(sid, str) else self.path.stem

    @property
    def errors(self) -> list[schema.Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> list[schema.Issue]:
        return [i for i in self.issues if i.level == "warning"]

    @property
    def timing_spec(self) -> str | None:
        return self.header.get("timingSpec")

    @property
    def is_core(self) -> bool:
        """No T0 stamps: timingSpec 'v2-core', or no row carries tExp."""
        return self.timing_spec == TIMING_CORE or self.time_axis != "tExp"

    @property
    def robot_tools(self) -> list[str]:
        return robot_tools_of(self.header)

    @property
    def tool_models(self) -> dict[str, dict]:
        """Per tool: the tracker's calib model (file, sha12) and the relay's toolCalib sha256_12."""
        return tool_models_of(self.header)


def tool_models_of(hdr: dict) -> dict[str, dict]:
    tracker_tools = _d(_d(_d(hdr.get("tracker")).get("cfg")).get("tools"))
    tool_calib = _d(hdr.get("toolCalib"))
    out = {}
    for tool in schema.TOOLS:
        calib = _d(_d(tracker_tools.get(tool)).get("calib"))
        out[tool] = {"file": calib.get("file"), "sha12": calib.get("sha12"),
                     "relay_sha12": _d(tool_calib.get(tool)).get("sha256_12")}
    return out


def robot_tools_of(hdr: dict) -> list[str]:
    """Robot-mounted marker blocks the tracker ran: `endo2`, or a robot drill model tracked as "drill"."""
    tracker_tools = _d(_d(_d(hdr.get("tracker")).get("cfg")).get("tools"))
    stream = _d(hdr.get("stream"))
    names = {*tracker_tools, *_strs(stream.get("tools")), *_d(_d(stream.get("cfg")).get("tools"))}
    out = sorted(n for n in names if isinstance(n, str) and n in ROBOT_TOOLS)
    for tool, model in tool_models_of(hdr).items():
        file = str(model["file"] or "")
        shas = sorted({model["sha12"], model["relay_sha12"]} & set(ROBOT_MODEL_SHA12))
        if "robot" in file.lower():
            out.append(f"{tool}:{file}")
        elif shas:
            out.append(f"{tool}:{shas[0]}")
    return out


def header_warnings(hdr: dict) -> list[str]:
    """Dataset-identity gates that warn (the rejecting ones are decided per episode, see episode_v2)."""
    out = []
    if hdr.get("timingSpec") == TIMING_CORE:
        out.append("timingSpec 'v2-core' (no T0 stamps): excluded unless --allow-core")
    elif hdr.get("timingSpec") != TIMING_T0:
        out.append(f"timingSpec {hdr.get('timingSpec')!r} is neither 'v2-t0' nor 'v2-core': read as T0 when rows carry tExp")
    if hdr.get("baseline") is None:
        out.append("baseline is null: the dataset baseline (tracker/FE cfgHash) was not pinned for this recording")
    for key in ("baselineOverride", "t0Override"):
        if hdr.get(key):
            out.append(f"{key}: the start gate was overridden by hand")
    if robot := robot_tools_of(hdr):
        out.append(f"tracker ran robot-mounted marker blocks {robot}: excluded unless --allow-robot-tools")
    for tool, model in tool_models_of(hdr).items():
        relay, tracker = model["relay_sha12"], model["sha12"]
        if relay and tracker and relay != tracker:
            out.append(f"{tool}: toolCalib sha256_12 {relay} != tracker.cfg calib sha12 {tracker} (relay and tracker models differ)")
        elif tracker and not relay:
            out.append(f"{tool}: toolCalib has no sha256_12 — the tracker's calib {tracker} cannot be cross-checked")
    carved = _d(_d(hdr.get("replay")).get("carvedBeforeStart"))
    if any(carved.values()):
        out.append(f"replay.carvedBeforeStart {carved}: cut rows alone do not reproduce the scene")
    return out


def load_recording(path: str | Path) -> Recording:
    path = Path(path)
    rec = Recording(path=path)
    stats = Counter()

    def issue(level, line, msg):
        rec.issues.append(schema.Issue(level, line, msg))

    parsed: list[tuple[int, str, dict]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, text in enumerate(fh, start=1):
            text = text.rstrip("\r\n")
            if not text.strip():
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as e:
                issue("warning", line_no, f"invalid JSON skipped: {e.msg}")
                stats["invalid_json"] += 1
                continue
            if isinstance(row, dict):
                parsed.append((line_no, text, row))
    rows = _order_by_r(parsed, stats, rec)

    headers = [item for item in rows if item[2].get("k") == "hdr"]
    if not headers:
        issue("error", None, "no hdr row")
        rec.stats = dict(stats)
        return rec
    if len(headers) > 1:
        issue("error", headers[1][0], "duplicate hdr row")
    line_no, rec.header_line, hdr = headers[0]
    rec.header = hdr
    if not is_safe_session_id(rec.session_id):  # it names sidecar folders
        issue("error", line_no, f"sessionId {rec.session_id!r} is not a safe id ([A-Za-z0-9_.-]{{1,120}}, not dots only)")
    if hdr.get("schemaVersion") != schema.SCHEMA_VERSION:
        issue("error", line_no, f"schemaVersion {hdr.get('schemaVersion')!r} != {schema.SCHEMA_VERSION}")
    if hdr.get("quatOrder", "wxyz") != "wxyz":
        issue("error", line_no, f"quatOrder {hdr.get('quatOrder')!r} is not 'wxyz'")
    reg = _d(hdr.get("registration"))
    dynamic = bool(reg.get("dynamic"))
    T0 = _rigid(reg.get("T0"))
    if T0 is None and not dynamic:
        issue("error", line_no, "hdr.registration.T0 missing or not a rotation: tool rotations cannot be put in the scene")
    for msg in header_warnings(hdr):
        issue("warning", line_no, msg)

    f_rows: dict[str, list[dict]] = {t: [] for t in schema.TOOLS}
    kinds = Counter()
    for _, text, row in rows:
        kind = row.get("k")
        kinds[str(kind)] += 1
        if kind == "f":
            tool = row.get("tool", "endo")  # untagged = endo (protocol convention, toolOf)
            if isinstance(tool, str) and tool in f_rows:
                f_rows[tool].append(row)
            else:
                stats["unknown_tool_rows"] += 1
        elif kind == "cut":
            rec.cuts.append((text, row))
        elif kind == "ftr":
            rec.footer = row  # rows are in r order: the last footer wins
    if rec.footer is None:
        issue("warning", None, "no ftr row (session not finished)")
    if stats["unknown_tool_rows"]:
        issue("warning", None, f"{stats['unknown_tool_rows']} f rows with an unknown tool ignored")

    rec.time_axis = "tExp" if any("tExp" in row for rows_ in f_rows.values() for row in rows_) else "tSrc"
    if rec.time_axis != "tExp":
        issue("warning", None, "no row carries tExp: time axis falls back to tSrc (tracker send time)")
    for tool, tool_rows in f_rows.items():
        rec.tools[tool] = _tool_track(tool, tool_rows, rec.time_axis, T0, dynamic)
        counts = rec.tools[tool].counts
        if counts["no_time"]:
            issue("warning", None, f"{tool}: {counts['no_time']} rows without {rec.time_axis} dropped")
        if not rec.tools[tool].valid.any():
            issue("warning", None, f"{tool}: no valid pose rows")
        if rec.tools[tool].normal_shaft_deg < MIN_NORMAL_SHAFT_DEG:
            issue("warning", None, f"{tool}: marker normal only {rec.tools[tool].normal_shaft_deg:.1f} deg from the shaft — "
                                   "the tool frame's x axis (roll) is ill-defined for this tool model")
    stats.update({f"rows_{k}": v for k, v in kinds.items()})
    stats["unknown_kinds"] = sum(v for k, v in kinds.items() if k not in ("hdr", "f", "ev", "cut", "ftr"))
    rec.stats = dict(stats)
    return rec


def _order_by_r(parsed: list[tuple[int, str, dict]], stats: Counter, rec: Recording) -> list[tuple[int, str, dict]]:
    has_r = [_int(row.get("r")) for _, _, row in parsed]
    if not any(has_r):
        rec.issues.append(schema.Issue("warning", None, "rows carry no `r`: file order used"))
        return parsed
    if not all(has_r):
        stats["rows_without_r"] = has_r.count(False)
        rec.issues.append(schema.Issue("warning", None, f"{stats['rows_without_r']} rows without `r` dropped"))
    keyed = sorted((item[2]["r"], i) for i, (item, ok) in enumerate(zip(parsed, has_r, strict=True)) if ok)
    out: list[tuple[int, str, dict]] = []
    first: dict[int, dict] = {}
    for r, i in keyed:
        row = parsed[i][2]
        if r in first:
            stats["duplicate_r"] += 1
            stats["duplicate_r_conflicts"] += row != first[r]
            continue
        first[r] = row
        out.append(parsed[i])
    rs = sorted(first)
    stats["r_gaps"] = (rs[-1] - rs[0] + 1) - len(rs)
    if stats["duplicate_r"]:
        conflict = f", {stats['duplicate_r_conflicts']} with different content" if stats["duplicate_r_conflicts"] else ""
        rec.issues.append(
            schema.Issue("warning", None, f"{stats['duplicate_r']} re-sent rows (same r), first copy kept{conflict}")
        )
    if stats["r_gaps"]:
        rec.issues.append(schema.Issue("warning", None, f"{stats['r_gaps']} rows missing (gaps in r: lost chunks)"))
    return out


def _tool_track(tool: str, rows: list[dict], axis: str, T0, dynamic: bool) -> ToolTrack:
    counts = Counter(dict.fromkeys(COUNT_KEYS, 0))
    counts["rows"] = len(rows)
    timed = []
    for row in rows:
        if _num(row.get(axis)):
            timed.append(row)
        else:
            counts["no_time"] += 1
    timed.sort(key=lambda row: row[axis])  # stable: equal stamps keep r order
    kept: list[dict] = []
    for row in timed:
        if kept and row[axis] == kept[-1][axis]:
            counts["dup_time"] += 1
            continue
        kept.append(row)
    n = len(kept)
    col = lambda key, default: np.array([row.get(key, default) for row in kept], dtype=np.float64)  # noqa: E731
    t, t_src, r = col(axis, np.nan), col("tSrc", np.nan), col("r", -1).astype(np.int64)
    valid, has_pose = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
    tip, u, pos_s, cam = (np.full((n, 3), np.nan) for _ in range(4))
    R_full = np.full((n, 3, 3), np.nan)
    for k, row in enumerate(kept):
        counts["lost"] += bool(row.get("lost"))
        if row.get("lost"):
            continue
        scene = _d(row.get("scene"))
        T = _rigid(row.get("T")) if dynamic else T0
        if T is None:
            counts["no_T"] += 1
            continue
        if not (_vec(scene.get("tip"), 3) and _vec(scene.get("u"), 3) and _vec(row.get("quat"), 4) and _vec(row.get("pos"), 3)):
            counts["no_scene"] += 1
            continue
        if not (0.5 < np.linalg.norm(row["quat"]) < 1.5 and 0.5 < np.linalg.norm(scene["u"]) < 1.5):
            counts["bad_pose"] += 1  # a malformed quaternion / shaft: masked like any other invalid row
            continue
        R_T, t_T = T
        R_full[k] = R_T @ quat_to_matrix(np.asarray(row["quat"], dtype=np.float64))
        pos_s[k] = R_T @ np.asarray(row["pos"], dtype=np.float64) + t_T
        cam[k] = t_T  # the camera centre in scene coordinates
        tip[k], u[k] = scene["tip"], scene["u"]
        has_pose[k] = True
        if row.get("hold") == 1:  # jump-hold re-send: not a new measurement
            counts["hold"] += 1
        elif _num(row.get("vis")) and row["vis"] < MIN_VIS:
            counts["vis_lt4"] += 1
        else:
            valid[k] = True

    rot = np.full((n, 3, 3), np.nan)
    normal_shaft_deg = float("nan")
    if has_pose.any():
        u[has_pose] /= np.linalg.norm(u[has_pose], axis=1, keepdims=True)
        # Full rotation (quat × T.R) re-expressed with z on the shaft: x = marker-plane normal (tool local +z) ⟂ u.
        rot[has_pose] = tool_rotation(u[has_pose], R_full[has_pose][:, :, 2])
        cos = np.abs(np.sum(R_full[has_pose][:, :, 2] * u[has_pose], axis=1))
        normal_shaft_deg = float(np.degrees(np.arccos(np.clip(np.median(cos), 0.0, 1.0))))

    roll = np.full(n, np.nan)
    roll_viewer = np.zeros(n, dtype=bool)
    roll_err = np.full(n, np.nan)
    empty3 = np.empty((0, 3))
    entry_t, entry, entry_dir, entry_depth, entry_src = np.empty(0), empty3, empty3, np.empty(0), ""
    if tool == "endo":
        if has_pose.any():
            roll[has_pose] = viewer_roll_deg(R_full[has_pose], pos_s[has_pose], u[has_pose], cam[has_pose])
        recorded = _recorded_rolls(rows)
        for k, row in enumerate(kept):
            v = recorded.get(_msg_key(row.get("seq"), row.get("tSrc")))
            if v is None or not has_pose[k]:
                continue
            roll_err[k] = abs(wrap_deg(v - roll[k]))
            roll[k], roll_viewer[k] = v, True
        entry_t, entry, entry_dir, entry_depth, entry_src = _entries(kept, axis)

    trig, trig_known = np.zeros(n), np.zeros(n, dtype=bool)
    stamped: list[float] = []
    if tool == "drill":
        last = 0.0
        for k, row in enumerate(kept):
            if _int(row.get("btn")):
                last, trig_known[k] = float(row["btn"] & BTN_TRIG), True
            trig[k] = last
            if _d(row.get("proxy")).get("stamped") == 1:
                when = _d(row.get("drawn")).get(axis, row[axis])  # the engine used the drawn pose
                if _num(when):
                    stamped.append(float(when))
    return ToolTrack(
        tool=tool, t=t, t_src=t_src, r=r, valid=valid, tip=tip, rot=rot, roll=roll, roll_viewer=roll_viewer,
        roll_rule_err=roll_err, trig=trig, trig_known=trig_known, stamped_t=np.unique(np.asarray(stamped)),
        entry_t=entry_t, entry=entry, entry_dir=entry_dir, entry_depth=entry_depth, entry_src=entry_src,
        normal_shaft_deg=normal_shaft_deg, counts=dict(counts),
    )


def viewer_roll_deg(R_s: np.ndarray, pos_s: np.ndarray, u: np.ndarray, cam_s: np.ndarray) -> np.ndarray:
    """The viewer's endoscope roll (deg) per pose, rows in time order (useTrackerFilmDrive → trackerEndoMap).

    Marker-plane normal n0 = R_s·[0,0,1], flipped to face the camera (filmPlaneFacingSign: the sign of
    n0·(cam − pos); when |cos| < 0.1 the previous sign is kept), projected ⟂ u and measured about u from
    the roll zero axis a = scene +Z ⟂ u (+Y when u is within ~8° of ±Z) — filmPlaneUpScene → filmPoseToEndoAbs.
    """
    n0 = R_s[:, :, 2]
    los = cam_s - pos_s
    dist = np.linalg.norm(los, axis=1)
    c = np.sum(n0 * los, axis=1) / np.where(dist > 1e-6, dist, np.inf)
    decisive = np.where(np.abs(c) >= PLANE_FACING_MIN_COS, np.sign(c), 0.0)
    last = np.maximum.accumulate(np.where(decisive != 0, np.arange(len(c)), -1))
    # before the first decisive frame the viewer skips; fall back to the raw sign so every pose has a roll
    sign = np.where(last >= 0, decisive[np.maximum(last, 0)], np.where(c >= 0, 1.0, -1.0))
    n = n0 * sign[:, None]
    v = u / np.linalg.norm(u, axis=1, keepdims=True)
    tgt = _perp(n, v)
    ref = np.where((np.abs(v[:, 2]) > 0.99)[:, None], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    a = _perp(ref, v)
    return np.degrees(np.arctan2(np.sum(np.cross(a, tgt) * v, axis=1), np.sum(a * tgt, axis=1)))


def wrap_deg(d):
    return (np.asarray(d, dtype=np.float64) + 180.0) % 360.0 - 180.0


def _perp(v: np.ndarray, axis: np.ndarray) -> np.ndarray:
    w = v - np.sum(v * axis, axis=1, keepdims=True) * axis
    norm = np.linalg.norm(w, axis=1, keepdims=True)
    return np.where(norm > 1e-9, w / np.where(norm > 1e-9, norm, 1.0), np.nan)


def _recorded_rolls(rows: list[dict]) -> dict[tuple, float]:
    """(seq, tSrc) of a message → the roll the viewer applied for it: `real.roll` of that row, or the
    `drawn.roll` of any row whose screen showed that message (drawn.seq / drawn.tSrc)."""
    out: dict[tuple, float] = {}
    for row in rows:
        for roll_of, key in ((_d(row.get("real")), _msg_key(row.get("seq"), row.get("tSrc"))),
                             (_d(row.get("drawn")), _msg_key(_d(row.get("drawn")).get("seq"), _d(row.get("drawn")).get("tSrc")))):
            if key is not None and _num(roll_of.get("roll")):
                out.setdefault(key, float(roll_of["roll"]))
    return out


def _msg_key(seq, t_src) -> tuple | None:
    """A message identity (relay seq, tracker tSrc) — tSrc disambiguates a seq reset."""
    return (seq, t_src) if _int(seq) and _num(t_src) else None


def _entries(rows: list[dict], axis: str):
    """Viewer entry points (skin portal) of the endoscope: from `drawn` (every v2-t0 row), else `real`."""
    for source in ("drawn", "real"):
        t, entry, view, depth = [], [], [], []
        for row in rows:
            d = _d(row.get(source))
            if d.get("lost") or not (_vec(d.get("entry"), 3) and _vec(d.get("viewDir"), 3) and _num(d.get("depth"))):
                continue
            t.append(row[axis])
            entry.append(d["entry"])
            view.append(d["viewDir"])
            depth.append(d["depth"])
        if t:
            as3 = lambda x: np.asarray(x, dtype=np.float64).reshape(-1, 3)  # noqa: E731
            return np.asarray(t, dtype=np.float64), as3(entry), as3(view), np.asarray(depth, dtype=np.float64), source
    empty3 = np.empty((0, 3))
    return np.empty(0), empty3, empty3, np.empty(0), ""


def _rigid(d) -> tuple[np.ndarray, np.ndarray] | None:
    """{R: row-major 9, t: 3} → (R, t) when R is a rotation (RᵀR ≈ I, det ≈ +1; rows carry 6 dp), else None."""
    if not isinstance(d, dict) or not _vec(d.get("R"), 9) or not _vec(d.get("t"), 3):
        return None
    R = np.asarray(d["R"], dtype=np.float64).reshape(3, 3)
    if np.abs(R.T @ R - np.eye(3)).max() > 1e-3 or abs(np.linalg.det(R) - 1.0) > 1e-3:
        return None
    return R, np.asarray(d["t"], dtype=np.float64)


def is_safe_session_id(sid: str) -> bool:
    return bool(SESSION_ID_RE.fullmatch(sid)) and bool(sid.strip("."))


def _d(x) -> dict:
    return x if isinstance(x, dict) else {}


def _strs(x) -> list[str]:
    return [v for v in x if isinstance(v, str)] if isinstance(x, list) else []


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _vec(x, n: int) -> bool:
    return isinstance(x, list) and len(x) == n and all(_num(v) for v in x)
