"""Viewer session recording (recorder v2 JSONL) → replay events for recording_player. ROS-free (numpy only), so it is
unit-tested without ROS; replay_timeline.py paces the events and the node only turns them into messages.

Field meaning (viewer schema sessionRecordSchemaV2*.ts):
  - row order: `r` (every row, monotone). A missing r = a lost chunk; the same r twice = a retransmission → the first
    one in the file wins. File line order is arrival order, not authority.
  - pos / quat / tip / shaft / up / raw: D435 camera frame, mm, quat [w, x, y, z] tool → camera. `tip` is filtered on
    its own (not rigid with pos/quat, up to ~3 mm); `raw` = before filter / still / hold / gyro fusion.
  - scene / drawn / cut.c: viewer scene (LPS), mm. hdr.registration.T0 = T_scene←cam {R: row-major 9, t: mm}; a dynamic
    registration puts the T it applied on every f row.
  - tExp: the camera frame's stamp (epoch ms) = the dataset time axis. tSrc / tEmit: tracker send (~+25 ms). tMono:
    the viewer's clock (performance.now) when the recorder wrote / received the row.
  - drawn: what the viewer had on screen when the row reached the recorder (drawn.tExp = the camera frame it showed).
  - cut: the stamps the drill engine applied in one tick (c = centres, full precision; rMm; mod = voxels changed).

One epoch-ms timeline, two clocks:
  - tracker axis: an f row's pose (and its registration T) sits at its tExp (tSrc when the row has none: v2-core).
  - screen axis: drawn, cut and ev rows sit at t0Wall + (tMono − t0Mono) (viewer and tracker share the host clock,
    hdr.clock.sameHost), never before the tracker time they refer to — so the drawn tool trails the tracked one by the
    lag the operator saw, and a cut appears when the drawn drill is there. Without tMono / t0 they fall back to the
    tracker axis: drawn → the row's tExp, cut → cut.tExp (else the previous drill row's drawn.tExp, else the previous
    f row), ev → the previous f row.
Events are sorted by (t, r); per tool the poses keep r order (tExp is monotone per tool; across tools a row held for the
`real` join can be written up to ~50 ms late).

Output is in ROS units: metres, quaternion (x, y, z, w); stamps stay epoch ms (convert.split_stamp_ms splits them).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .convert import R_to_xyzw, quat_wxyz_to_xyzw

TRACKING = "~/il-robot/tracking"   # = tracker_bridge.TRACKING
# Ball models of the replayed tools. A viewer recording's "drill" is the hand drill (tool_calib_drill.json); the live
# bridge draws "drill" with the robot's block B (tool_calib_drill_robotR.json). The player compares each file's sha with
# the recording's hdr.toolCalib and warns on a mismatch.
REPLAY_TOOL_MODELS = {"endo": f"{TRACKING}/tool_calib.json", "drill": f"{TRACKING}/tool_calib_drill.json"}
_SUB = {"tf": 0, "pose": 1, "drawn": 2}   # order of the events one f row yields (TF before the pose that needs it)
_ROW_ERRORS = (AttributeError, KeyError, TypeError, ValueError)   # a malformed row: skipped, never the whole file


@dataclass(slots=True)
class Event:
    t: float      # ms on the recording timeline (epoch)
    r: int        # the row it came from
    kind: str     # tf | pose | drawn | cut | ev
    data: dict


def _order(e: Event):
    return e.t, e.r, _SUB.get(e.kind, 0)


def _obj(x) -> dict:
    return x if isinstance(x, dict) else {}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


@dataclass
class Recording:
    header: dict
    events: list[Event]          # sorted by (t, r)
    stats: dict
    warnings: list[str]
    T0: np.ndarray | None        # T_scene←cam, 4×4, metres (hdr.registration.T0)
    dynamic: bool                # registration.dynamic: per-row T → "tf" events
    footer: dict | None = None
    static_T: np.ndarray | None = None   # the registration to publish once: T0, unless the rows carry their own T

    @property
    def session_id(self) -> str:
        return str(self.header.get("sessionId") or "")

    @property
    def t_first(self) -> float:
        return self.events[0].t if self.events else 0.0

    @property
    def t_last(self) -> float:
        return self.events[-1].t if self.events else 0.0

    @property
    def duration_s(self) -> float:
        return (self.t_last - self.t_first) / 1000.0

    @property
    def tools(self) -> list[str]:
        return sorted(self.stats["f_rows"])

    @property
    def drill_radius_m(self) -> float | None:
        r = _obj(self.header.get("drill")).get("radiusMm")
        return float(r) / 1000.0 if _num(r) and r > 0 else None

    def tool_calib_sha(self, tool: str) -> str | None:
        return _obj(_obj(self.header.get("toolCalib")).get(tool)).get("sha256_12")


# ── rows ────────────────────────────────────────────────────────────────────────────────────────────────────────────
def order_rows(lines: Iterable[str | bytes]) -> tuple[list[dict], dict]:
    """JSONL lines → rows sorted by `r`, a repeated r dropped (the first in file order wins), plus counts."""
    seen: dict[int, dict] = {}
    bad = no_r = dup = 0
    for ln in lines:
        if isinstance(ln, bytes):
            ln = ln.decode("utf-8")
        ln = ln.strip()
        if not ln:
            continue
        try:
            d = json.loads(ln)
        except ValueError:
            bad += 1
            continue
        if not isinstance(d, dict):
            bad += 1
            continue
        r = d.get("r")
        if isinstance(r, bool) or not isinstance(r, int):
            no_r += 1
            continue
        if r in seen:
            dup += 1
            continue
        seen[r] = d
    rs = sorted(seen)
    gaps = [(a + 1, b - 1) for a, b in zip(rs, rs[1:]) if b > a + 1]
    return [seen[r] for r in rs], {"rows": len(rs), "bad_lines": bad, "no_r": no_r, "dup_r": dup,
                                   "missing_r": sum(b - a + 1 for a, b in gaps), "gap_ranges": gaps[:10]}


def _vec3_m(v) -> list[float]:
    a = np.asarray(v, float).reshape(-1)
    if a.shape != (3,) or not np.all(np.isfinite(a)):
        raise ValueError(f"bad 3-vector {v!r}")
    return [float(x) / 1000.0 for x in a]


def _unit(v) -> list[float] | None:
    a = np.asarray(v, float).reshape(-1)
    n = float(np.linalg.norm(a)) if a.shape == (3,) and np.all(np.isfinite(a)) else 0.0
    return [float(x) / n for x in a] if n > 1e-9 else None


def _quat_xyzw(q) -> tuple[float, float, float, float]:
    a = np.asarray(q, float).reshape(-1)
    if a.shape != (4,) or not np.all(np.isfinite(a)) or np.linalg.norm(a) < 0.5:   # recorded quaternions are unit
        raise ValueError(f"bad quaternion {q!r}")
    return quat_wxyz_to_xyzw(a)


def _nested(row: dict, key: str) -> dict:
    """A nested object of a row ({} when absent); anything else makes the row malformed."""
    v = row.get(key)
    if v is None:
        return {}
    if not isinstance(v, dict):
        raise ValueError(f"{key} is not an object")
    return v


# ── rigid transforms ────────────────────────────────────────────────────────────────────────────────────────────────
def rigid_to_matrix(T: dict) -> np.ndarray:
    """Recorder RigidT {R: 9 row-major (or 3×3), t: mm} → 4×4 in mm; R re-orthonormalised (row T is rounded to 6 dp)."""
    R = np.asarray(T["R"], float).reshape(3, 3)
    t = np.asarray(T["t"], float).reshape(3)
    if not (np.all(np.isfinite(R)) and np.all(np.isfinite(t))):
        raise ValueError("registration T: non-finite")
    U, _, Vt = np.linalg.svd(R)
    Ro = U @ Vt
    if np.linalg.det(Ro) < 0 or np.abs(Ro - R).max() > 1e-3:
        raise ValueError("registration T: R is not a rotation")
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = Ro, t
    return M


def to_metres(M: np.ndarray) -> np.ndarray:
    out = np.array(M, float)
    out[:3, 3] /= 1000.0
    return out


def invert_rigid(M: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = M[:3, :3].T
    out[:3, 3] = -M[:3, :3].T @ M[:3, 3]
    return out


def registration_tf(T_scene_cam: np.ndarray, parent: str) -> tuple[bool, tuple, tuple]:
    """Registration T_scene←cam (4×4, m) → one TF edge (scene_is_parent, translation, quaternion xyzw).

    A TF transform maps child coordinates into its parent frame (p_parent = R·p_child + t), so
      parent "scene":  frame_id = scene,  child = camera, transform = T_scene←cam
      parent "camera": frame_id = camera, child = scene,  transform = T_cam←scene = T_scene←cam⁻¹
    ("camera" keeps the tree a tree when the camera frame already has a parent — calib_tf's <arm>_base_link.)"""
    if parent == "scene":
        M = np.asarray(T_scene_cam, float)
    elif parent == "camera":
        M = invert_rigid(np.asarray(T_scene_cam, float))
    else:
        raise ValueError(f"tf parent must be 'scene' or 'camera', got {parent!r}")
    q = np.asarray(R_to_xyzw(M[:3, :3]), float)
    q /= np.linalg.norm(q)
    return parent == "scene", tuple(float(v) for v in M[:3, 3]), tuple(float(v) for v in q)


def quat_xyzw_to_R(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


# ── one row → event payloads ────────────────────────────────────────────────────────────────────────────────────────
def row_time(row: dict) -> tuple[float | None, bool]:
    """(time on the tracker axis, fell back to tSrc) of an f row."""
    if row.get("tExp") is not None:
        return float(row["tExp"]), False
    if row.get("tSrc") is not None:
        return float(row["tSrc"]), True
    return None, False


def frame_fields(row: dict, t_ms: float) -> dict:
    """f row → TrackedPose / PoseStamped fields, shaped like convert.tap_to_fields (metres, xyzw, lost / held …) plus
    the raw epoch-ms times the player maps onto its clock: stamp_ms (= t_ms) and t_send_ms (tEmit, else tSrc).
    `frame` = the relay's per-tool counter `seq` (the recording has no camera frame number — the frame is the stamp).
    t_arrival / t_pose are tap-only (0). pose_raw falls back to the filtered pose when the row has no `raw`.
    A `raw` that is not an object or a non-unit quaternion raises ValueError (the row is skipped)."""
    raw = _nested(row, "raw")
    t_send = row.get("tEmit", row.get("tSrc"))
    out = {"tool": row.get("tool") or "endo", "frame": max(0, int(row.get("seq") or 0)), "lost": bool(row.get("lost")),
           "stamp_ms": float(t_ms), "stamp": float(t_ms) / 1000.0,
           "t_send_ms": float(t_send) if t_send is not None else 0.0, "t_arrival": 0.0, "t_pose": 0.0,
           "visible": min(255, max(0, int(row.get("vis") or 0))), "rmse": float(row.get("rmse") or 0.0),
           "still": bool(row.get("still")), "held": bool(row.get("hold"))}
    out["t_send"] = out["t_send_ms"] / 1000.0
    if not out["lost"] and row.get("pos") is not None and row.get("quat") is not None:
        out["pos"] = _vec3_m(row["pos"])
        out["quat"] = _quat_xyzw(row["quat"])
        out["pos_raw"] = _vec3_m(raw.get("pos", row["pos"]))
        out["quat_raw"] = _quat_xyzw(raw.get("quat", row["quat"]))
        out["tip"] = _vec3_m(row["tip"]) if row.get("tip") else None
        out["tip_raw"] = _vec3_m(raw["tip"]) if raw.get("tip") else None
        out["has_raw"] = raw.get("pos") is not None and raw.get("quat") is not None
    return out


def drawn_data(row: dict, radius_m: float | None = None) -> dict | None:
    """f row `drawn` → the tool as the viewer drew it (scene, m): endo = scope tip (the virtual camera) + view
    direction, drill = ball centre B + shaft axis u (+ the drill radius in effect, radius_m). None when the pose is
    absent (endo view degenerate / not drawn)."""
    d = _nested(row, "drawn")
    tool = row.get("tool") or "endo"
    p, u = (d.get("B"), d.get("u")) if tool == "drill" else (d.get("tip"), d.get("viewDir"))
    if p is None:
        return None
    lag = (float(row["tExp"]) - float(d["tExp"])) if row.get("tExp") is not None and d.get("tExp") is not None \
        else None
    return {"tool": tool, "p": _vec3_m(p), "dir": _unit(u) if u is not None else None,
            "entry": _vec3_m(d["entry"]) if d.get("entry") is not None else None,
            "roll_deg": d.get("roll"), "lost": bool(d.get("lost")), "drill_on": bool(d.get("drillOn")),
            "radius_m": radius_m if tool == "drill" else None,
            "seq": d.get("seq"), "t_exp": d.get("tExp"), "lag_ms": lag}


def cut_data(row: dict) -> dict:
    """cut row → stamp centres in the scene frame (m, (n, 3)), sphere radius (m), target / label / op / dv / mod."""
    op = row.get("op")
    c = row.get("c")
    pts = np.zeros((0, 3)) if op == "reset" or c is None else np.asarray(c, float).reshape(-1, 3) / 1000.0
    if not np.all(np.isfinite(pts)):
        raise ValueError("cut: non-finite centre")
    r = row.get("rMm")
    return {"target": row.get("target") or "bone", "label": row.get("label"), "op": op, "points": pts,
            "radius_m": float(r) / 1000.0 if r is not None else None, "fn": row.get("fn"),
            "t_exp": float(row["tExp"]) if row.get("tExp") is not None else None,
            "dv": row.get("dv"), "mod": int(row.get("mod") or 0)}


# ── whole file ──────────────────────────────────────────────────────────────────────────────────────────────────────
class _Walk:
    """State while walking the rows in r order. Each row method parses first and changes state only after the row has
    parsed, so a malformed row (any of _ROW_ERRORS) is skipped without a trace."""

    def __init__(self, hdr: dict, drawn: bool, dynamic: bool):
        self.drawn, self.dynamic = drawn, dynamic
        self.n = dict.fromkeys(("lost", "hold", "no_raw", "t_fallback", "screen_fallback", "bad_rows", "unknown_kind",
                                "cut_rows", "cut_points", "cut_mod", "cut_t_fallback", "soft_resets", "tf_conflict",
                                "drill_size_changes"), 0)
        self.f_rows: dict[str, int] = {}
        self.ev_names: dict[str, int] = {}
        self.events: list[Event] = []
        self.pending: list[Event] = []            # cut / ev rows waiting for the first f row's time
        self.last_f_t = self.last_drill_drawn_t = self.last_cut_t = None
        self.tf_at: dict[float, np.ndarray] = {}  # one TF per stamp (both tools share a frame; tf2 rejects repeats)
        self.last_drawn: dict[str, str] = {}
        r = _obj(hdr.get("drill")).get("radiusMm")
        self.radius_m = float(r) / 1000.0 if _num(r) and r > 0 else None   # the drill size in effect (ev can change it)
        t0w, t0m = hdr.get("t0Wall"), hdr.get("t0Mono")
        self.mono_to_epoch = float(t0w) - float(t0m) if _num(t0w) and _num(t0m) else None
        self.footer = None

    def screen(self, row: dict) -> float | None:
        """The row's moment on the viewer's clock, as epoch ms (None without tMono / t0)."""
        m = row.get("tMono")
        return self.mono_to_epoch + float(m) if self.mono_to_epoch is not None and _num(m) else None

    def _add(self, e: Event):
        (self.pending if e.t is None else self.events).append(e)

    def f_row(self, row: dict):
        r = row["r"]
        t, fell_back = row_time(row)
        if t is None:
            raise ValueError("f row without tExp / tSrc")
        fields = frame_fields(row, t)
        tool = fields["tool"]
        T = to_metres(rigid_to_matrix(row["T"])) if self.dynamic and row.get("T") is not None else None
        dr = _nested(row, "drawn")
        lag = (t - float(dr["tExp"])) if dr.get("tExp") is not None and not fell_back else None
        d_t = float(dr["tExp"]) if tool == "drill" and dr.get("tExp") is not None else None
        d_ev = None
        if self.drawn and dr:
            key = json.dumps(dr, sort_keys=True)
            if self.last_drawn.get(tool) != key:
                try:
                    d_ev = drawn_data(row, self.radius_m)
                except _ROW_ERRORS:
                    self.n["bad_rows"] += 1        # the pose still plays; only the drawn marker is skipped
                self.last_drawn[tool] = key
        s = self.screen(row)
        # ── the row parsed: from here on, state changes
        self.f_rows[tool] = self.f_rows.get(tool, 0) + 1
        self.n["t_fallback"] += fell_back
        self.n["lost"] += fields["lost"]
        self.n["hold"] += fields["held"]
        self.n["no_raw"] += "pos" in fields and not fields["has_raw"]
        if T is not None:
            if t in self.tf_at:
                self.n["tf_conflict"] += not np.allclose(T, self.tf_at[t], atol=1e-6)
            else:
                self.tf_at[t] = T
                self.events.append(Event(t, r, "tf", {"T": T}))
        self.events.append(Event(t, r, "pose", {"fields": fields, "screen_lag_ms": lag}))
        if d_ev is not None:
            self.n["screen_fallback"] += s is None
            self.events.append(Event(t if s is None else max(t, s), r, "drawn", d_ev))
        if d_t is not None:
            self.last_drill_drawn_t = d_t
        self.last_f_t = t
        if self.pending:
            for e in self.pending:
                e.t = t
            self.events.extend(self.pending)
            self.pending.clear()

    def cut_row(self, row: dict):
        d = cut_data(row)
        s = self.screen(row)
        if s is not None:
            t = s if d["t_exp"] is None else max(d["t_exp"], s)
        elif d["t_exp"] is not None:
            t = d["t_exp"]
        else:
            t = self.last_drill_drawn_t if self.last_drill_drawn_t is not None else self.last_f_t
        if t is not None and self.last_cut_t is not None:
            t = max(t, self.last_cut_t)          # cut rows are written in engine-tick order: keep it
        self.n["screen_fallback"] += s is None
        self.n["cut_t_fallback"] += d["t_exp"] is None
        if t is not None:
            self.last_cut_t = t
        self.n["cut_rows"] += 1
        self.n["cut_points"] += len(d["points"])
        self.n["cut_mod"] += d["mod"]
        self.n["soft_resets"] += d["op"] == "reset"
        self._add(Event(t, row["r"], "cut", d))

    def ev_row(self, row: dict):
        name = str(row.get("ev"))
        s = self.screen(row)
        if name == "drill_size_changed" and _num(row.get("d")) and row["d"] > 0:
            self.radius_m = float(row["d"]) / 1000.0   # later drawn drill balls use the new size
            self.n["drill_size_changes"] += 1
        self.n["screen_fallback"] += s is None
        self.ev_names[name] = self.ev_names.get(name, 0) + 1
        self._add(Event(s if s is not None else self.last_f_t, row["r"], "ev",
                        {"ev": name, "tool": row.get("tool"), "row": row}))


def _header_warnings(hdr: dict, st: dict) -> list[str]:
    warn = []
    if hdr.get("schemaVersion") != 2:
        warn.append(f"schemaVersion {hdr.get('schemaVersion')!r} (read as 2)")
    if hdr.get("timingSpec") != "v2-t0":
        warn.append(f"timingSpec {hdr.get('timingSpec')!r}: not a T0 recording (rows without tExp are timed by tSrc)")
    carved = [k for k, v in _obj(_obj(hdr.get("replay")).get("carvedBeforeStart")).items() if v]
    if carved:
        warn.append(f"carved before start ({', '.join(carved)}): the cut rows alone do not rebuild the scene")
    if "endo2" in _obj(_obj(_obj(hdr.get("tracker")).get("cfg")).get("tools")):
        warn.append("tracker cfg has endo2 (robot rig): 'endo' may be the robot's marker block")
    if _obj(hdr.get("clock")).get("sameHost") is False:
        warn.append("clock.sameHost false: screen times (drawn / cut / ev) may be offset from the tracker's")
    if st["dup_r"]:
        warn.append(f"{st['dup_r']} repeated r (retransmitted chunk) dropped")
    if st["missing_r"]:
        warn.append(f"{st['missing_r']} rows missing (lost chunks) {st['gap_ranges']}")
    return warn


def parse(lines: Iterable[str | bytes], *, drawn: bool = True) -> Recording:
    """JSONL lines → Recording (events on one epoch-ms timeline). drawn=False skips the drawn events."""
    rows, st = order_rows(lines)
    if not rows or rows[0].get("k") != "hdr":
        raise ValueError("recording has no header row (r=0, k='hdr')")
    hdr = rows[0]
    if str(hdr.get("quatOrder", "wxyz")) != "wxyz":
        raise ValueError(f"quatOrder {hdr.get('quatOrder')!r}: only 'wxyz' is understood")
    warn = _header_warnings(hdr, st)
    reg = _obj(hdr.get("registration"))
    T0 = None
    if reg.get("T0") is None:
        warn.append("no registration T0: no scene TF")
    else:
        try:
            T0 = to_metres(rigid_to_matrix(reg["T0"]))
        except _ROW_ERRORS:
            warn.append("registration T0 malformed: no scene TF")
    dynamic = bool(reg.get("dynamic"))
    w = _Walk(hdr, drawn, dynamic)
    if w.mono_to_epoch is None:
        warn.append("no t0Wall / t0Mono: drawn, cut and ev rows are timed on the tracker axis")
    for row in rows[1:]:
        k = row.get("k")
        try:
            if k == "f":
                w.f_row(row)
            elif k == "cut":
                w.cut_row(row)
            elif k == "ev":
                w.ev_row(row)
            elif k == "ftr":
                w.footer = row
            else:
                w.n["unknown_kind"] += 1
        except _ROW_ERRORS:
            w.n["bad_rows"] += 1
    if w.pending:
        warn.append(f"{len(w.pending)} cut/ev rows but no f row to time them: dropped")
    events = sorted(w.events, key=_order)
    n = w.n
    if n["bad_rows"]:
        warn.append(f"{n['bad_rows']} malformed rows skipped")
    if n["t_fallback"]:
        warn.append(f"{n['t_fallback']} f rows without tExp: timed by tSrc (~+25 ms send offset)")
    has_tf = any(e.kind == "tf" for e in events)
    if dynamic and not has_tf:
        warn.append("registration is dynamic but no row carries T: " +
                    ("T0 used as a static registration" if T0 is not None else "no scene TF"))
    static_T = T0 if T0 is not None and (not dynamic or not has_tf) else None
    for tool, cnt in _obj(_obj(w.footer).get("tools")).items():
        if isinstance(cnt, dict) and cnt.get("nFrames") is not None and cnt["nFrames"] != w.f_rows.get(tool, 0):
            warn.append(f"{tool}: footer nFrames {cnt['nFrames']} ≠ {w.f_rows.get(tool, 0)} f rows read")
    kinds: dict[str, int] = {}
    for e in events:
        kinds[e.kind] = kinds.get(e.kind, 0) + 1
    st.update(n, f_rows=w.f_rows, events=kinds, ev_names=w.ev_names)
    return Recording(hdr, events, st, warn, T0, dynamic, w.footer, static_T)


def load(path: str | Path, *, drawn: bool = True) -> Recording:
    with open(Path(path).expanduser(), encoding="utf-8") as fh:
        return parse(fh, drawn=drawn)


# ── tool models, topics, summary ────────────────────────────────────────────────────────────────────────────────────
def topic_namespace(ns: str) -> str:
    """'/replay', 'replay', '/replay/' → '/replay'; '' or '/' → '' (the live topic names)."""
    ns = (ns or "").strip().strip("/")
    return f"/{ns}" if ns else ""


def load_ball_model(path: str | Path) -> tuple[np.ndarray, float, str]:
    """Tracker tool calibration → (ball centres (n, 3) m in the tool frame, ball radius m, sha256_12 of the file bytes
    — the fingerprint the relay writes into hdr.toolCalib)."""
    raw = Path(path).expanduser().read_bytes()
    d = json.loads(raw)
    pts = np.asarray(d["points_local"], float).reshape(-1, 3) / 1000.0
    return pts, float(d.get("sphere_radius_mm", 7.0)) / 1000.0, hashlib.sha256(raw).hexdigest()[:12]


def balls_in_camera(points_local_m: np.ndarray, pos_m, quat_xyzw) -> np.ndarray:
    return np.asarray(points_local_m, float) @ quat_xyzw_to_R(quat_xyzw).T + np.asarray(pos_m, float)


def summary(rec: Recording) -> list[str]:
    """Human-readable lines (warnings not included — rec.warnings)."""
    s, h = rec.stats, rec.header
    return [f"{rec.session_id or '?'} · {h.get('recorderVersion')} · {h.get('timingSpec')} · {rec.duration_s:.2f} s · "
            f"registration {_obj(h.get('registration')).get('method')} {'dynamic' if rec.dynamic else 'static'}",
            f"rows {s['rows']} (repeated r {s['dup_r']}, missing r {s['missing_r']}, bad lines {s['bad_lines']}, "
            f"malformed {s['bad_rows']})",
            "events " + ", ".join(f"{k} {v}" for k, v in sorted(s["events"].items())),
            "f rows " + ", ".join(f"{k} {v}" for k, v in sorted(s["f_rows"].items()))
            + f" · lost {s['lost']} · hold {s['hold']} · no raw {s['no_raw']} · timed by tSrc {s['t_fallback']}",
            f"cuts {s['cut_rows']} rows · {s['cut_points']} stamps · mod {s['cut_mod']} voxels · "
            f"without tExp {s['cut_t_fallback']} · soft resets {s['soft_resets']} · "
            f"screen time missing {s['screen_fallback']} · drill size changes {s['drill_size_changes']}"]


if __name__ == "__main__":   # python -m spinearm_bridge.recording <file.jsonl>
    import sys

    _rec = load(sys.argv[1])
    print("\n".join(summary(_rec) + [f"WARN {w}" for w in _rec.warnings]))
