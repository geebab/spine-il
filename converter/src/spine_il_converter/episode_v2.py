"""Recorder v2 session → training episode(s) on the camera-frame (`tExp`) time axis.

Frames sit on a 30 fps grid anchored on the first camera frame where both tools are valid. Per frame k (t_k):

- `observation.state` / `action` (19): endo tip(3) + rot6d(6), drill tip(3) + rot6d(6), `drill_trig`, in the basal
  frame. action = the next frame's state (so it is valid when mask[k + 1] is).
- `observation.endo_roll_deg`: the viewer's endoscope roll for the pose at t_k, unwrapped within the episode.
- `label.drill_cut`: 1 when the drill engine stamped with a drill pose from a camera frame in (t_{k−1}, t_k]. A label
  on pose time: the engine applies a stamp later (0551: p50 ~120 ms), so a policy at t_k could not have seen it.
- `mask` (endo, drill, drill_trig_known): a pose is valid at t_k when t_k is a valid row's frame or lies between two
  consecutive valid rows ≤ max_gap_ms apart. Otherwise the frame holds the last valid sample and its mask is 0 —
  lost / hold / no-scene / bad / vis<4 rows are never bridged by interpolation.
- `diag.tsrc_minus_texp_ms` (endo, drill): tracker send time − camera frame of the latest row (diagnostic only).

A masked run longer than max_gap_ms excludes the recording, or with `split_on_gap` cuts it into pieces (never
bridged) that are filtered on their own and share one split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import numpy as np

from .episode import ConvertConfig, EpisodeResult
from .features import STATE_NAMES
from .geometry import GeometryError, basal_frame, matrix_to_quat, normalize, points_to_frame, quat_to_matrix, rot6d, slerp
from .recorder_v2 import FORMAT_REC2, Recording, ToolTrack, wrap_deg

# state[18]: the drill trigger (ZR) — what an arm can command. Actual stamping is the `label.drill_cut` column.
REC_STATE_NAMES = STATE_NAMES[:-1] + ["drill_trig"]
MASK_NAMES = ("endo", "drill", "drill_trig_known")
DEFAULT_TASK = "biportal endoscopic spine surgery with a tracked endoscope and drill"
# Basal origin = the endoscope's skin portal: the earliest BASAL_WINDOW_MS window, while the scope is inserted
# (depth > 0), holding ≥ BASAL_MIN_VALUES distinct recorded entry values that all lie within BASAL_RADIUS_MM of their
# median (the viewer re-issues its entry only past FILM_ENTRY_UPDATE_MM = 2 mm). Counting distinct values makes a
# screen that did not redraw, or an entry re-used after a failed skin cast, add no evidence.
BASAL_WINDOW_MS = 1000.0
BASAL_RADIUS_MM = 2.0
BASAL_MIN_VALUES = 8
PORTAL_SPREAD_WARN_MM = 10.0  # chosen portal vs the median of all inserted entries: more means a re-insertion elsewhere
SPLIT_MIN_S = 3.0  # --split-on-gap: shorter pieces are dropped
LONG_INTERP_FACTOR = 2.5  # interpolation across > 2.5 × the median row spacing is counted (tracker skipped frames)


@dataclass
class RecEpisode:
    session_id: str  # the recording's sessionId (pieces of one recording share it — and their split)
    source: Path
    task: str
    t0_ms: float  # epoch ms of frame 0 on the time axis
    time_axis: str  # "tExp" (or "tSrc" for an allowed v2-core file)
    state: np.ndarray  # (N, 19) float32
    action: np.ndarray  # (N, 19) float32 — absolute state of the next frame
    endo_roll: np.ndarray  # (N, 1) deg, unwrapped within the episode
    drill_cut: np.ndarray  # (N, 1) 0/1 label on pose time
    tracker_time: np.ndarray  # (N, 1) s since frame 0
    mask: np.ndarray  # (N, 3) 0/1
    t_src_lag: np.ndarray  # (N, 2) ms
    header_line: str  # verbatim hdr row
    cut_lines: list[str]  # verbatim cut rows, r order
    piece: int | None = None  # index among the recording's pieces (--split-on-gap), else None
    stats: dict = field(default_factory=dict)
    format: ClassVar[str] = FORMAT_REC2

    @property
    def n_frames(self) -> int:
        return len(self.state)


@dataclass
class RecPieces:
    """--split-on-gap: the pieces of one recording that passed the filters, in time order."""

    session_id: str
    source: Path
    pieces: list[RecEpisode]
    stats: dict
    format: ClassVar[str] = FORMAT_REC2

    @property
    def n_frames(self) -> int:
        return sum(p.n_frames for p in self.pieces)


def episodes_of(ep: RecEpisode | RecPieces) -> list[RecEpisode]:
    return ep.pieces if isinstance(ep, RecPieces) else [ep]


class _Reject(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(reason)
        self.reason, self.detail = reason, detail


@dataclass
class _Sampled:
    tip: np.ndarray
    rot: np.ndarray
    roll: np.ndarray
    valid: np.ndarray
    long_interp: np.ndarray


@dataclass
class _Session:
    """Per-recording values every piece reads."""

    grid: np.ndarray
    step: float
    endo: _Sampled
    drill: _Sampled
    trig: np.ndarray
    known: np.ndarray
    cut: np.ndarray  # drill_cut on the full grid
    cut_t: np.ndarray
    cut_source: str
    cut_unplaced: int
    lag: np.ndarray
    origin: np.ndarray
    R_b: np.ndarray
    basal: dict
    warnings: list[str]


def build_rec_episode(rec: Recording, cfg: ConvertConfig = ConvertConfig()) -> EpisodeResult:
    try:
        return EpisodeResult(rec.session_id, rec.path, _build(rec, cfg))
    except _Reject as r:
        return EpisodeResult(rec.session_id, rec.path, None, r.reason, r.detail)


def _build(rec: Recording, cfg: ConvertConfig) -> RecEpisode | RecPieces:
    ftr = rec.footer
    if rec.errors:
        raise _Reject("schema", str(rec.errors[0]))
    if rec.is_core and not cfg.allow_core:
        raise _Reject("timing_core", f"timingSpec {rec.timing_spec!r}, time axis {rec.time_axis}")
    if rec.robot_tools and not cfg.allow_robot_tools:
        raise _Reject("robot_tools", f"the tracker ran robot-mounted marker blocks {rec.robot_tools}")
    if ftr is None:
        raise _Reject("no_footer")
    if ftr.get("operatorVerdict") == "bad":
        raise _Reject("verdict_bad", str(ftr.get("verdictReason") or ""))
    if rec.session_id in cfg.replay_infeasible:
        raise _Reject("replay_infeasible")
    endo, drill = rec.tools["endo"], rec.tools["drill"]
    warnings = [i.message for i in rec.warnings]
    for tr in (endo, drill):
        if tr.n_rows == 0:
            raise _Reject("no_rows", f"no {tr.tool} f rows")
        if not tr.valid.any():
            raise _Reject("no_valid_frames", f"{tr.tool} has no valid pose row")
        if tr.lost_frac > cfg.max_lost_frac:  # rig health; frames are gated by the masked fraction inside the episode
            warnings.append(f"{tr.tool}: {tr.lost_frac:.1%} of the recording's rows are lost (> {cfg.max_lost_frac:.0%})")
    origin, R_b, basal = _basal(rec)
    if basal["rule"] != "inserted_stable_window":
        warnings.append(f"basal origin from {basal['source']}/{basal['rule']}: no stable inserted entry window")
    if basal.get("inserted_median_dist_mm", 0.0) > PORTAL_SPREAD_WARN_MM:
        warnings.append(f"basal portal is {basal['inserted_median_dist_mm']} mm from the median inserted entry: "
                        "the scope was inserted at more than one place")

    step = 1000.0 / cfg.fps
    t_lo = max(tr.t[tr.valid][0] for tr in (endo, drill))
    t_hi = min(tr.t[tr.valid][-1] for tr in (endo, drill))
    if t_hi - t_lo < step:
        raise _Reject("too_short", "the two tool streams do not overlap")
    grid = t_lo + step * np.arange(int(np.floor((t_hi - t_lo) / step + 1e-9)) + 1)
    se, sd = _sample(endo, grid, cfg.max_gap_ms), _sample(drill, grid, cfg.max_gap_ms)
    both = se.valid & sd.valid
    good = np.flatnonzero(both)
    if good.size == 0:
        raise _Reject("no_valid_frames", "no frame with both tools valid")
    segments, longest = _segments(both, good[0], good[-1] + 1, step, cfg.max_gap_ms)
    if len(segments) > 1 and not cfg.split_on_gap:
        raise _Reject("gap", f"{longest} consecutive masked frames ({(longest + 1) * step:.0f} ms between valid frames)")
    cut_t, cut_source, cut_unplaced = _cut_times(rec)
    if cut_unplaced:
        warnings.append(f"{cut_unplaced} cut rows without {rec.time_axis} are missing from drill_cut")
    sess = _Session(
        grid=grid, step=step, endo=se, drill=sd,
        trig=_hold(drill.t, drill.trig, grid, 0.0),
        known=_hold(drill.t, drill.trig_known.astype(np.float64), grid, 0.0),
        cut=_interval_hits(cut_t, grid, step), cut_t=cut_t, cut_source=cut_source, cut_unplaced=cut_unplaced,
        lag=np.stack([_hold(tr.t, tr.t_src - tr.t, grid, np.nan) for tr in (endo, drill)], axis=1),
        origin=origin, R_b=R_b, basal=basal, warnings=warnings,
    )
    pieces, dropped = [], []
    for k, (a, b) in enumerate(segments):
        try:
            pieces.append(_piece(rec, cfg, sess, a, b, k if cfg.split_on_gap else None))
        except _Reject as r:
            if not cfg.split_on_gap:
                raise
            dropped.append({"piece": k, "reason": r.reason, "detail": r.detail,
                            "s": [round((grid[a] - grid[0]) / 1000.0, 3), round((grid[b - 1] - grid[0]) / 1000.0, 3)]})
    if not pieces:
        raise _Reject("no_piece", "; ".join(f"#{d['piece']} {d['reason']}: {d['detail']}" for d in dropped))
    _warn_work_outside(rec, sess, pieces)
    if not cfg.split_on_gap:
        return pieces[0]
    for p in pieces:
        p.stats.update(n_pieces=len(pieces), dropped_pieces=dropped)
    summary = {"format": FORMAT_REC2, "pieces": [{"piece": p.piece, "n_frames": p.n_frames, "t0_ms": p.t0_ms,
                                                   "duration_s": p.stats["duration_s"]} for p in pieces],
               "dropped_pieces": dropped, "n_frames": sum(p.n_frames for p in pieces), "warnings": pieces[0].stats["warnings"]}
    return RecPieces(rec.session_id, rec.path, pieces, summary)


def _piece(rec: Recording, cfg: ConvertConfig, sess: _Session, a: int, b: int, piece: int | None) -> RecEpisode:
    g = sess.grid[a:b]
    n = len(g) - 1  # the last grid frame only serves as the action target of the one before it
    if n < cfg.min_frames:
        raise _Reject("too_short", f"{n + 1} frames < {cfg.min_frames + 1}")
    if piece is not None and g[n] - g[0] < SPLIT_MIN_S * 1000.0:
        raise _Reject("too_short", f"{(g[n] - g[0]) / 1000.0:.2f} s < {SPLIT_MIN_S:g} s")
    frames = slice(a, a + n)
    masked = {name: int((~s.valid[frames]).sum()) for name, s in (("endo", sess.endo), ("drill", sess.drill))}
    for name, count in masked.items():
        if count / n > cfg.max_masked_frac:
            raise _Reject("masked", f"{name} masked in {count}/{n} frames ({count / n:.1%} > {cfg.max_masked_frac:.0%})")
    sl = slice(a, b)
    parts = []
    for s in (sess.endo, sess.drill):
        parts += [points_to_frame(s.tip[sl], sess.origin, sess.R_b), rot6d(np.einsum("ji,njk->nik", sess.R_b, s.rot[sl]))]
    state = np.concatenate(parts + [sess.trig[sl][:, None]], axis=1)
    roll = np.degrees(np.unwrap(np.radians(sess.endo.roll[frames])))
    if not (np.isfinite(state).all() and np.isfinite(roll).all()):
        raise _Reject("nonfinite", "non-finite pose or roll")
    mask = np.stack([sess.endo.valid[frames], sess.drill.valid[frames], sess.known[frames] > 0.5], axis=1)

    hdr, ftr, endo, drill = rec.header, rec.footer, rec.tools["endo"], rec.tools["drill"]
    in_piece = (sess.cut_t > g[0] - sess.step) & (sess.cut_t <= g[n - 1])
    f32 = lambda x: np.ascontiguousarray(x, dtype=np.float32)  # noqa: E731
    stats = {
        "format": FORMAT_REC2,
        "time_axis": rec.time_axis,
        "timing_spec": rec.timing_spec,
        "recorder_version": hdr.get("recorderVersion"),
        "piece": piece,
        "n_frames": n,
        "duration_s": round((g[n] - g[0]) / 1000.0, 3),
        "start_s": round((g[0] - sess.grid[0]) / 1000.0, 3),
        "operator_verdict": ftr.get("operatorVerdict"),
        "end_reason": ftr.get("endReason"),
        **{f"{t.tool}_rows": t.n_rows for t in (endo, drill)},
        **{f"{t.tool}_lost_frac": round(t.lost_frac, 4) for t in (endo, drill)},
        **{f"{t.tool}_invalid_rows": {k: v for k, v in t.counts.items() if k != "rows" and v} for t in (endo, drill)},
        **{f"{name}_masked_frames": count for name, count in masked.items()},
        **{f"{name}_masked_frac": round(count / n, 4) for name, count in masked.items()},
        **{f"{name}_long_interp_frames": int(s.long_interp[frames].sum()) for name, s in (("endo", sess.endo), ("drill", sess.drill))},
        "endo_roll_range_deg": [round(float(roll.min()), 3), round(float(roll.max()), 3)],
        "endo_roll_viewer_frac": round(float(endo.roll_viewer[endo.valid].mean()), 4),
        "endo_roll_rule_err_max_deg": _nanmax(endo.roll_rule_err),
        "drill_trig_frac": round(float(sess.trig[frames].mean()), 4),
        "drill_trig_known_frac": round(float(sess.known[frames].mean()), 4),
        "drill_cut_frac": round(float(sess.cut[frames].mean()), 4),
        "drill_cut_source": sess.cut_source,
        "cut_rows": len(rec.cuts),
        "cut_rows_in_episode": int(in_piece.sum()),
        "cut_rows_unplaced": sess.cut_unplaced,
        "cut_mod_sum": int(sum(row.get("mod", 0) for _, row in rec.cuts if isinstance(row.get("mod"), int))),
        **{f"{t.tool}_tsrc_minus_texp_ms_p50": _nanmedian(t.t_src - t.t) for t in (endo, drill)},
        **{f"{t.tool}_normal_shaft_deg": round(t.normal_shaft_deg, 2) for t in (endo, drill)},
        "basal": {**sess.basal, "origin": _r(sess.origin), "x": _r(sess.R_b[:, 0], 6), "y": _r(sess.R_b[:, 1], 6),
                  "z": _r(sess.R_b[:, 2], 6)},
        "identity": {
            "mesh_id": hdr.get("meshId"),
            "case_group_key": hdr.get("caseGroupKey"),
            "tool_models": rec.tool_models,
            "tracker_cfg_hash": _dict(hdr.get("tracker")).get("cfgHash"),
            "fe_cfg_hash": _dict(hdr.get("fe")).get("cfgHash"),
            "build_git_sha": _dict(hdr.get("build")).get("gitSha"),
        },
        "warnings": list(sess.warnings),
        "parse": rec.stats,
    }
    return RecEpisode(
        session_id=rec.session_id,
        source=rec.path,
        task=_task(hdr, cfg),
        t0_ms=float(g[0]),
        time_axis=rec.time_axis,
        state=f32(state[:n]),
        action=f32(state[1 : n + 1]),
        endo_roll=f32(roll[:, None]),
        drill_cut=f32(sess.cut[frames][:, None]),
        tracker_time=f32(((g[:n] - g[0]) / 1000.0)[:, None]),
        mask=f32(mask),
        t_src_lag=f32(sess.lag[frames]),
        header_line=rec.header_line,
        cut_lines=[line for line, _ in rec.cuts],
        piece=piece,
        stats=stats,
    )


def _warn_work_outside(rec: Recording, sess: _Session, pieces: list[RecEpisode]) -> None:
    """Stamps or ZR presses outside the converted frames mean the episode(s) miss part of the work."""
    covered = [(p.t0_ms - sess.step, p.t0_ms + (p.n_frames - 1) * sess.step) for p in pieces]
    inside = lambda t: np.any([(t > lo) & (t <= hi) for lo, hi in covered], axis=0)  # noqa: E731
    drill = rec.tools["drill"]
    pressed = drill.t[(drill.trig > 0) & drill.trig_known]
    notes = []
    if (n_cut := int((~inside(sess.cut_t)).sum()) if sess.cut_t.size else 0):
        notes.append(f"{n_cut} drill stamps ({sess.cut_source}) fall outside the converted frames")
    if (n_zr := int((~inside(pressed)).sum()) if pressed.size else 0):
        notes.append(f"{n_zr} drill rows with ZR held fall outside the converted frames")
    for p in pieces:
        p.stats["warnings"] += notes


def _segments(both: np.ndarray, lo: int, hi: int, step: float, max_gap_ms: float) -> tuple[list[tuple[int, int]], int]:
    """[a, b) frame ranges of the valid span, cut at masked runs spanning more than max_gap_ms between valid frames.
    Every range starts and ends on a frame where both tools are valid. Also returns the longest masked run."""
    out, start, last_valid, run, longest = [], lo, lo, 0, 0
    for k in range(lo, hi):
        if both[k]:
            if run and (run + 1) * step > max_gap_ms:
                out.append((start, last_valid + 1))
                start = k
            run, last_valid = 0, k
        else:
            run += 1
            longest = max(longest, run)
    out.append((start, last_valid + 1))
    return out, longest


def _sample(tr: ToolTrack, grid: np.ndarray, max_gap_ms: float) -> _Sampled:
    """Pose at each grid time: linear tip / slerp rotation / linear roll between consecutive valid rows, else held."""
    t, ok = tr.t, tr.valid
    n = len(t)
    i0 = np.searchsorted(t, grid, side="right") - 1
    a0, a1 = np.clip(i0, 0, n - 1), np.clip(i0 + 1, 0, n - 1)
    inside = i0 >= 0
    exact = inside & (t[a0] == grid) & ok[a0]
    dt = t[a1] - t[a0]
    interp = inside & ~exact & (i0 + 1 < n) & ok[a0] & ok[a1] & (dt <= max_gap_ms)
    valid = exact | interp
    last_valid = np.maximum.accumulate(np.where(ok, np.arange(n), -1))
    held = np.where(inside, last_valid[a0], -1)
    held = np.where(held >= 0, held, np.flatnonzero(ok)[0])
    a = np.where(valid, a0, held)
    b = np.where(interp, a1, a)
    w = np.where(interp, (grid - t[a]) / np.where(interp, dt, 1.0), 0.0)
    tip = tr.tip[a] + w[:, None] * (tr.tip[b] - tr.tip[a])
    rot = quat_to_matrix(slerp(matrix_to_quat(tr.rot[a]), matrix_to_quat(tr.rot[b]), w))
    roll = wrap_deg(tr.roll[a] + w * wrap_deg(tr.roll[b] - tr.roll[a]))
    spacing = np.median(np.diff(t)) if n > 2 else np.inf
    return _Sampled(tip=tip, rot=rot, roll=roll, valid=valid, long_interp=interp & (dt > LONG_INTERP_FACTOR * spacing))


def _hold(t: np.ndarray, values: np.ndarray, grid: np.ndarray, default: float) -> np.ndarray:
    """Zero-order hold: the value of the latest row at or before each grid time."""
    if len(t) == 0:
        return np.full(len(grid), default)
    idx = np.searchsorted(t, grid, side="right") - 1
    return np.where(idx >= 0, values[np.clip(idx, 0, None)], default)


def _interval_hits(times: np.ndarray, grid: np.ndarray, step: float) -> np.ndarray:
    """1 for frames k whose interval (t_k − step, t_k] contains at least one of `times`."""
    out = np.zeros(len(grid))
    k = np.searchsorted(grid, times, side="left")
    keep = (k < len(grid)) & (times > grid[0] - step)
    out[k[keep]] = 1.0
    return out


def _cut_times(rec: Recording) -> tuple[np.ndarray, str, int]:
    """Stamp times on the session axis — bone and soft-tissue stamps alike: `cut` rows when the recorder writes them
    (hdr.replay.cutRows, rec ≥ 2.3), else drill rows with proxy.stamped = 1 (placed at their drawn frame).
    Returns (times, source, cut rows without a time on the axis)."""
    if rec.cuts or _dict(rec.header.get("replay")).get("cutRows"):
        rows = [row for _, row in rec.cuts if row.get("op") != "reset"]
        times = [row[rec.time_axis] for row in rows if _num(row.get(rec.time_axis))]
        return np.sort(np.asarray(times, dtype=np.float64)), "cut_rows", len(rows) - len(times)
    stamped = rec.tools["drill"].stamped_t
    return stamped, ("proxy_stamped" if stamped.size else "none"), 0


def _basal(rec: Recording) -> tuple[np.ndarray, np.ndarray, dict]:
    """Inserted stable entry window → median of the inserted entries → hdr.basal → reject."""
    found = _portal(rec.tools["endo"])
    if found is not None:
        origin, z, info = found
        target = origin + z
    else:
        b = _dict(rec.header.get("basal"))  # PLAN-0 header field
        if not ({"entryWorld", "targetWorld"} <= b.keys()):
            raise _Reject("basal", "no inserted viewer entry point (drawn/real with depth > 0) and no hdr.basal")
        origin, target, info = b["entryWorld"], b["targetWorld"], {"source": "header", "rule": "header"}
    try:
        origin, R_b = basal_frame(origin, target)
    except (GeometryError, ValueError, TypeError) as e:
        raise _Reject("basal", str(e)) from None
    return origin, R_b, info


def _portal(endo: ToolTrack):
    """The skin portal from the viewer's entry points while the scope is inserted, or None without such points."""
    inserted = endo.entry_depth > 0
    if not inserted.any():
        return None
    t, entry, view = endo.entry_t[inserted], endo.entry[inserted], endo.entry_dir[inserted]
    _, first = np.unique(entry, axis=0, return_index=True)  # each distinct recorded value once, at its first row
    keep = np.sort(first)
    t, entry, view = t[keep], entry[keep], view[keep]
    t_first = float(endo.t[0]) if len(endo.t) else float(t[0])
    everywhere = np.median(entry, axis=0)
    spread = lambda origin: round(float(np.linalg.norm(origin - everywhere)), 2)  # noqa: E731
    j = 0
    for i in range(len(t)):
        j = max(j, i)
        while j + 1 < len(t) and t[j + 1] <= t[i] + BASAL_WINDOW_MS:
            j += 1
        if j - i + 1 < BASAL_MIN_VALUES or t[j] - t[i] < 0.9 * BASAL_WINDOW_MS:
            continue
        med = np.median(entry[i : j + 1], axis=0)
        if np.linalg.norm(entry[i : j + 1] - med, axis=1).max() <= BASAL_RADIUS_MM:
            info = {"source": endo.entry_src, "rule": "inserted_stable_window", "values": j - i + 1,
                    "window_s": [round(float(t[i] - t_first) / 1000.0, 3), round(float(t[j] - t_first) / 1000.0, 3)],
                    "inserted_median_dist_mm": spread(med)}
            return med, normalize(np.median(view[i : j + 1], axis=0)), info
    info = {"source": endo.entry_src, "rule": "inserted_median", "values": len(t), "inserted_median_dist_mm": 0.0}
    return everywhere, normalize(np.median(view, axis=0)), info


def _task(hdr: dict, cfg: ConvertConfig) -> str:
    if cfg.task:
        return cfg.task
    diagnosis, approach = _dict(hdr.get("diagnosis")).get("text"), _dict(hdr.get("approach")).get("text")
    return f"{diagnosis}: {approach}" if diagnosis and approach else DEFAULT_TASK


def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and bool(np.isfinite(x))


def _nanmax(a: np.ndarray) -> float | None:
    return round(float(np.nanmax(a)), 6) if np.isfinite(a).any() else None


def _nanmedian(a: np.ndarray) -> float | None:
    return round(float(np.nanmedian(a)), 3) if np.isfinite(a).any() else None


def _r(v, nd: int = 3) -> list[float]:
    return [round(float(x), nd) for x in v]
