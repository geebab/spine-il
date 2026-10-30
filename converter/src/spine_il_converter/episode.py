"""Session → training episode (PLAN-0 §6.2–6.5): filters, 30 fps resampling, basal frame, features."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import schema
from .geometry import GeometryError, basal_frame, normalize, points_to_frame, rot6d
from .resample import interpolate_track, match_nearest, phase_locked_grid
from .recorder_v2 import Recording
from .session import Session


@dataclass(frozen=True)
class ConvertConfig:
    fps: int = schema.FPS
    max_gap_ms: float = 250.0
    align_tol_ms: float = 20.0
    max_lost_frac: float = 0.05
    min_frames: int = 30
    basal_parity_tol: float = 1e-6
    image_size: tuple[int, int] = (256, 256)  # (height, width)
    replay_infeasible: frozenset[str] = frozenset()
    allow_core: bool = False  # recorder v2: accept timingSpec 'v2-core' files (no tExp → tSrc time axis)
    allow_robot_tools: bool = False  # recorder v2: accept files whose tracker also ran robot marker blocks (endo2)
    task: str | None = None  # recorder v2: task text (default: hdr diagnosis/approach, else a generic one)
    max_masked_frac: float = 0.05  # recorder v2: per tool, masked frames inside an episode (any invalid-row reason)
    split_on_gap: bool = False  # recorder v2: cut at masked runs > max_gap_ms into pieces instead of excluding


@dataclass
class Episode:
    session_id: str
    source: Path
    task: str
    t0_src_ms: float
    state: np.ndarray  # (N, 19) float32
    action: np.ndarray  # (N, 19) float32 — absolute state of the next frame
    phase: np.ndarray  # (N, 5) one-hot
    exposure: np.ndarray  # (N, 2)
    progress: np.ndarray  # (N, 3)
    goal: np.ndarray  # (6,)
    tracker_time: np.ndarray  # (N, 1) seconds since the episode's first frame
    image_paths: dict[str, list[Path]]
    stats: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.state)


@dataclass
class EpisodeResult:
    session_id: str
    source: Path
    episode: Episode | None
    reason: str | None = None  # None ⇔ included
    detail: str = ""

    @property
    def included(self) -> bool:
        return self.episode is not None


def image_path(data_root: Path, session_id: str, cam: str, frame_idx: int) -> Path:
    base = data_root / "frames" / session_id / cam
    plain = base / f"{frame_idx}.jpg"
    return plain if plain.exists() else base / f"{frame_idx:06d}.jpg"


def build_episode(session: Session | Recording, data_root: Path, cfg: ConvertConfig = ConvertConfig()) -> EpisodeResult:
    if getattr(session, "format", None) == "rec2":  # real recorder v2 file (recorder_v2.Recording)
        from .episode_v2 import build_rec_episode

        return build_rec_episode(session, cfg)
    sid, src = session.session_id, session.path

    def reject(reason: str, detail: str = "") -> EpisodeResult:
        return EpisodeResult(sid, src, None, reason, detail)

    if session.errors:
        return reject("schema", str(session.errors[0]))
    hdr, ftr = session.header, session.footer
    if ftr is None:
        return reject("no_footer")
    if ftr.get("verdict") == "bad":
        return reject("verdict_bad", str(ftr.get("verdictReason", "")))
    if ftr.get("failFlags"):
        return reject("fail_flags", ",".join(map(str, ftr["failFlags"])))
    if sid in cfg.replay_infeasible:
        return reject("replay_infeasible")
    for tool in schema.TOOLS:
        if session.poses[tool].lost_frac > cfg.max_lost_frac:
            return reject("tracker_lost", f"{tool} lost {session.poses[tool].lost_frac:.1%}")

    m = session.metrics
    human_phases = [p for p in ftr.get("phases") or [] if {"phase", "tMonoStart", "tMonoEnd"} <= p.keys()]
    phase_source = "footer" if human_phases else "metrics"
    phases_present = {int(p["phase"]) for p in human_phases} if human_phases else set(m.phase.tolist())
    if schema.LAMINECTOMY_PHASE not in phases_present:
        return reject("no_phase3", f"phases from {phase_source}: {sorted(phases_present)}")

    try:
        origin, R_b = basal_frame(hdr["basal"]["entryWorld"], hdr["basal"]["targetWorld"])
    except (GeometryError, KeyError, TypeError) as e:
        return reject("basal", str(e))
    if (bf := hdr.get("basalFrame")) is not None:
        ref = np.column_stack([bf["xAxis"], bf["yAxis"], bf["zAxis"]])
        err = max(np.abs(ref - R_b).max(), np.abs(np.asarray(bf["origin"]) - origin).max())
        if err > cfg.basal_parity_tol:
            return reject("basal_parity", f"header basalFrame differs by {err:.3g}")

    areas = hdr.get("areas") or []
    if not areas:
        return reject("no_area")
    try:
        centroid = np.mean(np.asarray(areas[0]["worldPoints"], dtype=np.float64), axis=0)
        normal = normalize(areas[0]["planeNormal"])
    except (KeyError, ValueError, GeometryError) as e:
        return reject("area", str(e))
    goal = np.concatenate([points_to_frame(centroid, origin, R_b), normal @ R_b])

    endo, drill = session.poses["endo"], session.poses["drill"]
    t_min = max(endo.t_src[0], drill.t_src[0])
    t_max = min(endo.t_src[-1], drill.t_src[-1])
    grid = phase_locked_grid(session.images["endoscope"].t_src_endo, t_min, t_max, cfg.fps)
    if len(grid) < 2:
        return reject("too_short", "no overlap between tool streams and captures")

    rs = {tool: interpolate_track(session.poses[tool], grid, cfg.max_gap_ms) for tool in schema.TOOLS}
    # During a short tracker dropout the viewer keeps rendering the last pose, so captures carry a stale
    # tSrcEndo. Bridged frames therefore accept the nearest capture within max_gap_ms (counted as stale).
    bridged = rs["endo"].bridged | rs["drill"].bridged
    metric_idx, _ = _match(grid, m.t_src_endo, cfg, bridged)
    img_idx, stale = {}, np.zeros(len(grid), dtype=bool)
    for cam in schema.CAMS:
        img_idx[cam], cam_stale = _match(grid, session.images[cam].t_src_endo, cfg, bridged)
        stale |= cam_stale
    paths = {cam: [None] * len(grid) for cam in schema.CAMS}
    valid = rs["endo"].valid & rs["drill"].valid & (metric_idx >= 0)
    for cam in schema.CAMS:
        valid &= img_idx[cam] >= 0
        for k in np.flatnonzero(valid):
            p = image_path(data_root, sid, cam, int(session.images[cam].frame_idx[img_idx[cam][k]]))
            if p.exists():
                paths[cam][k] = p
            else:
                valid[k] = False

    good = np.flatnonzero(valid)
    if good.size == 0:
        return reject("no_valid_frames")
    lo, hi = good[0], good[-1] + 1
    if not valid[lo:hi].all():
        holes = np.flatnonzero(~valid[lo:hi])
        return reject(
            "gap",
            f"{holes.size} invalid frames inside the episode, first at +{(grid[lo + holes[0]] - grid[lo]) / 1000:.2f}s",
        )
    if hi - lo < cfg.min_frames + 1:
        return reject("too_short", f"{hi - lo} frames < {cfg.min_frames + 1}")
    sl = slice(lo, hi)
    grid = grid[sl]

    parts = []
    for tool in schema.TOOLS:
        tip_b = points_to_frame(rs[tool].tip[sl], origin, R_b)
        rot_b = np.einsum("ji,njk->nik", R_b, rs[tool].rot[sl])  # R_bᵀ · R_scene←tool
        parts += [tip_b, rot6d(rot_b)]
    state = np.concatenate(parts + [rs["drill"].on[sl][:, None]], axis=1)

    mi = metric_idx[sl]
    phase_ids = _phase_ids(m.t_mono[mi], human_phases) if human_phases else m.phase[mi]
    phase = np.eye(len(schema.PHASE_NAMES))[np.clip(phase_ids, 0, len(schema.PHASE_NAMES) - 1)]
    progress = np.stack(
        [m.removed_mm3[mi] / 1000.0, m.drill_tip_in_view[mi], m.drill_tip_to_bone_mm[mi] / 100.0], axis=1
    )
    img_h, img_w = cfg.image_size
    for cam in hdr["capture"].get("cams", []):
        if cam.get("name") in schema.CAMS and (cam.get("h"), cam.get("w")) != (img_h, img_w):
            return reject("image_size", f"{cam['name']} is {cam.get('w')}x{cam.get('h')}, expected {img_w}x{img_h}")

    f32 = lambda a: np.ascontiguousarray(a, dtype=np.float32)  # noqa: E731
    n = len(grid) - 1  # the last grid frame only serves as the action target of the one before it
    episode = Episode(
        session_id=sid,
        source=src,
        task=f"{hdr['diagnosis']['text']}: {hdr['approach']['text']}",
        t0_src_ms=float(grid[0]),
        state=f32(state[:n]),
        action=f32(state[1 : n + 1]),
        phase=f32(phase[:n]),
        exposure=f32(m.exposure[mi][:n]),
        progress=f32(progress[:n]),
        goal=f32(goal),
        tracker_time=f32(((grid[:n] - grid[0]) / 1000.0)[:, None]),
        image_paths={cam: [paths[cam][k] for k in range(lo, lo + n)] for cam in schema.CAMS},
        stats={
            "n_frames": n,
            "duration_s": round((grid[n] - grid[0]) / 1000.0, 3),
            "phase_source": phase_source,
            "recovery": bool(hdr.get("recovery") or ftr.get("recovery")),
            **{f"{t}_lost_frac": round(session.poses[t].lost_frac, 4) for t in schema.TOOLS},
            **{f"{t}_bridged_frames": int(rs[t].bridged[sl][:n].sum()) for t in schema.TOOLS},
            "stale_capture_frames": int(stale[sl][:n].sum()),
            **{f"{t}_rows_without_up": session.poses[t].n_up_missing for t in schema.TOOLS},
        },
    )
    return EpisodeResult(sid, src, episode)


def _match(grid: np.ndarray, ref: np.ndarray, cfg: ConvertConfig, bridged: np.ndarray):
    strict = match_nearest(grid, ref, cfg.align_tol_ms)
    relaxed = match_nearest(grid, ref, cfg.max_gap_ms)
    use_relaxed = (strict < 0) & bridged & (relaxed >= 0)
    return np.where(use_relaxed, relaxed, strict), use_relaxed


def _phase_ids(t_mono: np.ndarray, phases: list[dict]) -> np.ndarray:
    """Phase per frame from footer intervals [tMonoStart, tMonoEnd); frames outside inherit the nearest."""
    starts = np.array([p["tMonoStart"] for p in phases], dtype=np.float64)
    order = np.argsort(starts)
    starts = starts[order]
    ids = np.array([int(phases[i]["phase"]) for i in order])
    pos = np.clip(np.searchsorted(starts, t_mono, side="right") - 1, 0, len(ids) - 1)
    return ids[pos]
