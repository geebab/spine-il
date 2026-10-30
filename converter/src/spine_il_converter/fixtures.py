"""Synthetic recorder-v2 sessions with analytic ground truth (PLAN-0 §6.6).

Tool tips move on straight lines at constant velocity and roll about their fixed shaft axis at a
constant rate, so linear/slerp resampling is exact and converted states can be compared with
`tool_truth` to float precision. Tracker timestamps jitter, captures lag the tracker, and gaps /
lost rows / bad verdicts etc. are injectable to exercise every filter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from . import schema
from .geometry import SUPERIOR, basal_frame, matrix_to_quat, normalize, tool_rotation

T0_SRC_MS = 1_757_900_000_000.0  # realistic epoch-ms magnitude (float32 would lose it)
MONO_OFFSET_MS = 81_234.5
PHASE_FRACTIONS = (0.0, 0.10, 0.30, 0.45, 0.85)  # start of phases 0..4 as a fraction of the duration
HUMAN_PHASE_SHIFT_S = 0.05  # footer (human-confirmed) boundaries differ slightly from auto ones
_R_CAM_SCENE = np.diag([1.0, -1.0, -1.0])  # camera looks down at the table
_T_CAM_SCENE = np.array([0.0, 0.0, 600.0])


@dataclass(frozen=True)
class FixtureSpec:
    session_id: str
    date: str = "2026-09-15"
    duration_s: float = 3.0
    tracker_hz: float = 90.0
    capture_hz: float = 30.0
    seed: int = 0
    gaps: tuple[tuple[float, float], ...] = ()  # (start_s, length_ms): both tools unseen
    lost_frac: float = 0.0
    verdict: str = "good"
    fail_flags: tuple[str, ...] = ()
    with_phase3: bool = True
    with_footer: bool = True
    schema_version: int = schema.SCHEMA_VERSION
    corrupt_basal_frame: bool = False
    recovery: bool = False
    omit_up: bool = False
    entry: tuple[float, float, float] = (45.0, 95.0, 160.0)
    target: tuple[float, float, float] = (38.0, 60.0, 165.0)
    image_hw: tuple[int, int] = (256, 256)
    expected_reason: str | None = field(default=None, compare=False)


def tool_truth(spec: FixtureSpec, tool: str, t_src_ms) -> tuple[np.ndarray, np.ndarray]:
    """Scene-frame tip (N, 3) and rotation (N, 3, 3) of `tool` at tracker times (ms)."""
    t = (np.atleast_1d(np.asarray(t_src_ms, dtype=np.float64)) - T0_SRC_MS) / 1000.0
    entry, target = np.asarray(spec.entry), np.asarray(spec.target)
    if tool == "endo":
        home, goal, roll_rate = entry + [12.0, 55.0, 25.0], target + [0.0, 8.0, 4.0], np.deg2rad(20.0)
    else:
        home, goal, roll_rate = entry + [-14.0, 60.0, -20.0], target + [0.0, 10.0, -4.0], np.deg2rad(-35.0)
    tip = home + np.outer(t / spec.duration_s, goal - home)
    R0 = tool_rotation(normalize(goal - home), SUPERIOR)
    a = roll_rate * t
    c, s = np.cos(a), np.sin(a)
    Rz = np.zeros((len(t), 3, 3))
    Rz[:, 0, 0], Rz[:, 0, 1], Rz[:, 1, 0], Rz[:, 1, 1], Rz[:, 2, 2] = c, -s, s, c, 1.0
    return tip, R0 @ Rz


def auto_phase(spec: FixtureSpec, t_src_ms) -> np.ndarray:
    frac = (np.atleast_1d(t_src_ms) - T0_SRC_MS) / 1000.0 / spec.duration_s
    phase = np.searchsorted(PHASE_FRACTIONS, frac, side="right") - 1
    phase = np.clip(phase, 0, 4)
    return phase if spec.with_phase3 else np.where(phase == 3, 2, phase)


def image_value(cam: str, frame_idx: int) -> int:
    """Flat gray level encoding the capture index — survives lossy video round trips within a few units."""
    v = 40 + (frame_idx * 13) % 170
    return v if cam == "endoscope" else 255 - v


def write_session(data_root: Path, spec: FixtureSpec) -> Path:
    rng = np.random.default_rng(spec.seed)
    data_root = Path(data_root)
    out = data_root / "sessions" / spec.date / f"{spec.session_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []  # body rows, sorted by tMono before writing

    # --- tracker rows (both tools share the tracker clock) ---
    n_ticks = int(spec.duration_s * spec.tracker_hz) + 1
    t_ticks = T0_SRC_MS + np.arange(n_ticks) * 1000.0 / spec.tracker_hz + rng.uniform(-0.5, 0.5, n_ticks)
    t_ticks[0] = T0_SRC_MS
    rel_s = (t_ticks - T0_SRC_MS) / 1000.0
    seen = np.ones(n_ticks, dtype=bool)
    for start_s, length_ms in spec.gaps:
        seen &= ~((rel_s >= start_s) & (rel_s < start_s + length_ms / 1000.0))
    valid_times = {}
    seq = 0
    for tool in schema.TOOLS:
        tip, R = tool_truth(spec, tool, t_ticks)
        lost = rng.random(n_ticks) < spec.lost_frac
        lost[0] = lost[-1] = False
        on = (auto_phase(spec, t_ticks) == 3).astype(int)
        valid_times[tool] = t_ticks[seen & ~lost]
        for i in np.flatnonzero(seen):
            seq += 1
            base = {"k": "f", "i": seq, "seq": seq, "tool": tool, "tSrc": float(t_ticks[i]),
                    "tMono": float(t_ticks[i] - T0_SRC_MS + MONO_OFFSET_MS + 3.0)}
            if lost[i]:
                rows.append({**base, "lost": 1})
                continue
            r_cam = _R_CAM_SCENE @ R[i]
            scene = {"tip": tip[i].tolist(), "dir": R[i][:, 2].tolist()}
            if not spec.omit_up:
                scene["up"] = R[i][:, 0].tolist()
            row = {**base, "pos": (_R_CAM_SCENE @ tip[i] + _T_CAM_SCENE).tolist(),
                   "quat": matrix_to_quat(r_cam).tolist(), "tip": (_R_CAM_SCENE @ tip[i] + _T_CAM_SCENE).tolist(),
                   "rmse": 0.3, "vis": 4, "split": 0, "still": 0, "scene": scene}
            if tool == "endo":
                row["endo"] = {"viewDir": scene["dir"], "depth": float(np.linalg.norm(tip[i] - spec.entry)), "roll": 0.0}
            else:
                row["drill"] = {"B": tip[i].tolist(), "depth": 0.0, "on": int(on[i]), "stick": [0.0, 0.0], "buttons": {}}
            rows.append(row)

    # --- viewer ticks: one m row + one img row per camera at the capture rate ---
    phase_start_s = np.array(PHASE_FRACTIONS) * spec.duration_s
    t3, t4 = phase_start_s[3], phase_start_s[4]
    n_caps = int(spec.duration_s * spec.capture_hz)
    for j in range(n_caps):
        render = T0_SRC_MS + 5.0 + j * 1000.0 / spec.capture_hz + rng.uniform(-2.0, 2.0)
        t_endo = _latest(valid_times["endo"], render)
        t_drill = _latest(valid_times["drill"], render)
        if t_endo is None or t_drill is None:
            continue
        t_mono = render - T0_SRC_MS + MONO_OFFSET_MS + 8.0
        ts = (t_endo - T0_SRC_MS) / 1000.0
        u = float(np.clip((ts - t3) / (t4 - t3), 0.0, 1.0)) if spec.with_phase3 else 0.0
        to_bone = 30.0 * (1 - ts / t3) if ts < t3 else (0.0 if ts < t4 else 30.0 * (ts - t4) / (spec.duration_s - t4))
        common = {"tMono": t_mono, "tSrcEndo": t_endo, "tSrcDrill": t_drill}
        rows.append({"k": "m", **common, "phase": int(auto_phase(spec, t_endo)[0]),
                     "exposureEndo": 0.9 * u, "exposureWork": 0.85 * u, "removedMm3": 250.0 * u,
                     "drillTipInView": 1, "drillTipToBoneMm": max(to_bone, 0.0),
                     "endoTipToTargetMm": 0.0, "endoDepth": 0.0, "drillDepth": 0.0})
        for cam in schema.CAMS:
            path = data_root / "frames" / spec.session_id / cam / f"{j}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            h, w = spec.image_hw
            Image.fromarray(np.full((h, w, 3), image_value(cam, j), dtype=np.uint8)).save(path, quality=90)
            rows.append({"k": "img", "cam": cam, "frameIdx": j, **common, "jpgBytes": path.stat().st_size})

    rows.sort(key=lambda row: row["tMono"])
    rows = [_header(spec), *rows, *([_footer(spec, phase_start_s)] if spec.with_footer else [])]
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return out


def standard_specs(seed: int = 0) -> list[FixtureSpec]:
    """Three includable sessions plus one session per rejection reason."""
    def spec(sid: str, i: int, **kw) -> FixtureSpec:
        d = np.array([i * 1.5, -i * 0.8, i * 1.1])
        return FixtureSpec(session_id=sid, seed=seed + i, entry=tuple(np.add((45.0, 95.0, 160.0), d)),
                           target=tuple(np.add((38.0, 60.0, 165.0), d * 0.5)), **kw)

    return [
        spec("s_good_0", 0),
        spec("s_good_recovery", 1, recovery=True),
        spec("s_good_bridged", 2, gaps=((1.5, 150.0),), lost_frac=0.02),
        spec("s_bad_verdict", 3, verdict="bad", expected_reason="verdict_bad"),
        spec("s_fail_flags", 4, fail_flags=("out_of_view_drilling",), expected_reason="fail_flags"),
        spec("s_no_phase3", 5, with_phase3=False, expected_reason="no_phase3"),
        spec("s_tracker_lost", 6, lost_frac=0.12, expected_reason="tracker_lost"),
        spec("s_long_gap", 7, gaps=((1.5, 400.0),), expected_reason="gap"),
        spec("s_basal_parity", 8, corrupt_basal_frame=True, expected_reason="basal_parity"),
        spec("s_no_footer", 9, with_footer=False, expected_reason="no_footer"),
        spec("s_schema_v1", 10, schema_version=1, expected_reason="schema"),
    ]


def write_standard_set(data_root: Path, seed: int = 0) -> list[tuple[FixtureSpec, Path]]:
    return [(s, write_session(data_root, s)) for s in standard_specs(seed)]


def _header(spec: FixtureSpec) -> dict:
    origin, R = basal_frame(spec.entry, spec.target)
    if spec.corrupt_basal_frame:
        R = R.copy()
        R[:, 0] = normalize(R[:, 0] + [0.0, 0.0, 1e-3])
    entry, target = np.asarray(spec.entry), np.asarray(spec.target)
    center = target + [0.0, -6.0, 0.0]
    square = [center + d for d in ([5, 0, 5], [-5, 0, 5], [-5, 0, -5], [5, 0, -5])]
    h, w = spec.image_hw
    return {
        "k": "hdr", "schemaVersion": spec.schema_version, "recorderVersion": "rec-2.0.0-fixture",
        "sessionId": spec.session_id, "operator": "fixture", "portalMode": "ube", "meshSha256": "0" * 64,
        "diagnosis": {"level": "L4-5", "side": "Lt", "type": "HIVD", "text": "Left L4-5 HNP"},
        "approach": {"level": "L4-5", "side": "Lt", "procedure": "laminectomy_ipsilateral",
                     "text": "left L4-5 laminectomy (ipsilateral)"},
        "areas": [{"id": "ar_fixture", "worldPoints": [p.tolist() for p in square], "planeOrigin": center.tolist(),
                   "planeNormal": [0.0, 1.0, 0.0], "source": "clinic-2d1", "halfThicknessMm": 1.0}],
        "basal": {"entryWorld": entry.tolist(), "targetWorld": target.tolist()},
        "basalFrame": {"origin": origin.tolist(), "xAxis": R[:, 0].tolist(), "yAxis": R[:, 1].tolist(),
                       "zAxis": R[:, 2].tolist()},
        "capture": {"cams": [{"name": c, "w": w, "h": h, "hz": spec.capture_hz} for c in schema.CAMS],
                    "jpegQuality": 90},
        "streamHz": spec.tracker_hz, "recovery": spec.recovery,
        "t0Wall": T0_SRC_MS, "t0Mono": MONO_OFFSET_MS,
    }


def _footer(spec: FixtureSpec, phase_start_s: np.ndarray) -> dict:
    bounds = list(phase_start_s + np.where(np.arange(5) > 0, HUMAN_PHASE_SHIFT_S, 0.0)) + [spec.duration_s]
    phases = []
    for p in range(5):
        if p == 3 and not spec.with_phase3:
            continue  # phase 2 then lasts until phase 4 starts
        phases.append({"phase": p, "tMonoStart": bounds[p] * 1000.0 + MONO_OFFSET_MS,
                       "tMonoEnd": bounds[p + 1] * 1000.0 + MONO_OFFSET_MS, "source": "human"})
    return {"k": "ftr", "phases": phases, "verdict": spec.verdict, "verdictReason": "",
            "failFlags": list(spec.fail_flags), "recovery": spec.recovery,
            "exposureEnd": {"endo": 0.9, "work": 0.85}, "removedMm3": 250.0}


def _latest(times: np.ndarray, t: float) -> float | None:
    i = np.searchsorted(times, t, side="right") - 1
    return float(times[i]) if i >= 0 else None
