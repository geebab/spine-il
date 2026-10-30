"""Load one recorder v2 JSONL session into numpy arrays, collecting schema issues instead of raising."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import schema
from .geometry import GeometryError, tool_rotation
from .recorder_v2 import FORMAT_REC2, Recording, detect_format, load_recording


@dataclass
class PoseTrack:
    """Valid (non-lost) samples of one tool, sorted by tSrc with duplicates removed."""

    t_src: np.ndarray  # (N,) ms
    tip: np.ndarray  # (N, 3) scene LPS mm
    rot: np.ndarray  # (N, 3, 3) scene ← tool
    on: np.ndarray  # (N,) drill on 0/1 (zeros for the endoscope)
    n_rows: int  # all rows for this tool, including lost ones
    n_lost: int
    n_up_missing: int

    @property
    def lost_frac(self) -> float:
        return self.n_lost / self.n_rows if self.n_rows else 1.0


@dataclass
class Metrics:
    t_mono: np.ndarray
    t_src_endo: np.ndarray
    phase: np.ndarray  # auto-detected phase 0..4
    exposure: np.ndarray  # (N, 2) endo, work
    removed_mm3: np.ndarray
    drill_tip_in_view: np.ndarray
    drill_tip_to_bone_mm: np.ndarray


@dataclass
class ImageIndex:
    frame_idx: np.ndarray
    t_src_endo: np.ndarray


@dataclass
class Session:
    path: Path
    header: dict = field(default_factory=dict)
    footer: dict | None = None
    poses: dict[str, PoseTrack] = field(default_factory=dict)
    metrics: Metrics | None = None
    images: dict[str, ImageIndex] = field(default_factory=dict)
    issues: list[schema.Issue] = field(default_factory=list)

    @property
    def session_id(self) -> str:
        return str(self.header.get("sessionId", self.path.stem))

    @property
    def errors(self) -> list[schema.Issue]:
        return [i for i in self.issues if i.level == "error"]


def load_session(path: str | Path) -> Session | Recording:
    """Real recorder files (the default) load as a `Recording`; PLAN-0 synthetic fixtures as a `Session`."""
    path = Path(path)
    if detect_format(path) == FORMAT_REC2:
        return load_recording(path)
    s = Session(path=path)
    pose_rows: dict[str, list[tuple[int, dict]]] = {t: [] for t in schema.TOOLS}
    metric_rows: list[dict] = []
    image_rows: dict[str, list[dict]] = {c: [] for c in schema.CAMS}

    def err(line, msg):
        s.issues.append(schema.Issue("error", line, msg))

    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                err(line_no, f"invalid JSON: {e.msg}")
                continue
            kind = row.get("k")
            if kind == "hdr":
                if s.header:
                    err(line_no, "duplicate hdr row")
                s.header = row
                if miss := schema.missing(row, schema.HEADER_REQUIRED):
                    err(line_no, f"hdr missing {miss}")
                if row.get("schemaVersion") != schema.SCHEMA_VERSION:
                    err(line_no, f"schemaVersion {row.get('schemaVersion')!r} != {schema.SCHEMA_VERSION}")
            elif kind == "f":
                if miss := schema.missing(row, schema.POSE_REQUIRED):
                    err(line_no, f"f missing {miss}")
                elif row["tool"] not in schema.TOOLS:
                    err(line_no, f"unknown tool {row['tool']!r}")
                elif not row.get("lost"):
                    scene = row.get("scene") or {}
                    if miss := schema.missing(scene, schema.POSE_SCENE_REQUIRED):
                        err(line_no, f"f.scene missing {miss}")
                    elif row["tool"] == "drill" and "on" not in (row.get("drill") or {}):
                        err(line_no, "drill f row missing drill.on")
                    else:
                        pose_rows[row["tool"]].append((line_no, row))
                else:
                    pose_rows[row["tool"]].append((line_no, row))
            elif kind == "m":
                if miss := schema.missing(row, schema.METRIC_REQUIRED):
                    err(line_no, f"m missing {miss}")
                else:
                    metric_rows.append(row)
            elif kind == "img":
                if miss := schema.missing(row, schema.IMAGE_REQUIRED):
                    err(line_no, f"img missing {miss}")
                elif row["cam"] not in schema.CAMS:
                    s.issues.append(schema.Issue("warning", line_no, f"ignoring unknown cam {row['cam']!r}"))
                else:
                    image_rows[row["cam"]].append(row)
            elif kind == "ftr":
                s.footer = row
                if miss := schema.missing(row, schema.FOOTER_REQUIRED):
                    err(line_no, f"ftr missing {miss}")
            elif kind == "ev":
                pass
            else:
                s.issues.append(schema.Issue("warning", line_no, f"unknown row kind {kind!r}"))

    if not s.header:
        err(None, "no hdr row")
    if s.footer is None:
        s.issues.append(schema.Issue("warning", None, "no ftr row (session not finished)"))

    for tool, rows in pose_rows.items():
        try:
            s.poses[tool] = _pose_track(tool, rows)
        except GeometryError as e:
            err(None, f"{tool}: {e}")
        if not s.poses.get(tool) or len(s.poses[tool].t_src) == 0:
            err(None, f"no valid pose rows for tool {tool!r}")
        elif s.poses[tool].n_up_missing:
            s.issues.append(
                schema.Issue("warning", None, f"{tool}: {s.poses[tool].n_up_missing} rows without scene.up (roll from +S)")
            )
    if metric_rows:
        s.metrics = _metrics(metric_rows)
    else:
        err(None, "no m rows")
    for cam, rows in image_rows.items():
        if not rows:
            err(None, f"no img rows for cam {cam!r}")
            continue
        order = np.argsort([r["tSrcEndo"] for r in rows], kind="stable")
        s.images[cam] = ImageIndex(
            frame_idx=np.array([rows[i]["frameIdx"] for i in order], dtype=np.int64),
            t_src_endo=np.array([rows[i]["tSrcEndo"] for i in order], dtype=np.float64),
        )
    return s


def _pose_track(tool: str, rows: list[tuple[int, dict]]) -> PoseTrack:
    valid = [r for _, r in rows if not r.get("lost")]
    n_rows, n_lost = len(rows), len(rows) - len(valid)
    if not valid:
        empty3 = np.empty((0, 3))
        return PoseTrack(np.empty(0), empty3, np.empty((0, 3, 3)), np.empty(0), n_rows, n_lost, 0)
    t = np.array([r["tSrc"] for r in valid], dtype=np.float64)
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.append(np.diff(t) > 0, True)  # keep the last of duplicate timestamps
    idx = order[keep]
    t = t[keep]
    scenes = [valid[i]["scene"] for i in idx]
    tip = np.array([sc["tip"] for sc in scenes], dtype=np.float64)
    direction = np.array([sc["dir"] for sc in scenes], dtype=np.float64)
    has_up = np.array(["up" in sc for sc in scenes])
    up = np.array([sc.get("up", [0.0, 0.0, 1.0]) for sc in scenes], dtype=np.float64)
    on = (
        np.array([float(valid[i]["drill"]["on"]) for i in idx])
        if tool == "drill"
        else np.zeros(len(idx))
    )
    return PoseTrack(t, tip, tool_rotation(direction, up), on, n_rows, n_lost, int((~has_up).sum()))


def _metrics(rows: list[dict]) -> Metrics:
    rows = sorted(rows, key=lambda r: r["tMono"])
    col = lambda key: np.array([r[key] for r in rows], dtype=np.float64)  # noqa: E731
    return Metrics(
        t_mono=col("tMono"),
        t_src_endo=col("tSrcEndo"),
        phase=col("phase").astype(np.int64),
        exposure=np.stack([col("exposureEndo"), col("exposureWork")], axis=1),
        removed_mm3=col("removedMm3"),
        drill_tip_in_view=col("drillTipInView"),
        drill_tip_to_bone_mm=col("drillTipToBoneMm"),
    )
