"""Canonical render script: what the renderer feeds the viewer, sample by sample.

A script is tracker-level and camera-frame, i.e. exactly what the real tracker + relay would have sent:
per tool, time-stamped samples with pos / quat (wxyz, tool→cam) / tip / shaft, or `lost`. Everything the
viewer derives (skin entry, soft fulcrum, fixture clamp, drill proxy) is left to the viewer.

Serialised as JSONL: one `{"k": "hdr", ...}` line, then `{"k": "s", ...}` samples and `{"k": "imu", ...}` rows.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Iterator

import numpy as np

from .geom import Registration, matrix_to_quat, quat_to_matrix, slerp, unit

CASE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")  # FE case keys (makeSessionIdV2 keeps [A-Za-z0-9_.-])


def check_case(case: object) -> str:
    """A viewer case key — goes into the viewer URL query, so it must be a plain key (no `&`, `=`, `/`)."""
    if not isinstance(case, str) or not CASE_RE.fullmatch(case):
        raise ValueError(f"case must match {CASE_RE.pattern}, got {case!r}")
    return case


SCRIPT_KIND = "spine-render-script"
SCRIPT_VERSION = 1
TOOLS = ("endo", "drill")
# The viewer's drill burr sizes (FE engine/hooks/drilling/useDrillMouseLoop.ts DRILL_SIZES): key → carve radius (mm).
# Its default is large; the banner's "Drill size" radiogroup (소/중/대) selects one.
DRILL_SIZES = {"small": 1.5, "medium": 2.0, "large": 4.0}
DRILL_SIZE_LABELS = {"small": "소", "medium": "중", "large": "대"}


def rig_default() -> dict:
    return json.loads(resources.files("spine_render.data").joinpath("rig_default.json").read_text())


@dataclass
class Sample:
    t: float  # ms, tracker clock (recordings: tSrc; authored inputs: 0-based)
    tool: str
    pos: list | None = None  # cam mm, constellation origin
    quat: list | None = None  # [w,x,y,z] tool → cam
    tip: list | None = None  # cam mm
    shaft: list | None = None  # cam unit, handle → tip
    lost: bool = False
    filled: bool = False  # interpolated across a short tracking gap
    trig: bool | None = None  # drill trigger (ZR)
    extra: dict = field(default_factory=dict)  # rmse / vis / split / still, source row index …

    @property
    def valid(self) -> bool:
        return not self.lost and self.pos is not None and self.quat is not None

    def to_row(self) -> dict:
        row = {"k": "s", "t": round(self.t, 3), "tool": self.tool}
        if self.valid:
            row.update(pos=_r(self.pos, 3), quat=_r(self.quat, 7), tip=_r(self.tip, 3), shaft=_r(self.shaft, 6))
        else:
            row["lost"] = 1
        if self.filled:
            row["filled"] = 1
        if self.trig is not None:
            row["trig"] = int(bool(self.trig))
        row.update(self.extra)
        return row

    @classmethod
    def from_row(cls, row: dict) -> "Sample":
        known = {"k", "t", "tool", "pos", "quat", "tip", "shaft", "lost", "filled", "trig"}
        trig = row.get("trig")
        return cls(t=float(row["t"]), tool=row["tool"], pos=row.get("pos"), quat=row.get("quat"), tip=row.get("tip"),
                   shaft=row.get("shaft"), lost=bool(row.get("lost", 0)), filled=bool(row.get("filled", 0)),
                   trig=None if trig is None else bool(trig), extra={k: v for k, v in row.items() if k not in known})


@dataclass
class Script:
    case: str
    registration: Registration
    tools: dict  # tool → {"tipLocal": [3], "shaftLocal": [3]}
    samples: list[Sample]
    gravity_up_cam: list | None = None
    fe: dict = field(default_factory=dict)  # viewer URL knobs recorded with the session (authored: the header's own)
    drill: dict = field(default_factory=dict)  # {"sizeKey": small|medium|large} — the drill burr the viewer carves with
    source: dict = field(default_factory=dict)
    imu: list[dict] = field(default_factory=list)

    def sort(self) -> "Script":
        self.samples.sort(key=lambda s: (s.t, TOOLS.index(s.tool) if s.tool in TOOLS else 9))
        return self

    @property
    def t_range(self) -> tuple[float, float]:
        ts = [s.t for s in self.samples]
        return (min(ts), max(ts)) if ts else (0.0, 0.0)

    def by_tool(self, tool: str) -> list[Sample]:
        return [s for s in self.samples if s.tool == tool]

    def header(self) -> dict:
        return {"k": "hdr", "kind": SCRIPT_KIND, "version": SCRIPT_VERSION, "case": self.case,
                "registration": self.registration.to_dict(), "tools": self.tools, "gravityUpCam": self.gravity_up_cam,
                "fe": self.fe, "drill": self.drill, "source": self.source, "quatOrder": "wxyz",
                "frames": {"pos/tip": "cam mm", "quat": "tool→cam", "shaft": "cam unit", "t": "ms"}}

    def dump(self, path: str | Path) -> None:
        with open(path, "w") as f:
            for row in self.rows():
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def rows(self) -> Iterator[dict]:
        yield self.header()
        for s in self.samples:
            yield s.to_row()
        for r in self.imu:
            yield {"k": "imu", **r}

    @classmethod
    def load(cls, path: str | Path) -> "Script":
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        return cls.from_rows(rows)

    @classmethod
    def from_rows(cls, rows: list[dict]) -> "Script":
        h = rows[0]
        if h.get("kind") != SCRIPT_KIND:
            raise ValueError("not a spine-render script")
        return cls(case=h["case"], registration=Registration.from_dict(h["registration"]), tools=h["tools"],
                   samples=[Sample.from_row(r) for r in rows if r.get("k") == "s"], gravity_up_cam=h.get("gravityUpCam"),
                   fe=h.get("fe") or {}, drill=h.get("drill") or {}, source=h.get("source") or {},
                   imu=[{k: v for k, v in r.items() if k != "k"} for r in rows if r.get("k") == "imu"])


def scene_pose_to_sample(t_ms: float, tool: str, pos_s, R_s, reg: Registration, calib: dict, **kw) -> Sample:
    """Scene-frame tool pose → camera-frame sample (what the tracker would report)."""
    R_c = reg.R.T @ np.asarray(R_s, float)
    pos_c = reg.to_cam(pos_s)
    tip_c = pos_c + R_c @ np.asarray(calib["tipLocal"], float)
    shaft_c = unit(R_c @ np.asarray(calib["shaftLocal"], float))
    return Sample(t=float(t_ms), tool=tool, pos=pos_c.tolist(), quat=matrix_to_quat(R_c).tolist(), tip=tip_c.tolist(),
                  shaft=shaft_c.tolist(), **kw)


def interpolate_sample(a: Sample, b: Sample, t: float, calib: dict | None) -> Sample:
    """Pose at time t between two valid samples of one tool (position lerp, rotation slerp)."""
    s = 0.0 if b.t == a.t else (t - a.t) / (b.t - a.t)
    q = slerp(a.quat, b.quat, s)
    pos = (1 - s) * np.asarray(a.pos) + s * np.asarray(b.pos)
    tip = (1 - s) * np.asarray(a.tip) + s * np.asarray(b.tip)
    if calib and calib.get("shaftLocal") is not None:
        shaft = unit(quat_to_matrix(q) @ np.asarray(calib["shaftLocal"], float))
    else:
        shaft = unit((1 - s) * np.asarray(a.shaft) + s * np.asarray(b.shaft))
    return Sample(t=float(t), tool=a.tool, pos=pos.tolist(), quat=q.tolist(), tip=tip.tolist(), shaft=shaft.tolist())


def _r(v, nd):
    return None if v is None else [round(float(x), nd) for x in v]
