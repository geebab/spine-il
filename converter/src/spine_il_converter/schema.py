"""Recorder schema v2 — the subset of fields the converter relies on (PLAN-0 §5.3).

The recorder (the Altician viewer + relay `ws_relay.py`) writes one JSONL file per session:
`hdr` (once) → `f` / `m` / `img` / `ev` (repeated) → `ftr` (once). Timestamps: `tSrc*` are tracker
wall-clock milliseconds, `tMono` is the browser's performance.now() in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

SCHEMA_VERSION = 2
TOOLS = ("endo", "drill")
CAMS = ("endoscope", "scene")
PHASE_NAMES = ("home", "approach", "advance", "laminectomy", "exit")
LAMINECTOMY_PHASE = 3
FPS = 30
ROBOT_TYPE = "tracked_bi_so101_dummy_v1"

HEADER_REQUIRED = ("schemaVersion", "sessionId", "diagnosis", "approach", "basal", "areas", "capture")
POSE_REQUIRED = ("tool", "tSrc")
POSE_SCENE_REQUIRED = ("tip", "dir")  # `up` is strongly recommended; see geometry.tool_rotation fallback
METRIC_REQUIRED = (
    "tMono",
    "tSrcEndo",
    "phase",
    "exposureEndo",
    "exposureWork",
    "removedMm3",
    "drillTipInView",
    "drillTipToBoneMm",
)
IMAGE_REQUIRED = ("cam", "frameIdx", "tMono", "tSrcEndo")
FOOTER_REQUIRED = ("phases", "verdict", "failFlags")


@dataclass(frozen=True)
class Issue:
    level: str  # "error" | "warning"
    line: int | None
    message: str

    def __str__(self) -> str:
        where = f"line {self.line}: " if self.line is not None else ""
        return f"[{self.level}] {where}{self.message}"


def missing(row: dict, keys: tuple[str, ...]) -> list[str]:
    return [k for k in keys if k not in row]
