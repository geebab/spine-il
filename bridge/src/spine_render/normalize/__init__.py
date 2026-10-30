"""Authored input → Script: trajectory, start+delta, or a saved script. Recordings are not normalized here — the viewer
replays them itself (recref.py / replay.py)."""

from __future__ import annotations

import json
from pathlib import Path

from ..script import SCRIPT_KIND, Script
from .delta import FORMAT as DELTA_FORMAT
from .delta import from_delta, integrate
from .recording import is_path, is_recording, read_rows
from .trajectory import FORMAT as TRAJ_FORMAT
from .trajectory import from_trajectory

__all__ = ["normalize", "from_trajectory", "from_delta", "integrate"]

RECORDING_NOT_NORMALIZED = ("a recorder session is not converted in Python: submit the file as is — the viewer replays "
                            "it (mode 'replay'; mode 'simulate' with diagnostic=true re-runs the recorded hand through "
                            "the current engine)")


def normalize(src, case: str | None = None) -> tuple[Script, dict | None]:
    """Returns (script, absolute_trajectory_or_None). `src`: path, JSON/JSONL text/bytes, dict, or list of rows.
    Recorder sessions raise ValueError (see RECORDING_NOT_NORMALIZED)."""
    if is_path(src):
        src = Path(src).read_text()
    if isinstance(src, bytes):
        src = src.decode()
    if isinstance(src, str):
        s = src.strip()
        try:
            src = json.loads(s)  # a single JSON document
        except json.JSONDecodeError:
            src = read_rows(s)  # JSONL
    if isinstance(src, dict):
        fmt = src.get("format")
        if fmt == DELTA_FORMAT or ("deltas" in src and "start" in src):
            return from_delta(src, case)
        if fmt == TRAJ_FORMAT or "frames" in src:
            return from_trajectory(src, case), None
        raise ValueError("unknown JSON input: expected format spine-render-trajectory or spine-render-delta")
    rows = list(src)
    if rows and rows[0].get("kind") == SCRIPT_KIND:
        if (rows[0].get("source") or {}).get("type") == "recording":
            raise ValueError("this script was converted from a recording by an older spine-render; "
                             + RECORDING_NOT_NORMALIZED)
        return Script.from_rows(rows), None
    if is_recording(rows):
        raise ValueError(RECORDING_NOT_NORMALIZED)
    raise ValueError("unknown JSONL input: expected a spine-render script (recorder sessions are replayed, not normalized)")
