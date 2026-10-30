"""Recorder schema v2 JSONL (tracker rig sessions): input detection only.

Recordings are not converted in Python. Replay rules — rows → tracker messages, cut rows, drawn poses, the recorded-hand
re-simulation — live in the viewer (FE) in one place; the render hands it the file (recref.RecordingRef →
window.__ubeRender.replay) and only picks the frame times to capture. These helpers are shared with the other inputs.
"""

from __future__ import annotations

import json
from pathlib import Path


def is_path(src) -> bool:
    """True when src names an existing file (a JSON/JSONL document passed as text is not a path)."""
    if isinstance(src, Path):
        return src.exists()
    if not isinstance(src, str) or "\n" in src or len(src) > 1024 or src.lstrip().startswith(("{", "[")):
        return False
    try:
        return Path(src).exists()
    except OSError:
        return False


def read_rows(src) -> list[dict]:
    if is_path(src):
        text = Path(src).read_text()
    elif isinstance(src, bytes):
        text = src.decode()
    elif isinstance(src, str):
        text = src
    else:  # already parsed
        return list(src)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def is_recording(rows: list[dict]) -> bool:
    """A recorder v2 session: a header row with a schemaVersion (wherever the relay's chunk order put it)."""
    return any(isinstance(r, dict) and r.get("k") == "hdr" and "schemaVersion" in r for r in rows)
