#!/usr/bin/env python3
"""The hand tool calibration a recording was made with.

    python robot/hand_calib.py RECORDING.jsonl     →  {"endo": "<file>", "drill": "<file>"}  (one JSON line; rc 1 if unknown)

A rev.4 recording's frame rows are the tools' marker blocks. Where the tip is comes from that day's tool calibration
(tool_calib.json, tool_calib_drill.json: tip_local), and the header names it: toolCalib.<tool>.sha256_12 and .tipLocal.
The 09-28 demos and the 10-06 demos were made with different calibrations — the same marker blocks, the drill tip 5.1 mm
apart — so whatever turns rows into tips outside the viewer (the start-pair search, the arms' replay of a recording or of
a policy's stream) has to take the file the recording names. With a fixed file a 10-06 recording's drill tip sits 5 mm
off: the paired start gets the wrong tip-to-tip vector, and the replayed tip wanders with the tool's rotation.

Known files: ~/il-robot/calib/tracker_models/hand_*/ first, then ~/il-robot/runs/hand_incoming_*/ (a batch of recordings
is staged there together with its calibration). A policy recording (inference/policy2rec.py) carries the header of the
demo it started from, so it resolves the same way.
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

FILES = {"endo": "tool_calib.json", "drill": "tool_calib_drill.json"}
ROOTS = ("il-robot/calib/tracker_models/hand_*", "il-robot/runs/hand_incoming_*")
TIP_TOL_MM = 1e-6


def known(home: Path | None = None) -> dict:
    """{tool: [(sha256's first 12 hex, tip_local mm, path)]} of every calibration file under ROOTS, in ROOTS' order."""
    home = home or Path.home()
    out: dict = {t: [] for t in FILES}
    for root in ROOTS:
        for d in sorted(glob.glob(str(home / root))):
            for tool, name in FILES.items():
                f = Path(d) / name
                if f.is_file():
                    raw = f.read_bytes()
                    out[tool].append((hashlib.sha256(raw).hexdigest()[:12], json.loads(raw).get("tip_local"), f))
    return out


def header(recording) -> dict:
    with open(recording) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("k") == "hdr":
                return d
    raise LookupError(f"{recording}: no header row")


def for_header(hdr: dict, home: Path | None = None) -> dict:
    """{tool: Path} for a recording header: by the file hash, else by the tip (a header without hashes)."""
    kn, out = known(home), {}
    for tool in FILES:
        tc = (hdr.get("toolCalib") or {}).get(tool) or {}
        sha, tip = tc.get("sha256_12"), tc.get("tipLocal")
        hit = next((f for s, _, f in kn[tool] if sha and s == sha), None)
        if hit is None and not sha and tip is not None:
            hit = next((f for _, t, f in kn[tool] if t is not None and len(t) == len(tip)
                        and max(abs(a - b) for a, b in zip(t, tip)) <= TIP_TOL_MM), None)
        if hit is None:
            raise LookupError(f"no {FILES[tool]} with sha256 {sha} (tip {tip}) under {' or '.join('~/' + r for r in ROOTS)} "
                              f"— stage the calibration this recording was made with")
        out[tool] = hit
    return out


def for_recording(recording, home: Path | None = None) -> dict:
    return for_header(header(recording), home)


if __name__ == "__main__":
    try:
        print(json.dumps({t: str(f) for t, f in for_recording(sys.argv[1]).items()}))
    except (LookupError, OSError, IndexError) as e:
        print(f"hand_calib: {e}", file=sys.stderr)
        sys.exit(1)
