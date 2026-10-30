"""A pause switch for batch jobs, held from outside the service.

A render job shares the machine with whatever else runs on it. On the robot rig the headless recorder page of a robot
run uses the same GPU, and a render beside it stalled the recorder's drill loop for 0.5-1.1 s; until 10-09 a robot run
therefore waited for the render to end (a 5-panel video: up to an hour). Now the render waits for the robot: while the
pause file exists and is fresh, a job stops between two frames and goes on when the file is gone.

    file     $UBE_RENDER_PAUSE_FILE, default $XDG_RUNTIME_DIR/ube-render/pause   (tmpfs: gone after a reboot)
    content  JSON {"reason": "..."} — or anything: only its presence and its age count
    fresh    touched within FRESH_S seconds. Whoever asks for the pause keeps touching the file, so a pause whose owner
             died ends by itself and a job never waits for nobody

Between two frames nothing is in flight with the browser (no call that could time out), and a job's page runs on the fake
clock: a job that waited an hour renders the frames it would have rendered (tests/test_pause.py, and the e2e check in
the bridge README). What is not paused: a job's setup and its packaging (they end within a minute), and step sessions
(/v1/step) — their client decides when a step is made.

Who writes the file on the rig: robot/eval/render_gate.py (it watches the arms and the robot's processes).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Callable

FRESH_S = 15.0


def pause_file() -> Path:
    p = os.environ.get("UBE_RENDER_PAUSE_FILE")
    if p:
        return Path(p)
    return Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}") / "ube-render" / "pause"


def pause_reason(now: float | None = None) -> str | None:
    """Why jobs have to wait right now — None when nothing (or only a stale file) asks for it."""
    f = pause_file()
    try:
        age = (time.time() if now is None else now) - f.stat().st_mtime
    except OSError:
        return None
    if age > FRESH_S:
        return None
    try:
        doc = json.loads(f.read_text())
        return str(doc.get("reason") or "pause file") if isinstance(doc, dict) else "pause file"
    except (OSError, ValueError):
        return "pause file"


async def wait_clear(on_wait: Callable[[str | None, float], None] | None = None, poll_s: float = 0.5) -> float:
    """Return at once when nothing asks for a pause (one stat call); otherwise wait until the pause is over. `on_wait`
    is told the reason and the time the wait began, once at its start and whenever the reason changes, and (None, t0)
    at its end. Returns the seconds waited."""
    reason = pause_reason()
    if reason is None:
        return 0.0
    t0, told = time.time(), None
    while reason is not None:
        if on_wait and reason != told:
            on_wait(reason, t0)
            told = reason
        await asyncio.sleep(poll_s)
        reason = pause_reason()
    if on_wait:
        on_wait(None, t0)
    return time.time() - t0
