"""Script samples → tracker WebSocket messages (joycon protocol v1), exactly what ws_relay.py would broadcast."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .script import Sample, Script


def buttons(trig: bool | None) -> dict:
    b = {"zr": False, "capture": False, "a": False}
    if trig is not None:
        b["trig"] = bool(trig)
    return b


def to_message(s: Sample, seq: int, up: list | None) -> dict:
    if s.valid:
        m = {"type": "pose", "v": 1, "t": s.t, "seq": seq, "tool": s.tool, "pos": s.pos, "quat": s.quat,
             "tip": s.tip, "shaft": s.shaft, "buttons": buttons(s.trig)}
        if up is not None:
            m["up"] = up
        for k in ("rmse", "vis", "split"):
            if k in s.extra:
                m[k] = s.extra[k]
        if "still" in s.extra:
            m["still"] = bool(s.extra["still"])
        return m
    return {"type": "pose_lost", "v": 1, "t": s.t, "seq": seq, "tool": s.tool, "buttons": buttons(s.trig)}


def messages(script: Script) -> Iterator[tuple[float, dict]]:
    """(t_ms, message) in time order, with a per-tool seq like the relay."""
    seq: dict[str, int] = {}
    for s in script.samples:
        n = seq.get(s.tool, 0)
        seq[s.tool] = n + 1
        yield s.t, to_message(s, n, script.gravity_up_cam)


def capture_grid(script: Script, fps: float, t0: float | None = None) -> list[float]:
    """Output frame times (ms): every 1/fps from the first sample (phase-locked to the input clock)."""
    lo, hi = script.t_range
    start = lo if t0 is None else t0
    step = 1000.0 / fps
    n = int(np.floor((hi - start) / step + 1e-9)) + 1
    return [start + i * step for i in range(max(n, 0))]
