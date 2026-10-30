"""Start pose + per-step deltas → absolute trajectory (returned too, so users see what was rendered).

    {"format": "spine-render-delta", "version": 1, "case": "altician_lumbar1",
     "frame": "tool",                 # "tool": dpos/drot in the tool's own axes; "scene": in scene LPS axes
     "dt_ms": 33.333,                 # default step; a delta may carry its own "dt_ms"
     "start":  {"endo": {"tip": [..], "u": [..], "roll": 0}, "drill": {"tip": [..], "quat": [..], "trig": false}},
     "deltas": [{"endo": {"dpos": [0,0,1], "drot_deg": [0,0.5,0]}, "drill": {"trig": true}}, ...]}

Rotations turn about the tool tip (the pose's origin), so a pure drot keeps the tip where it is.
"""

from __future__ import annotations

import numpy as np

from ..geom import matrix_to_quat, rotvec_to_matrix
from ..script import Script
from .trajectory import FORMAT as TRAJ_FORMAT
from .trajectory import _setup, from_trajectory, scene_pose

FORMAT = "spine-render-delta"


def integrate(obj: dict) -> dict:
    """Delta document → absolute trajectory document (tip + quat per tool per step)."""
    if obj.get("format") not in (None, FORMAT):
        raise ValueError(f"expected format {FORMAT}")
    reg, tools = _setup(obj)
    frame = obj.get("frame", "tool")
    if frame not in ("tool", "scene"):
        raise ValueError("frame must be 'tool' or 'scene'")
    state, trig = {}, False
    for tool, p in (obj.get("start") or {}).items():
        pos_s, R_s, _ = scene_pose(p, reg, tools[tool])
        state[tool] = [pos_s + R_s @ np.asarray(tools[tool]["tipLocal"], float), R_s]
        trig = bool(p.get("trig", trig)) if tool == "drill" else trig
    if not state:
        raise ValueError("start needs at least one tool pose")

    def snap(t_s):
        fr = {"t": round(t_s, 6)}
        for tool, (tip, R) in state.items():
            fr[tool] = {"tip": [float(x) for x in tip], "quat": [float(x) for x in matrix_to_quat(R)]}
            if tool == "drill":
                fr[tool]["trig"] = trig
        return fr

    t, frames = 0.0, [snap(0.0)]
    for d in obj.get("deltas", []):
        t += float(d.get("dt_ms", obj.get("dt_ms", 1000.0 / 30.0))) / 1000.0
        for tool, dd in d.items():
            if tool not in state:
                continue
            tip, R = state[tool]
            dR = rotvec_to_matrix(np.radians(dd.get("drot_deg", [0, 0, 0])))
            dp = np.asarray(dd.get("dpos", [0, 0, 0]), float)
            if frame == "tool":
                tip, R = tip + R @ dp, R @ dR
            else:
                tip, R = tip + dp, dR @ R
            state[tool] = [tip, R]
            if tool == "drill" and "trig" in dd:
                trig = bool(dd["trig"])
        frames.append(snap(t))
    return {"format": TRAJ_FORMAT, "version": 1, "case": obj.get("case", "altician_lumbar1"),
            "registration": obj.get("registration"), "tools": obj.get("tools"),
            "feed_hz": obj.get("feed_hz", None), "fe": obj.get("fe"), "drill": obj.get("drill"), "frames": frames}


def from_delta(obj: dict, case: str | None = None) -> tuple[Script, dict]:
    traj = integrate(obj)
    script = from_trajectory(traj, case)
    script.source["type"] = "delta"
    return script, traj
