"""User-authored trajectory (scene frame, LPS mm) → Script.

    {"format": "spine-render-trajectory", "version": 1, "case": "altician_lumbar1",
     "registration": {"R": [9], "t": [3]},        # optional, T_scene←cam; default = the rig's
     "tools": {"drill": {"tipLocal": [3], "shaftLocal": [3]}},   # optional overrides
     "feed_hz": 90,                                # resample keyframes to a tracker-like rate; null = as given
     "drill": {"sizeKey": "medium"},               # optional: burr small (r 1.5) | medium (r 2) | large (r 4, viewer default)
     "fe": {"bite": 1, "softFulcrum": true},       # optional: drill-proxy knobs / rawabs / softFulcrum (flags.py)
     "frames": [
       {"t": 0.0, "endo":  {"tip": [x,y,z], "u": [ux,uy,uz], "roll": 0},
                  "drill": {"tip": [x,y,z], "quat": [w,x,y,z], "trig": false}},
       {"t": 0.5, "endo":  {"lost": true}}
     ]}

t is in seconds. A tool pose is either tip + u (shaft direction, handle → tip) + roll (deg, the viewer's endoscope roll)
or tip + quat (tool → scene rotation). A tool missing from a frame simply has no keyframe there.
"""

from __future__ import annotations

import numpy as np

from ..geom import Registration, pose_from_tip_u_roll, quat_to_matrix
from ..script import DRILL_SIZES, Sample, Script, check_case, interpolate_sample, rig_default, scene_pose_to_sample

FORMAT = "spine-render-trajectory"
DEFAULT_FEED_HZ = 90.0
ROLL_TOL_DEG = 0.01


def _setup(obj: dict) -> tuple[Registration, dict]:
    rig = rig_default()
    reg = Registration.from_dict(obj.get("registration") or rig["registration"])
    tools = {k: dict(v) for k, v in rig["tools"].items()}
    for k, v in (obj.get("tools") or {}).items():
        tools.setdefault(k, {}).update(v)
    return reg, tools


def scene_pose(p: dict, reg: Registration, calib: dict) -> tuple[np.ndarray, np.ndarray, float]:
    """(pos_s, R_s, roll residual deg) for one authored tool pose."""
    tip = np.asarray(p["tip"], float)
    if "quat" in p:
        R_s = quat_to_matrix(p["quat"])
        return tip - R_s @ np.asarray(calib["tipLocal"], float), R_s, 0.0
    if "u" in p:
        return pose_from_tip_u_roll(tip, p["u"], float(p.get("roll", 0.0)), calib["tipLocal"], calib["shaftLocal"],
                                    reg.cam_centre)
    raise ValueError("a tool pose needs tip + u (+ roll) or tip + quat")


def _drill_header(obj: dict) -> dict:
    """{"sizeKey"} from the trajectory's "drill" (or "fe": {"drill"}) header, validated; {} = the viewer default."""
    d = obj.get("drill")
    if d is None and isinstance(obj.get("fe"), dict) and isinstance(obj["fe"].get("drill"), dict):
        d = obj["fe"]["drill"]
    if d is None:
        return {}
    if not isinstance(d, dict):
        raise ValueError(f'"drill" must be an object like {{"sizeKey": "medium"}}, got {d!r}')
    key = d.get("sizeKey")
    if key is None:
        return {}
    if key not in DRILL_SIZES:
        raise ValueError(f"drill.sizeKey must be one of {tuple(DRILL_SIZES)}, got {key!r}")
    return {"sizeKey": key}


def from_trajectory(obj: dict, case: str | None = None) -> Script:
    if obj.get("format") not in (None, FORMAT):
        raise ValueError(f"expected format {FORMAT}")
    reg, tools = _setup(obj)
    keys, warnings, trig = [], [], False
    for i, fr in enumerate(obj["frames"]):
        t_ms = float(fr.get("t", i / float(obj.get("fps", 30)))) * 1000.0
        for tool in ("endo", "drill"):
            p = fr.get(tool)
            if p is None:
                continue
            if tool == "drill" and "trig" in p:
                trig = bool(p["trig"])
            if p.get("lost"):
                keys.append(Sample(t=t_ms, tool=tool, lost=True, trig=trig if tool == "drill" else None))
                continue
            pos_s, R_s, err = scene_pose(p, reg, tools[tool])
            if err > ROLL_TOL_DEG:
                warnings.append(f"frame {i} {tool}: roll {p.get('roll')}° is off by {err:.3f}° "
                                "(marker plane cannot face the camera there)")
            keys.append(scene_pose_to_sample(t_ms, tool, pos_s, R_s, reg, tools[tool],
                                             trig=trig if tool == "drill" else None, extra={"key": i}))
    script = Script(case=check_case(case or obj.get("case", "altician_lumbar1")), registration=reg, tools=tools, samples=keys,
                    gravity_up_cam=obj.get("gravityUpCam", rig_default().get("gravityUpCam")), fe=obj.get("fe") or {},
                    drill=_drill_header(obj),
                    source={"type": "trajectory", "warnings": warnings, "keyframes": len(keys)}).sort()
    hz = obj.get("feed_hz", DEFAULT_FEED_HZ)
    if hz:
        script.samples = resample(script, float(hz))
        script.sort()
    return script


def resample(script: Script, hz: float) -> list[Sample]:
    """Keyframes → samples every 1/hz s per tool (lerp/slerp between valid keyframes; a lost keyframe holds lost
    until the next keyframe; the drill trigger is held). Keyframe times are kept exactly."""
    out: list[Sample] = []
    step = 1000.0 / hz
    for tool in {s.tool for s in script.samples}:
        keys = script.by_tool(tool)
        calib = script.tools.get(tool)
        for a, b in zip(keys, keys[1:]):
            n = max(1, int(np.ceil((b.t - a.t) / step - 1e-9)))
            for k in range(n):
                t = a.t + (b.t - a.t) * k / n
                if k == 0:
                    out.append(a)
                elif a.valid and b.valid:
                    s = interpolate_sample(a, b, t, calib)
                    s.trig, s.extra = a.trig, {"interp": 1}
                    out.append(s)
                else:
                    out.append(Sample(t=t, tool=tool, lost=True, trig=a.trig))
        if keys:
            out.append(keys[-1])
    return out
