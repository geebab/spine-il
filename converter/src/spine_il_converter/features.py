"""LeRobot v3 feature spec for the tracked bimanual SO-101 dataset (PLAN-0 §6.4)."""

from __future__ import annotations

from . import schema

_AXES = ("x", "y", "z")


def _tool_names(tool: str) -> list[str]:
    return [f"{tool}.pos.{a}" for a in _AXES] + [f"{tool}.rot6d.{i}" for i in range(6)]


# "drill_on" must stay a unique substring: LeRobot's relative-action exclusion matches by substring.
STATE_NAMES = _tool_names("endo") + _tool_names("drill") + ["drill_on"]


def dataset_features(image_hw: tuple[int, int] = (256, 256)) -> dict:
    h, w = image_hw
    vec = lambda names: {"dtype": "float32", "shape": (len(names),), "names": list(names)}  # noqa: E731
    video = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    return {
        "observation.state": vec(STATE_NAMES),
        "action": vec(STATE_NAMES),
        **{f"observation.images.{cam}": dict(video) for cam in schema.CAMS},
        "observation.phase": vec(schema.PHASE_NAMES),
        "observation.exposure": vec(("endo", "work")),
        "observation.progress": vec(("removed_cm3", "drill_tip_in_view", "drill_tip_to_bone_dm")),
        "observation.goal": vec(("area.x", "area.y", "area.z", "area.nx", "area.ny", "area.nz")),
        "observation.tracker_time": vec(("t_s",)),
    }
