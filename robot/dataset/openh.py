"""The Open-H-Embodiment v2 layout of the export (export_lerobot.py --layout openh).

Conventions: github.com/open-h/open-h-embodiment (c8f8632: README, AGENTS.md, templates/dataset_template.md), checked by
its scripts/validation/validate_formatting.py (needs lerobot == 0.6.0: ~/il-robot/.venv-openh). The episodes, the time
grid, the pose sampling and the images are export_lerobot.py's own — only what a frame holds and the dataset's metadata
differ from our training layout (mm, wxyz, 15 values):

  observation.state / action     (16,) float32: one block [x_m, y_m, z_m, qx, qy, qz, qw, gripper] per tool, endoscope
                                 then drill — tip position in metres and unit quaternion xyzw (tool marker frame →
                                 scene), in the scene frame (patient CT, LPS; registration T0 of the session).
                                 gripper: endoscope 0 (nothing to actuate), drill = burr trigger 0/1.
                                 action[k] = observation.state[k+1]: no setpoint was commanded (the tools are
                                 hand-held; a robot episode replays a recording), the next measured pose is the target.
  observation.images.endoscope   the viewer's endoscope screen (ours: observation.images.endo)
  observation.meta.host_stamp_ns        (1,) int64 Unix ns — the frame's time on the tracking camera's clock (tExp grid)
  observation.meta.<tool>_stamp_ns      (1,) int64 — the tracker sample the pose starts from (the last one at or before
                                        the frame; the pose is interpolated from it to the next one)
  observation.meta.tool.<tool>          string — the instrument
  observation.meta.phase_index          (1,) int64 1..6 (phases.py);  observation.meta.track_valid  (2,) as track.valid
  observation.meta.joint_positions      (12,) rad, …_target, …_valid — only when robot episodes are exported
  instruction.text                      string — the phase's instruction; the LeRobot task = the episode's task (English)

After finalize(): info.json splits (train / val / test = index ranges, so the episodes are written in that order),
meta/calibration/camera_intrinsics.json (the scope profile of the images' medium) and meta/README.md (openh_readme.md
filled in; what only the team knows stays marked TODO).
"""

from __future__ import annotations

import json
import string
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import phases as PH  # noqa: E402
    import sources as S  # noqa: E402
else:
    from . import phases as PH
    from . import sources as S

TOLERANCE_S = 0.1    # the README's value; LeRobotDataset.create(tolerance_s=) — lerobot does not store it in info.json
MIN_FPS, MIN_SIZE = 20, 480   # suggested by Open-H (below: validator warnings, not errors)
ARMS = (("endo", "endoscope"), ("drill", "drill"))   # our tool → the block's name, in block order
BLOCK = ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw", "gripper")
STATE_NAMES = [f"{name}.{c}" for _, name in ARMS for c in BLOCK]
VIEW = {"endo": "endoscope"}
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
JOINT_NAMES = [f"left_{j}" for j in JOINTS] + [f"right_{j}" for j in JOINTS]   # left arm = endoscope, right arm = drill
TASK = S.TASK_EPISODE.split(" / ")[-1]
INSTRUCTION = {k: v.split(" / ")[-1] for k, v in PH.PHASES.items()}
CIRCLE_FRAC = 0.68 * 2 ** 0.5 / 2   # image circle radius / side of the endoscope screen (viewer layout; 123 px of 256 measured)
README_TEMPLATE = Path(__file__).resolve().parent / "openh_readme.md"


def image_key(view: str) -> str:
    return f"observation.images.{VIEW.get(view, view)}"


def features(views=("endo",), size: int = 480, image_dtype: str = "video", joints: bool = False) -> dict:
    f = {image_key(v): {"dtype": image_dtype, "shape": (size, size, 3), "names": ["height", "width", "channels"]} for v in views}
    stamp = {"dtype": "int64", "shape": (1,), "names": ["ns"]}
    text = {"dtype": "string", "shape": (1,), "names": ["value"]}
    f.update({
        "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "action": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
        "observation.meta.host_stamp_ns": dict(stamp),
        **{f"observation.meta.{name}_stamp_ns": dict(stamp) for _, name in ARMS},
        **{f"observation.meta.tool.{name}": dict(text) for _, name in ARMS},
        "observation.meta.phase_index": {"dtype": "int64", "shape": (1,), "names": ["phase"]},
        "observation.meta.track_valid": {"dtype": "float32", "shape": (2,), "names": [name for _, name in ARMS]},
        "instruction.text": {"dtype": "string", "shape": (1,), "names": ["value"],
                             "description": "Natural language instruction of the frame's phase"},
    })
    if joints:
        f.update({"observation.meta.joint_positions": {"dtype": "float32", "shape": (12,), "names": JOINT_NAMES},
                  "observation.meta.joint_positions_target": {"dtype": "float32", "shape": (12,), "names": JOINT_NAMES},
                  "observation.meta.joint_positions_valid": {"dtype": "float32", "shape": (1,), "names": ["valid"]}})
    return f


def pose16(state: np.ndarray) -> np.ndarray:
    """(…, 15) ours [endo xyz mm, quat wxyz, drill xyz mm, quat wxyz, trigger] → (…, 16) two Open-H blocks."""
    s = np.asarray(state, np.float64)
    out = np.zeros(s.shape[:-1] + (16,), np.float64)
    for b, a in enumerate((0, 7)):
        out[..., 8 * b:8 * b + 3] = s[..., a:a + 3] * 1e-3
        out[..., 8 * b + 3:8 * b + 6] = s[..., a + 4:a + 7]
        out[..., 8 * b + 6] = s[..., a + 3]
    out[..., 15] = s[..., 14]
    return out.astype(np.float32)


def stamp_ns(t_ms: float) -> np.ndarray:
    """Epoch ms (float, the recorder's tExp) → Unix ns, exact to the microsecond."""
    return np.array([int(round(float(t_ms) * 1000.0)) * 1000], np.int64)


def tool_text(meta: dict) -> dict:
    """observation.meta.tool.* of an episode (export_lerobot.build_frames' meta: the recording's drill size)."""
    r, key = meta.get("drill_radius_mm"), meta.get("drill_size")
    burr = "spherical burr" + (f", radius {float(r):g} mm" if r is not None else "") + (f" ({key})" if key else "")
    held = "on an SO-101 arm" if meta.get("source") == "robot" else "hand-held"
    return {"endoscope": f"rigid endoscope replica, {held}, optically tracked",
            "drill": f"drill replica with a {burr}, {held}, optically tracked"}


def frame(fr: dict, i: int, tools: dict, joints: bool) -> dict:
    """Frame i of an episode (export_lerobot.build_frames' arrays + "pose16") without its images."""
    ph = int(fr["phase"][i])
    f = {"observation.state": fr["pose16"][i], "action": fr["pose16"][i + 1],
         "observation.meta.host_stamp_ns": stamp_ns(fr["tq"][i]),
         **{f"observation.meta.{name}_stamp_ns": stamp_ns(fr["src_t"][i, c]) for c, (_, name) in enumerate(ARMS)},
         **{f"observation.meta.tool.{name}": tools[name] for _, name in ARMS},
         "observation.meta.phase_index": np.array([ph], np.int64),
         "observation.meta.track_valid": fr["valid"][i],
         "instruction.text": INSTRUCTION[ph], "task": TASK}
    if joints:
        f.update({"observation.meta.joint_positions": np.radians(fr["joints"][i]).astype(np.float32),
                  "observation.meta.joint_positions_target": np.radians(fr["target"][i]).astype(np.float32),
                  "observation.meta.joint_positions_valid": fr["jvalid"][i]})
    return f


# ---------------------------------------------------------------- splits
def assign_splits(eps: list, val: set | None = None, test: set | None = None, log=print) -> list:
    """Give every planned episode a split (ep.meta["split"]) and return them in the order train, val, test — info.json
    holds index ranges. val / test: short session ids (or run tags); without them the last tenth of the episodes (at
    least one) is the test split and the tenth before it the validation split, in recording order."""
    named = bool(val or test)
    if not named:
        k = max(1, round(0.1 * len(eps))) if len(eps) >= 3 else 0
        if not k:
            log(f"[openh] {len(eps)} episodes: too few for a val / test split — all train")
        val = {id(e) for e in eps[len(eps) - 2 * k:len(eps) - k]}
        test = {id(e) for e in eps[len(eps) - k:]}
    for e in eps:
        keys = {e.sess.sid, e.sess.meta.get("tag")} if named else {id(e)}
        e.meta["split"] = "test" if keys & (test or set()) else "val" if keys & (val or set()) else "train"
    return sorted(eps, key=lambda e: ("train", "val", "test").index(e.meta["split"]))   # stable: recording order inside


def split_ranges(written: list[dict]) -> dict:
    out = {}
    for name in ("train", "val", "test"):
        ix = [m["episode_index"] for m in written if m.get("split") == name]
        if ix:
            if ix != list(range(ix[0], ix[-1] + 1)):
                raise ValueError(f"split {name} is not one index range: {ix}")
            out[name] = f"{ix[0]}:{ix[-1] + 1}"
    return out


# ---------------------------------------------------------------- calibration
def intrinsics(optics: str, size: int) -> dict:
    """The endoscope image's camera model at size × size px: the viewer's built-in scope profile of that medium
    (inference/rollout.py SCOPE: Kannala-Brandt = OpenCV fisheye, calibrated on a 1920×1080 image) scaled to the stored
    image. The viewer draws the profile's image circle centred on its square endoscope screen with radius CIRCLE_FRAC
    of the side, and the dataset image is that screen resized — so scale = CIRCLE_FRAC · size / circle radius and a
    calibration pixel (u, v) lands on (u − circle x) · scale + size/2 − 0.5 (pixel centres at integers)."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
    import rollout as R
    cx0, cy0, r0 = R.SCOPE_CIRCLE
    p = R.SCOPE[optics]; fx, fy, cx, cy = p["K"]
    s = CIRCLE_FRAC * size / r0
    return {"model": "pinhole", "width": size, "height": size,
            "fx": round(fx * s, 4), "fy": round(fy * s, 4),
            "cx": round(size / 2 + (cx - cx0) * s - 0.5, 4), "cy": round(size / 2 + (cy - cy0) * s - 0.5, 4),
            "distortion_model": "opencv_fisheye", "distortion_coeffs": [float(k) for k in p["k"]],
            "medium": optics, "valid_theta_max_deg": p["theta_max"],
            "image_circle": {"cx": size / 2 - 0.5, "cy": size / 2 - 0.5, "r": round(CIRCLE_FRAC * size, 3)},
            "note": "simulated endoscope: the renderer's lens model, not a calibration of these frames. theta_d = "
                    "theta (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8), x right, y down, z forward; outside "
                    "the image circle the frame is masked black. Pixels of on-screen overlays (orientation notch, "
                    "burr contact marker) are part of the image."}


# ---------------------------------------------------------------- README
RENDERING = {
    "recorded": "The images are rendered after the session by replaying the recording in the viewer: the instrument poses "
                "that were on the operator's screen and the recorded bone-removal events are applied again, so every frame "
                "shows the bone surface the operator saw at that instant.",
    "simulated": "The images are rendered after the session, by feeding the dataset's own poses to the viewer at the dataset "
                 "rate; the bone surface they show is that re-simulation's, not the one the operator saw.",
    "mixed": "The images are rendered after the session; some episodes replay the recorded bone-removal events, others "
             "re-simulate the cutting from the dataset's poses (meta/spine_il/export.json: image_cuts).",
    None: "This export has no images.",
}


def readme_text(name: str, written: list[dict], fps: int, size: int | None, optics: str | None, splits: dict,
                has_intrinsics: bool, cuts: str | None = None) -> str:
    n = sum(m["n_frames"] for m in written)
    hand = [m for m in written if m["source"] == "hand"]
    robot = [m for m in written if m["source"] == "robot"]
    days = sorted("-".join((d[:4], d[4:6], d[6:8])) for d in (m["session_id"].split("_")[1] for m in written))   # s_<date>T<time>_…
    ph = sorted({int(k) for m in written for k in m["phase_counts"]})
    trims = [m["trim"] for m in written if "trim" in m]
    trim = ("Hand episodes are trimmed to the part two robot arms can play back: from "
            f"{trims[0]['pre_s']:g} s before the first bone cut (not before both instruments are through the skin) to "
            f"{abs(trims[0]['post_s']):g} s {'before' if trims[0]['post_s'] < 0 else 'after'} both instruments are out of "
            "the skin again; the operator's ready pose before and after is not included."
            if trims else "Episodes are whole recordings: ready pose, skin entry, advance, drilling, withdrawal, return.")
    src = f"{len(hand)} hand-held demonstration(s)" + (f" and {len(robot)} robot episode(s) (two SO-101 arms replaying a "
                                                       "recording or a policy's output)" if robot else "")
    box = lambda on: "[x]" if on else "[ ]"
    return string.Template(README_TEMPLATE.read_text(encoding="utf-8")).substitute(
        name=name, hours=f"{n / fps / 3600:.3f}", episodes=len(written), frames=n, fps=fps, sources=src,
        image=(f"{size}×{size} px, {fps} fps, MP4" if size else "no images in this export"),
        medium=optics or "n/a", period_from=days[0], period_to=days[-1], rendering=RENDERING[cuts if size else None],
        operators=len({m.get("operator") for m in hand if m.get("operator")}) or "TODO(team)",
        phases=", ".join(f"{k} = {INSTRUCTION[k]}" for k in ph), trim=trim,
        splits=", ".join(f"{k} {v}" for k, v in splits.items()) or "none",
        hand_box=box(bool(hand)), robot_box=box(bool(robot)), joint_box=box(bool(robot)),
        intrinsics=(f"OpenCV fisheye (Kannala-Brandt) model of the simulated endoscope for `{image_key('endo')}` in "
                    "`meta/calibration/camera_intrinsics.json` (the renderer's lens model of the "
                    f"{optics} medium, scaled to the stored image size)" if has_intrinsics else "not provided"),
        tolerance=f"{TOLERANCE_S:g}")


def finish(ds, out: Path, written: list[dict], fps: float, views, size: int, optics: str | None, log=print,
           cuts: str | None = None) -> None:
    """Splits, camera intrinsics and README of a finalized dataset (after finalize(): its writers own meta/)."""
    from lerobot.datasets.io_utils import write_info
    sp = split_ranges(written)
    ds.meta.info.splits = sp
    write_info(ds.meta.info, ds.root)
    for k in ("val", "test"):
        if k not in sp:
            log(f"[openh] no {k} split (the validator warns)")
    cal = {}
    if views and optics in ("air", "water"):
        cal = {image_key(v): intrinsics(optics, size) for v in views if v == "endo"}
        d = out / "meta" / "calibration"; d.mkdir(parents=True, exist_ok=True)
        (d / "camera_intrinsics.json").write_text(json.dumps(cal, indent=2))
    elif views:
        log(f"[openh] no camera intrinsics: the images' medium is {optics!r} (render_step.py --optics air|water)")
    (out / "meta" / "README.md").write_text(readme_text(out.name, written, int(round(fps)), size if views else None, optics, sp,
                                                        bool(cal), cuts), encoding="utf-8")
    for what, ok, low in (("fps", fps >= MIN_FPS, f"{fps:g} < {MIN_FPS}"), ("image", not views or size >= MIN_SIZE, f"{size} px < {MIN_SIZE}")):
        if not ok:
            log(f"[openh] {what} below the suggested minimum ({low}): the validator warns")
    log(f"[openh] splits {sp}, intrinsics for {list(cal) or 'no camera'}, README → {out / 'meta/README.md'}")


def scrub(m: dict, operators: dict) -> dict:
    """A sidecar row without what should not leave the lab: the operator's name → surgeon<N> (the public recordings' name,
    dataset/public_recordings.py --operator surgeon1), file paths → file names."""
    m = dict(m)
    if m.get("operator"):
        m["operator"] = operators.setdefault(m["operator"], f"surgeon{len(operators) + 1}")
    if m.get("recording"):
        m["recording"] = Path(m["recording"]).name
    return m


# ---------------------------------------------------------------- validation
def validate(out: Path, repo_id: str, log=print) -> dict:
    """Read the dataset back: block layout, units, action = next state, stamps on the frame grid, texts, splits."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id, root=out)
    side = [json.loads(x) for x in (out / "meta/spine_il/episodes.jsonl").read_text().splitlines() if x.strip()]
    hf = ds.hf_dataset.with_format("numpy")
    st, ac = np.asarray(hf["observation.state"], float), np.asarray(hf["action"], float)
    epi = np.asarray(hf["episode_index"]).reshape(-1)
    ph = np.asarray(hf["observation.meta.phase_index"]).reshape(-1)
    ns = np.asarray(hf["observation.meta.host_stamp_ns"]).reshape(-1).astype(np.int64)
    info = json.loads((out / "meta/info.json").read_text())
    rep = {"layout": "openh", "episodes": ds.meta.total_episodes, "frames": ds.meta.total_frames, "fps": ds.meta.fps,
           "hours": round(ds.meta.total_frames / ds.meta.fps / 3600, 4), "tasks": len(ds.meta.tasks),
           "features": {k: list(v["shape"]) for k, v in ds.meta.features.items()}, "splits": info.get("splits"),
           "state_names": info["features"]["observation.state"]["names"]}
    rep["quat_norm_err_max"] = float(max(np.abs(np.linalg.norm(st[:, a:a + 4], axis=1) - 1).max() for a in (3, 11)))
    rep["nan"] = {"state": int(np.isnan(st).sum()), "action": int(np.isnan(ac).sum())}
    rep["state_min"], rep["state_max"] = np.round(st.min(0), 4).tolist(), np.round(st.max(0), 4).tolist()
    rep["position_abs_max_m"] = float(np.abs(st[:, [0, 1, 2, 8, 9, 10]]).max())   # metres: a spine scene stays under 2
    rep["endoscope_gripper_all_zero"] = bool((st[:, 7] == 0).all() and (ac[:, 7] == 0).all())
    rep["action_is_next_state"] = bool(all(np.allclose(ac[np.flatnonzero(epi == e)[:-1]], st[np.flatnonzero(epi == e)[1:]], atol=1e-6)
                                           for e in np.unique(epi)))
    step = 1e9 / ds.meta.fps
    rep["host_stamp_step_err_ns_max"] = float(max(np.abs(np.diff(ns[epi == e]) - step).max() for e in np.unique(epi)))
    rows = []
    for m in side:
        w = epi == m["episode_index"]
        rows.append({"ep": m["episode_index"], "src": m["source"], "id": S.short_id(m["session_id"]), "split": m.get("split"),
                     "n": int(w.sum()), "duty_ds": round(float(st[w, 15].mean()), 3),
                     "duty_src": None if m["trigger_duty_source"] is None else round(m["trigger_duty_source"], 3),
                     "phase_monotonic": bool(np.all(np.diff(ph[w]) >= 0)), "phases": sorted(set(ph[w].tolist())),
                     "host_stamp0_ms": round(float(ns[w][0]) / 1e6, 3), "t_frame0_ms": m["t_frame0_ms"]})
    rep["per_episode"] = rows
    rep["duty_abs_err_max"] = max((abs(r["duty_ds"] - r["duty_src"]) for r in rows if r["duty_src"] is not None), default=None)
    rep["host_stamp0_err_ms_max"] = max(abs(r["host_stamp0_ms"] - r["t_frame0_ms"]) for r in rows)
    item = ds[0]
    rep["sample_keys"] = {k: list(getattr(v, "shape", ())) for k, v in item.items() if hasattr(v, "shape")}
    rep["sample_task"], rep["sample_instruction"] = item.get("task"), item.get("instruction.text")
    rep["sample_tool"] = {name: item.get(f"observation.meta.tool.{name}") for _, name in ARMS}
    rep["files"] = {f: (out / "meta" / f).exists() for f in ("README.md", "calibration/camera_intrinsics.json")}
    return rep
