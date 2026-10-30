"""Input normalisation: authored trajectories and deltas (recordings are replayed by the viewer, not normalized) —
the geometry they use checked against what the rig recorded."""

import glob
import json
import os

import numpy as np
import pytest

from recsynth import make_session

from spine_render import from_delta, from_trajectory, integrate, normalize
from spine_render.geom import (Registration, endo_roll_deg, matrix_to_quat, pose_from_tip_u_roll, quat_to_matrix,
                               rotvec_to_matrix, unit)
from spine_render.messages import capture_grid, messages
from spine_render.script import Script, rig_default

SESSIONS_ROOT = os.environ.get("SPINE_REC_SESSIONS", os.path.expanduser("~/il-robot/viewer-data/sessions"))  # must be inside RENDER_API_ALLOWED_ROOTS
SESSIONS = sorted(glob.glob(os.path.join(SESSIONS_ROOT, "2026-09-23", "*.jsonl")))
needs_sessions = pytest.mark.skipif(not SESSIONS, reason="rig recordings not on this machine")
RIG = rig_default()
REG = Registration.from_dict(RIG["registration"])


def test_quat_matrix_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        R = rotvec_to_matrix(rng.normal(size=3))
        assert np.allclose(quat_to_matrix(matrix_to_quat(R)), R, atol=1e-12)


@pytest.mark.parametrize("tool", ["endo", "drill"])
def test_tip_u_roll_inverse_roundtrip(tool):
    """Any real pose → (tip, u, roll) → inverse → the same tip, u and roll."""
    calib = RIG["tools"][tool]
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(300):
        R_true = rotvec_to_matrix(rng.normal(size=3))
        tip = REG.to_scene([0, 0, 450]) + rng.normal(scale=40, size=3)
        pos_true = tip - R_true @ calib["tipLocal"]
        u = R_true @ calib["shaftLocal"]
        roll = endo_roll_deg(R_true, pos_true, u, REG.cam_centre)
        pos, R, err = pose_from_tip_u_roll(tip, u, roll, calib["tipLocal"], calib["shaftLocal"], REG.cam_centre)
        assert np.allclose(R @ calib["shaftLocal"], u, atol=1e-9)
        assert np.allclose(pos + R @ calib["tipLocal"], tip, atol=1e-9)
        worst = max(worst, err, abs(((endo_roll_deg(R, pos, u, REG.cam_centre) - roll) + 180) % 360 - 180))
    assert worst < 1e-4


def test_unreachable_roll_is_reported():
    """The marker is one-sided: rolling the tool 180° about its shaft turns it away from the camera, so for a
    given shaft direction only a 180° window of rolls exists. The opposite roll must come back as a warning."""
    calib = RIG["tools"]["endo"]
    tip = REG.to_scene([0, 0, 450])
    u = unit([-0.656, -0.683, -0.321])
    reachable = [r for r in range(-180, 180, 15)
                 if pose_from_tip_u_roll(tip, u, r, calib["tipLocal"], calib["shaftLocal"], REG.cam_centre)[2] < 1e-6]
    assert 9 <= len(reachable) <= 15  # about half of the circle
    bad = next(r for r in range(-180, 180, 15) if r not in reachable)
    traj = from_trajectory({"frames": [{"t": 0, "endo": {"tip": list(tip), "u": list(u), "roll": bad}}]})
    assert traj.source["warnings"] and "roll" in traj.source["warnings"][0]


@needs_sessions
@pytest.mark.parametrize("path", SESSIONS)
def test_viewer_geometry_matches_recordings(path):
    """What authored inputs rely on, checked on rig recordings (read directly — they are not normalized): T0·tip ==
    scene.tip, the viewer roll / view direction from the recorded pose == the drive's `real`, and the rig's
    shaftLocal == the one the recorded poses imply."""
    rows = [json.loads(line) for line in open(path)]
    h = next(r for r in rows if r.get("k") == "hdr")
    reg = Registration.from_dict((h.get("registration") or {}).get("T0") or RIG["registration"])
    n, worst, loc = 0, 0.0, {}
    for r in rows:
        if r.get("k") != "f" or r.get("lost") or "quat" not in r:
            continue
        R = quat_to_matrix(r["quat"])
        if "scene" in r and "tip" in r:
            worst = max(worst, float(np.linalg.norm(reg.to_scene(r["tip"]) - np.asarray(r["scene"]["tip"]))))
        if "shaft" in r:
            loc.setdefault(r["tool"], []).append(R.T @ np.asarray(r["shaft"]))
        if r["tool"] == "endo" and "real" in r and "shaft" in r:
            u = unit(reg.R @ np.asarray(r["shaft"]))
            roll = endo_roll_deg(reg.R @ R, reg.to_scene(r["pos"]), u, reg.cam_centre)
            assert abs(((roll - r["real"]["roll"]) + 180) % 360 - 180) < 1e-3
            assert np.allclose(u, r["real"]["viewDir"], atol=1e-4)
            n += 1
    assert worst < 1e-3  # 3-dp rounding
    assert n > 100
    for tool, v in loc.items():
        assert np.allclose(unit(np.median(np.asarray(v), axis=0)), RIG["tools"][tool]["shaftLocal"], atol=2e-4)


def test_trajectory_resample_and_messages():
    traj = {"format": "spine-render-trajectory", "version": 1, "case": "altician_lumbar1", "feed_hz": 90, "frames": [
        {"t": 0.0, "endo": {"tip": [30, 139, 1052], "u": [-0.656, -0.683, -0.321], "roll": 10},
         "drill": {"tip": [38, 93, 975], "u": [-0.476, -0.303, 0.826], "roll": 0, "trig": False}},
        {"t": 1.0, "endo": {"tip": [30, 139, 1062], "u": [-0.656, -0.683, -0.321], "roll": 40},
         "drill": {"tip": [38, 93, 985], "u": [-0.476, -0.303, 0.826], "roll": 0, "trig": True}},
        {"t": 1.5, "endo": {"lost": True}},
    ]}
    script = from_trajectory(traj)
    endo = script.by_tool("endo")
    assert len(endo) == 90 + 45 + 1 and endo[-1].lost and endo[-2].lost and not endo[89].lost
    mid = endo[45]  # t = 0.5 s → halfway in tip and roll
    reg = script.registration
    tip_s = reg.to_scene(mid.tip)
    assert np.allclose(tip_s, [30, 139, 1057], atol=1e-6)
    R_s = reg.R @ quat_to_matrix(mid.quat)
    assert abs(endo_roll_deg(R_s, reg.to_scene(mid.pos), unit(reg.R @ np.asarray(mid.shaft)), reg.cam_centre) - 25) < 0.05
    assert script.by_tool("drill")[-1].trig is True and script.by_tool("drill")[0].trig is False
    msgs = [m for _, m in messages(script)]
    assert {m["type"] for m in msgs} == {"pose", "pose_lost"}
    assert all(m["v"] == 1 and "buttons" in m for m in msgs)
    assert len(capture_grid(script, 30)) == 46  # 0 … 1.5 s


def test_delta_tool_and_scene_frames():
    start = {"endo": {"tip": [30, 139, 1052], "u": [-0.656, -0.683, -0.321], "roll": 0}}
    fwd = {"format": "spine-render-delta", "frame": "tool", "dt_ms": 100, "start": start,
           "deltas": [{"endo": {"dpos": [0, 0, 0], "drot_deg": [0, 0, 0]}}] * 2}
    script, traj = from_delta(fwd)
    assert len(traj["frames"]) == 3 and traj["frames"][-1]["t"] == pytest.approx(0.2)
    # a tool-frame translation along shaftLocal advances the tip along u by the same distance
    shaft = RIG["tools"]["endo"]["shaftLocal"]
    adv = integrate({"frame": "tool", "start": start, "deltas": [{"endo": {"dpos": list(np.asarray(shaft) * 5)}}]})
    d = np.asarray(adv["frames"][1]["endo"]["tip"]) - np.asarray(adv["frames"][0]["endo"]["tip"])
    assert np.allclose(d, unit([-0.656, -0.683, -0.321]) * 5, atol=1e-3)
    # scene-frame rotation turns about the tip: tip stays, orientation changes
    rot = integrate({"frame": "scene", "start": start, "deltas": [{"endo": {"drot_deg": [0, 0, 30]}}]})
    assert np.allclose(rot["frames"][1]["endo"]["tip"], rot["frames"][0]["endo"]["tip"])
    q0, q1 = (quat_to_matrix(f["endo"]["quat"]) for f in rot["frames"])
    assert np.allclose(q1, rotvec_to_matrix(np.radians([0, 0, 30])) @ q0, atol=1e-9)


def test_normalize_autodetect_and_script_roundtrip(tmp_path):
    script, traj = normalize({"frames": [{"t": 0, "endo": {"tip": [0, 0, 1000], "u": [0, 0, 1], "roll": 0}},
                                         {"t": 0.2, "endo": {"tip": [0, 0, 1010], "u": [0, 0, 1], "roll": 0}}]})
    assert traj is None and script.source["type"] == "trajectory"
    out = tmp_path / "s.jsonl"
    script.dump(out)
    again, _ = normalize(str(out))
    assert len(again.samples) == len(script.samples) and again.tools.keys() == script.tools.keys()
    assert len(Script.load(out).samples) == len(script.samples)
    d, traj = normalize({"start": {"endo": {"tip": [0, 0, 1000], "u": [0, 0, 1]}}, "deltas": []})
    assert d.source["type"] == "delta" and traj["format"] == "spine-render-trajectory"


def test_recordings_are_not_normalized(tmp_path):
    """A recorder session is handed to the viewer as is: normalize() refuses it, and a script an older version
    converted from a recording too."""
    rows = make_session(n=5)
    for src in (rows, "".join(json.dumps(r) + "\n" for r in rows)):
        with pytest.raises(ValueError, match="not converted in Python"):
            normalize(src)
    old = tmp_path / "old.jsonl"
    old.write_text(json.dumps({"k": "hdr", "kind": "spine-render-script", "version": 1, "case": "altician_lumbar1",
                               "registration": REG.to_dict(), "tools": RIG["tools"], "source": {"type": "recording"}})
                   + "\n" + json.dumps({"k": "s", "t": 0, "tool": "endo", "lost": 1}) + "\n")
    with pytest.raises(ValueError, match="converted from a recording"):
        normalize(str(old))
