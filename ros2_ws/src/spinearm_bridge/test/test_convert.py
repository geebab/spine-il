import math

import numpy as np

from spinearm_bridge.convert import R_to_xyzw, split_stamp_ms, state_to_fields, tap_to_fields


def test_tap_units_and_stamp():
    m = {"tool": "endo", "fn": 7, "ts": 1790000000123.5, "ta": 1790000000133.0, "tp": 1790000000140.0,
         "tsend": 1790000000141.0, "pos": [100.0, -20.0, 500.0], "quat": [1, 0, 0, 0], "raw_pos": [101, -20, 500],
         "raw_quat": [0, 1, 0, 0], "tip": [0, 0, 666.0], "raw_tip": [1, 0, 666], "vis": 4, "rmse": 0.3}
    f = tap_to_fields(m)
    assert f["stamp"] == 1790000000.1235 and f["frame"] == 7 and not f["lost"]
    assert f["pos"] == [0.1, -0.02, 0.5] and f["quat"] == (0.0, 0.0, 0.0, 1.0) and f["quat_raw"] == (1.0, 0.0, 0.0, 0.0)
    assert f["tip"] == [0.0, 0.0, 0.666]
    assert split_stamp_ms(f["stamp_ms"]) == (1790000000, 123500000)
    assert tap_to_fields({"tool": "endo2", "fn": 8, "ts": 1.0, "lost": 1})["lost"]


def test_state_offsets_and_clock():
    s = {"t": 100.0, "tr": 100.002, "tw": 100.004, "gseq": 5, "grx": 99.99, "tick": 3, "state": "MOVING",
         "pos": {"shoulder_pan": 10.0, "shoulder_lift": -60.0, "elbow_flex": 50.0, "wrist_flex": 70.0,
                 "wrist_roll": 0.0, "gripper": -40.0}, "cmd": None, "goal": None, "load": {}, "temp": {}}
    f = state_to_fields(s, 1000.0, [0, 5.0, -7.5, 2.3, 2.6, 0])
    assert abs(f["stamp"] - 1100.001) < 1e-9 and abs(f["t_write"] - 1100.004) < 1e-9 and f["goal_seq"] == 5
    assert abs(f["joint_rad"][1] - math.radians(-55.0)) < 1e-12 and math.isnan(f["command"][0])


def test_rotation_to_quaternion():
    for ang in (0.3, 2.9, math.pi):
        c, s = math.cos(ang), math.sin(ang)
        x, y, z, w = R_to_xyzw(np.array([[1, 0, 0], [0, c, -s], [0, s, c]]))
        assert abs(abs(w) - math.cos(ang / 2)) < 1e-9 and abs(abs(x) - math.sin(ang / 2)) < 1e-9


def test_command_is_inverse_of_state():
    from spinearm_bridge.convert import JOINTS, command_to_goal
    off = [1.0, -7.9, 2.5, 0.0, 0.0, 3.0]
    arm = [10.0, -60.0, 50.0, 70.0, 0.0, -40.0]
    rad = [math.radians(a + o) for a, o in zip(arm, off)]   # what joint_states publishes
    got = command_to_goal(JOINTS, rad, off)
    assert all(abs(got[j] - a) < 1e-9 for j, a in zip(JOINTS, arm))
    assert command_to_goal(["elbow_flex"], [0.0], off) == {"elbow_flex": -2.5}   # subset allowed
    for bad in ((["nope"], [0.0]), (["elbow_flex"], [float("nan")]), (["elbow_flex", "gripper"], [0.0])):
        try:
            command_to_goal(*bad, off)
        except ValueError:
            continue
        raise AssertionError(bad)


def test_visual_urdf_marker_and_arm():
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from spinearm_bridge.visual_urdf import build, match_balls, _rpy
    # rpy round trip
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        r, p, yw = _rpy(R)
        Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
        Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
        Rz = np.array([[math.cos(yw), -math.sin(yw), 0], [math.sin(yw), math.cos(yw), 0], [0, 0, 1]])
        assert np.allclose(Rz @ Ry @ Rx, R, atol=1e-9)
    # ball matching recovers a known pose and a shuffled correspondence
    src = np.array([[61.1, 115.3, 19.0], [21.9, 6.0, 19.0], [6.0, 63.0, 19.0], [33.0, 150.1, 19.0]])
    a = 0.7
    Rt = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    dst = (src @ Rt.T + [5.0, -3.0, 2.0])[[2, 0, 3, 1]]
    R, t, e, perm = match_balls(src, dst)
    assert e.max() < 1e-9 and perm == (1, 3, 0, 2) and np.allclose(R, Rt)
    # a model built from the real files: every original visual lands, the marker adds 4 spheres + the block
    root = Path(__file__).resolve().parents[4]   # repo root
    kin = root / "robot/lerobot_robot_spinearm/lerobot_robot_spinearm/assets/so101_marker.urdf"
    vis = root / "ros2_ws/src/spinearm_description/urdf/so101_new_calib.urdf"
    X = np.eye(4)
    X[:3, 3] = [0.09, 0.0, 0.0]
    urdf, rep = build(kin, vis, X, dst, 7.0, "package://spinearm_description/meshes/markers/m.stl", src)
    r = ET.fromstring(urdf)
    n_orig = sum(len(ln.findall("visual")) for ln in ET.parse(vis).getroot().findall("link"))
    mk = [ln for ln in r.findall("link") if ln.get("name") == "marker_frame_link"][0]
    assert sum(len(ln.findall("visual")) for ln in r.findall("link")) == n_orig + 5
    assert len(mk.findall("visual/geometry/sphere")) == 4 and len(mk.findall("visual/geometry/mesh")) == 1
    assert all(m.get("filename").startswith("package://spinearm_description/meshes/")
               for m in r.iter("mesh"))
    sph = [np.array(list(map(float, v.find("origin").get("xyz").split()))) for v in mk.findall("visual")
           if v.find("geometry/sphere") is not None]
    assert np.allclose(sorted(s[0] for s in sph), sorted(0.09 + dst[:, 0] / 1000.0))
    assert rep["residual_mm"] and max(rep["residual_mm"]) < 1e-6


def test_visual_urdf_link_prefix():
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from spinearm_bridge.visual_urdf import build
    root = Path(__file__).resolve().parents[4]
    kin = root / "robot/lerobot_robot_spinearm/lerobot_robot_spinearm/assets/so101_marker.urdf"
    vis = root / "ros2_ws/src/spinearm_description/urdf/so101_new_calib.urdf"
    r = ET.fromstring(build(kin, vis, np.eye(4), np.zeros((4, 3)) + [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
                            link_prefix="left_")[0])
    names = {ln.get("name") for ln in r.findall("link")}
    assert "left_base_link" in names and "left_marker_frame_link" in names and all(n.startswith("left_") for n in names)
    assert all(j.find(t).get("link") in names for j in r.findall("joint") for t in ("parent", "child"))
    mk = [ln for ln in r.findall("link") if ln.get("name") == "left_marker_frame_link"][0]
    assert len(mk.findall("visual/geometry/sphere")) == 4
