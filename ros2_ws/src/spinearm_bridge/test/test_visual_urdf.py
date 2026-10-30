import math

import numpy as np

from spinearm_bridge.visual_urdf import _T, _Rz, jaw_zero, mount_cad, mount_rods

N = np.array([0.9944, -0.1054, 0.0])
MOUNT = {"O_stl_mm": [63.2, 107.3], "slot_stl_mm": [63.2, 97.8], "hole4_mm": [4.24, -54.38, 0.0],
         "hole3_mm": [5.31, -44.44, 0.0], "hole_axis": N.tolist(), "finger_in_x_mm": -8.3, "rod_d_mm": 1.5,
         "O_depth_mm": 12.0, "slot_depth_mm": 5.0}


def _mesh(T, p_stl_mm):
    return (T @ np.r_[np.asarray(p_stl_mm, float) / 1000.0, 1.0])[:3] * 1000.0


def test_mount_cad_assembles_block_on_the_jaw():
    T = mount_cad(MOUNT)
    n = N / np.linalg.norm(N)
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3)) and abs(np.linalg.det(T[:3, :3]) - 1) < 1e-12
    assert np.allclose(_mesh(T, [63.2, 107.3, 0]), MOUNT["hole4_mm"])              # O on hole #4
    assert np.allclose(T[:3, 2], n)                                                  # back face flush on the hole face
    slot = _mesh(T, [63.2, 97.8, 0]) - np.array(MOUNT["hole4_mm"])
    h43 = np.subtract(MOUNT["hole3_mm"], MOUNT["hole4_mm"])
    h43 -= (h43 @ n) * n                                                             # the hole line within the face
    assert abs(slot @ n) < 1e-9 and np.allclose(np.cross(slot, h43), 0, atol=1e-9) and slot @ h43 > 0
    # a ball seat 18.4 mm above the back face stands 18.4 mm off the jaw face
    assert abs((_mesh(T, [61.1, 115.3, 18.4]) - MOUNT["hole4_mm"]) @ n - 18.4) < 1e-9


def test_rods_run_on_the_hole_axes_through_the_block():
    n = N / np.linalg.norm(N)
    for (T, r, length), hole, depth in zip(mount_rods(MOUNT), ("hole4_mm", "hole3_mm"), (12.0, 5.0)):
        h = np.array(MOUNT[hole])
        c, axis = T[:3, 3] * 1000.0, T[:3, 2]
        assert abs(r - 0.00075) < 1e-12 and np.allclose(axis, n)
        assert np.linalg.norm(np.cross(c - h, n)) < 1e-9                             # centred on the hole axis
        top, bottom = c + axis * length * 500.0, c - axis * length * 500.0
        assert abs((top - h) @ n - depth) < 1e-9 and abs(bottom[0] + 8.3) < 1e-9


def test_jaw_zero_recovers_the_turn():
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 3))
    for deg in (51.6, -20.0):
        B = A @ _Rz(math.radians(deg))[:3, :3].T
        assert abs(math.degrees(jaw_zero(A, B)) - deg) < 1e-9


def test_seat_turns_about_O_and_slides_in_the_block_plane():
    flat = mount_cad(MOUNT)
    seated = mount_cad({**MOUNT, "seat": {"turn_deg": -4.87, "slide_mm": [0.38, -1.96]}})
    O = [63.2, 107.3, 0]
    moved = _mesh(seated, O) - _mesh(flat, O)                       # O slides by exactly the slide, in the face
    assert abs(np.linalg.norm(moved) - math.hypot(0.38, -1.96)) < 1e-9
    n = N / np.linalg.norm(N)
    assert abs(moved @ n) < 1e-9 and np.allclose(seated[:3, 2], flat[:3, 2])          # still flush
    turned = mount_cad({**MOUNT, "seat": {"turn_deg": 10.0, "slide_mm": [0, 0]}})
    assert np.allclose(_mesh(turned, O), MOUNT["hole4_mm"])                        # a pure turn keeps O on hole #4
    c = float(turned[:3, 0] @ flat[:3, 0])
    assert abs(math.degrees(math.acos(c)) - 10.0) < 1e-9
    assert np.allclose(mount_cad({**MOUNT, "seat": {"turn_deg": 5}}, seated=False), flat)


def test_model_zero_turns_the_joint_in_the_model_only():
    import xml.etree.ElementTree as ET
    from pathlib import Path
    from spinearm_bridge.visual_urdf import build
    kin = Path(__import__("os").environ.get("SPINE_IL_REPO", str(Path.home() / "spine-il"))) / "robot/lerobot_robot_spinearm/lerobot_robot_spinearm/assets/so101_marker.urdf"
    vis = Path(__import__("os").environ.get("SPINE_IL_REPO", str(Path.home() / "spine-il"))) / "ros2_ws/src/spinearm_description/urdf/so101_new_calib.urdf"
    if not (kin.exists() and vis.exists()):
        return
    a, _ = build(kin, vis)
    b, _ = build(kin, vis, zero_deg={"elbow_flex": 5.0})
    ja = {j.get("name"): j for j in ET.fromstring(a).findall("joint")}
    jb = {j.get("name"): j for j in ET.fromstring(b).findall("joint")}
    Ta, Tb = _T(ja["elbow_flex"].find("origin")), _T(jb["elbow_flex"].find("origin"))
    assert np.allclose(Ta[:3, 3], Tb[:3, 3])                                       # same place, turned about the axis
    ang = math.degrees(math.acos(max(-1, min(1, (np.trace(Ta[:3, :3].T @ Tb[:3, :3]) - 1) / 2))))
    assert abs(ang - 5.0) < 1e-3                                                   # rpy written to 6 digits
    assert ET.tostring(ja["wrist_flex"]) == ET.tostring(jb["wrist_flex"])
