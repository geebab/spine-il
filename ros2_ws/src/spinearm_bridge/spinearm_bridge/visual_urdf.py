"""Robot model with visuals, assembled from calibration at launch (ROS2-1-design.md §3.3).

    kinematic URDF (so101_marker.urdf — what FK/IK and the hand-eye fit use)
  + SO-101 visuals copied link by link from TheRobotStudio's so101_new_calib.urdf (same link frames)
  + on marker_frame_link: the marker block mesh and its 4 balls, placed by the arm's calibration:
        marker_frame_link ←X— constellation frame (tracker tool calib, points_local) ←T— block STL frame
    X comes from the hand-eye / kincal JSON; T is fitted here by matching the block's ball seats (STL frame, from
    ~/il-robot/calib/markers.json) to the tool calibration's measured ball centres.
  + with a mount (markers.json "mount") the marker is instead drawn as assembled, from the drawings only: the block
    flush on the moving jaw's hole face, its reference O on jaw hole #4, the rod from hole #3 in the slot behind its
    arrow, the balls on their seats. Only the jaw's zero (joint 6, not in any drawing) is taken from the calibration:
    the angle that brings the drawn balls closest to the calibrated ones. What is left between the two is the
    model-vs-measurement error to find (report "mount").

The marker is drawn on marker_frame_link (not as extra links), so it cannot collide with calib_tf's static frames.
Everything is plain numpy + ElementTree — testable without ROS.
"""

from __future__ import annotations

import itertools
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

PKG = "package://spinearm_description"
COLORS = {"3d_printed": "0.46 0.33 0.86 1", "sts3215": "0.1 0.1 0.1 1",   # the arms are printed purple
          "marker_block": "0.18 0.18 0.2 1", "marker_ball": "0.92 0.92 0.92 1", "steel": "0.62 0.64 0.68 1"}
JAW_LINK = "moving_jaw_so101_v1_link"


def kabsch(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """R, t with B ≈ R·A + t (proper rotation)."""
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def match_balls(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]]:
    """Rigid fit of ball set `src` onto `dst` with unknown correspondence (≤ 6 balls: all permutations).
    Returns R, t, per-ball residual, perm (src[i] ↔ dst[perm[i]]) minimising the worst residual."""
    best = None
    for perm in itertools.permutations(range(len(dst)), len(src)):
        D = dst[list(perm)]
        R, t = kabsch(src, D)
        e = np.linalg.norm(src @ R.T + t - D, axis=1)
        if best is None or e.max() < best[2].max():
            best = (R, t, e, perm)
    return best


def _rpy(R: np.ndarray) -> tuple[float, float, float]:
    """URDF rpy (fixed-axis X-Y-Z, R = Rz(y)·Ry(p)·Rx(r))."""
    p = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    if abs(math.cos(p)) > 1e-9:
        return math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])
    return math.atan2(-R[1, 2], R[1, 1]), p, 0.0


def _origin(T: np.ndarray) -> ET.Element:
    r, p, y = _rpy(T[:3, :3])
    return ET.Element("origin", xyz=" ".join(f"{v:.6g}" for v in T[:3, 3]), rpy=f"{r:.6g} {p:.6g} {y:.6g}")


def _T(origin: ET.Element | None) -> np.ndarray:
    """4×4 of a URDF <origin> (identity when absent)."""
    T = np.eye(4)
    if origin is None:
        return T
    r, p, y = (float(x) for x in origin.get("rpy", "0 0 0").split())
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    T[:3, 3] = [float(x) for x in origin.get("xyz", "0 0 0").split()]
    return T


def _Rz(a: float) -> np.ndarray:
    T = np.eye(4)
    T[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
    return T


def _unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _z_to(n: np.ndarray) -> np.ndarray:
    """Rotation taking +z onto unit n (URDF cylinders run along their z)."""
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(z, n), float(z @ n)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]) / s
    return np.eye(3) + s * K + (1 - c) * K @ K


def _visual(T: np.ndarray, geom: ET.Element, material: str) -> ET.Element:
    v = ET.Element("visual")
    v.append(_origin(T))
    g = ET.SubElement(v, "geometry")
    g.append(geom)
    ET.SubElement(v, "material", name=material)
    return v


def mount_cad(mount: dict, seated: bool = True) -> np.ndarray:
    """Block (STL frame, metres) in the jaw-mesh frame as assembled (markers.json "mount", jaw-mesh mm): back face
    flush on the jaw's hole face (normal `hole_axis` — the holes are drilled square to it), O on hole #4's face point,
    O→slot along hole #4→#3. With `seat` {turn_deg, slide_mm} (and `seated`) the block also sits the way the pins let
    it — turned about O and slid in its own plane, still flush — as fitted to the camera (fit_asbuilt.py)."""
    n = _unit(mount["hole_axis"])
    h4, h3 = np.asarray(mount["hole4_mm"], float), np.asarray(mount["hole3_mm"], float)
    u = _unit((h3 - h4) - ((h3 - h4) @ n) * n)
    O = np.r_[mount["O_stl_mm"], 0.0]
    sb = _unit(np.r_[mount["slot_stl_mm"], 0.0] - O)
    ez = np.array([0.0, 0.0, 1.0])
    T = np.eye(4)
    T[:3, :3] = np.c_[n, u, np.cross(n, u)] @ np.c_[ez, sb, np.cross(ez, sb)].T
    T[:3, 3] = (h4 - T[:3, :3] @ O) / 1000.0
    seat = mount.get("seat") if seated else None
    if seat:
        a = math.radians(float(seat.get("turn_deg", 0.0)))
        D = np.eye(4)
        D[:2, :2] = [[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]
        D[:2, 3] = (O[:2] - D[:2, :2] @ O[:2] + np.asarray(seat.get("slide_mm", (0.0, 0.0)), float)) / 1000.0
        T = T @ D
    return T


def mount_rods(mount: dict) -> list[tuple[np.ndarray, float, float]]:
    """The two rods in the jaw-mesh frame, as (T mesh←cylinder, radius m, length m): on the hole axes from the finger's
    inner face (x = finger_in_x_mm) out through the flush block — hole #4's through O to the block top (O_depth_mm),
    hole #3's to the bottom of the slot (slot_depth_mm)."""
    n = _unit(mount["hole_axis"])
    out = []
    for hole, depth in (("hole4_mm", mount.get("O_depth_mm", 10.0)), ("hole3_mm", mount.get("slot_depth_mm", 5.0))):
        h = np.asarray(mount[hole], float)
        l0, l1 = (float(mount.get("finger_in_x_mm", -8.3)) - h[0]) / n[0], float(depth)
        T = np.eye(4)
        T[:3, :3] = _z_to(n)
        T[:3, 3] = (h + n * (l0 + l1) / 2) / 1000.0
        out.append((T, float(mount.get("rod_d_mm", 1.5)) / 2000.0, (l1 - l0) / 1000.0))
    return out


def jaw_zero(A: np.ndarray, B: np.ndarray) -> float:
    """Turn about the jaw link's z (joint 6) that best maps points A onto B (jaw-link frame, least squares)."""
    return math.atan2(float((A[:, 0] * B[:, 1] - A[:, 1] * B[:, 0]).sum()), float((A[:, :2] * B[:, :2]).sum()))


def marker_visuals(X: np.ndarray, balls_const_mm: np.ndarray, radius_mm: float,
                   mesh_uri: str | None = None, balls_stl_mm: np.ndarray | None = None) -> tuple[list[ET.Element], dict]:
    """Visuals for marker_frame_link. X = T_link←const (m). Balls at their calibrated centres; the block mesh (if
    given) placed by fitting its ball seats (STL mm) onto the calibrated centres. Returns (visuals, fit report)."""
    out, report = [], {}
    for p in np.asarray(balls_const_mm, float):
        T = np.eye(4)
        T[:3, 3] = (X @ np.r_[p / 1000.0, 1.0])[:3]
        out.append(_visual(T, ET.Element("sphere", radius=f"{radius_mm / 1000.0:.6g}"), "marker_ball"))
    if mesh_uri and balls_stl_mm is not None:
        R, t, e, perm = match_balls(np.asarray(balls_stl_mm, float), np.asarray(balls_const_mm, float))
        T_const_stl = np.eye(4)
        T_const_stl[:3, :3], T_const_stl[:3, 3] = R, t / 1000.0     # mm → m (the mesh is scaled by 0.001)
        out.append(_visual(X @ T_const_stl, ET.Element("mesh", filename=mesh_uri, scale="0.001 0.001 0.001"),
                           "marker_block"))
        report = {"residual_mm": [round(float(v), 3) for v in e], "perm": list(perm)}
    return out, report


def build(kin_urdf: str | Path, visuals_urdf: str | Path, X: np.ndarray | None = None,
          balls_const_mm=None, radius_mm: float = 7.0, mesh_uri: str | None = None, balls_stl_mm=None,
          colors: dict | None = None, link_prefix: str = "", jaw_deg: float = 0.0,
          mount: dict | None = None, zero_deg: dict | None = None) -> tuple[str, dict]:
    """URDF text: kinematic model + SO-101 visuals + the marker.
    link_prefix renames every link (e.g. "left_") so the model's link names equal the TF frame names — a viewer then
    needs no frame-prefix setting (robot_state_publisher must then run with frame_prefix "").
    With `mount` (and the ball seats) the marker is drawn as assembled on the moving jaw — block, balls on their seats,
    the two rods — and the jaw's zero about joint 6 (not in any drawing, and not observable in the calibration either:
    X absorbs it) is the turn that brings those balls closest to the calibrated ones (X, balls_const_mm); report
    "mount" gives it and the per-ball gaps that remain. Without `mount`: balls at the calibrated centres on
    marker_frame_link, the block fitted to them, the jaw turned by jaw_deg. Kinematics are untouched either way.
    zero_deg {joint: deg} turns those joints' zeros in this model only (the as-built fit's joint zeros; the arm bridge
    keeps converting commands with the calibration's own offsets)."""
    robot = ET.parse(kin_urdf).getroot()
    src = ET.parse(visuals_urdf).getroot()
    for j in robot.findall("joint"):
        dz = (zero_deg or {}).get(j.get("name"))
        if dz:
            ax = _unit([float(v) for v in j.find("axis").get("xyz").split()])
            a_ = math.radians(float(dz))
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            Rj_ = np.eye(4)
            Rj_[:3, :3] = np.eye(3) + math.sin(a_) * K + (1 - math.cos(a_)) * K @ K
            o = j.find("origin")
            T_new = _T(o) @ Rj_                                   # = the joint turned by dz at zero
            if o is not None:
                j.remove(o)
            j.insert(0, _origin(T_new))
    colors = {**COLORS, **(colors or {})}
    for m in robot.findall("material"):
        robot.remove(m)
    for i, (name, rgba) in enumerate(colors.items()):
        m = ET.Element("material", name=name)
        ET.SubElement(m, "color", rgba=rgba)
        robot.insert(i, m)
    vis = {ln.get("name"): ln.findall("visual") for ln in src.findall("link")}
    links = {ln.get("name"): ln for ln in robot.findall("link")}
    report, on_jaw = {}, []                                    # on_jaw: (T jaw-mesh ← visual, geometry, material)
    V_jaw = _T(vis[JAW_LINK][0].find("origin")) if vis.get(JAW_LINK) else np.eye(4)
    cad = bool(mount) and balls_stl_mm is not None
    if cad:
        T_ms = mount_cad(mount)
        seats = np.asarray(balls_stl_mm, float)
        if mount.get("balls") == "tracker" and balls_const_mm is not None:
            # balls where they really sit on the block: the tracker's own constellation, laid on the seats (a ball
            # standing off its ring, like the drill's D3 +2.3 mm, is drawn standing off)
            const = np.asarray(balls_const_mm, float)
            R_, t_, _, perm_ = match_balls(seats, const)
            seats = (const[list(perm_)] - t_) @ R_                              # const → block STL frame, seat order
        seats = seats + np.array([0.0, 0.0, float(mount.get("ball_lift_mm", 0.0))])   # balls standing off the block
        P = (T_ms @ np.c_[seats / 1000.0, np.ones(len(seats))].T).T[:, :3]     # balls on their seats, jaw-mesh frame
        if X is not None and balls_const_mm is not None:
            const = np.asarray(balls_const_mm, float)
            _, _, e, perm = match_balls(seats, const)
            F_m = _T(next(j for j in robot.findall("joint") if j.get("name") == "marker_frame_joint").find("origin"))
            A = P @ V_jaw[:3, :3].T + V_jaw[:3, 3]
            B = (F_m @ np.asarray(X, float) @ np.c_[const[list(perm)] / 1000.0, np.ones(len(const))].T).T[:, :3]
            jaw_deg = float(mount["jaw_deg"]) if "jaw_deg" in mount else math.degrees(jaw_zero(A, B))
            gap = np.linalg.norm(A @ _Rz(math.radians(jaw_deg))[:3, :3].T - B, axis=1) * 1000.0
            report["mount"] = {"jaw_deg": round(jaw_deg, 2), "jaw_from": "markers.json" if "jaw_deg" in mount else
                               "fit to the calibrated balls", "seat": mount.get("seat"),
                               "ball_gap_mm": [round(float(g), 2) for g in gap],
                               "gap_rms_mm": round(float(np.sqrt((gap ** 2).mean())), 2),
                               "seat_residual_mm": [round(float(v), 3) for v in e], "perm": list(perm)}
        if mesh_uri:
            on_jaw.append((T_ms, ET.Element("mesh", filename=mesh_uri, scale="0.001 0.001 0.001"), "marker_block"))
        for p in P:
            T = np.eye(4)
            T[:3, 3] = p
            on_jaw.append((T, ET.Element("sphere", radius=f"{radius_mm / 1000.0:.6g}"), "marker_ball"))
        for T, r, length in mount_rods(mount):
            on_jaw.append((T, ET.Element("cylinder", radius=f"{r:.6g}", length=f"{length:.6g}"), "steel"))
    Rj = _Rz(math.radians(jaw_deg))
    for name, vs in vis.items():
        if name not in links:
            continue
        for v in vs:
            for mesh in v.iter("mesh"):
                mesh.set("filename", f"{PKG}/meshes/so101/{Path(mesh.get('filename')).name}")
            if jaw_deg and name == JAW_LINK:
                o = v.find("origin")
                T = _T(o)
                if o is not None:
                    v.remove(o)
                v.insert(0, _origin(Rj @ T))
            links[name].append(v)
    for T, geom, material in on_jaw:
        links[JAW_LINK].append(_visual(Rj @ V_jaw @ T, geom, material))
    if not cad and X is not None and balls_const_mm is not None and "marker_frame_link" in links:
        mv, rep = marker_visuals(np.asarray(X, float), np.asarray(balls_const_mm, float), radius_mm,
                                 mesh_uri, balls_stl_mm)
        report.update(rep)
        for v in mv:
            links["marker_frame_link"].append(v)
    if link_prefix:
        for ln in robot.findall("link"):
            ln.set("name", link_prefix + ln.get("name"))
        for j in robot.findall("joint"):
            for tag in ("parent", "child"):
                e = j.find(tag)
                e.set("link", link_prefix + e.get("link"))
    return ET.tostring(robot, encoding="unicode"), report


def build_for_arm(arm: str, kin_urdf: str | Path, visuals_urdf: str | Path, calib_path: str | Path,
                  markers_path: str | Path, mesh_dir: str | Path, link_prefix: str = "") -> tuple[str, dict]:
    """Launch helper: read ~/il-robot/calib/markers.json {arm: {stl, tool_calib, balls_stl_mm, mount}} and the arm's
    calibration; fall back to arm visuals only when either is missing (the report says why)."""
    markers = json.loads(Path(markers_path).expanduser().read_text()) if Path(markers_path).expanduser().exists() else {}
    m = markers.get(arm)
    calib = Path(calib_path).expanduser()
    if not m or not calib.exists():
        urdf, _ = build(kin_urdf, visuals_urdf, link_prefix=link_prefix)
        return urdf, {"marker": None, "why": "no marker entry" if not m else f"no calibration {calib}"}
    tool = json.loads(Path(m["tool_calib"]).expanduser().read_text())
    cal = json.loads(calib.read_text())
    X = np.asarray(cal["T_link_const"], float)
    stl = Path(mesh_dir) / m["stl"]
    uri = f"{PKG}/meshes/markers/{m['stl']}" if stl.exists() else None
    urdf, rep = build(kin_urdf, visuals_urdf, X, tool["points_local"], float(tool.get("sphere_radius_mm", 7.0)),
                      uri, m.get("balls_stl_mm"), link_prefix=link_prefix, jaw_deg=float(m.get("jaw_visual_deg", 0.0)),
                      mount=m.get("mount"), zero_deg=cal.get("model_zero_deg"))
    return urdf, {"marker": m["stl"] if uri else None, "fit": rep,
                  **({} if uri else {"why": f"marker mesh not installed: {stl}"})}
