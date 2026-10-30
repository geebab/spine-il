"""Safe zone for moving both arms while the tracker watches them (2026-09-26, user definitions).

safe    : all 4 balls of each arm's marker are seen by both IR imagers — inside the image with a margin, within the
          depth range the tracker has tracked, facing the camera within `max_view_deg`, no line of sight blocked by
          either robot — and no part of either robot model (marker block, balls and rods included) comes near the
          floor, the camera, the other arm or itself. Moves are straight lines in joint space checked point by point,
          so the way back along the same line is safe too.
optimal : safe, and all 4 balls 35–45 cm from the camera (the tracker's working distance, "target 35-45cm").

The geometry is the viewer's robot model (spinearm_bridge.visual_urdf with the markers.json mount), each arm placed
by its own calibration (base ← camera, joint offsets). Plain numpy; the planner/runner is robot/verify_two_arm.py.
"""

from __future__ import annotations

import math
import os
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
MARKER_MESHES = ("tool_scope_A.stl", "tool_drill_B.stl")


def _complete(p: Path) -> bool:
    """A model is usable only with the marker-block meshes: without them build() silently drops the block visual and
    the safe zone loses that collision volume (2026-09-27). The team share carries bounding-box stand-ins."""
    return (p / "urdf/so101_new_calib.urdf").is_file() and all((p / "meshes/markers" / m).is_file() for m in MARKER_MESHES)


def _find_share() -> Path:
    """The robot model (URDF + meshes): $SPINEARM_DESCRIPTION, then the owner's colcon install (exact marker designs),
    then the team share (/srv/il-robot/description — same geometry, marker blocks as bounding-box stand-ins).
    Teammates cannot read the owner's home, so they land on the share without building ros2_ws."""
    owner = Path(__import__("os").environ.get("SPINE_IL_REPO", str(Path.home() / "spine-il"))) / "ros2_ws/install/spinearm_description/share/spinearm_description"
    for c in (os.environ.get("SPINEARM_DESCRIPTION"), owner, "/srv/il-robot/description"):
        if c and _complete(Path(c)):
            return Path(c)
    return owner  # not found: build() raises below (importing this module must not fail)


SHARE = _find_share()
GROUPS = {"base_link": "base", "shoulder_link": "shoulder", "upper_arm_link": "upper", "lower_arm_link": "lower",
          "wrist_link": "wrist", "gripper_link": "gripper", "moving_jaw_so101_v1_link": "jaw"}
MARKER_MATERIALS = ("marker_block", "marker_ball")   # drawn on the jaw link, but a group of their own (the rods sit in
                                                     # the jaw's holes: they count as jaw)
# non-adjacent part pairs that can meet when the arm folds (adjacent links touch at their joints by design)
SELF_PAIRS = ([("marker", g) for g in ("base", "shoulder", "upper", "lower", "wrist", "gripper")]
              + [("jaw", g) for g in ("base", "shoulder", "upper", "lower")]
              + [("gripper", g) for g in ("base", "shoulder", "upper")] + [("wrist", "base"), ("wrist", "shoulder"),
                                                                            ("lower", "base")])
FLOOR_GROUPS = ("upper", "lower", "wrist", "gripper", "jaw", "marker")


@dataclass
class Limits:
    floor_mm: float = 25.0          # model points above each arm's base plane (the table)
    self_mm: float = 8.0            # SELF_PAIRS apart (or no closer than at the start pose, if that is closer)
    arms_mm: float = 40.0           # the two arms apart (calibration + lag margin)
    camera_mm: float = 150.0        # model points from the camera
    margin_px: float = 15.0         # ball centres inside both IR images by this much (a ball is ~8-10 px in radius)
    window_px: tuple = (270.0, 575.0, 80.0, 350.0)  # u0, u1, v0, v1 (left IR): where balls are actually detected —
    # 790k tracked frames (2026-09-27): ~100 % inside, falling to 0-44 % below v 400, left of u 265 and right of u 580,
    # where the balls turn dim; a geometric view alone is not enough
    apart_px: float = 30.0          # the markers' image boxes this far apart (two constellations side by side → the
                                    # tracker loses both, 2026-09-27)
    z_mm: tuple = (280.0, 620.0)    # ball depth the tracker has tracked (home is ~57 cm)
    max_view_deg: float = 40.0      # losses per frame x7 beyond 40° (760k tracked frames, 2026-09-26)
    sep_px: float = 20.0            # any two balls apart in both images (no merged blobs)
    los_mm: float = 7.0             # line of sight to a ball clear of other geometry by the ball radius
    optimal_z_mm: tuple = (350.0, 450.0)
    target_margin_px: float = 50.0  # targets (strict) sit well inside the zone, so moves between them can stay in it
    target_window_px: float = 20.0
    target_view_deg: float = 32.0


def _stl(path: Path) -> np.ndarray:
    d = path.read_bytes()
    n = struct.unpack("<I", d[80:84])[0]
    a = np.frombuffer(d[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3, 3).astype(float)


def _voxel(P: np.ndarray, step: float) -> np.ndarray:
    _, i = np.unique(np.floor(P / step).astype(np.int64), axis=0, return_index=True)
    return P[np.sort(i)]


def _surface(tris: np.ndarray, step: float) -> np.ndarray:
    """Points covering a triangle soup every ~step (vertices + area-weighted samples, voxel-thinned)."""
    area = np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1) / 2
    k = np.minimum(np.ceil(area / (step * step / 4)).astype(int), 400)
    rng = np.random.default_rng(0)
    idx = np.repeat(np.arange(len(tris)), k)
    r1, r2 = rng.random(len(idx)), rng.random(len(idx))
    s = np.sqrt(r1)
    P = (1 - s)[:, None] * tris[idx, 0] + (s * (1 - r2))[:, None] * tris[idx, 1] + (s * r2)[:, None] * tris[idx, 2]
    return _voxel(np.vstack([tris.reshape(-1, 3), P]), step)


def _sphere(r: float, step: float) -> np.ndarray:
    n = max(24, int(4 * math.pi * r * r / (step * step / 2)))
    i = np.arange(n) + 0.5
    phi, th = np.arccos(1 - 2 * i / n), math.pi * (1 + 5 ** 0.5) * i
    return r * np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def _cylinder(r: float, length: float, step: float) -> np.ndarray:
    nz, na = max(2, int(length / step) + 1), max(8, int(2 * math.pi * r / step) + 1)
    z, a = np.meshgrid(np.linspace(-length / 2, length / 2, nz), np.linspace(0, 2 * math.pi, na, endpoint=False))
    return np.c_[r * np.cos(a).ravel(), r * np.sin(a).ravel(), z.ravel()]


def _T(origin: ET.Element | None) -> np.ndarray:
    T = np.eye(4)
    if origin is None:
        return T
    r, p, y = (float(x) for x in origin.get("rpy", "0 0 0").split())
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    T[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], [-sp, cp * sr, cp * cr]]
    T[:3, 3] = [float(x) for x in origin.get("xyz", "0 0 0").split()]
    return T


def _axis_rot(axis: np.ndarray, a: float) -> np.ndarray:
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    T = np.eye(4)
    T[:3, :3] = np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
    return T


def _apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def min_dist(A: np.ndarray, B: np.ndarray, cutoff: float = math.inf) -> float:
    """Smallest distance between two point sets (m); returns early (> cutoff) when their boxes are farther apart."""
    gap = np.maximum(0.0, np.maximum(A.min(0) - B.max(0), B.min(0) - A.max(0)))
    if float(np.linalg.norm(gap)) > cutoff:
        return float(np.linalg.norm(gap))
    best = math.inf
    bb = (B * B).sum(1)
    for i in range(0, len(A), 2048):
        a = A[i:i + 2048]
        d2 = (a * a).sum(1)[:, None] + bb[None, :] - 2 * a @ B.T
        best = min(best, float(d2.min()))
    return math.sqrt(max(best, 0.0))


class Arm:
    """One arm of the viewer's model: forward kinematics of every link and the model's surface points."""

    def __init__(self, name: str, urdf: str, offsets_deg: dict, T_world_base: np.ndarray, step_m: float = 0.006,
                 share: Path = SHARE):
        self.name, self.T_wb = name, np.asarray(T_world_base, float)
        self.off = np.array([float(offsets_deg.get(j, 0.0)) for j in JOINTS])
        root = ET.fromstring(urdf)
        pre = f"{name}_"
        strip = (lambda s: s[len(pre):] if s.startswith(pre) else s)
        self.joints = [(strip(j.find("parent").get("link")), strip(j.find("child").get("link")), _T(j.find("origin")),
                        np.array([float(v) for v in (j.find("axis").get("xyz") if j.find("axis") is not None
                                                     else "0 0 1").split()]),
                        j.get("name") if j.get("type") in ("revolute", "continuous") else None)
                       for j in root.findall("joint")]
        self.parts: dict[str, list[tuple[str, np.ndarray]]] = {}
        balls = []
        for ln in root.findall("link"):
            link = strip(ln.get("name"))
            for v in ln.findall("visual"):
                mat = v.find("material").get("name") if v.find("material") is not None else ""
                group = "marker" if mat in MARKER_MATERIALS else GROUPS.get(link)     # rods ("steel") → jaw
                if group is None:
                    continue
                To, g = _T(v.find("origin")), v.find("geometry")[0]
                if g.tag == "mesh":
                    f = g.get("filename").replace("package://spinearm_description", str(share))
                    s = [float(x) for x in g.get("scale", "1 1 1").split()]
                    P = _surface(_stl(Path(f)) * np.array(s), step_m)
                elif g.tag == "sphere":
                    P = _sphere(float(g.get("radius")), step_m)
                    if mat == "marker_ball":
                        balls.append((link, To[:3, 3].copy()))
                elif g.tag == "cylinder":
                    P = _cylinder(float(g.get("radius")), float(g.get("length")), step_m)
                else:
                    continue
                self.parts.setdefault(group, []).append((link, _apply(To, P)))
        if len(balls) != 4:
            raise ValueError(f"{name}: expected 4 marker balls in the model, got {len(balls)}")
        self.ball_link = balls[0][0]
        self.balls_local = np.array([b for _, b in balls])
        blk = np.vstack([P for l, P in self.parts["marker"] if l == self.ball_link])
        c = self.balls_local.mean(0)
        _, _, Vt = np.linalg.svd(self.balls_local - c)
        n = Vt[2]
        self.normal_local = n if (c - blk.mean(0)) @ n > 0 else -n              # out of the block, ball side
        self.radius = 0.007

    def fk(self, q_deg) -> dict[str, np.ndarray]:
        """Link → 4×4 in the base frame; q_deg = arm angles (server convention, JOINTS order)."""
        q = dict(zip(JOINTS, np.radians(np.asarray(q_deg, float) + self.off)))
        T = {"base_link": np.eye(4)}
        pending = list(self.joints)
        while pending:
            rest = []
            for parent, child, To, axis, jn in pending:
                if parent not in T:
                    rest.append((parent, child, To, axis, jn))
                    continue
                Tc = T[parent] @ To
                if jn is not None:
                    Tc = Tc @ _axis_rot(axis / np.linalg.norm(axis), q.get(jn.removeprefix(f"{self.name}_"), 0.0))
                T[child] = Tc
            if len(rest) == len(pending):
                break
            pending = rest
        return T

    def balls(self, q_deg) -> tuple[np.ndarray, np.ndarray]:
        """Ball centres and block normal (world) only — cheap, for screening."""
        Tb = self.T_wb @ self.fk(q_deg)[self.ball_link]
        return _apply(Tb, self.balls_local), Tb[:3, :3] @ self.normal_local

    def pose(self, q_deg) -> dict:
        """Everything the checks need at q: part points (base frame and world), balls and block normal (world)."""
        T = self.fk(q_deg)
        base = {g: np.vstack([_apply(T[l], P) for l, P in parts]) for g, parts in self.parts.items()}
        Tb = T[self.ball_link]
        return {"base": base, "world": {g: _apply(self.T_wb, P) for g, P in base.items()},
                "balls": _apply(self.T_wb @ Tb, self.balls_local),
                "normal": (self.T_wb @ Tb)[:3, :3] @ self.normal_local}


@dataclass
class Camera:
    T_world_cam: np.ndarray            # left base ← camera (the left arm's calibration), m
    fx: float = 425.2
    fy: float = 425.2
    cx: float = 424.0
    cy: float = 240.0
    w: int = 848
    h: int = 480
    baseline_m: float = 0.05012        # right IR imager at +x

    def cam(self, P_world: np.ndarray) -> np.ndarray:
        return _apply(np.linalg.inv(self.T_world_cam), P_world)

    def centres(self) -> np.ndarray:
        return _apply(self.T_world_cam, np.array([[0.0, 0, 0], [self.baseline_m, 0, 0]]))

    def pixels(self, Pc: np.ndarray) -> np.ndarray:
        """(n, 3): u left, u right, v."""
        z = Pc[:, 2]
        return np.c_[self.fx * Pc[:, 0] / z + self.cx, self.fx * (Pc[:, 0] - self.baseline_m) / z + self.cx,
                     self.fy * Pc[:, 1] / z + self.cy]


def parts_dist(A: dict, B: dict, cutoff: float) -> float:
    """Smallest distance between two arms' part clouds, skipping part pairs whose boxes are beyond `cutoff` and the
    base–base pair (both bolted to the table: 32 mm apart at all times)."""
    return min(min_dist(a, b, cutoff) for ga, a in A.items() for gb, b in B.items() if not ga == gb == "base")


def _seg_clear(p0: np.ndarray, p1: np.ndarray, P: np.ndarray) -> float:
    d = p1 - p0
    t = np.clip(((P - p0) @ d) / (d @ d), 0.0, 1.0)
    return float(np.linalg.norm(P - (p0 + t[:, None] * d), axis=1).min()) if len(P) else math.inf


def view(cam: Camera, poses: dict[str, dict], lim: Limits, see=None) -> dict:
    """Per arm in `see` (default all): can the tracker see all 4 balls? (depth, image margin, facing angle, line of
    sight past every robot), plus the balls of all arms kept apart in both images."""
    out, pix_all = {}, []
    everything = np.vstack([P for p in poses.values() for P in p["world"].values()])
    centres = cam.centres()
    for arm, p in poses.items():
        if see is not None and arm not in see:
            pix_all.append(cam.pixels(cam.cam(p["balls"])))
            continue
        B = p["balls"]
        Bc = cam.cam(B)
        px = cam.pixels(Bc)
        pix_all.append(px)
        margin = float(min(px[:, 0].min(), px[:, 1].min(), cam.w - px[:, 0].max(), cam.w - px[:, 1].max(),
                           px[:, 2].min(), cam.h - px[:, 2].max()))
        u0, u1, v0, v1 = lim.window_px
        window = float(min(px[:, 0].min() - u0, u1 - px[:, 0].max(), px[:, 2].min() - v0, v1 - px[:, 2].max()))
        to_cam = centres[0] - B.mean(0)
        ang = math.degrees(math.acos(max(-1.0, min(1.0, float(p["normal"] @ to_cam / np.linalg.norm(to_cam))))))
        los = math.inf
        for b in B:
            others = everything[np.linalg.norm(everything - b, axis=1) > 0.010]
            for c in centres:
                los = min(los, _seg_clear(c, b, others))
        out[arm] = {"z_mm": (Bc[:, 2] * 1000).round(1).tolist(), "margin_px": round(margin, 1),
                    "window_px": round(window, 1), "px": px.round(0).tolist(),
                    "view_deg": round(ang, 1), "los_mm": round(los * 1000, 1)}
    boxes = {}
    for arm, p in poses.items():
        px = cam.pixels(cam.cam(p["balls"]))
        boxes[arm] = [(px[:, c].min(), px[:, c].max(), px[:, 2].min(), px[:, 2].max()) for c in (0, 1)]
    for arm in out:
        gaps = []
        for other, bo in boxes.items():
            if other == arm:
                continue
            for (a0, a1, b0, b1), (c0, c1, d0, d1) in zip(boxes[arm], bo):
                gaps.append(max(c0 - a1, a0 - c1, d0 - b1, b0 - d1))     # > 0: apart by that much
        out[arm]["apart_px"] = round(float(min(gaps)), 1) if gaps else math.inf
    P = np.vstack(pix_all)
    sep = math.inf
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            sep = min(sep, float(np.hypot(P[i, 0] - P[j, 0], P[i, 2] - P[j, 2])),
                      float(np.hypot(P[i, 1] - P[j, 1], P[i, 2] - P[j, 2])))
    out["sep_px"] = round(sep, 1)
    return out


class Zone:
    """Checks for a configuration of both arms {arm: q_deg} against Limits. `set_start` relaxes the self, floor and
    arm-to-arm limits to what the start (rest) pose already has — the folded home is physically fine, so moves may
    leave it and come back — while `strict` (the target poses) keeps the full limits and arms at least
    `target_arms_mm` apart."""

    def __init__(self, arms: dict[str, Arm], cam: Camera, lim: Limits | None = None, target_arms_mm: float = 50.0):
        self.arms, self.cam, self.lim, self.target_arms_mm = arms, cam, lim or Limits(), target_arms_mm
        self.self_floor = {a: {} for a in arms}
        self.floor_floor = {a: {} for a in arms}
        self.arms_floor = None

    def set_start(self, q: dict[str, np.ndarray]) -> None:
        poses = {a: arm.pose(q[a]) for a, arm in self.arms.items()}
        for a, p in poses.items():
            b = p["base"]
            for g1, g2 in SELF_PAIRS:
                d = min_dist(b[g1], b[g2]) * 1000
                if d < self.lim.self_mm:
                    self.self_floor[a][(g1, g2)] = d - 1.0
            for g in FLOOR_GROUPS:
                z = float(b[g][:, 2].min()) * 1000
                if z < self.lim.floor_mm:
                    self.floor_floor[a][g] = z - 1.0
        names = list(poses)
        if len(names) == 2:
            d = parts_dist(poses[names[0]]["world"], poses[names[1]]["world"], self.lim.arms_mm / 1000) * 1000
            if d < self.lim.arms_mm:
                self.arms_floor = d - 2.0

    def single(self, a: str, q, strict: bool = True, optimal: bool = False) -> tuple[bool, list, dict]:
        """One arm alone (itself, floor, camera, its own view) — the screening step for candidate poses."""
        p = self.arms[a].pose(q)
        ok, why, m = self._arm(a, p, strict)
        v = view(self.cam, {a: p}, self.lim)[a]
        why += self._view_why(a, v, optimal, strict)
        return not why, why, v

    def _arm(self, a: str, p: dict, strict: bool) -> tuple[bool, list, dict]:
        lim, why, b = self.lim, [], p["base"]
        for g in FLOOR_GROUPS:
            z = float(b[g][:, 2].min()) * 1000
            if z < (lim.floor_mm if strict else self.floor_floor[a].get(g, lim.floor_mm)):
                why.append(f"{a}:{g} {z:.0f} mm over the floor")
        for g1, g2 in SELF_PAIRS:
            need = self.self_floor[a].get((g1, g2), lim.self_mm)     # pairs already close at rest (the marker
            # beside the jaw's servo horn, the folded gripper): no closer than there, in targets too
            d = min_dist(b[g1], b[g2], cutoff=need / 1000) * 1000
            if d < need:
                why.append(f"{a}:{g1}-{g2} {d:.0f} mm")
        cam_c = self.cam.centres()[0]
        dc = min(float(np.linalg.norm(P - cam_c, axis=1).min()) for P in p["world"].values()) * 1000
        if dc < lim.camera_mm:
            why.append(f"{a} {dc:.0f} mm from the camera")
        return not why, why, {}

    def _view_why(self, a: str, v: dict, optimal: bool, strict: bool = False) -> list:
        lim, why, z = self.lim, [], v["z_mm"]
        if min(z) < lim.z_mm[0] or max(z) > lim.z_mm[1]:
            why.append(f"{a} balls at {min(z):.0f}-{max(z):.0f} mm")
        if v["margin_px"] < (lim.target_margin_px if strict else lim.margin_px):
            why.append(f"{a} {v['margin_px']:.0f} px from the image edge")
        if v["window_px"] < (lim.target_window_px if strict else 0.0):
            why.append(f"{a} {-v['window_px']:.0f} px out of the detection window")
        if v.get("apart_px", math.inf) < lim.apart_px:
            why.append(f"{a} marker {v['apart_px']:.0f} px from the other marker in the image")
        if v["view_deg"] > (lim.target_view_deg if strict else lim.max_view_deg):
            why.append(f"{a} faces {v['view_deg']:.0f}° off the camera")
        if v["los_mm"] < lim.los_mm:
            why.append(f"{a} line of sight {v['los_mm']:.0f} mm")
        if optimal and (min(z) < lim.optimal_z_mm[0] or max(z) > lim.optimal_z_mm[1]):
            why.append(f"{a} not optimal ({min(z):.0f}-{max(z):.0f} mm)")
        return why

    def check(self, q: dict[str, np.ndarray], optimal=(), strict: bool = False, see=None) -> tuple[bool, list, dict]:
        """All arms together: each arm's limits, the arms apart, the markers of the arms in `see` (default: all) seen
        with every robot as a possible blocker, their balls apart in the images; `optimal` = arms that must also be in
        the 35–45 cm band. An arm resting at its start pose is simply an obstacle (see = the moving arm)."""
        lim, why, m = self.lim, [], {}
        see = list(q) if see is None else list(see)
        poses = {a: arm.pose(q[a]) for a, arm in self.arms.items()}
        for a, p in poses.items():
            why += self._arm(a, p, strict)[1]
        names = list(poses)
        if len(names) == 2:
            need = self.target_arms_mm if strict else (lim.arms_mm if self.arms_floor is None
                                                      else min(lim.arms_mm, self.arms_floor))
            d = parts_dist(poses[names[0]]["world"], poses[names[1]]["world"], need / 1000) * 1000
            m["arms_mm"] = round(d, 1)
            if d < need:
                why.append(f"arms {d:.0f} mm apart")
        if see:
            v = view(self.cam, poses, lim, see)
            m.update(v)
            for a in see:
                why += self._view_why(a, v[a], a in optimal, strict)
            if v["sep_px"] < lim.sep_px:
                why.append(f"balls {v['sep_px']:.0f} px apart")
        return not why, why, m

    def path(self, qa: dict, qb: dict, step_deg: float = 2.0, see=None) -> tuple[bool, list, int]:
        """Straight line in joint space (all arms together) checked every ≤ step_deg."""
        span = max(float(np.abs(np.asarray(qb[a]) - np.asarray(qa[a])).max()) for a in qa)
        n = max(1, int(math.ceil(span / step_deg)))
        for k in range(1, n + 1):
            f = k / n
            ok, why, _ = self.check({a: np.asarray(qa[a]) + f * (np.asarray(qb[a]) - np.asarray(qa[a])) for a in qa},
                                    see=see)
            if not ok:
                return False, why, k
        return True, [], n
