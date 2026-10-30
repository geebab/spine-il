#!/usr/bin/env python3
"""Fit the as-built robot to the camera, both arms at once (2026-09-27, user instructions).

What may move (everything else is the drawing):
  - the camera (one, 6)                       — "the robot a little forward"
  - the right base relative to the left (6)   — held to the tape: outer foot centres 207 mm, moving-jaw centres at
                                                home 143 mm (the user's measurements)
  - per arm: joint zeros lift/elbow/wrist_flex/wrist_roll (4), the jaw zero (1)
  - per arm: the marker block's seat on the jaw (3) — flush on the jaw face (no tilt, no lift); it may turn and slide
    only as far as the pins allow: the rod in jaw hole #4 stays inside the block's O hole, the rod in hole #3 inside
    the slot behind the arrow.
Data: the verify run's stops (each ball's raw triangulated centre) [+ the hand-eye poses].

  fit_asbuilt.py <verify_run_dir> [--runs <dir> ...] [--homes <file> ...] [--rod-d 1.5] [--with-handeye] [--save]

Balls are placed where they really sit on the block (the tracker's own constellation laid on the seats) and the joint
zeros are fitted for the model only (`model_zero_deg`: the arm bridge keeps the hand-eye offsets for commands).
Every fit starts from the hand-eye calibration, so fits do not stack.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_two_arm as V  # noqa: E402
from spinearm_bridge.visual_urdf import JAW_LINK, _T, match_balls, mount_cad  # noqa: E402

TAPE_FEET_MM, TAPE_JAWS_MM = 207.0, 143.0
SIG_TABLE_MM, SIG_TABLE_DEG = 1.5, 0.5   # both bases stand on the same table: same height, same tilt
SIG_ZERO_DEG = 3.0                       # joint zeros stay near the hand-eye ones (fitted over 32 poses each)
SIG_LINK_MM = 2.0                        # link lengths stay near the drawing
LINK_JOINTS = ("elbow_flex", "wrist_flex")   # their origins hold the upper-arm and the forearm length
SIG_FEET, SIG_JAWS = 0.3, 0.3        # the tape is the truth here (user): held to ±0.3 mm; foot centre = middle of each pad
FOOT = {"left": np.array([21.2, 34.2, 0.0]), "right": np.array([21.2, -34.9, 0.0])}   # outer foot, own base, mm


def _rot2(t):
    return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])


class Arm:
    def __init__(self, name, model: V.sz.Arm, mount: dict, seats_mm, jaw0_deg, V_jaw, rod_d, const_mm=None):
        self.name, self.m, self.mount = name, model, mount
        self.C = mount_cad(mount, seated=False)                     # jaw mesh ← block STL (drawing), m
        self.seats = np.asarray(seats_mm, float)
        if const_mm is not None:                                    # balls as they sit: the tracker's constellation
            const = np.asarray(const_mm, float)
            R, t, _, perm = match_balls(self.seats, const)
            self.seats = (const[list(perm)] - t) @ R
        self.jaw0 = math.radians(jaw0_deg)
        self.Vj = V_jaw
        O = np.r_[mount["O_stl_mm"], 0.0]
        self.O = O[:2]
        sb = np.r_[mount["slot_stl_mm"], 0.0] - O
        self.sb = sb[:2] / np.linalg.norm(sb[:2])
        self.slot_c = np.asarray(mount["slot_stl_mm"], float)
        self.slot_half = (mount.get("slot_len_mm", 7.08) - mount.get("slot_w_mm", 2.1)) / 2
        self.slot_r = mount.get("slot_w_mm", 2.1) / 2 - rod_d / 2
        self.r_O = (mount.get("O_hole_d_mm", 5.5) - rod_d) / 2
        Ci = np.linalg.inv(self.C)
        self.q3 = ((Ci @ np.r_[np.asarray(mount["hole3_mm"]) / 1000, 1])[:3] * 1000)[:2]   # rod 3 in the block, drawn
        # the moving jaw's centre (for the tape): its own mesh only — the rods drawn in it would pull the centre out
        jaw_pts = [P for l, P in model.parts["jaw"] if len(P) > 150]
        self.jaw_c = np.vstack(jaw_pts).mean(0) if jaw_pts else np.zeros(3)                # jaw centre, jaw link

    def dev(self, th, s):
        """Block STL → block STL: turn by th about O, then slide by s (mm) in the block plane."""
        D = np.eye(4)
        R = _rot2(th)
        D[:2, :2] = R
        D[:2, 3] = (self.O - R @ self.O + s) / 1000
        return D

    def balls_jaw(self, djaw, th, s, dh=0.0, tl=None):
        a = self.jaw0 + djaw
        Rz = np.eye(4)
        Rz[:2, :2] = _rot2(a)
        T = Rz @ self.Vj @ self.C @ self.dev(th, s)
        if tl is not None:                                          # face tilt about the block's x/y axes through O
            To = np.eye(4)
            To[:3, 3] = np.r_[self.O, 0.0] / 1000
            T = T @ To @ V._exp(np.r_[tl[0], tl[1], 0, 0, 0, 0]) @ np.linalg.inv(To)
        B = self.seats + np.array([0.0, 0.0, dh])                   # dh: all balls standing higher off the block
        return B / 1000 @ T[:3, :3].T + T[:3, 3]

    def clearance(self, th, s):
        """(O hole room left, slot room left), mm: < 0 = the pins would have to bend."""
        o_room = self.r_O - float(np.linalg.norm(s))
        p3 = self.O + _rot2(-th) @ (self.q3 - self.O - s)             # rod 3 as seen in the block
        d = p3 - self.slot_c
        a = float(np.clip(d @ self.sb, -self.slot_half, self.slot_half))
        return o_room, self.slot_r - float(np.linalg.norm(d - a * self.sb))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--runs", nargs="*", default=[], help="more run dirs (all their events*.jsonl records)")
    ap.add_argument("--homes", nargs="*", default=[], help="static samples (runs/homes/*.json), weighted ×3")
    ap.add_argument("--seat-balls", action="store_true", help="balls on their drawn seats instead of as they sit")
    ap.add_argument("--loo-every", type=int, default=3, help="leave-one-out on every n-th stop (0 = skip)")
    ap.add_argument("--max-shape-mm", type=float, default=1.0,
                    help="drop samples whose 4 measured balls fit the tracker's own constellation worse than this "
                         "(rigidly): the dim block-A balls come out bent beyond ~45 cm (0.5 → 1.2 mm), a measurement "
                         "error no model should chase")
    ap.add_argument("--measure", choices=["pose", "raw"], default="pose",
                    help="pose = the tracker's pose × constellation (what the viewer draws, and what the hand-eye used); "
                         "raw = each ball's triangulated centre")
    ap.add_argument("--unlock", action="store_true", help="allow --save over a locked model (markers.json _lock)")
    ap.add_argument("--links", action="store_true",
                    help="fit the upper-arm and forearm lengths too (printed parts: a mm or two off the drawing)")
    ap.add_argument("--allow-tilt", action="store_true",
                    help="test: the jaw face (so the flush block) tilted off the drawing, about O (2 per arm)")
    ap.add_argument("--split-lift", type=float, default=None,
                    help="test: a second block seat for poses with shoulder_lift below this (folded) — does a loose "
                         "block sit differently when the arm folds?")
    ap.add_argument("--rod-d", type=float, default=1.5, help="steel rod diameter, mm (jaw holes Ø1.5, slot 2.1 wide)")
    ap.add_argument("--with-handeye", action="store_true", help="fit the hand-eye poses too (left ones are from 09-23)")
    ap.add_argument("--save", action="store_true", help="write ~/il-robot/calib/asbuilt.json")
    ap.add_argument("--no-zeros", action="store_true",
                    help="keep the joint zeros of the hand-eye calibration (the ROS bridge converts commands with them)")
    a = ap.parse_args()
    run = Path(a.run_dir)
    lock = json.loads((V.CALIB / "markers.json").read_text()).get("_lock")
    if a.save and lock and not a.unlock:
        sys.exit(f"the model is locked ({lock.get('locked_at')}, {lock.get('by')}): fit without --save, or --unlock "
                 f"after the user asks (snapshot {lock.get('snapshot')})")
    P = json.loads((run / "plan.json").read_text())
    models, cam, info = V.load_models(kind="handeye")               # always start from the hand-eye calibration
    M = json.loads((V.CALIB / "markers.json").read_text())
    vis = ET.parse(str(V.sz.SHARE / "urdf/so101_new_calib.urdf")).getroot()
    Vj = _T([l for l in vis.findall("link") if l.get("name") == JAW_LINK][0].find("visual").find("origin"))
    tools = {x: json.loads(Path(json.loads((V.CALIB / f"handeye_{x}.json").read_text())["tool_calib"]).expanduser()
                           .read_text())["points_local"] for x in ("left", "right")}
    arms = {x: Arm(x, models[x], M[x]["mount"], M[x]["balls_stl_mm"], info[x]["model"]["fit"]["mount"]["jaw_deg"],
                   Vj, a.rod_d, None if a.seat_balls else tools[x]) for x in ("left", "right")}
    Z0 = np.array(info["left"]["T_base_cam"])
    TLR0 = np.array(info["right"]["T_world_base"])
    home = {x: np.array(v) for x, v in P["start"].items()}
    data = []                                                       # (arm, q, measured balls cam mm, weight, kind)
    perms = {x: info[x]["model"]["fit"]["mount"]["perm"] for x in arms}
    for d in [run] + [Path(r) for r in a.runs]:
        for ev in sorted(d.glob("events*.jsonl")):
            meas = V.stops_measured(d, None, ev.name)
            for x in arms:
                for s in meas[x]:
                    src_b = s["pose_balls"] if a.measure == "pose" else s["balls"]
                    if s["q"] is not None and len(s["balls"]) == 4 and len(src_b) == 4:
                        data.append((x, s["q"], np.array([src_b[perms[x][i]] for i in range(4)]), 1.0,
                                     "stop" if d == run else "other", f"{d.name}/{ev.stem}"))
    for hf in a.homes:
        for smp in json.loads(Path(hf).read_text())["samples"]:
            x = smp["arm"]
            b = {int(k): v for k, v in (smp["pose_balls"] if a.measure == "pose" else smp["balls"]).items()}
            data.append((x, np.array(smp["q"]), np.array([b[perms[x][i]] for i in range(4)]), 3.0, "home",
                         Path(hf).stem))
    if a.with_handeye:
        for x in arms:
            hd = json.loads((V.CALIB / f"handeye_{x}.json").read_text())
            pts = np.array(json.loads(Path(hd["tool_calib"]).expanduser().read_text())["points_local"], float)
            for smp in json.loads(Path(hd["samples"]).read_text())["samples"]:
                To = np.array(json.loads(smp["obs"]) if isinstance(smp["obs"], str) else smp["obs"])
                data.append((x, np.array(smp["q_enc"], float),
                             (pts[perms[x]] / 1000 @ To[:3, :3].T + To[:3, 3]) * 1000, 1.0, "handeye", "handeye"))
    def shape_rms(x, m_):                                           # measured balls vs the tracker's constellation
        P_ = np.asarray(tools[x], float)[perms[x]]
        R_, t_ = V._kabsch(P_, m_)
        return float(np.sqrt(((P_ @ R_.T + t_ - m_) ** 2).sum(1).mean()))
    if a.max_shape_mm > 0:                                          # hard cut
        dropped = [(d_[0], d_[5], round(shape_rms(d_[0], d_[2]), 2)) for d_ in data
                   if shape_rms(d_[0], d_[2]) > a.max_shape_mm]
        data = [d_ for d_ in data if shape_rms(d_[0], d_[2]) <= a.max_shape_mm]
        if dropped:
            print(f"[fit] dropped {len(dropped)} bent measurements (> {a.max_shape_mm} mm): {dropped}")
    else:                                                           # soft: a bent measurement counts less
        data = [(x, q, m_, w / max(1.0, shape_rms(x, m_) / 0.7), kind, src) for (x, q, m_, w, kind, src) in data]
    print(f"[fit] data: " + ", ".join(f"{k} {sum(1 for d_ in data if d_[4] == k)}"
                                      for k in ("stop", "other", "home", "handeye")), flush=True)
    NA = 9                                                          # per arm: dq(4), djaw, th, sx, sy, ball lift
    NS = 3 if a.split_lift is not None else 0                       # per arm: second seat (th, sx, sy) when folded
    NT = 2 if a.allow_tilt else 0                                   # per arm: face tilt about the block x, y through O
    NL = 2 if a.links else 0                                        # per arm: upper-arm, forearm length change (mm)
    base_joints = {x: list(arms[x].m.joints) for x in arms}

    def links(p, x):
        i = ("left", "right").index(x)
        b = 12 + 2 * NA + 2 * NS + 2 * NT
        return p[b + i * NL: b + (i + 1) * NL]

    def set_links(x, dl):
        """Stretch the joint origins that carry the link lengths (along their own offset)."""
        js = []
        for parent, child, To, axis, jn in base_joints[x]:
            if jn and jn.removeprefix(f"{x}_") in LINK_JOINTS and dl is not None:
                k = LINK_JOINTS.index(jn.removeprefix(f"{x}_"))
                To = To.copy()
                L = float(np.linalg.norm(To[:3, 3]))
                To[:3, 3] *= (L + dl[k] / 1000) / L
            js.append((parent, child, To, axis, jn))
        arms[x].m.joints = js

    def unpack(p):
        Zc = Z0 @ V._exp(p[:6])
        TLR = TLR0 @ V._exp(p[6:12])
        per = {x: p[12 + i * NA: 12 + (i + 1) * NA] for i, x in enumerate(("left", "right"))}
        return Zc, TLR, per

    def seat2(p, x):
        i = ("left", "right").index(x)
        return p[12 + 2 * NA + i * NS: 12 + 2 * NA + (i + 1) * NS]

    def tilt(p, x):
        i = ("left", "right").index(x)
        b = 12 + 2 * NA + 2 * NS
        return p[b + i * NT: b + (i + 1) * NT]

    def world(x, q, pa, TLR, pts_jaw=None, s2=None, tl=None, dl=None):
        if NL:
            set_links(x, dl)
        T = arms[x].m.fk(np.asarray(q, float) + np.r_[0, pa[:4], 0])[arms[x].m.ball_link]
        if pts_jaw is None and s2 is not None and q[1] < a.split_lift:
            B = arms[x].balls_jaw(pa[4], s2[0], s2[1:3], pa[8])
        else:
            B = arms[x].balls_jaw(pa[4], pa[5], pa[6:8], pa[8], tl) if pts_jaw is None else pts_jaw
        W = B @ T[:3, :3].T + T[:3, 3]
        return W if x == "left" else W @ TLR[:3, :3].T + TLR[:3, 3]

    def predict(p, x, q):
        Zc, TLR, per = unpack(p)
        W = world(x, q, per[x], TLR, s2=seat2(p, x) if NS else None, tl=tilt(p, x) if NT else None,
                  dl=links(p, x) if NL else None)
        Zi = np.linalg.inv(Zc)
        return (W @ Zi[:3, :3].T + Zi[:3, 3]) * 1000

    def tape(p):
        Zc, TLR, per = unpack(p)
        fl = FOOT["left"] / 1000
        fr = TLR[:3, :3] @ (FOOT["right"] / 1000) + TLR[:3, 3]
        feet = float(np.linalg.norm(fl - fr)) * 1000
        jl = world("left", home["left"], per["left"], TLR, arms["left"].jaw_c[None], dl=links(p, "left") if NL else None)[0]
        jr = world("right", home["right"], per["right"], TLR, arms["right"].jaw_c[None],
                   dl=links(p, "right") if NL else None)[0]
        return feet, float(np.linalg.norm(jl - jr)) * 1000

    def resid(p, skip=None):
        if a.no_zeros:
            p = p.copy()
            for i in range(2):
                p[12 + i * NA: 12 + i * NA + 4] = 0.0
        r = [(predict(p, x, q) - m_).ravel() * w for k, (x, q, m_, w, kind, src) in enumerate(data) if k != skip]
        feet, jaws = tape(p)
        r.append(np.array([(feet - TAPE_FEET_MM) / SIG_FEET, (jaws - TAPE_JAWS_MM) / SIG_JAWS]))
        _, TLR, per = unpack(p)
        zl = TLR[:3, 2]                                               # the right base's up axis, in the left base
        r.append(np.array([TLR[2, 3] * 1000 / SIG_TABLE_MM,
                           math.degrees(math.acos(max(-1.0, min(1.0, float(zl[2]))))) / SIG_TABLE_DEG]))
        if not a.no_zeros:
            r.append(np.concatenate([per[x][:4] for x in arms]) / SIG_ZERO_DEG)
        if NL:
            r.append(np.concatenate([links(p, x) for x in arms]) / SIG_LINK_MM)
        for x in arms:
            o, sl = arms[x].clearance(per[x][5], per[x][6:8])
            r.append(np.array([min(0.0, o), min(0.0, sl)]) * 30.0)    # the pins do not bend: stiff penalty
            if NS:
                s2 = seat2(p, x)
                o, sl = arms[x].clearance(s2[0], s2[1:3])
                r.append(np.array([min(0.0, o), min(0.0, sl)]) * 30.0)
        return np.concatenate(r)

    def report(p, label):
        Zc, TLR, per = unpack(p)
        feet, jaws = tape(p)
        out = {"label": label}
        for x in arms:
            E = np.array([np.linalg.norm(predict(p, x, q) - m_, axis=1) for (xx, q, m_, w, kind, src) in data if xx == x
                          and kind == "stop"])
            H = [np.linalg.norm(predict(p, x, q) - m_, axis=1) for (xx, q, m_, w, kind, src) in data if xx == x
                 and kind == "handeye"]
            HM = [np.linalg.norm(predict(p, x, q) - m_, axis=1) for (xx, q, m_, w, kind, src) in data if xx == x
                  and kind == "home"]
            OT = [np.linalg.norm(predict(p, x, q) - m_, axis=1) for (xx, q, m_, w, kind, src) in data if xx == x
                  and kind == "other"]
            o, sl = arms[x].clearance(per[x][5], per[x][6:8])
            out[x] = {"stops_rms_mm": round(float(np.sqrt((E ** 2).mean())), 2), "stops_max_mm": round(float(E.max()), 2),
                      "per_ball_rms_mm": np.sqrt((E ** 2).mean(0)).round(2).tolist(),
                      "handeye_rms_mm": round(float(np.sqrt((np.array(H) ** 2).mean())), 2) if H else None,
                      "home_mm": np.array(HM).round(1).tolist() if HM else None,
                      "other_rms_mm": round(float(np.sqrt((np.array(OT) ** 2).mean())), 2) if OT else None,
                      "zero_change_deg": np.round(per[x][:4], 2).tolist(), "jaw_change_deg": round(math.degrees(per[x][4]), 2),
                      "block_turn_deg": round(math.degrees(per[x][5]), 2), "block_slide_mm": np.round(per[x][6:8], 2).tolist(),
                      "ball_lift_mm": round(float(per[x][8]), 2),
                      "o_hole_room_mm": round(o, 2), "slot_room_mm": round(sl, 2)}
        dcam = np.linalg.inv(Z0) @ Zc                                 # the camera move, in the old camera frame
        robot_fwd = (Z0[:3, 3] - Zc[:3, 3]) * 1000                    # camera → robot shift, left base frame
        out["camera"] = {"move_mm_left_base": np.round((Zc[:3, 3] - Z0[:3, 3]) * 1000, 2).tolist(),
                         "robot_shift_mm_left_base": np.round(robot_fwd, 2).tolist(),
                         "turn_deg": round(math.degrees(math.acos(max(-1, min(1, (np.trace(dcam[:3, :3]) - 1) / 2)))), 3)}
        dR = np.linalg.inv(TLR0) @ TLR
        out["right_base"] = {"move_mm": np.round((TLR[:3, 3] - TLR0[:3, 3]) * 1000, 2).tolist(),
                             "turn_deg": round(math.degrees(math.acos(max(-1, min(1, (np.trace(dR[:3, :3]) - 1) / 2)))), 3),
                             "offset_mm": np.round(TLR[:3, 3] * 1000, 1).tolist()}
        out["tape"] = {"feet_mm": round(feet, 1), "jaws_mm": round(jaws, 1)}
        return out

    p0 = np.zeros(12 + 2 * NA + 2 * NS + 2 * NT + 2 * NL)
    before = report(p0, "as drawn (current calibration)")
    p = V._lm(resid, p0, iters=80)
    if a.no_zeros:
        for i in range(2):
            p[12 + i * NA: 12 + i * NA + 4] = 0.0
    after = report(p, f"as built (rod Ø{a.rod_d}{', joint zeros kept' if a.no_zeros else ''})")
    stops = [k for k, d in enumerate(data) if d[4] == "stop"][::a.loo_every] if a.loo_every else []
    loo = {x: [] for x in arms}
    for k in stops:
        pk = V._lm(lambda q_: resid(q_, skip=k), p, iters=25)
        if a.no_zeros:
            for i in range(2):
                pk[12 + i * NA: 12 + i * NA + 4] = 0.0
        x, q, m_ = data[k][0], data[k][1], data[k][2]
        loo[x].append(np.linalg.norm(predict(pk, x, q) - m_, axis=1))
    for x in arms:
        if loo[x]:
            after[x]["leave_one_out_rms_mm"] = round(float(np.sqrt((np.array(loo[x]) ** 2).mean())), 2)
    by = {}
    for (x, q, m_, w, kind, src) in data:
        by.setdefault((x, src), []).append(np.linalg.norm(predict(p, x, q) - m_, axis=1))
    after["by_source"] = {f"{x}:{src}": round(float(np.sqrt((np.array(v) ** 2).mean())), 2) for (x, src), v in by.items()}
    for r in (before, after):
        print(f"== {r['label']}: tape feet {r['tape']['feet_mm']} mm (207), jaws {r['tape']['jaws_mm']} mm (143)")
        for x in arms:
            print(f"   {x:5s}: stops rms {r[x]['stops_rms_mm']} mm (max {r[x]['stops_max_mm']}, per ball {r[x]['per_ball_rms_mm']})"
                  + (f", left-out {r[x]['leave_one_out_rms_mm']}" if "leave_one_out_rms_mm" in r[x] else "")
                  + (f", hand-eye {r[x]['handeye_rms_mm']}" if r[x]["handeye_rms_mm"] is not None else "")
                  + (f", other runs {r[x]['other_rms_mm']}" if r[x]["other_rms_mm"] is not None else "")
                  + (f", HOME per ball {r[x]['home_mm']}" if r[x]["home_mm"] is not None else ""))
            if r is after:
                print(f"          zeros {r[x]['zero_change_deg']}°, jaw {r[x]['jaw_change_deg']}°, block turned "
                      f"{r[x]['block_turn_deg']}° slid {r[x]['block_slide_mm']} mm (room: O hole {r[x]['o_hole_room_mm']}, "
                      f"slot {r[x]['slot_room_mm']} mm), balls {r[x]['ball_lift_mm']:+.2f} mm higher")
        if r is after:
            if NL:
                for x in arms:
                    print(f"   {x} link length change: upper arm {links(p, x)[0]:+.2f} mm, forearm {links(p, x)[1]:+.2f} mm")
            if NT:
                for x in arms:
                    tl = tilt(p, x)
                    print(f"   {x} face tilt about the block x/y: {math.degrees(tl[0]):+.2f}° / {math.degrees(tl[1]):+.2f}°")
            if NS:
                for x in arms:
                    s2 = seat2(p, x)
                    print(f"   {x} folded seat (lift < {a.split_lift}): turned {math.degrees(s2[0]):.2f}° slid "
                          f"{np.round(s2[1:3], 2).tolist()} mm vs extended {r[x]['block_turn_deg']}° {r[x]['block_slide_mm']}")
            print("   by source: " + ", ".join(f"{k} {v} ({len(by[tuple(k.split(':', 1))])})" for k, v in r["by_source"].items()))
            print(f"   camera moved {r['camera']['move_mm_left_base']} mm (left base) / {r['camera']['turn_deg']}° → robot "
                  f"relative to the camera {r['camera']['robot_shift_mm_left_base']} mm; right base moved "
                  f"{r['right_base']['move_mm']} mm / {r['right_base']['turn_deg']}° (now at {r['right_base']['offset_mm']})")
    if a.save:
        import time
        Zc, TLR, per = unpack(p)
        path = V.CALIB / "asbuilt.json"
        path.write_text(json.dumps({
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": str(run), "rod_d_mm": a.rod_d, "with_handeye": a.with_handeye,
            "joint_zeros_kept": a.no_zeros, "T_leftbase_cam": Zc.tolist(), "T_leftbase_rightbase": TLR.tolist(),
            "arms": {x: {"joint_zero_change_deg": np.round(per[x][:4], 4).tolist(),
                         "jaw_deg": round(math.degrees(arms[x].jaw0 + per[x][4]), 3),
                         "seat": {"turn_deg": round(math.degrees(per[x][5]), 3), "slide_mm": np.round(per[x][6:8], 3).tolist()}}
                     for x in arms}, "before": before, "after": after}, indent=1))
        # markers.json: the jaw zero and the block's seat, as built (the model draws them)
        Mj = json.loads((V.CALIB / "markers.json").read_text())
        for x in arms:
            mt = Mj[x]["mount"]
            mt["jaw_deg"] = round(math.degrees(arms[x].jaw0 + per[x][4]), 3)
            mt["seat"] = {"turn_deg": round(math.degrees(per[x][5]), 3), "slide_mm": np.round(per[x][6:8], 3).tolist()}
            mt["ball_lift_mm"] = round(float(per[x][8]), 3)
            if a.seat_balls:
                mt.pop("balls", None)
            else:
                mt["balls"] = "tracker"
            mt.setdefault("O_hole_d_mm", 5.5)
            mt.setdefault("slot_w_mm", 2.1)
            mt.setdefault("slot_len_mm", {"left": 7.08, "right": 8.08}[x])
            mt["rod_d_mm"] = a.rod_d
        (V.CALIB / "markers.json").write_text(json.dumps(Mj, indent=1, ensure_ascii=False))
        # asbuilt_<arm>.json: the hand-eye file with the fitted camera / base placement and the matching marker frame
        # (X = marker_frame_link ← tool constellation, from the seated block), joint zeros untouched
        kin = ET.parse(str(V.KIN)).getroot()
        F_m = _T(next(j for j in kin.findall("joint") if j.get("name") == "marker_frame_joint").find("origin"))
        for x in arms:
            hd = json.loads((V.CALIB / f"handeye_{x}.json").read_text())
            mt = Mj[x]["mount"]
            Rz = np.eye(4)
            Rz[:2, :2] = _rot2(math.radians(mt["jaw_deg"]))
            Lz = np.eye(4)
            Lz[2, 3] = float(mt.get("ball_lift_mm", 0.0)) / 1000        # the balls stand this much off the block
            T_jl_stl = Rz @ Vj @ mount_cad(mt) @ Lz
            pts = np.array(json.loads(Path(hd["tool_calib"]).expanduser().read_text())["points_local"], float)
            R, t, e, perm = match_balls(np.array(Mj[x]["balls_stl_mm"], float), pts)
            S = np.eye(4)
            S[:3, :3], S[:3, 3] = R, t / 1000
            if not a.no_zeros:
                hd["model_zero_deg"] = dict(zip(("shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"),
                                                np.round(per[x][:4], 4).tolist()))
            hd.update(kind="asbuilt", created=time.strftime("%Y-%m-%dT%H:%M:%S"), source=str(path),
                      T_base_cam=(Zc if x == "left" else np.linalg.inv(TLR) @ Zc).tolist(),
                      T_link_const=(np.linalg.inv(F_m) @ T_jl_stl @ np.linalg.inv(S)).tolist(),
                      note="hand-eye joint zeros; camera/base placement and marker frame from the as-built fit")
            (V.CALIB / f"asbuilt_{x}.json").write_text(json.dumps(hd, indent=1))
        print(f"→ {path}, markers.json (jaw_deg, seat), asbuilt_left.json, asbuilt_right.json")

if __name__ == "__main__":
    main()
