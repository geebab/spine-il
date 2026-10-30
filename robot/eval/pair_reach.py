#!/usr/bin/env python3
"""How much of the demos' drilling can the two arms follow from one robot start pair? (no robot, no viewer.)

    python robot/eval/pair_reach.py --dataset DS --pair RID [--anchor-s S] [--pairs-file F] [--every-s 2] [-o OUT.json]

A closed-loop run starts the arms at a paired start pose that belongs to ONE hand recording (the gate pair, anchored at
that recording's pose at --anchor-s), and the inference safety gate (robot/inference/safety_gate.py) lets a commanded
pose through only if the arms can play it back from there: IK within 5 mm / 10°, clearance, safe zone, arms apart, the
80 mm replay sphere. A policy trained on other recordings wants to work where THEY worked. 10-07: the 24-demo policy,
started on the 3868 pair, asked for the new demos' endoscope orientation, the left arm could not turn that far, and the
gate held the tools for 30 min.

This walks the gate from the pair's anchor to every demo's drilling frames (phase 4, one every --every-s; the IK follows
the path as it does in a run) and reports, per demo, the share of drilling frames the gate would pass and what stops the
rest. The IK solution is carried along also through failing frames — the question here is where the arms can be, not what
a held run would do next.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT / "inference"))
sys.path.insert(0, str(ROBOT))


def episodes(ds: Path):
    """[(sidecar meta, states N×15, phase N)] of a --state-phase dataset, by episode index."""
    import pyarrow.parquet as pq
    meta = [json.loads(l) for l in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()]
    rows = collections.defaultdict(list)
    for f in sorted(glob.glob(str(ds / "data/*/*.parquet"))):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"])
        for e, k, s in zip(t.column("episode_index").to_pylist(), t.column("frame_index").to_pylist(),
                           t.column("observation.state").to_pylist()):
            rows[e].append((k, s))
    out = []
    for e, m in enumerate(meta):
        s = np.asarray([r[1] for r in sorted(rows[e], key=lambda r: r[0])], float)
        if s.shape[1] < 21:
            raise SystemExit(f"{ds.name}: the state has no phase columns (a --state-phase dataset is needed)")
        out.append((m, s[:, :15], s[:, 15:21].argmax(axis=1) + 1))
    return out


def slerp_path(a: np.ndarray, b: np.ndarray, step_mm: float):
    """Frames from a (excluded) to b (included): positions linear, orientations slerp, at most step_mm a step."""
    from safety import _slerp
    n = max(1, int(np.ceil(max(np.linalg.norm(b[o:o + 3] - a[o:o + 3]) for o in (0, 7)) / step_mm)))
    for j in range(1, n + 1):
        f = a + (b - a) * (j / n)
        for o in (3, 10):
            f[o:o + 4] = _slerp(a[o:o + 4], b[o:o + 4], j / n)
        yield f


def ik_errors(gate, v: np.ndarray, q: dict) -> dict:
    """Position (mm) and orientation (°) residual of the joint solution q against the target of frame v, per arm."""
    from safety_gate import ARMS
    out = {}
    for a, (_, off) in ARMS.items():
        k = gate.K[a]; Rbc = k.Z[:3, :3]
        Ri, pi = gate._cam(v, off); R0, P0 = gate.H0[a]
        T = gate.he.compose(Rbc @ (Ri @ R0.T) @ Rbc.T @ gate.FK0[a][:3, :3], gate.FK0[a][:3, 3] + Rbc @ (pi - P0))
        Ta = k.ik.fk(q[a])
        if getattr(gate, "axis", False):   # axis_mode: the angle between the shafts
            c = np.clip(float((Ta[:3, :3] @ k.axis) @ (T[:3, :3] @ k.axis)), -1, 1)
        else:
            c = np.clip((np.trace(Ta[:3, :3].T @ T[:3, :3]) - 1) / 2, -1, 1)
        out[a] = (1000 * float(np.linalg.norm(Ta[:3, 3] - T[:3, 3])), float(np.degrees(np.arccos(c))))
    return out


def axis_mode(gate, weight: float, w_post: float) -> None:
    """What if the arms aligned the tools' SHAFTS (endo_follow --ik-axis: tip exact + shaft axis, roll free) instead of
    following the full orientation softly? The gate's arm models get that IK and its orientation test becomes the angle
    between the shafts. Analysis only: the follower and the inference gate are not changed by this."""
    import common as C
    from lerobot_robot_spinearm.ee import TipIK
    from safety_gate import ARMS
    rig = json.loads((ROBOT.parent / "bridge/src/spine_render/data/rig_default.json").read_text())["tools"]
    for arm, (tool, _) in ARMS.items():
        k = gate.K[arm]; b = C.BOUNDS[arm]
        lim = {j: (-b[j] + k.off[i] + 3.0, b[j] + k.off[i] - 3.0) for i, j in enumerate(C.JOINTS)}
        ax = np.asarray(rig[tool]["shaftLocal"], float)
        k.ik = TipIK(k.urdf, C.he.TIP_FRAME, lim, w_ori=0.1, w_post=w_post, vmax_dps=3000, rate_hz=90, axis_local=ax, w_axis=weight)
        k.ik.set_posture(gate.q[arm]); k.axis = ax / np.linalg.norm(ax)
    gate.axis = True


def check_axis(gate, v: np.ndarray) -> tuple[bool, str, dict]:
    """RobotGate.check with the orientation test on the shaft axis (axis_mode)."""
    from safety_gate import ARMS, CR_MIN, ARMS_MIN_MM, PERR_MM, OERR_DEG, RADIUS_M
    he, q, why = gate.he, {}, []
    for a, (tool, off) in ARMS.items():
        k = gate.K[a]; Rbc = k.Z[:3, :3]
        Ri, pi = gate._cam(v, off); R0, P0 = gate.H0[a]
        dp = pi - P0
        if np.linalg.norm(dp) > RADIUS_M:
            return False, f"{tool} {np.linalg.norm(dp) * 1000:.0f} mm from its start (> 80 mm replay sphere)", {}
        T = he.compose(Rbc @ (Ri @ R0.T) @ Rbc.T @ gate.FK0[a][:3, :3], gate.FK0[a][:3, 3] + Rbc @ dp)
        qa = k.ik.ik(gate.q[a], T, iters=10)
        Ta = k.ik.fk(qa)
        perr = 1000 * np.linalg.norm(Ta[:3, 3] - T[:3, 3])
        oerr = float(np.degrees(np.arccos(np.clip(float((Ta[:3, :3] @ k.axis) @ (T[:3, :3] @ k.axis)), -1, 1))))
        if perr > PERR_MM or oerr > OERR_DEG:
            why.append(f"{a} IK {perr:.1f} mm / {oerr:.1f}°")
        if gate.CL.mm(qa) < CR_MIN:
            why.append(f"{a} clearance {gate.CL.mm(qa):.0f} mm")
        if (k.margin(qa - k.off) < 0).any():
            why.append(f"{a} joint limit")
        q[a] = qa
    qz = {a: q[a] - gate.K[a].off for a in ARMS}
    ok, zwhy, _ = gate.zone.check(qz, strict=False, see=("left", "right"))
    if not ok:
        why += list(zwhy[:2])
    po = {a: gate.arms[a].pose(qz[a])["world"] for a in ARMS}
    d = 1000 * gate.sz.parts_dist(po["left"], po["right"], 0.07)
    if d < ARMS_MIN_MM:
        why.append(f"arms {d:.0f} mm apart")
    return (not why), "; ".join(map(str, why)), q


def kind(why: str) -> list:
    """The gate's reasons as categories."""
    ks = []
    for w in why.split("; "):
        w = w.strip()
        if not w:
            continue
        if " IK " in w:
            ks.append(w.split()[0] + " IK")
        elif "clearance" in w:
            ks.append(w.split()[0] + " clearance")
        elif "joint limit" in w:
            ks.append(w.split()[0] + " joint limit")
        elif "replay sphere" in w:
            ks.append(w.split()[0] + " 80 mm sphere")
        elif "apart" in w:
            ks.append("arms apart")
        else:
            ks.append("zone: " + w[:40])
    return ks


def fit_rotation(gate_K: dict, T0: dict, anchor: np.ndarray, pj: dict) -> tuple[np.ndarray, dict]:
    """The rotation Q (recording camera → robot camera) that turns the anchor's two tools onto the arms' own tools at the
    pair — the bone placed under the robot's tools as it lay under the surgeon's — and what is left per tool (°).
    A recording from another rig has its own camera axes; with them the arms' tools sit 10–20° off the hand's."""
    from safety_gate import ARMS, _q2R
    R0 = np.asarray(T0["R"], float).reshape(3, 3)
    Rh = {a: R0.T @ _q2R(anchor[off + 3:off + 7]) for a, (_, off) in ARMS.items()}
    Rr = {a: gate_K[a].T_cam(np.asarray(pj["q_left" if a == "left" else "q_right"], float))[:3, :3] for a in ARMS}
    U, _, Vt = np.linalg.svd(sum(Rr[a] @ Rh[a].T for a in ARMS))
    Q = U @ np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))]) @ Vt
    ang = lambda A, B: float(np.degrees(np.arccos(np.clip((np.trace(A.T @ B) - 1) / 2, -1, 1))))
    return Q, {"turn_deg": round(ang(np.eye(3), Q), 1), "off_before_deg": {a: round(ang(Rh[a], Rr[a]), 1) for a in ARMS},
               "off_after_deg": {a: round(ang(Q @ Rh[a], Rr[a]), 1) for a in ARMS}}


def q_angle(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Rotation angle (°) between unit quaternions, rows of qa against rows of qb → (len(qa), len(qb))."""
    d = np.abs(qa @ qb.T)
    return np.degrees(2 * np.arccos(np.clip(d, 0.0, 1.0)))


def anchors(eps, fps: float, every_s: float, pre_s: float) -> list:
    """Which recording's start would be the most central one? No arms here: per demo, the pose pre_s before its drilling
    starts (where a pair's window starts) against every demo's drilling frames — how far the tools would have to turn
    (and move) from that anchor. A 5-DOF arm follows a turn only in part, so a small turn is the first thing to ask for."""
    rid = lambda m: Path(m["recording"]).stem[-4:]
    every = max(1, int(round(every_s * fps)))
    drill = [(rid(m), st[np.nonzero(ph == 4)[0][::every]]) for m, st, ph in eps if (ph == 4).any()]
    rows = []
    for e, (m, st, ph) in enumerate(eps):
        k4 = np.nonzero(ph == 4)[0]
        if not len(k4):
            continue
        k = max(0, int(k4[0] - round(pre_s * fps)))
        row = {"episode": e, "rid": rid(m), "date": Path(m["recording"]).parent.name, "anchor_frame": k,
               "anchor_s": round((m["t_frame0_ms"] - m["t_exp_first_ms"]) / 1000.0 + k / fps, 2)}
        for tool, o in (("endo", 0), ("drill", 7)):
            per = [float(np.median(q_angle(st[k:k + 1, o + 3:o + 7], d[:, o + 3:o + 7]))) for _, d in drill]   # one value a demo
            mm = [float(np.median(np.linalg.norm(d[:, o:o + 3] - st[k, o:o + 3], axis=1))) for _, d in drill]
            row[tool] = {"deg_median": round(float(np.median(per)), 1), "deg_max": round(max(per), 1),
                         "demos_within_10deg": int(sum(v <= 10.0 for v in per)), "mm_median": round(float(np.median(mm)), 1),
                         "mm_max": round(max(mm), 1)}
        rows.append(row)
    return sorted(rows, key=lambda r: (-min(r["endo"]["demos_within_10deg"], r["drill"]["demos_within_10deg"]), r["endo"]["deg_max"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anchors", action="store_true", help="no arms: rank every demo's start (--pre-s before its drilling) "
                    "by how far the tools turn from it to all the demos' drilling")
    ap.add_argument("--pre-s", type=float, default=2.0)
    ap.add_argument("--dataset", required=True, help="a --state-phase dataset (the phase one-hot marks the drilling frames)")
    ap.add_argument("--pair", default=None, help="recording id (4 hex) whose robot start pair is checked")
    ap.add_argument("--anchor-s", type=float, default=None, help="session time of the pair's anchor (default: pairstart's window start)")
    ap.add_argument("--pairs-file", default="~/il-robot/runs/jitter_iter/work/pairstart/pairs_by_recording_chosen.json")
    ap.add_argument("--windows", default="~/il-robot/runs/jitter_iter/work/pairstart/windows.json")
    ap.add_argument("--q-left", default=None, help="six joint angles (deg, comma separated) instead of --pairs-file")
    ap.add_argument("--q-right", default=None)
    ap.add_argument("--fit-rotation", action="store_true", help="place the bone on the arms' tools: the scene's rotation is "
                    "fitted to the pair (both tools) instead of taken from the recording's camera — for a recording of another rig")
    ap.add_argument("--registration", default=None, help="fit_registration.py's file for the pair's recording (as "
                    "rollout.py --gate-registration) instead of fitting here")
    ap.add_argument("--ik-axis", action="store_true", help="analysis: the arms align the tools' shafts (endo_follow --ik-axis; "
                    "tip exact + shaft axis, roll free) and the gate tests the shaft angle — what turning the tools would reach")
    ap.add_argument("--axis-weight", type=float, default=1.0)
    ap.add_argument("--axis-w-post", type=float, default=0.1)
    ap.add_argument("--tag", default=None, help="a name for this variant in the output (e.g. the arm pair's origin)")
    ap.add_argument("--every-s", type=float, default=2.0)
    ap.add_argument("--step-mm", type=float, default=1.0, help="largest tool step between two IK solutions")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--only", default=None, help="comma separated recording ids (default: every episode)")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    ds = Path(a.dataset).expanduser()
    eps = episodes(ds)
    rid = lambda m: Path(m["recording"]).stem[-4:]
    if a.anchors:
        rows = anchors(eps, a.fps, a.every_s, a.pre_s)
        for r in rows:
            print(f"{r['rid']} {r['date'][5:]} ep {r['episode']:2d} @ {r['anchor_s']:6.2f} s | endo turn median {r['endo']['deg_median']:4.1f}° "
                  f"max {r['endo']['deg_max']:4.1f}° ({r['endo']['demos_within_10deg']:2d} demos ≤ 10°), {r['endo']['mm_median']:4.1f} mm | drill "
                  f"{r['drill']['deg_median']:4.1f}° max {r['drill']['deg_max']:4.1f}° ({r['drill']['demos_within_10deg']:2d}), {r['drill']['mm_median']:4.1f} mm")
        if a.out:
            Path(a.out).expanduser().write_text(json.dumps({"dataset": str(ds), "pre_s": a.pre_s, "anchors": rows}, indent=1))
        return
    if not a.pair:
        raise SystemExit("--pair RID (or --anchors)")
    from safety_gate import RobotGate, ARMS, OERR_DEG, PERR_MM
    mine = [i for i, (m, _, _) in enumerate(eps) if rid(m) == a.pair]
    if not mine:
        raise SystemExit(f"{a.pair}: no episode of this recording in {ds.name}")
    if a.q_left and a.q_right:
        pj = {"q_left": [float(x) for x in a.q_left.split(",")], "q_right": [float(x) for x in a.q_right.split(",")]}
    else:
        pj = json.loads(Path(a.pairs_file).expanduser().read_text())   # {rid: pair} (pairs_by_recording_chosen.json) or
        pj = pj["best"] if "best" in pj else pj[a.pair]                 # the search's own pair_clear_<rid>.json {"best": pair}
    anchor_s = a.anchor_s
    if anchor_s is None:   # the pair's window start: the 09-28 table (pairstart/recs.py) or new_recording.py's windows.json
        import recs
        win = Path(a.windows).expanduser()
        w = json.loads(win.read_text()) if win.exists() else {}
        if a.pair not in w and a.pair not in recs.REC:
            raise SystemExit(f"{a.pair}: no window start known — give --anchor-s")
        anchor_s = float(w[a.pair]["from_s"] if a.pair in w else recs.REC[a.pair])
    m0, s0, _ = eps[mine[0]]
    k_an = int(round((anchor_s - (m0["t_frame0_ms"] - m0["t_exp_first_ms"]) / 1000.0) * a.fps))
    if not 0 <= k_an < len(s0):
        raise SystemExit(f"anchor {anchor_s} s: frame {k_an} is outside episode {mine[0]} (0..{len(s0) - 1})")
    with open(m0["recording"]) as fh:
        hdr = next(d for d in map(json.loads, fh) if d.get("k") == "hdr")
    T0, fitted = hdr["registration"]["T0"], None
    if a.registration:
        g = json.loads(Path(a.registration).expanduser().read_text())
        T0, fitted = {"R": g["R"], "t": g["t"]}, {"file": str(a.registration), **(g.get("fit") or {})}
    elif a.fit_rotation:
        import common as C   # pairstart, on sys.path through safety_gate
        Q, fitted = fit_rotation({arm: C.Kin(arm, margin=3.0, w_ori=0.1, w_post=0.1) for arm in ARMS}, T0, s0[k_an], pj)
        T0 = {"R": (np.asarray(T0["R"], float).reshape(3, 3) @ Q.T).tolist(), "t": T0["t"]}
        print(f"fitted rotation {fitted['turn_deg']}°: tools off the hand's {fitted['off_before_deg']} → {fitted['off_after_deg']}", flush=True)
    gate = RobotGate(s0[k_an], T0, pj["q_left"], pj["q_right"])
    q_anchor = {arm: gate.q[arm].copy() for arm in ARMS}
    check = gate.check
    if a.ik_axis:
        axis_mode(gate, a.axis_weight, a.axis_w_post)
        check = lambda f: check_axis(gate, f)
    anchor = s0[k_an].copy()
    only = set(a.only.split(",")) if a.only else None
    every = max(1, int(round(a.every_s * a.fps)))
    res, t0 = {"dataset": str(ds), "pair": a.pair, "tag": a.tag, "anchor_s": anchor_s, "anchor_frame": k_an, "q": pj, "every_s": a.every_s,
               "fit_rotation": fitted, "ik_axis": {"weight": a.axis_weight, "w_post": a.axis_w_post} if a.ik_axis else None,
               "limits": {"ik_mm": PERR_MM, "ik_deg": OERR_DEG}, "demos": []}, time.time()
    for e, (m, st, ph) in enumerate(eps):
        if only and rid(m) not in only:
            continue
        ks = np.nonzero(ph == 4)[0][::every]
        if not len(ks):
            continue
        gate.q.update({arm: q_anchor[arm].copy() for arm in ARMS})
        prev, n_ok, why_n, oe, pe, lost = anchor, 0, collections.Counter(), {arm: [] for arm in ARMS}, {arm: [] for arm in ARMS}, 0
        dq = {arm: 0.0 for arm in ARMS}
        for k in ks:
            ok, why, q = False, "", {}
            for f in slerp_path(prev, st[k], a.step_mm):   # the arms follow the tools there, 1 mm at a time
                ok, why, q = check(f)
                if q:
                    gate.q.update(q)
            prev = st[k]
            if not q:   # outside the replay sphere: no joint solution to measure
                lost += 1
                why_n.update(kind(why))
                continue
            er = ik_errors(gate, st[k], q)
            for arm in ARMS:
                pe[arm].append(er[arm][0]); oe[arm].append(er[arm][1])
                dq[arm] = max(dq[arm], float(np.abs(q[arm] - q_anchor[arm]).max()))
            n_ok += bool(ok)
            why_n.update(set(kind(why)))
        row = {"episode": e, "rid": rid(m), "date": Path(m["recording"]).parent.name, "n": int(len(ks)), "pass": round(n_ok / len(ks), 3),
               "ik_deg": {arm: [round(float(np.median(v)), 1), round(float(np.max(v)), 1)] if v else None for arm, v in oe.items()},
               "ik_mm_max": {arm: round(float(np.max(v)), 1) if v else None for arm, v in pe.items()},
               "joint_max_deg": {arm: round(v, 1) for arm, v in dq.items()},
               "stops": {k: round(v / len(ks), 3) for k, v in why_n.most_common()}}
        res["demos"].append(row)
        print(f"{row['rid']} {row['date'][5:]} ep {e:2d} | {row['n']:3d} frames | pass {row['pass']:.2f} | IK° med/max left "
              f"{row['ik_deg']['left']} right {row['ik_deg']['right']} | joints ≤ {row['joint_max_deg']['left']:.0f}/{row['joint_max_deg']['right']:.0f}° | "
              f"{row['stops']}  [{time.time() - t0:.0f} s]", flush=True)
    by = collections.defaultdict(list)
    for r in res["demos"]:
        by[r["date"]].append(r)
    res["by_date"] = {d: {"demos": len(v), "frames": sum(r["n"] for r in v),
                          "pass": round(sum(r["pass"] * r["n"] for r in v) / sum(r["n"] for r in v), 3),
                          "pass_min": min(r["pass"] for r in v)} for d, v in sorted(by.items())}
    n = sum(r["n"] for r in res["demos"])
    res["pass"] = round(sum(r["pass"] * r["n"] for r in res["demos"]) / n, 3) if n else None
    print(json.dumps({"pair": a.pair, "anchor_s": anchor_s, "pass": res["pass"], "by_date": res["by_date"]}))
    if a.out:
        Path(a.out).expanduser().write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
