"""Top pairs: closest parts, rest->start paths, perturbations, full offline replay of the 3868 recording (both arms)."""
import sys, time, pickle, json, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
from common import _exp
import verify_two_arm as V
from lerobot_robot_spinearm import safezone as sz
A = pickle.load(open(P / "stageA.pkl", "rb")); B = pickle.load(open(P / "stageB.pkl", "rb"))
HR = A["HR"]; d = A["d"]
arms, cam, info = V.load_models(); z = sz.Zone(arms, cam); z.set_start(NEUTRAL)
K = {a: Kin(a, margin=3.0, w_ori=0.1, w_post=0.1) for a in ("left", "right")}   # replay IK (endo_follow defaults here)
KP = {a: Kin(a, margin=3.0, w_ori=1.0, w_post=1e-3) for a in ("left", "right")}  # perturbation IK
hs = hand_start()
import copy
zr = sz.Zone(arms, cam); zr.set_start(NEUTRAL)
for a in zr.self_floor:
    for kk in list(zr.self_floor[a]): zr.self_floor[a][kk] -= 1.5     # near-rest model dips (current start fails too)
def paths(qL, qR, zz):
    N = NEUTRAL
    ok1, w1, _ = zz.path({"left": N["left"], "right": N["right"]}, {"left": N["left"], "right": qR}, see=("right",))
    ok2, w2, _ = zz.path({"left": N["left"], "right": qR}, {"left": qL, "right": qR}, see=("left", "right"))
    if not ok2:
        ok2b, w2b, _ = zz.path({"left": N["left"], "right": qR}, {"left": qL, "right": qR}, see=("left",))
        return [ok1, w1[:2]], [ok2, w2[:2]], [ok2b, w2b[:2]]
    return [ok1, w1[:2]], [ok2, w2[:2]], [ok2, w2[:2]]
print("current start paths strict", paths(CUR["left"], CUR["right"], z), "relaxed", paths(CUR["left"], CUR["right"], zr), flush=True)
NTOP = int(sys.argv[1]) if len(sys.argv) > 1 else 20

def score(p):
    m = p["m"]
    return (p["l"]["oerr"] + p["r"]["oerr"] - 0.3 * min(m["arms_mm"], 80) - 0.5 * min(p["l"]["marg"], 20)
            - 0.5 * min(p["r"]["marg"], 20) + 0.1 * (np.abs(p["l"]["q"] - CUR["left"])[:5].sum() + np.abs(p["r"]["q"] - CUR["right"])[:5].sum()) / 5)
def score2(p):
    return p["l"]["oerr"] + p["r"]["oerr"] - 2.0 * min(p["l"]["marg"], 20) - 2.0 * min(p["r"]["marg"], 20) - 0.3 * min(p["m"]["arms_mm"], 80)
sel = []
for key in (score, score2):
    n0 = len(sel)
    for p in sorted([p for p in B if p["ok"]], key=key):
        if all(np.abs(p["l"]["q"] - s["l"]["q"]).max() + np.abs(p["r"]["q"] - s["r"]["q"]).max() > 6 for s in sel):
            sel.append(p)
        if len(sel) - n0 >= NTOP: break

def closest(q):
    po = {a: arms[a].pose(q[a])["world"] for a in arms}; best = (1e9, "")
    for ga, A_ in po["left"].items():
        for gb, B_ in po["right"].items():
            if ga == gb == "base": continue
            dd = sz.min_dist(A_, B_, 0.2) * 1000
            if dd < best[0]: best = (dd, f"L.{ga}-R.{gb}")
    return round(best[0], 1), best[1]

def perturb(q, other):
    out = []
    for a, oa in (("left", "right"), ("right", "left")):
        k = KP[a]; T0 = k.T_cam(q[a])
        moves = [("t", s * 0.015 * e) for e in np.eye(3) for s in (1, -1)] + \
                [("r", s * np.radians(10) * e) for e in np.eye(3)[:2] for s in (1, -1)]
        for kind, v in moves:
            R = T0[:3, :3] if kind == "t" else _exp(v) @ T0[:3, :3]
            p = T0[:3, 3] + (v if kind == "t" else 0)
            qa = k.solve(q[a], R, p, n=150, post=q[a])
            perr = 1000 * np.linalg.norm(k.tip_cam(qa) - p)
            okc, why, m = z.check({a: qa, oa: q[oa]}, strict=False, see=("left", "right"))
            out.append(dict(arm=a, kind=kind, v=(v * (1000 if kind == "t" else 57.2958)).round(1).tolist(), ok=bool(okc and perr < 2),
                            why=why[:2], perr=round(perr, 1), arms_mm=m.get("arms_mm"), marg=round(float(K[a].margin(qa).min()), 1)))
    return out

def replay(q0, dt=0.1, check_every=5, radius=0.08):
    ts = np.arange(0, 54.4, dt); q = {}; FK0 = {}
    for a in K:
        K[a].ik.set_posture(q0[a] + K[a].off); q[a] = q0[a] + K[a].off; FK0[a] = K[a].ik.fk(q[a])
    bad, amin, mmin, perr_max, sepr = [], 1e9, {a: np.full(6, 1e9) for a in K}, 0.0, []
    amin30 = 1e9
    for i, t in enumerate(ts):
        tips = {}
        for a in K:
            rp = hs[TOOL[a]]; Rbc = K[a].Z[:3, :3]
            Ri, pi, _ = rec.sample_pose(rp, t, "linear", 0.25)
            dp = Ri @ rp["tip_local_m"] + pi - rp["H0_tip"]; dR = Ri @ rp["H0_R"].T
            if np.linalg.norm(dp) > radius: dp *= radius / np.linalg.norm(dp)
            T = he.compose(Rbc @ dR @ Rbc.T @ FK0[a][:3, :3], FK0[a][:3, 3] + Rbc @ dp)
            q[a] = K[a].ik.ik(q[a], T, iters=10)
            Ta = K[a].ik.fk(q[a]); perr_max = max(perr_max, 1000 * np.linalg.norm(Ta[:3, 3] - T[:3, 3]))
            mmin[a] = np.minimum(mmin[a], K[a].margin(q[a] - K[a].off))
            tips[a] = (K[a].Zi @ Ta)[:3, 3]
        sepr.append(1000 * np.linalg.norm(tips["right"] - tips["left"]))
        if i % check_every == 0:
            okc, why, m = z.check({a: q[a] - K[a].off for a in K}, strict=False, see=("left", "right"))
            amin = min(amin, m.get("arms_mm", 1e9))
            if t < 30: amin30 = min(amin30, m.get("arms_mm", 1e9))
            if not okc: bad.append((round(t, 1), why[:2]))
    return dict(n_check=len(ts[::check_every]), n_bad=len(bad), n_bad_first30s=sum(b[0] < 30 for b in bad), bad=bad[:6],
                arms_min=round(amin, 1), arms_min_first30s=round(amin30, 1),
                marg_min={a: dict(zip(JOINTS, v.round(1).tolist())) for a, v in mmin.items()}, perr_max=round(perr_max, 1),
                sep_mm=[round(float(np.min(sepr)), 1), round(float(np.median(sepr)), 1), round(float(np.max(sepr)), 1)])

from lerobot_robot_spinearm import recording as rec
out = []; t0 = time.time()
for n, p in enumerate(sel):
    q = {"left": p["l"]["q"], "right": p["r"]["q"]}
    r = dict(rank_score=round(score(p), 1), P_endo_mm=(A["Ps"][p["ip"]] * 1000).round(1).tolist(),
             q_left=q["left"].round(2).tolist(), q_right=q["right"].round(2).tolist(),
             tip_endo_mm=(p["l"]["tip"] * 1000).round(1).tolist(), tip_drill_mm=(p["r"]["tip"] * 1000).round(1).tolist(),
             vec_mm=p["vec"].round(1).tolist(), vec_target_mm=(d * 1000).round(1).tolist(),
             vec_err_mm=round(float(np.linalg.norm(p["vec"] - d * 1000)), 2), sep_mm=round(float(np.linalg.norm(p["vec"])), 1),
             ori_err_deg={"left": round(p["l"]["oerr"], 1), "right": round(p["r"]["oerr"], 1)},
             ori_mode={"left": p["l"]["ori"], "right": p["r"]["ori"]},
             joint_margin_deg={"left": dict(zip(JOINTS, K["left"].margin(q["left"]).round(1).tolist())),
                               "right": dict(zip(JOINTS, K["right"].margin(q["right"]).round(1).tolist()))},
             dq_from_current={a: (q[a] - CUR[a]).round(1).tolist() for a in q},
             strict_check=dict(ok=p["ok"], why=p["why"], arms_mm=p["m"]["arms_mm"], sep_px=p["m"]["sep_px"],
                               **{a: {k: p["m"][a][k] for k in ("z_mm", "view_deg", "window_px", "margin_px", "apart_px", "los_mm")} for a in q}))
    r["closest_parts"] = closest(q)
    ps = paths(q["left"], q["right"], z); pr = paths(q["left"], q["right"], zr)
    r["paths"] = dict(strict=dict(zip(("right_first", "left_second_see_both", "left_second_see_left"), ps)),
                      relaxed_1p5mm=dict(zip(("right_first", "left_second_see_both", "left_second_see_left"), pr)))
    ok1, ok2, ok2b = pr[0][0], pr[1][0], pr[2][0]
    pt = perturb(q, None); r["perturb_ok"] = f"{sum(x['ok'] for x in pt)}/{len(pt)}"
    r["perturb_fail"] = [x for x in pt if not x["ok"]][:6]
    r["perturb_arms_min"] = min(x["arms_mm"] for x in pt); r["perturb_marg_min"] = min(x["marg"] for x in pt)
    r["replay"] = replay(q)
    out.append(r)
    print(n, f"{time.time()-t0:.0f}s", r["closest_parts"], "paths", ok1, ok2, ok2b, "pert", r["perturb_ok"], r["perturb_arms_min"],
          "replay bad", r["replay"]["n_bad"], r["replay"]["n_bad_first30s"], "/", r["replay"]["n_check"], "arms_min", r["replay"]["arms_min"], {a: min(v.values()) for a, v in r["replay"]["marg_min"].items()}, flush=True)
    json.dump(out, open(P / "stageC.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
