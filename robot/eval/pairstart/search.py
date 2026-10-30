"""Per recording: re-target stage B/C pairs to that recording's start vector, strict pair check, then paths +
30 s replay (relaxed zone, every 0.1 s) for the best few; final pick re-verified at every recorded sample."""
import sys, time, pickle, json, math, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import recs
import verify_two_arm as V
from lerobot_robot_spinearm import safezone as sz
rid = sys.argv[1]; NFULL = int(sys.argv[2]) if len(sys.argv) > 2 else 12
R = recs.load(rid); vt = R["vec"]; HR = {"left": R["R0"]["endo"], "right": R["R0"]["drill"]}
B = pickle.load(open(P / "stageB.pkl", "rb"))
arms, cam, info = V.load_models(); z = sz.Zone(arms, cam); z.set_start(NEUTRAL)
zr = sz.Zone(arms, cam); zr.set_start(NEUTRAL)
for a in zr.self_floor:
    for kk in list(zr.self_floor[a]): zr.self_floor[a][kk] -= 1.5
KS = {a: Kin(a, margin=5.0, w_ori=1.0, w_post=1e-3) for a in ("left", "right")}
CAP = {"left": {"shoulder_pan": (-55, 55), "wrist_roll": (-90, 90), "gripper": (-59.8, -30)},
       "right": {"shoulder_pan": (-55, 55), "wrist_roll": (-90, 90), "gripper": (-59.8, -26)}}
for a, k in KS.items():
    for j, (lo, hi) in CAP[a].items():
        i = JOINTS.index(j)
        k.ik.robot.set_joint_limits(j, float(np.deg2rad(max(lo, k.lo[i] + 5) + k.off[i])), float(np.deg2rad(min(hi, k.hi[i] - 5) + k.off[i])))
K = {a: Kin(a, margin=3.0, w_ori=0.1, w_post=0.1) for a in ("left", "right")}      # replay IK (endo_follow)
t0 = time.time()
def log(*a): print(rid, f"{time.time()-t0:.0f}s", *a, flush=True)

seeds = []
for p in B:
    if p["ok"] and all(np.abs(p["l"]["q"] - s["l"]["q"]).max() + np.abs(p["r"]["q"] - s["r"]["q"]).max() > 1.0 for s in seeds):
        seeds.append(p)
log("seeds", len(seeds), "target", (vt * 1000).round(1))

# ---- per-arm variants
FR = ((1.0, 0.0), (0.5, 0.5), (0.0, 1.0))           # (left share, right share) of the vector correction
ORI = ("keep", "half", "hand")
var = {}
for si, p in enumerate(seeds):
    D = vt - (p["r"]["tip"] - p["l"]["tip"])
    for a, sgn, key in (("left", -1, "l"), ("right", 1, "r")):
        s = p[key]
        for fi, fr in enumerate(FR):
            f = fr[0] if a == "left" else fr[1]
            tgt = s["tip"] + sgn * f * D
            for on in ORI:
                Rt = s["R"] if on == "keep" else (slerp(s["R"], HR[a], 0.5) if on == "half" else HR[a])
                if on == "keep" and f == 0:
                    q = s["q"].copy()
                else:
                    q = KS[a].solve(s["q"], Rt, tgt, n=250, post=s["q"])
                T = KS[a].T_cam(q); perr = 1000 * np.linalg.norm(T[:3, 3] - tgt)
                if perr > 0.5: continue
                ok, why, v = z.single(a, q, strict=True)
                if not ok: continue
                var[(si, a, fi, on)] = dict(q=q, tip=T[:3, 3], R=T[:3, :3], oerr=rot_angle(T[:3, :3], HR[a]),
                                            marg=float(KS[a].margin(q).min()), dq=float(np.abs(q - CUR[a])[:5].sum()))
    if si % 50 == 0: log("variants", si, len(var))
log("variants done", len(var))

cands = []
for si in range(len(seeds)):
    for fi in range(3):
        for oL in ORI:
            for oR in ORI:
                l, r = var.get((si, "left", fi, oL)), var.get((si, "right", fi, oR))
                if l is None or r is None: continue
                vec = r["tip"] - l["tip"]; ve = 1000 * np.linalg.norm(vec - vt)
                if ve > 3: continue
                pre = l["oerr"] + r["oerr"] - 1.0 * min(l["marg"], 15) - 1.0 * min(r["marg"], 15) + 0.05 * (l["dq"] + r["dq"])
                cands.append(dict(si=si, fi=fi, ori=(oL, oR), l=l, r=r, vec=vec, ve=ve, pre=pre))
cands.sort(key=lambda c: c["pre"]); log("pair candidates", len(cands))
okc = []
for c in cands:
    if any(np.abs(c["l"]["q"] - o["l"]["q"]).max() + np.abs(c["r"]["q"] - o["r"]["q"]).max() < 4 for o in okc): continue
    ok, why, m = z.check({"left": c["l"]["q"], "right": c["r"]["q"]}, strict=True, see=("left", "right"))
    if ok:
        c["m"] = m; okc.append(c)
    if len(okc) >= 3 * NFULL: break
log("strict-ok diverse", len(okc))

def paths(qL, qR, zz):
    N = NEUTRAL
    ok1, w1, _ = zz.path({"left": N["left"], "right": N["right"]}, {"left": N["left"], "right": qR}, see=("right",))
    ok2, w2, _ = zz.path({"left": N["left"], "right": qR}, {"left": qL, "right": qR}, see=("left", "right"))
    ok2b, w2b, _ = (ok2, w2, 0) if ok2 else zz.path({"left": N["left"], "right": qR}, {"left": qL, "right": qR}, see=("left",))
    return {"right_first": [ok1, w1[:2]], "left_second_see_both": [ok2, w2[:2]], "left_second_see_left": [ok2b, w2b[:2]]}

def armdist(q, cut=0.07):
    po = {a: arms[a].pose(q[a])["world"] for a in arms}
    return 1000 * sz.parts_dist(po["left"], po["right"], cut)

def closest(q):
    po = {a: arms[a].pose(q[a])["world"] for a in arms}; best = (1e9, "")
    for ga, A_ in po["left"].items():
        for gb, B_ in po["right"].items():
            if ga == gb == "base": continue
            dd = sz.min_dist(A_, B_, 0.2) * 1000
            if dd < best[0]: best = (dd, f"L.{ga}-R.{gb}")
    return round(best[0], 1), best[1]

def replay(q0, times, radius=0.08, cut=0.06):
    """Robot tip target = start tip + hand tip displacement (camera → base via R_bc), orientation dR·R_start, TipIK."""
    q, FK0 = {}, {}
    for a in K:
        K[a].ik.set_posture(q0[a] + K[a].off); q[a] = q0[a] + K[a].off; FK0[a] = K[a].ik.fk(q[a])
    bad, amin, mmin, perr_max, sepr, oerr_max = [], 1e9, {a: np.full(6, 1e9) for a in K}, 0.0, [], {a: 0.0 for a in K}
    viewmin = {a: dict(margin_px=1e9, window_px=1e9, view_deg_max=0, los_mm=1e9) for a in K}
    for t in times:
        tips = {}
        for a in K:
            tool = TOOL[a]; rp = R["rp"][tool]; Rbc = K[a].Z[:3, :3]
            Ri, pi, _ = rec.sample_pose(rp, t + R["sh"][tool], "linear", 0.25)
            dp = Ri @ rp["tip_local_m"] + pi - R["P0"][tool]; dR = Ri @ R["R0"][tool].T
            if np.linalg.norm(dp) > radius: dp *= radius / np.linalg.norm(dp)
            T = he.compose(Rbc @ dR @ Rbc.T @ FK0[a][:3, :3], FK0[a][:3, 3] + Rbc @ dp)
            q[a] = K[a].ik.ik(q[a], T, iters=10)
            Ta = K[a].ik.fk(q[a]); perr_max = max(perr_max, 1000 * np.linalg.norm(Ta[:3, 3] - T[:3, 3]))
            oerr_max[a] = max(oerr_max[a], rot_angle(Ta[:3, :3], T[:3, :3]))
            mmin[a] = np.minimum(mmin[a], K[a].margin(q[a] - K[a].off))
            tips[a] = (K[a].Zi @ Ta)[:3, 3]
        sepr.append(1000 * np.linalg.norm(tips["right"] - tips["left"]))
        qa = {a: q[a] - K[a].off for a in K}
        ok, why, m = zr.check(qa, strict=False, see=("left", "right"))
        amin = min(amin, armdist(qa, cut))
        for a in K:
            v = m[a]; vm = viewmin[a]
            vm["margin_px"] = min(vm["margin_px"], v["margin_px"]); vm["window_px"] = min(vm["window_px"], v["window_px"])
            vm["view_deg_max"] = max(vm["view_deg_max"], v["view_deg"]); vm["los_mm"] = min(vm["los_mm"], v["los_mm"])
        if not ok: bad.append((round(float(t), 2), why[:2]))
    return dict(n_check=len(times), n_bad=len(bad), bad=bad[:6], arms_min=round(amin, 1),
                marg_min={a: dict(zip(JOINTS, v.round(1).tolist())) for a, v in mmin.items()}, perr_max=round(perr_max, 2),
                ori_track_err_max_deg={a: round(v, 1) for a, v in oerr_max.items()}, view=viewmin,
                sep_mm=[round(float(np.min(sepr)), 1), round(float(np.median(sepr)), 1), round(float(np.max(sepr)), 1)])

grid = np.arange(0, R["T"] + 1e-9, 0.1)
full = []
for n, c in enumerate(okc[:NFULL]):
    q = {"left": c["l"]["q"], "right": c["r"]["q"]}
    c["paths"] = {"strict": paths(q["left"], q["right"], z), "relaxed_1p5mm": paths(q["left"], q["right"], zr)}
    c["closest"] = closest(q)
    c["rep"] = replay(q, grid)
    pr = c["paths"]["relaxed_1p5mm"]; c["paths_ok"] = bool(pr["right_first"][0] and pr["left_second_see_left"][0])
    c["jm"] = {a: min(c["rep"]["marg_min"][a].values()) for a in q}
    full.append(c)
    log(n, "seed", c["si"], "fr", FR[c["fi"]], c["ori"], "ve", round(c["ve"], 2), "oerr", round(c["l"]["oerr"], 1), round(c["r"]["oerr"], 1),
        "arms0", c["closest"], "paths", c["paths_ok"], "replay bad", c["rep"]["n_bad"], "/", c["rep"]["n_check"], "armsmin", c["rep"]["arms_min"], "jm", c["jm"])

def rank(c):
    return (c["ve"] > 3, c["rep"]["n_bad"] > 0, not c["paths_ok"], -math.floor(min(c["rep"]["arms_min"], 60) / 5),
            min(c["jm"].values()) < 5, c["l"]["oerr"] + c["r"]["oerr"], c["l"]["dq"] + c["r"]["dq"])
full.sort(key=rank)
# final: the winner through every recorded (drill) sample
tall = R["rp"]["drill"]["t"]
for c in full[:3]:
    q = {"left": c["l"]["q"], "right": c["r"]["q"]}
    c["rep_all"] = replay(q, tall, cut=0.05)
    c["rep_all"]["arms_min"] = min(c["rep_all"]["arms_min"], c["rep"]["arms_min"])
    log("final", "seed", c["si"], "every-sample bad", c["rep_all"]["n_bad"], "/", c["rep_all"]["n_check"], "armsmin", c["rep_all"]["arms_min"])
    if c["rep_all"]["n_bad"] == 0: break

def out(c):
    q = {"left": c["l"]["q"], "right": c["r"]["q"]}; rp = c.get("rep_all", c["rep"])
    return dict(q_left=q["left"].round(2).tolist(), q_right=q["right"].round(2).tolist(),
                vec_target_mm=(vt * 1000).round(1).tolist(), vec_mm=(c["vec"] * 1000).round(1).tolist(), vec_err_mm=round(c["ve"], 2),
                sep_mm=round(float(np.linalg.norm(c["vec"]) * 1000), 1),
                ori_err_deg={"left": round(c["l"]["oerr"], 1), "right": round(c["r"]["oerr"], 1)},
                arms_min_mm_start=c["closest"][0], closest_parts_start=c["closest"][1], strict_start_arms_mm=c["m"]["arms_mm"],
                arms_min_mm_replay=rp["arms_min"], replay_unsafe=rp["n_bad"], replay_n_checked=rp["n_check"], replay_bad=rp["bad"],
                replay_every_sample=("rep_all" in c),
                joint_margin_min_deg={a: min(rp["marg_min"][a].values()) for a in q},
                joint_margin_min_joint={a: min(rp["marg_min"][a], key=rp["marg_min"][a].get) for a in q},
                joint_margin_start_deg={a: round(float(KS[a].margin(q[a]).min()), 1) for a in q},
                replay_perr_max_mm=rp["perr_max"], replay_ori_track_err_max_deg=rp["ori_track_err_max_deg"], replay_sep_mm=rp["sep_mm"],
                replay_view=rp["view"], paths=c["paths"], paths_ok_relaxed=c["paths_ok"],
                dq_from_current={a: (q[a] - CUR[a]).round(1).tolist() for a in q},
                seed=dict(stageB_index=c["si"], shift_share=FR[c["fi"]], ori_mode=c["ori"]))
res = dict(best=out(full[0]), others=[out(c) for c in full[1:6]], n_seeds=len(seeds), n_pair_candidates=len(cands), n_strict_ok=len(okc),
           n_full=len(full))
json.dump(res, open(P / f"pair_{rid}.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else (o.tolist() if hasattr(o, "tolist") else str(o)))
log("DONE best", res["best"]["q_left"], res["best"]["q_right"], "unsafe", res["best"]["replay_unsafe"], "armsmin", res["best"]["arms_min_mm_replay"])
