"""Exact arm-to-arm minimum (cutoff 70 mm) of each best pair over its 0.1 s replay grid (the every-sample pass used a
50 mm cutoff, so its value is only a lower bound)."""
import sys, json, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import recs
import verify_two_arm as V
from lerobot_robot_spinearm import safezone as sz
arms, cam, info = V.load_models()
K = {a: Kin(a, margin=3.0, w_ori=0.1, w_post=0.1) for a in ("left", "right")}
rid = sys.argv[1]; R = recs.load(rid)
f = P / f"pair_{rid}.json"; D = json.load(open(f)); b = D["others"][0]
q0 = {"left": np.array(b["q_left"]), "right": np.array(b["q_right"])}
q, FK0 = {}, {}
for a in K:
    K[a].ik.set_posture(q0[a] + K[a].off); q[a] = q0[a] + K[a].off; FK0[a] = K[a].ik.fk(q[a])
amin, where = 1e9, None
zr = sz.Zone(arms, cam); zr.set_start(NEUTRAL)
[zr.self_floor[a].__setitem__(kk, zr.self_floor[a][kk] - 1.5) for a in zr.self_floor for kk in list(zr.self_floor[a])]
nbad = 0; M = {a: 1e9 for a in K}
for t in R["rp"]["drill"]["t"]:
    for a in K:
        tool = TOOL[a]; rp = R["rp"][tool]; Rbc = K[a].Z[:3, :3]
        Ri, pi, _ = rec.sample_pose(rp, t + R["sh"][tool], "linear", 0.25)
        dp = Ri @ rp["tip_local_m"] + pi - R["P0"][tool]; dR = Ri @ R["R0"][tool].T
        if np.linalg.norm(dp) > 0.08: dp *= 0.08 / np.linalg.norm(dp)
        q[a] = K[a].ik.ik(q[a], he.compose(Rbc @ dR @ Rbc.T @ FK0[a][:3, :3], FK0[a][:3, 3] + Rbc @ dp), iters=10)
    po = {a: arms[a].pose(q[a] - K[a].off)["world"] for a in K}
    d = 1000 * sz.parts_dist(po["left"], po["right"], 0.05)
    nbad += not zr.check({a: q[a] - K[a].off for a in K}, strict=False, see=("left", "right"))[0]
    for a in K: M[a] = min(M[a], float(K[a].margin(q[a] - K[a].off).min()))
    if d < amin: amin, where = d, round(float(t), 1)
b["replay_unsafe_every_sample"] = nbad; b["replay_n_every_sample"] = len(R["rp"]["drill"]["t"]); b["joint_margin_min_every_sample"] = {a: round(v, 1) for a, v in M.items()}
b["arms_min_mm_every_sample_lb"] = round(amin, 1)
json.dump(D, open(f, "w"), indent=1)
print(rid, "alt", nbad, len(R["rp"]["drill"]["t"]), round(amin, 1), M)
