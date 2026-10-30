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
f = P / f"pair_clear_{rid}.json"; D = json.load(open(f)); b = D["best"]
q0 = {"left": np.array(b["q_left"]), "right": np.array(b["q_right"])}
q, FK0 = {}, {}
for a in K:
    K[a].ik.set_posture(q0[a] + K[a].off); q[a] = q0[a] + K[a].off; FK0[a] = K[a].ik.fk(q[a])
amin, where = 1e9, None
for t in np.arange(0, R["T"] + 1e-9, 0.1):
    for a in K:
        tool = TOOL[a]; rp = R["rp"][tool]; Rbc = K[a].Z[:3, :3]
        Ri, pi, _ = rec.sample_pose(rp, t + R["sh"][tool], "linear", 0.25)
        dp = Ri @ rp["tip_local_m"] + pi - R["P0"][tool]; dR = Ri @ R["R0"][tool].T
        _r = float(__import__('os').environ.get('PS_RADIUS_M', '0.08'))
        if np.linalg.norm(dp) > _r: dp *= _r / np.linalg.norm(dp)
        q[a] = K[a].ik.ik(q[a], he.compose(Rbc @ dR @ Rbc.T @ FK0[a][:3, :3], FK0[a][:3, 3] + Rbc @ dp), iters=10)
    po = {a: arms[a].pose(q[a] - K[a].off)["world"] for a in K}
    d = 1000 * sz.parts_dist(po["left"], po["right"], 0.07)
    if d < amin: amin, where = d, round(float(t), 1)
b["arms_min_mm_replay_every_sample_lower_bound"] = b["arms_min_mm_replay"]
b["arms_min_mm_replay"] = round(amin, 1); b["arms_min_replay_t_s"] = where
json.dump(D, open(f, "w"), indent=1)
print(rid, "arms_min grid", round(amin, 1), "at", where, "every-sample lb", b["arms_min_mm_replay_every_sample_lower_bound"], flush=True)
