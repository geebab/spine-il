"""Clearance (jaw/gripper frames -> motor 2, as endo_follow) of the current chosen pairs: start + replay min
(TipIK w_ori 0.1 / post 0.1, 0.1 s grid and every recorded sample)."""
import sys, json, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import recs
from lerobot_robot_spinearm.ee import Clearance
CL = Clearance()
K = {a: Kin(a, margin=3.0, w_ori=0.1, w_post=0.1) for a in ("left", "right")}

def clear_replay(R, q0, times, radius=0.08):
    q, FK0 = {}, {}
    for a in K:
        K[a].ik.set_posture(q0[a] + K[a].off); q[a] = q0[a] + K[a].off; FK0[a] = K[a].ik.fk(q[a])
    cmin = {a: (1e9, None) for a in K}
    for t in times:
        for a in K:
            tool = TOOL[a]; rp = R["rp"][tool]; Rbc = K[a].Z[:3, :3]
            Ri, pi, _ = rec.sample_pose(rp, t + R["sh"][tool], "linear", 0.25)
            dp = Ri @ rp["tip_local_m"] + pi - R["P0"][tool]; dR = Ri @ R["R0"][tool].T
            if np.linalg.norm(dp) > radius: dp *= radius / np.linalg.norm(dp)
            T = he.compose(Rbc @ dR @ Rbc.T @ FK0[a][:3, :3], FK0[a][:3, 3] + Rbc @ dp)
            q[a] = K[a].ik.ik(q[a], T, iters=10)
            c = CL.mm(q[a])
            if c < cmin[a][0]: cmin[a] = (c, round(float(t), 1))
    return cmin

if __name__ == "__main__":
    d = json.load(open(P / "pairs_by_recording_chosen.json"))
    out = {}
    for rid, b in d.items():
        R = recs.load(rid)
        q0 = {"left": np.array(b["q_left"]), "right": np.array(b["q_right"])}
        cs = {a: round(CL.mm(q0[a] + K[a].off), 1) for a in K}
        g = clear_replay(R, q0, np.arange(0, R["T"] + 1e-9, 0.1))
        e = clear_replay(R, q0, R["rp"]["drill"]["t"])
        out[rid] = dict(clear_start_mm=cs, clear_replay_min_mm={a: round(min(g[a][0], e[a][0]), 1) for a in K},
                        clear_replay_min_t_s={a: (g[a] if g[a][0] <= e[a][0] else e[a])[1] for a in K})
        print(rid, out[rid], flush=True)
    json.dump(out, open(P / "clear_current.json", "w"), indent=1)
