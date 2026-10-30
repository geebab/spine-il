"""Per-arm IK candidates on a camera-frame grid of endo tip points P (drill at P + d)."""
import sys, time, itertools, pickle, numpy as np
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import verify_two_arm as V
from lerobot_robot_spinearm import safezone as sz

hs = hand_start()
d = hs["drill"]["H0_tip"] - hs["endo"]["H0_tip"]                     # m, camera
HR = {"left": hs["endo"]["H0_R"], "right": hs["drill"]["H0_R"]}
K = {a: Kin(a) for a in ("left", "right")}
# tighter caps: pan / roll as the planner's ranges, gripper near what the calibrations used
CAP = {"left": {"shoulder_pan": (-55, 55), "wrist_roll": (-90, 90), "gripper": (-59.8, -30)},
       "right": {"shoulder_pan": (-55, 55), "wrist_roll": (-90, 90), "gripper": (-59.8, -26)}}
for a, k in K.items():
    for j, (lo, hi) in CAP[a].items():
        i = JOINTS.index(j)
        k.ik.robot.set_joint_limits(j, float(np.deg2rad(max(lo, k.lo[i] + 5) + k.off[i])), float(np.deg2rad(min(hi, k.hi[i] - 5) + k.off[i])))
arms, cam, info = V.load_models(); z = sz.Zone(arms, cam); z.set_start(NEUTRAL)

tipL0 = K["left"].tip_cam(CUR["left"]); tipR0 = K["right"].tip_cam(CUR["right"])
mid = (tipL0 + tipR0 - d) / 2                                        # endo P such that drill P+d is centred too
g = np.arange(-60, 61, 20) / 1000
gz = np.arange(-60, 41, 20) / 1000
Ps = [mid + np.array([dx, dy, dz]) for dx, dy, dz in itertools.product(g, np.arange(-40, 41, 20) / 1000, gz)]
print("mid", (mid * 1000).round(1), "d", (d * 1000).round(1), "n P", len(Ps), flush=True)

def seeds(a):
    c = CUR[a]; out = [c.copy()]
    for j, dv in (("wrist_roll", 40), ("wrist_roll", -40), ("shoulder_pan", 20), ("shoulder_pan", -20)):
        s = c.copy(); s[JOINTS.index(j)] += dv; out.append(s)
    return out

def ori_targets(a, q0):
    Rc = K[a].T_cam(q0)[:3, :3]
    return [("hand", HR[a], 1.0), ("half", slerp(Rc, HR[a], 0.5), 1.0), ("free", Rc, 1e-4)]

res = {"left": [], "right": []}; t0 = time.time()
for ip, Pe in enumerate(Ps):
    for a, tgt in (("left", Pe), ("right", Pe + d)):
        k = K[a]
        for si, s in enumerate(seeds(a)):
            for on, R, w in ori_targets(a, s):
                k.ik.w_ori = w
                q = k.solve(s, R, tgt, n=250, post=s)
                T = k.T_cam(q); perr = 1000 * np.linalg.norm(T[:3, 3] - tgt)
                if perr > 0.5:
                    continue
                ok, why, v = z.single(a, q, strict=True)
                res[a].append(dict(ip=ip, seed=si, ori=on, q=q, tip=T[:3, 3], R=T[:3, :3], perr=perr,
                                   oerr=rot_angle(T[:3, :3], HR[a]), marg=float(k.margin(q).min()),
                                   ok=ok, why=why, view=v))
    if ip % 20 == 0:
        print(ip, f"{time.time()-t0:.0f}s", {a: (len(r), sum(x['ok'] for x in r)) for a, r in res.items()}, flush=True)
pickle.dump(dict(Ps=Ps, d=d, HR=HR, mid=mid, res=res), open(P / "stageA.pkl", "wb"))
print("done", {a: (len(r), sum(x['ok'] for x in r)) for a, r in res.items()})
