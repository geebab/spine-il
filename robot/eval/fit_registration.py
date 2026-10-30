#!/usr/bin/env python3
"""Where to put the (virtual) bone for a recording made on another rig: a registration fitted to the arms' own tools.

    python fit_registration.py RID --anchor-s S [--arms-of 3868] [--dry]     →  pairstart/registration_<RID>.json

A robot-origin session has no phantom: the scene is hung on the arms' tools at the start (robot_origin_rec.py anchor
registration: R = the recording's T0.R, t from the drill tip). T0.R is the scene seen from the camera of the day of the
recording. For the 09-28 demos that is this rig's camera, and the arms take the hand's pose: at the 3868 pair the drill is
1° and the endoscope 10° off the hand's. The 10-06 demos were recorded on another rig — another camera pose (gravity 11°
apart in the camera frame), the phantom lying differently — and with their own T0.R the arms' tools sit 14–23° (endoscope)
and 7–15° (drill) off the hand's, although the two tools are arranged the same: one rotation of 12–16° for both tools
brings them within 2–5° of the arms' tools.

That rotation is this file's result. Q (recording camera → robot camera) is the rotation that best turns the hand's two
tools at the anchor onto the arms' two tools at a start pair (--arms-of: a verified pair, the arms' proven pose); the
registration to use on this rig is then T_now = (T0.R·Qᵀ, T0.t): the same scene, seen from this rig's camera. It is the
recorder loader's absolute mode (lerobot_robot_spinearm.recording.load_recording T_now, endo_follow --replay-registration):
p_now = T_now⁻¹·T0·p_rec = Q·p_rec. Consumers: pairstart/recs.py (the pair search then finds the hand's own arrangement
near the proven pose), inference/rollout.py --gate-registration (safety gate, robot link, the policy recording's header —
and through that header the robot session's registration).
Nothing here moves the robot.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
sys.path.insert(0, str(W / "pairstart"))
PS = Path.home() / "il-robot/runs/jitter_iter/work/pairstart"   # where the pairs, windows and registrations are kept


def angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def fit(Rh: dict, Rr: dict) -> np.ndarray:
    """The rotation Q minimising Σ‖Q·Rh[k] − Rr[k]‖² over the tools k (orthogonal Procrustes on the rotation matrices)."""
    U, _, Vt = np.linalg.svd(sum(Rr[k] @ Rh[k].T for k in Rh))
    return U @ np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))]) @ Vt


def registration(T0: dict, Rh: dict, Rr: dict) -> tuple[dict, np.ndarray, dict]:
    """(T_now {R row-major 9, t}, Q, what the fit did) from the recording's T0, the hand's tool rotations (recording
    camera) and the arms' (robot camera) — keys: the same tool names in both."""
    Q = fit(Rh, Rr)
    R0 = np.asarray(T0["R"], float).reshape(3, 3)
    info = {"turn_deg": round(angle(np.eye(3), Q), 2),
            "off_before_deg": {k: round(angle(Rh[k], Rr[k]), 2) for k in Rh},
            "off_after_deg": {k: round(angle(Q @ Rh[k], Rr[k]), 2) for k in Rh}}
    return {"R": (R0 @ Q.T).reshape(-1).tolist(), "t": [float(x) for x in np.asarray(T0["t"], float).reshape(3)]}, Q, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rid", help="the recording's 4-hex id")
    ap.add_argument("--anchor-s", type=float, required=True, help="session time of the pose the arms' tools are matched to "
                    "(the pair's window start)")
    ap.add_argument("--arms-of", default="3868", help="recording id of the verified start pair whose arm pose is the target")
    ap.add_argument("--pairs-file", default=str(PS / "pairs_by_recording_chosen.json"))
    ap.add_argument("--dry", action="store_true", help="print only")
    a = ap.parse_args()
    import common as C
    import hand_calib
    from lerobot_robot_spinearm import recording as rec
    hits = sorted(f for pat in (f"*/*_trk_{a.rid}.jsonl", f"policy/*/*_trk_{a.rid}.jsonl")
                  for f in glob.glob(str(Path.home() / "il-robot/viewer-data/sessions" / pat)) if "/robot/" not in f)
    if len(hits) != 1:
        raise SystemExit(f"{a.rid}: {len(hits)} recordings with this id")
    path = hits[0]
    hdr = rec.read_header(path)
    T0 = hdr["registration"]["T0"]
    cal = hand_calib.for_recording(path)
    hand, tips = {}, {}
    for arm, tool in C.TOOL.items():
        rp = rec.load_recording(path, tool, calib_path=str(cal[tool]), t_start_s=a.anchor_s, t_end_s=a.anchor_s + 1.0)
        hand[arm] = rp["R"][0]; tips[arm] = rp["R"][0] @ rp["tip_local_m"] + rp["pos"][0]
    pj = json.loads(Path(a.pairs_file).expanduser().read_text())[a.arms_of]
    K = {arm: C.Kin(arm, margin=3.0, w_ori=0.1, w_post=0.1) for arm in C.TOOL}
    Tr = {arm: K[arm].T_cam(np.asarray(pj["q_left" if arm == "left" else "q_right"], float)) for arm in K}
    T_now, Q, info = registration(T0, hand, {arm: Tr[arm][:3, :3] for arm in Tr})
    vh, vr = Q @ (tips["right"] - tips["left"]), Tr["right"][:3, 3] - Tr["left"][:3, 3]
    out = {"rid": a.rid, "recording": path, "anchor_s": a.anchor_s, "arms_of": a.arms_of, **T_now, "Q": Q.round(9).tolist(),
           "fit": {**info, "tip_vector_mm": {"hand_on_this_rig": (1000 * vh).round(1).tolist(), "arms": (1000 * vr).round(1).tolist(),
                                              "off": round(float(1000 * np.linalg.norm(vh - vr)), 1)}},
           "T0_recording": {"R": [float(x) for x in np.asarray(T0["R"], float).reshape(-1)], "t": [float(x) for x in np.asarray(T0["t"], float).reshape(-1)]}}
    print(json.dumps({k: out[k] for k in ("rid", "anchor_s", "arms_of", "fit")}))
    if not a.dry:
        f = PS / f"registration_{a.rid}.json"
        f.write_text(json.dumps(out, indent=1))
        print(f"→ {f}")


if __name__ == "__main__":
    main()
