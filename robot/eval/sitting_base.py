#!/usr/bin/env python3
"""A base for robot runs that does not stand on one recording: the representative pose of a sitting.

    python sitting_base.py NAME --dataset DS --episode E [--arms-of 3868] [--dry]

A robot run needs a base: where the (virtual) bone is put under the arms (registration), the arms' start pair, the pose
the safety gate is anchored at and the tools' start orientation (the arms hold a tool's orientation). Until 10-07 that was
one recording at one moment (3868 @ 17 s, 92c5 @ 65.4 s). User, 10-07: an evaluation should hang on single recordings as
little as possible — so the base is taken from all recordings of a sitting (one day's phantom registration; --episode
names the sitting by one of its episodes):

  representative pose   over every drilling frame (phase 4) of the sitting's recordings: per tool the median tip (scene mm)
                        and the mean orientation — the centre of what the demonstrations did there
  registration          the rotation that puts the bone under the arms so that these two tool orientations coincide with
                        the arms' tools at a proven pair (--arms-of; fit_registration.fit), and the translation that puts
                        the midpoint of the two tips on the arms' — the scene as this rig's camera sees it
  trajectory            the pose held, then both tips swept ±SWEEP mm along every scene axis and back (orientation fixed):
                        what the pair search replays through the safe zone (search_clear.py: paths + replay)

Written to pairstart/: base_<NAME>.json (state, spread, recordings, portals), registration_<NAME>.json (the format of
fit_registration.py: rollout.py --gate-registration, recs.py), base_<NAME>_traj.json (for inference/policy2rec.py).
Then, with no robot run beside it (CPU):
    python ../inference/policy2rec.py --traj pairstart/base_<NAME>_traj.json        → a synthetic recording <rid>
    python new_recording.py <rid> --from-s 0 --len-s 30 --no-iou --no-queue --force  → pairstart/pair_clear_<rid>.json
    python pair_reach.py …                                                           → which demos' drilling the gate passes
and robot runs with rollout.py / live_loop.py --gate-anchor-state pairstart/base_<NAME>.json --gate-registration … .
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
sys.path.insert(0, str(W.parent / "inference"))
PS = Path.home() / "il-robot/runs/jitter_iter/work/pairstart"
TOOLS = (("endo", 0), ("drill", 7))
SWEEP_MM, SPEED_MMS, HOLD_S, FPS = 5.0, 5.0, 3.0, 15.0


def mean_quat(qs: np.ndarray) -> np.ndarray:
    M = sum(np.outer(q, q) for q in qs)
    _, v = np.linalg.eigh(M)
    q = v[:, -1]
    return q if q @ qs[0] >= 0 else -q


def qang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1)))


def representative(S: np.ndarray) -> tuple[np.ndarray, dict]:
    """(15-vector, spread) of drilling states S (n × 15): median tips, mean orientations, trigger off."""
    rep, spread = np.zeros(15), {}
    for t, o in TOOLS:
        rep[o:o + 3] = np.median(S[:, o:o + 3], axis=0)
        rep[o + 3:o + 7] = mean_quat(S[:, o + 3:o + 7])
        ang = qang(S[:, o + 3:o + 7], rep[o + 3:o + 7])
        d = np.linalg.norm(S[:, o:o + 3] - rep[o:o + 3], axis=1)
        spread[t] = {"tip_mm": {"median": round(float(np.median(d)), 1), "p90": round(float(np.percentile(d, 90)), 1)},
                     "box_5_95_mm": [np.percentile(S[:, o:o + 3], 5, axis=0).round(1).tolist(), np.percentile(S[:, o:o + 3], 95, axis=0).round(1).tolist()],
                     "orientation_deg": {"median": round(float(np.median(ang)), 1), "p90": round(float(np.percentile(ang, 90)), 1),
                                         "within_10": round(float(np.mean(ang <= 10.0)), 3)}}
    return rep, spread


def registration_for(rep: np.ndarray, Rr: dict, tips_r: dict, q2R) -> tuple[dict, dict]:
    """({R, t} scene → this rig's camera as rollout / the recorder use it, what the fit did): the two tool orientations of
    `rep` onto the arms' (Rr, camera frame), the midpoint of the tips onto the arms' (tips_r, m)."""
    import fit_registration as FR
    arm_of = {"endo": "left", "drill": "right"}
    Rs = {arm_of[t]: q2R(rep[o + 3:o + 7]) for t, o in TOOLS}                      # tool → scene
    Qs = FR.fit(Rs, Rr)                                                           # Qs·Rs ≈ Rr: scene → camera = R_nowᵀ
    R_now = Qs.T
    mid_s = 0.5 * (rep[0:3] + rep[7:10]); mid_c = 0.5 * (tips_r["left"] + tips_r["right"]) * 1000.0
    t = mid_s - R_now @ mid_c
    vh = Qs @ (rep[7:10] - rep[0:3]); vr = (tips_r["right"] - tips_r["left"]) * 1000.0
    info = {"off_after_deg": {a: round(FR.angle(Qs @ Rs[a], Rr[a]), 2) for a in Rs},
            "tip_vector_mm": {"representative_on_this_rig": vh.round(1).tolist(), "arms": vr.round(1).tolist(), "off": round(float(np.linalg.norm(vh - vr)), 1)}}
    return {"R": R_now.reshape(-1).tolist(), "t": [float(x) for x in t]}, info


def sweep(rep: np.ndarray) -> list[list[float]]:
    """The pose held, then both tips together ±SWEEP_MM along each scene axis and back; orientation fixed, trigger off."""
    n_hold, n_leg = int(HOLD_S * FPS), max(2, int(round(SWEEP_MM / SPEED_MMS * FPS)))
    fr = [rep.copy() for _ in range(n_hold)]
    for ax in range(3):
        for sgn in (1.0, -1.0):
            for j in list(range(1, n_leg + 1)) + list(range(n_leg - 1, -1, -1)):
                f = rep.copy()
                for _, o in TOOLS:
                    f[o + ax] += sgn * SWEEP_MM * j / n_leg
                fr.append(f)
    fr += [rep.copy() for _ in range(n_hold)]
    return [[round(float(x), 5) for x in f] for f in fr]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="the base's name, e.g. C1006 (sitting of 10-06), A0928")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, required=True, help="one episode of the sitting (its header also lends case and tool models)")
    ap.add_argument("--arms-of", default="3868", help="the verified pair whose arm pose the registration is fitted to")
    ap.add_argument("--pairs-file", default=str(PS / "pairs_by_recording_chosen.json"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    import pandas as pd
    import common as C
    import rollout as RO
    ds = Path(a.dataset).expanduser()
    eps = [json.loads(l) for l in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()]
    use = RO.batch_episodes(ds, a.episode)
    df = pd.concat([pd.read_parquet(f, columns=["observation.state", "phase.index", "episode_index"]) for f in sorted(glob.glob(str(ds / "data/*/*.parquet")))])
    st = np.stack(df["observation.state"].to_numpy())[:, :15]
    ep, ph = df["episode_index"].to_numpy().ravel(), df["phase.index"].to_numpy().astype(int).ravel()
    S = st[np.isin(ep, use) & (ph == 4)]
    rep, spread = representative(S)
    pj = json.loads(Path(a.pairs_file).expanduser().read_text())[a.arms_of]
    K = {arm: C.Kin(arm, margin=3.0, w_ori=0.1, w_post=0.1) for arm in C.TOOL}
    Tr = {arm: K[arm].T_cam(np.asarray(pj["q_left" if arm == "left" else "q_right"], float)) for arm in K}
    T_now, fit = registration_for(rep, {arm: Tr[arm][:3, :3] for arm in Tr}, {arm: Tr[arm][:3, 3] for arm in Tr}, RO.q2R)
    recs = sorted({Path(eps[e]["recording"]).name for e in use})
    portals = {t: np.asarray(p, float).round(2).tolist() for t, p in RO.demo_portals(ds, use).items()}
    base = {"name": a.name, "state": [round(float(x), 5) for x in rep], "from": f"median tips and mean tool orientations of the {len(S)} drilling frames "
            f"of the {len(use)} episodes ({len(recs)} recordings) of the sitting of episode {a.episode}", "dataset": str(ds), "episodes": [int(e) for e in use],
            "recordings": recs, "handle_episode": a.episode, "spread": spread, "tips_apart_mm": round(float(np.linalg.norm(rep[0:3] - rep[7:10])), 1),
            "portals_mm": portals, "arms_of": a.arms_of, "registration": T_now, "fit": fit}
    print(json.dumps({k: base[k] for k in ("name", "state", "tips_apart_mm", "spread", "fit")}, ensure_ascii=False))
    if a.dry:
        return
    (PS / f"base_{a.name}.json").write_text(json.dumps(base, indent=1))
    (PS / f"registration_{a.name}.json").write_text(json.dumps({"rid": a.name, "base": str(PS / f"base_{a.name}.json"), "arms_of": a.arms_of, **T_now,
                                                               "fit": {"method": "representative pose of a sitting (sitting_base.py)", **fit}}, indent=1))
    traj = {"fps": FPS, "frames": sweep(rep), "ref_recording": eps[a.episode]["recording"],
            "meta": {"policy": f"sitting-base {a.name}", "dataset": str(ds), "episode": a.episode,
                     "registration": {"file": str(PS / f"registration_{a.name}.json"), "T0": T_now, "arms_of": a.arms_of, "fit": fit}}}
    (PS / f"base_{a.name}_traj.json").write_text(json.dumps(traj))
    print(f"→ {PS}/base_{a.name}.json, registration_{a.name}.json, base_{a.name}_traj.json ({len(traj['frames'])} frames, {len(traj['frames']) / FPS:.0f} s)")


if __name__ == "__main__":
    main()
