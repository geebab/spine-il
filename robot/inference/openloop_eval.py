#!/usr/bin/env python3
"""Open-loop check of a policy on the demonstrations: does the predicted action chunk move like the demo does?

    python robot/inference/openloop_eval.py --policy-path <ckpt>/pretrained_model --dataset <dataset> [--per-phase 60] [-o out.json]

For sampled demo frames (per phase) the policy predicts one action chunk from the demo's own image and state; the chunk's
net tool displacement over the first H steps (H = 1, 16, 40) is compared with the demo's:
  gain    |predicted| / |demo|  (median of the ratio, and the least-squares slope of predicted on demo along the demo's
          direction) — a policy that regresses to "stay still" shows gain ≪ 1
  cos     cosine between the predicted and the demo displacement (direction)
  trig    trigger agreement over the chunk
No renderer and no robot: the images are the dataset's. Delta-action policies only (action = per-step dpos + rotvec + trigger).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout as RO   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--per-phase", type=int, default=60)
    ap.add_argument("--phases", default="3,4,5")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", default=None, help="only frames of these episodes, e.g. 6 — the held-out demo of a "
                    "lerobot-train-rel.py --holdout-episodes training (every earlier number was measured on training frames)")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(4)
    ds = LeRobotDataset("local/x", root=str(Path(a.dataset).expanduser()), video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    ep = np.asarray(hf["episode_index"]).reshape(-1)
    ph = np.asarray(hf["phase.index"]).reshape(-1)
    act = np.asarray(hf["action"], float)
    pol, pre, post = RO.load_policy(a.policy_path)
    n_state = int(pol.config.input_features["observation.state"].shape[0])
    rp = Path(a.policy_path) / "action_rep.json"
    chunk_rel = rp.exists() and json.loads(rp.read_text()).get("rep") == "chunk_relative"   # cumulative chunk (lerobot-train-rel.py)
    step_norm = RO.load_step_norm(a.policy_path)
    done_out = bool(RO.policy_rep(a.policy_path).get("done_output"))   # such a policy's input folds the withdraw bit into drilling
    st_all = np.asarray(hf["observation.state"], float)
    eps_sel = None if a.episodes is None else {int(x) for x in a.episodes.split(",")}
    absolute = act.shape[1] == 15          # dataset with absolute actions: the demo's displacement from the states
    rng = np.random.default_rng(a.seed)
    HS = (1, 16, 40)
    res = {}
    for p in [int(x) for x in a.phases.split(",")]:
        # frames whose next 40 frames stay in the same episode
        ok = np.flatnonzero((ph == p) & (np.r_[ep[40:], -np.ones(40)] == ep))
        if eps_sel is not None:   # e.g. the episodes a --holdout-episodes training never saw
            ok = ok[np.isin(ep[ok], list(eps_sel))]
        if len(ok) == 0:
            continue
        idx = np.sort(rng.choice(ok, min(a.per_phase, len(ok)), replace=False))
        rows = {H: {"endo": [], "drill": []} for H in HS}
        trig = []
        for i in idx:
            it = ds[int(i)]
            st = it["observation.state"].float()
            if st.shape[0] != n_state:   # dataset without the phase one-hot, policy with it (or the reverse)
                st = torch.cat([st[:15], torch.eye(6)[p - 1]]) if n_state == 21 else st[:15]
            if done_out and st.shape[0] == 21:   # as trained (lerobot-train-rel.py merge_phase_input)
                st = st.clone(); st[18] = torch.clamp(st[18] + st[19], max=1.0); st[19] = 0.0
            obs = {"observation.state": st[None], "observation.images.endo": it["observation.images.endo"][None].float(),
                   "task": [it["task"]]}
            pol.reset()
            with torch.no_grad():
                ch = RO.predict_chunk(pol, pre, post, obs, step_norm)   # (chunk, 13), physical units
            dm = act[i:i + 40]
            for H in HS:
                for tool, o, so in (("endo", 0, 0), ("drill", 6, 7)):
                    pred = ch[H - 1, o:o + 3] if chunk_rel else ch[:H, o:o + 3].sum(0)
                    demo = dm[H - 1, so:so + 3] - st_all[i, so:so + 3] if absolute else dm[:H, o:o + 3].sum(0)
                    rows[H][tool].append((pred, demo))
            trig.append(float(((ch[:16, 12] > 0.5) == (dm[:16, -1] > 0.5)).mean()))
        out = {"n": int(len(idx)), "trigger_agree_16": round(float(np.mean(trig)), 3)}
        for H in HS:
            for tool in ("endo", "drill"):
                P = np.array([r[0] for r in rows[H][tool]]); D = np.array([r[1] for r in rows[H][tool]])
                nd, np_ = np.linalg.norm(D, axis=1), np.linalg.norm(P, axis=1)
                m = nd > 0.3 * (H ** 0.5)                       # demo actually moved (mm)
                along = np.einsum("ij,ij->i", P, D) / np.maximum(nd, 1e-9)   # predicted move along the demo's direction
                out[f"{tool}_H{H}"] = {
                    "demo_mm_median": round(float(np.median(nd)), 3), "pred_mm_median": round(float(np.median(np_)), 3),
                    "gain_slope": round(float((along[m] * nd[m]).sum() / max((nd[m] ** 2).sum(), 1e-9)), 3) if m.any() else None,
                    "cos_median": round(float(np.median(along[m] / np.maximum(np_[m], 1e-9))), 3) if m.any() else None,
                    "moved_frac": round(float(m.mean()), 3)}
        res[p] = out
        print(p, json.dumps(out), flush=True)
    if a.out:
        Path(a.out).expanduser().write_text(json.dumps({"policy": a.policy_path, "dataset": a.dataset, "phases": res}, indent=1))


if __name__ == "__main__":
    main()
