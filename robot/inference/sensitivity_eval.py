#!/usr/bin/env python3
"""How much does a policy's predicted action chunk change when its input barely changes?

    python robot/inference/sensitivity_eval.py --policy-path <ckpt>/pretrained_model --dataset <dataset> [--n 40] [-o out.json]

Background (2026-10-02): two closed-loop rollouts of the same checkpoint from the same start ended at 7 mm³ (stalled) and
145 mm³; the only difference was the rendered image — brightness off by one level (1/255) in 0.14 % of the pixels. This
script measures that sensitivity open loop, on demo frames of the drilling phase: the chunk is predicted for the frame as
is and for perturbed copies, and the change of the drill's net displacement over 16 and 40 steps is reported relative to
the displacement itself.
  pix1    ±1 level on 0.14 % of the pixels (the renderer's own run-to-run difference)
  noise1  Gaussian noise σ = 1 level on every pixel
  noise3  σ = 3 levels
  pos0.1  both tool tip positions ± 0.1 mm in the state
  pos0.5  ± 0.5 mm
Delta-action and chunk-relative policies. No renderer, no robot.
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
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--phase", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(4)
    ds = LeRobotDataset("local/x", root=str(Path(a.dataset).expanduser()), video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    ph = np.asarray(hf["phase.index"]).reshape(-1)
    pol, pre, post = RO.load_policy(a.policy_path)
    step_norm = RO.load_step_norm(a.policy_path)
    n_state = int(pol.config.input_features["observation.state"].shape[0])
    rp = Path(a.policy_path) / "action_rep.json"
    chunk_rel = rp.exists() and json.loads(rp.read_text()).get("rep") == "chunk_relative"
    rng = np.random.default_rng(a.seed)
    idx = np.sort(rng.choice(np.flatnonzero(ph == a.phase), a.n, replace=False))

    def chunk(st, im, task):
        obs = {"observation.state": st[None], "observation.images.endo": im[None], "task": [task]}
        pol.reset()
        with torch.no_grad():
            return RO.predict_chunk(pol, pre, post, obs, step_norm)

    def disp(ch, H):
        return ch[H - 1, 6:9] if chunk_rel else ch[:H, 6:9].sum(0)

    kinds = ("pix1", "noise1", "noise3", "pos0.1", "pos0.5")
    rel = {k: {16: [], 40: []} for k in kinds}
    base = {16: [], 40: []}
    for i in idx:
        it = ds[int(i)]
        st = it["observation.state"].float(); im = it["observation.images.endo"].float()
        if st.shape[0] != n_state:
            st = torch.cat([st[:15], torch.eye(6)[a.phase - 1]]) if n_state == 21 else st[:15]
        c0 = chunk(st, im, it["task"])
        for H in (16, 40):
            base[H].append(float(np.linalg.norm(disp(c0, H))))
        g = torch.Generator().manual_seed(int(i))
        for k in kinds:
            s2, i2 = st.clone(), im.clone()
            if k == "pix1":
                m = torch.rand(im.shape[1:], generator=g) < 0.0014
                i2 = (im + m[None] * (torch.randint(0, 2, im.shape[1:], generator=g) * 2 - 1)[None] / 255.0).clamp(0, 1)
            elif k.startswith("noise"):
                i2 = (im + torch.randn(im.shape, generator=g) * float(k[5:]) / 255.0).clamp(0, 1)
            else:
                d = float(k[3:])
                s2[0:3] += (torch.rand(3, generator=g) * 2 - 1) * d; s2[7:10] += (torch.rand(3, generator=g) * 2 - 1) * d
            c1 = chunk(s2, i2, it["task"])
            for H in (16, 40):
                rel[k][H].append(float(np.linalg.norm(disp(c1, H) - disp(c0, H))))
    out = {"policy": a.policy_path, "n": int(len(idx)), "phase": a.phase,
           "base_disp_mm_median": {H: round(float(np.median(base[H])), 3) for H in (16, 40)}}
    for k in kinds:
        out[k] = {f"H{H}": {"change_mm_median": round(float(np.median(rel[k][H])), 4),
                            "change_over_disp_median": round(float(np.median(np.array(rel[k][H]) / np.maximum(base[H], 1e-6))), 4)}
                  for H in (16, 40)}
        print(k, json.dumps(out[k]), flush=True)
    print("base", json.dumps(out["base_disp_mm_median"]))
    if a.out:
        Path(a.out).expanduser().write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
