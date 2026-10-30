#!/usr/bin/env python3
"""ACT with chunk-relative (cumulative) action targets — the representation GR00T/pi0 get from use_relative_actions.

    python robot/train/lerobot-train-rel.py --dataset ~/il-robot/datasets/spine_rev4_step_ph_trim --out out_rel \
        [--init <ckpt>/pretrained_model] [--steps 30000] [--batch 4] [--save-freq 5000]

Why (2026-10-02, inference/openloop_eval.py): with per-step delta targets (dataset --action-mode delta) the L1 loss treats
every 0.1–0.3 mm step on its own, and the small sweeping motion of drilling regresses toward "stay still" — the predicted
chunk moves the drill 0.33× the demo over 16 steps (0.64× over 40), while the large withdrawal is predicted at 0.9–0.96×.
Here the target of chunk step j is the pose j+1 frames ahead RELATIVE TO THE CURRENT STATE (position difference, rotation
vector of R_nowᵀ·R_then in the tool frame, trigger absolute): an error in the net displacement costs at every later step, so
under-moving is penalised. lerobot 0.6.1 offers this only for groot/pi0 (action − state in the processor); ACT has no such
step, hence this script — a plain loop around lerobot's ACTPolicy with the dataset's absolute actions converted per batch.

The dataset must hold ABSOLUTE actions (action = next state, 15 values; export_lerobot.py --action-mode absolute). The
checkpoint is a normal lerobot pretrained_model dir plus action_rep.json {"rep": "chunk_relative"}; inference/rollout.py
turns each predicted cumulative chunk back into per-step deltas.

--done-output (2026-10-07, user: "정책이 끝을 직접 내도록 … '드릴링 끝' 표시를 정책의 출력으로 학습"): a 14th value, drill.done —
1 for every chunk step that lies after the demonstration's last cut (phase.index ≥ 5), 0 before. Until then nothing in the
policy ended the drilling: the phase flag is an INPUT, in the data it flips at the last cut (hindsight), and on the robot
the harness flipped it with timers (pause / stall). With the flag given, the model never had to tell when the bone is
done. So with --done-output the input no longer says it either: the "withdraw" bit of the phase one-hot is folded into
the "drilling" bit (merge_phase_input), and the end has to be read from the image and the tools. The rollout ends the
drilling when the executed steps say done (inference/rollout.py done_step); action_rep.json carries "done_output".
--init accepts a checkpoint without the value (13 actions): load_partial keeps everything else.

--progress-output (2026-10-09, user: "끝 출력을 0/1 대신 진행도로 바꾼다"): the 14th value is drill.progress instead — the bone
removed up to the frame a chunk step leads to / the demonstration's final removed volume (the dataset's cut.progress,
export_lerobot.py: the replay render's removed volume per frame). The 0/1 flag had 24 examples of its step, was 1 in 4 % of
the chunk steps and on the robot never rose or rose too early (final test 10-08: 5 of 15 passes for both checkpoints, 0 of
10 on the 09-28 base); the progress is a target on every frame. The input is folded as with --done-output, and the
rollout ends the drilling when the predicted value stays above its end level (inference/rollout.py END_LEVEL).

The file name contains "lerobot-train" on purpose: eval/robot_origin_run.sh pauses processes matching that during a robot run.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

NAMES = [f"{tool}.{c}" for tool in ("endo", "drill") for c in ("dx", "dy", "dz", "rx", "ry", "rz")] + ["drill.trigger"]
DONE_NAME = "drill.done"          # --done-output: the 14th value
PROGRESS_NAME = "drill.progress"  # --progress-output: the 14th value
END_COLUMN = {"done": "phase.index", "progress": "cut.progress"}   # the dataset column each form of the 14th value is made from
END_NAME = {"done": DONE_NAME, "progress": PROGRESS_NAME}
PHASE_DRILL, PHASE_WITHDRAW = 4, 5   # dataset/phases.py; observation.state[15:21] = one-hot of phases 1-6


def qmul(a, b):
    aw, ax, ay, az = a.unbind(-1); bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def rotvec_rel(q_now, q_then):
    """Rotation vector of R_nowᵀ·R_then (tool frame), quaternions wxyz, broadcast over leading dims."""
    q_now = q_now / q_now.norm(dim=-1, keepdim=True); q_then = q_then / q_then.norm(dim=-1, keepdim=True)
    conj = q_now * torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q_now.dtype)
    q = qmul(conj.expand_as(q_then), q_then)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]; n = v.norm(dim=-1, keepdim=True)
    ang = 2 * torch.atan2(n, q[..., :1])
    return torch.where(n > 1e-9, v / n.clamp_min(1e-9) * ang, 2 * v)


def to_chunk_relative(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """state (B, ≥15) now, action (B, T, 15) absolute future states → (B, T, 13) relative to now."""
    out = []
    for off in (0, 7):
        out.append(action[..., off:off + 3] - state[:, None, off:off + 3])
        out.append(rotvec_rel(state[:, None, off + 3:off + 7], action[..., off + 3:off + 7]))
    out.append(action[..., 14:15])
    return torch.cat(out, -1)


def done_from_phase(phase: torch.Tensor) -> torch.Tensor:
    """The "drilling is over" target of chunk steps from the phase index (1-6) of the frames they lead to: 1 from the frame
    after the last cut on (phase 5 = withdraw, 6 = outside), 0 while approaching or drilling."""
    return (phase.float() >= PHASE_WITHDRAW).float()


def end_target(kind: str, column: torch.Tensor) -> torch.Tensor:
    """The 14th value of chunk steps from the dataset column of the frames they lead to: the flag from the phase index
    (done) or the removed fraction itself (progress)."""
    return done_from_phase(column) if kind == "done" else column.float()


def merge_phase_input(state: torch.Tensor) -> torch.Tensor:
    """The observed state of a --done-output policy: the phase one-hot (state[..., 15:21]) with the "withdraw" bit folded
    into the "drilling" bit — after the first cut the flag no longer says whether the drilling is over. A state without
    the one-hot (15 values) is returned as it is."""
    if state.shape[-1] != 21:
        return state
    st = state.clone()
    i4, i5 = 15 + PHASE_DRILL - 1, 15 + PHASE_WITHDRAW - 1
    st[..., i4] = torch.clamp(st[..., i4] + st[..., i5], max=1.0); st[..., i5] = 0.0
    return st


def load_partial(policy, init_dir) -> dict:
    """Weights of a checkpoint whose action dimension is smaller (13 → 14: a --done-output policy started from one without
    the value). Tensors of the same shape are copied; a tensor that only grew (the action head's rows, the VAE encoder's
    action projection's columns) keeps the old values in its first rows / columns and its fresh initialisation in the new
    one. Returns the counts {same, grown, kept_fresh} (kept_fresh = names not taken from the checkpoint)."""
    from safetensors.torch import load_file
    old = load_file(str(Path(init_dir) / "model.safetensors"))
    new = {k: v.clone() for k, v in policy.state_dict().items()}
    same, grown, fresh = 0, [], []
    for k, w in new.items():
        o = old.get(k)
        if o is None:
            fresh.append(k)
        elif o.shape == w.shape:
            new[k] = o.clone(); same += 1
        elif o.dim() == w.dim() and all(a <= b for a, b in zip(o.shape, w.shape)):
            w[tuple(slice(0, n) for n in o.shape)] = o; grown.append(k)
        else:
            fresh.append(k)
    policy.load_state_dict(new)
    return {"same": same, "grown": grown, "kept_fresh": fresh}


def chunk_stats(ds_root: Path, T: int, episodes=None, per_step: bool = False, done: bool = False, end: str | None = None) -> dict:
    """mean/std/min/max of the 13 relative values over every (frame, chunk step), padded like the loader (last frame held).
    done / end="done": a 14th column, done_from_phase of the frame each chunk step leads to (the frame after the current
    one + j); end="progress": that frame's cut.progress instead.
    episodes: only these (a --holdout-episodes training). per_step: also "step_mean" / "step_std" (T, 13) — one statistic
    per chunk step, as GR00T's relative-action processor keeps them: a 1-step target is ~0.35 mm and a 40-step target
    ~4 mm, so one pooled std under-weights the first steps in the loss (open-loop gain 0.42 at 1 step, 0.97 at 40)."""
    import glob
    import pyarrow.parquet as pq
    end = end or ("done" if done else None)
    st, ac, ep, fi, ph = [], [], [], [], []
    for f in sorted(glob.glob(str(ds_root / "data/*/*.parquet"))):
        t = pq.read_table(f, columns=["observation.state", "action", "episode_index", "frame_index"] + ([END_COLUMN[end]] if end else []))
        st += t.column("observation.state").to_pylist(); ac += t.column("action").to_pylist()
        ep += t.column("episode_index").to_pylist(); fi += t.column("frame_index").to_pylist()
        if end:
            ph += t.column(END_COLUMN[end]).to_pylist()
    st, ac, ep, fi = np.asarray(st, np.float32), np.asarray(ac, np.float32), np.asarray(ep), np.asarray(fi)
    ph = np.asarray(ph, np.float32).reshape(-1) if end else None
    n_act = 14 if end else 13
    rows = []
    for e in np.unique(ep):
        if episodes is not None and int(e) not in episodes:
            continue
        ix = np.flatnonzero(ep == e); ix = ix[np.argsort(fi[ix])]
        s, a = torch.tensor(st[ix]), torch.tensor(ac[ix])
        n = len(ix)
        idx = torch.clamp(torch.arange(n)[:, None] + torch.arange(T)[None, :], max=n - 1)     # (n, T)
        rel = to_chunk_relative(s, a[idx])                                                    # (n, T, 13)
        if end:   # the action of frame i is the state of frame i + 1: chunk step j leads to frame i + 1 + j
            p = torch.tensor(ph[ix])[torch.clamp(idx + 1, max=n - 1)]
            rel = torch.cat([rel, end_target(end, p)[..., None]], -1)
        rows.append(rel)
    x3 = torch.cat(rows).numpy().astype(np.float64)
    x = x3.reshape(-1, n_act)
    out = {"mean": x.mean(0), "std": x.std(0) + 1e-6, "min": x.min(0), "max": x.max(0),
           "q01": np.quantile(x, 0.01, 0), "q99": np.quantile(x, 0.99, 0), "count": np.array([len(x)])}
    if per_step:
        out["step_mean"], out["step_std"] = x3.mean(0), x3.std(0) + 1e-6
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default=None, help="pretrained_model dir to start from (same architecture, e.g. the per-step delta ACT)")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=40)
    ap.add_argument("--n-action-steps", type=int, default=16)
    ap.add_argument("--save-freq", type=int, default=5000)
    ap.add_argument("--log-freq", type=int, default=100)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--tolerance-s", type=float, default=0.01, help="how far a decoded video frame may be from the asked time "
                    "(frames are 1/fps apart). lerobot's 0.1 ms fails on video files longer than 1,024 s: the times are float32")
    ap.add_argument("--image-aug", action="store_true", help="LeRobot's image augmentation (brightness, contrast, saturation, hue, "
                    "sharpness, small affine; up to 3 per frame) — against run-to-run divergence from tiny rendering differences")
    ap.add_argument("--holdout-episodes", default=None, help="episodes left out of the training (and of the statistics), e.g. 6 — "
                    "then openloop_eval.py --episodes 6 measures the policy on a demo it never saw")
    ap.add_argument("--step-norm", action="store_true", help="normalise the 12 motion values with one mean/std per chunk step "
                    "(saved as step_norm.json next to the weights; the rollout undoes it) instead of one pooled over the chunk")
    ap.add_argument("--done-output", action="store_true", help="a 14th action value, drill.done: 1 for chunk steps after the "
                    "demonstration's last cut. The policy ends the drilling itself; the phase one-hot of the input no longer "
                    "tells drilling from withdrawing (see the docstring)")
    ap.add_argument("--progress-output", action="store_true", help="a 14th action value, drill.progress: the bone removed up to the "
                    "frame a chunk step leads to / the demonstration's total (the dataset's cut.progress) — instead of the 0/1 flag "
                    "of --done-output; the input is folded the same way")
    ap.add_argument("--state-noise-mm", type=float, default=0.0, help="uniform ± noise on both tool tips of the observed state; the "
                    "chunk-relative targets are taken from the noisy state, so the model also learns to steer back to the demo path")
    a = ap.parse_args()
    if a.done_output and a.progress_output:
        ap.error("--done-output and --progress-output are two forms of the one 14th value: give one")
    end = "done" if a.done_output else "progress" if a.progress_output else None
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); np.random.seed(a.seed)
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    root = Path(a.dataset).expanduser(); out = Path(a.out).expanduser()
    meta = LeRobotDatasetMetadata("local/x", root=root)
    if meta.features["action"]["shape"][0] != 15:
        raise SystemExit("the dataset must hold absolute actions (15): export_lerobot.py --action-mode absolute")
    n_state = meta.features["observation.state"]["shape"][0]
    img = meta.features["observation.images.endo"]["shape"]
    cfg = ACTConfig(chunk_size=a.chunk, n_action_steps=a.n_action_steps, device="cpu", push_to_hub=False,
                    input_features={"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(n_state,)),
                                    "observation.images.endo": PolicyFeature(type=FeatureType.VISUAL, shape=(img[2], img[0], img[1]))
                                    if img[-1] == 3 else PolicyFeature(type=FeatureType.VISUAL, shape=tuple(img))},
                    output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14 if end else 13,))})
    names = NAMES + ([END_NAME[end]] if end else [])
    if end and END_COLUMN[end] not in meta.features:
        raise SystemExit(f"--{end}-output needs the dataset's {END_COLUMN[end]}"
                         + (" (export_lerobot.py writes it from 10-09 on: export the dataset again)" if end == "progress" else ""))
    stats = {k: v for k, v in meta.stats.items() if k != "action"}
    hold = set() if a.holdout_episodes is None else {int(x) for x in a.holdout_episodes.split(",")}
    ex = Path(root) / "meta/spine_il/export.json"   # the endoscope medium of the images: the rollout renders the same one
    optics = json.loads(ex.read_text()).get("image_optics") if ex.exists() else None
    print(f"[rel] image optics of the dataset: {optics or 'not recorded (the recordings\' own flag)'}", flush=True)
    train_eps = [e for e in range(meta.total_episodes) if e not in hold]
    cs = chunk_stats(root, a.chunk, episodes=set(train_eps), per_step=a.step_norm, end=end)
    step_norm = None
    if a.step_norm:   # motion values: (x - mean_j) / std_j per chunk step j, then identity in the policy's own normaliser
        sm, ss = cs.pop("step_mean"), cs.pop("step_std")
        sm[:, 12:], ss[:, 12:] = 0.0, 1.0                                  # the trigger and the end value keep the pooled statistic
        step_norm = {"mean": torch.tensor(sm, dtype=torch.float32), "std": torch.tensor(ss, dtype=torch.float32)}
        for k, v0, v1 in (("mean", 0.0, None), ("std", 1.0, None), ("min", -5.0, None), ("max", 5.0, None), ("q01", -3.0, None), ("q99", 3.0, None)):
            cs[k] = np.concatenate([np.full(12, v0), cs[k][12:]])
        print("[rel] per-step action stats: std of the drill x at steps 1 / 16 / 40 =",
              [round(float(ss[j, 6]), 3) for j in (0, 15, a.chunk - 1)], flush=True)
    stats["action"] = cs
    print("[rel] action stats (cumulative over the chunk): std", np.round(stats["action"]["std"], 3).tolist(),
          f"| training episodes {train_eps}" + (f", held out {sorted(hold)}" if hold else ""), flush=True)
    if end == "done":
        print(f"[rel] done output: {float(cs['mean'][13]):.4f} of the chunk steps of the training episodes lie after the last cut", flush=True)
    elif end == "progress":
        print(f"[rel] progress output: mean {float(cs['mean'][13]):.3f}, std {float(cs['std'][13]):.3f} over the chunk steps of the training "
              f"episodes (0 = before the first cut, 1 = after the last)", flush=True)
    if a.init:
        n_init = int(json.loads((Path(a.init) / "config.json").read_text())["output_features"]["action"]["shape"][0])
        if n_init == len(names):
            policy = ACTPolicy.from_pretrained(a.init, config=cfg)
            print(f"[rel] weights from {a.init}", flush=True)
        else:   # a checkpoint without the done value: everything but the new row / column
            policy = ACTPolicy(cfg)
            lp = load_partial(policy, a.init)
            print(f"[rel] weights from {a.init} ({n_init} actions → {len(names)}): {lp['same']} tensors as they are, grown {lp['grown']}, "
                  f"fresh {lp['kept_fresh']}", flush=True)
    else:
        policy = ACTPolicy(cfg)
    policy.train()
    pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
    fps = meta.fps
    tf = None
    if a.image_aug:
        from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig
        tf = ImageTransforms(ImageTransformsConfig(enable=True))
    dts = {"action": [i / fps for i in range(a.chunk)]}
    if end:   # the action of frame i is the state of frame i + 1: the phase / progress of the frames the chunk steps lead to
        dts[END_COLUMN[end]] = [(i + 1) / fps for i in range(a.chunk)]
    ds = LeRobotDataset("local/x", root=root, delta_timestamps=dts, video_backend="pyav",
                        image_transforms=tf, episodes=train_eps if hold else None, tolerance_s=a.tolerance_s)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=0, drop_last=True)
    opt = torch.optim.AdamW(policy.get_optim_params(), lr=cfg.optimizer_lr, weight_decay=cfg.optimizer_weight_decay)
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_args.json").write_text(json.dumps({**vars(a), "action_names": names, "action_std": stats["action"]["std"].tolist()}, indent=1))

    def save(step: int) -> None:
        d = out / "checkpoints" / f"{step:06d}" / "pretrained_model"
        d.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(d); pre.save_pretrained(d); post.save_pretrained(d)
        (d / "action_rep.json").write_text(json.dumps({"rep": "chunk_relative", "names": names, "step": step, "dataset": str(root),
                                                       "holdout_episodes": sorted(hold), "step_norm": bool(a.step_norm),
                                                       # done_output = "the policy ends the drilling itself" (either form of the
                                                       # 14th value); end_output says which, the rollout takes its end level from it
                                                       **({"done_output": True, "end_output": end, "phase_input": "withdraw bit folded into drilling"} if end else {}),
                                                       **({"optics": optics} if optics in ("air", "water") else {})}))
        if step_norm is not None:
            (d / "step_norm.json").write_text(json.dumps({"mean": step_norm["mean"].tolist(), "std": step_norm["std"].tolist()}))
        last = out / "checkpoints" / "last"
        if last.is_symlink() or last.exists():
            last.unlink()
        last.symlink_to(f"{step:06d}")

    step, t0, run = 0, time.time(), []
    while step < a.steps:
        for batch in dl:
            if a.state_noise_mm > 0:
                st = batch["observation.state"].float().clone()
                for off in (0, 7):
                    st[:, off:off + 3] += (torch.rand(st.shape[0], 3) * 2 - 1) * a.state_noise_mm
                batch["observation.state"] = st
            rel = to_chunk_relative(batch["observation.state"].float(), batch["action"].float())
            if end:
                col = batch.pop(END_COLUMN[end]); batch.pop(END_COLUMN[end] + "_is_pad", None)
                rel = torch.cat([rel, end_target(end, col.reshape(col.shape[0], a.chunk))[..., None]], -1)
                batch["observation.state"] = merge_phase_input(batch["observation.state"].float())
            batch["action"] = rel
            if step_norm is not None:
                batch["action"] = (batch["action"] - step_norm["mean"]) / step_norm["std"]
            b = pre(batch)
            loss, info = policy.forward(b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step(); opt.zero_grad()
            step += 1; run.append(float(loss))
            if step % a.log_freq == 0:
                el = time.time() - t0
                print(f"[rel] step {step}/{a.steps} loss {np.mean(run[-a.log_freq:]):.4f} l1 {info.get('l1_loss', float('nan')):.4f} "
                      f"{el / step:.2f} s/step, {(a.steps - step) * el / step / 3600:.1f} h left", flush=True)
            if step % a.save_freq == 0 or step == a.steps:
                save(step)
            if step >= a.steps:
                break
    print("[rel] done", flush=True)


if __name__ == "__main__":
    main()
