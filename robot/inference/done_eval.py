#!/usr/bin/env python3
"""Does a --done-output policy say "the drilling is over" where the demonstration ended it? Open loop, no robot, no renderer.

    python robot/inference/done_eval.py --policy-path <ckpt>/pretrained_model --dataset <dataset> [--episodes 2,18]
        [--n-exec 20] [--offsets 0,10] [-o out.json]

A policy trained with train/lerobot-train-rel.py --done-output has a 14th action value, 1 for the chunk steps after the
demonstration's last cut. On the robot the harness ends the drilling when the steps about to be executed say so
(rollout.done_step). Here the same rule walks along a demonstration: the policy is asked every --n-exec frames (as a run
asks it, from the demo's own image and tools, phase flag folded as in training) and the first chunk that says "done" gives
the frame where a run would have ended. Against the demonstration's own end (the frame after its last cut):

  end_s        predicted end − the demo's end (s): negative = stopped early (bone left), positive = went on
  never        the policy did not say done before the episode's last frame (the trimmed data go on for a few seconds
               after the last cut — a policy that needs longer than that is not seen ending here)
  done_max     the highest done value over the executed steps, per observation — how the signal rises toward the end

--episodes = the held-out demos of the training (action_rep.json "holdout_episodes") by default: on training demos the
end is memorised. Several --offsets shift the observation grid (a run's chunk boundaries fall anywhere).
Teacher-forced: the tools are where the demonstrator had them. A run's own states differ — this says whether the model
reads the end from a demonstration's pictures at all, which comes before any robot run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout as RO   # noqa: E402

PHASE_DRILL, PHASE_WITHDRAW = 4, 5


def fold_phase(state: np.ndarray) -> np.ndarray:
    """The state as a done-output policy was trained on it: the "withdraw" bit of the phase one-hot folded into "drilling"."""
    s = np.array(state, float)
    if s.shape[-1] == 21:
        s[..., 15 + PHASE_DRILL - 1] = np.minimum(1.0, s[..., 15 + PHASE_DRILL - 1] + s[..., 15 + PHASE_WITHDRAW - 1])
        s[..., 15 + PHASE_WITHDRAW - 1] = 0.0
    return s


def demo_end(phase: np.ndarray) -> int | None:
    """Index (within the episode) of the first frame after the last cut, None when the episode holds none."""
    k = np.flatnonzero(np.asarray(phase) >= PHASE_WITHDRAW)
    return int(k[0]) if len(k) else None


def walk(done_of, n_frames: int, n_exec: int, offset: int, hold: int = RO.DONE_HOLD, level: float = RO.END_LEVEL["done"]) -> tuple[int | None, list]:
    """The run-time rule along an episode: done_of(t) → the done values of the chunk predicted at frame t. Returns the
    frame where the drilling would end (t + 1 + step: chunk step j leads to frame t + 1 + j) or None, and the trace
    [t, highest done value over the executed steps] of every observation made."""
    trace = []
    for t in range(offset, n_frames, n_exec):
        d = np.asarray(done_of(t), float)
        trace.append([t, round(float(d[:n_exec].max()), 3)])
        j = RO.done_step(d, n_exec, hold, level)
        if j is not None:
            return t + 1 + j, trace
    return None, trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy-path", required=True)
    ap.add_argument("--dataset", default=None, help="default: the checkpoint's own (action_rep.json)")
    ap.add_argument("--episodes", default=None, help="default: the training's held-out episodes")
    ap.add_argument("--n-exec", type=int, default=20)
    ap.add_argument("--offsets", default="0,10")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--end-level", type=float, default=None, help="the level above which the end value means \"over\" (default by "
                    "its form: rollout.END_LEVEL — 0.5 for the done flag, 0.91 for the progress)")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    rep = RO.policy_rep(a.policy_path)
    end_kind, end_level = RO.end_output(rep)
    if end_kind is None:
        raise SystemExit("this checkpoint has no end output (train/lerobot-train-rel.py --done-output / --progress-output)")
    if a.end_level:
        end_level = float(a.end_level)
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    torch.set_num_threads(a.threads)
    root = Path(a.dataset or rep["dataset"]).expanduser()
    eps = [int(x) for x in a.episodes.split(",")] if a.episodes else list(rep.get("holdout_episodes") or [])
    if not eps:
        raise SystemExit("no episodes: the training held none out — name some with --episodes")
    ds = LeRobotDataset("local/x", root=str(root), video_backend="pyav")
    hf = ds.hf_dataset.with_format("numpy")
    ep_all = np.asarray(hf["episode_index"]).reshape(-1); ph_all = np.asarray(hf["phase.index"]).reshape(-1)
    pol, pre, post = RO.load_policy(a.policy_path)
    step_norm = RO.load_step_norm(a.policy_path)
    fps = float(ds.meta.fps); res = {}
    for e in eps:
        ix = np.flatnonzero(ep_all == e); ph = ph_all[ix]; k_end = demo_end(ph)
        cache: dict = {}

        def done_of(t: int) -> np.ndarray:
            if t not in cache:
                it = ds[int(ix[t])]
                st = torch.tensor(fold_phase(it["observation.state"].numpy()), dtype=torch.float32)
                obs = {"observation.state": st[None], "observation.images.endo": it["observation.images.endo"][None].float(), "task": [it["task"]]}
                pol.reset()
                with torch.no_grad():
                    cache[t] = RO.predict_chunk(pol, pre, post, obs, step_norm)[:, 13]
            return cache[t]

        rows = []
        for off in [int(x) for x in a.offsets.split(",")]:
            k, trace = walk(done_of, len(ix), a.n_exec, off, level=end_level)
            rows.append({"offset": off, "end_frame": k, "end_s": None if k is None or k_end is None else round((k - k_end) / fps, 2),
                         "never": k is None, "observations": len(trace), "done_max_last": [x[1] for x in trace[-8:]]})
            print(f"episode {e} offset {off}: " + (f"done at frame {k} = {rows[-1]['end_s']:+.2f} s against the demo's end (frame {k_end})" if k is not None and k_end is not None
                                                    else f"done at frame {k} (the episode has no frame after its last cut)" if k is not None
                                                    else f"never (episode of {len(ix)} frames, the demo's end at {k_end})")
                  + f" | done_max of the last observations {rows[-1]['done_max_last']}", flush=True)
        res[e] = {"frames": int(len(ix)), "demo_end_frame": k_end, "drilling_s": round(float((ph == PHASE_DRILL).sum()) / fps, 1), "walks": rows}
    ends = [w["end_s"] for r in res.values() for w in r["walks"] if w["end_s"] is not None]
    summ = {"walks": sum(len(r["walks"]) for r in res.values()), "never": sum(w["never"] for r in res.values() for w in r["walks"]),
            "early_over_2s": sum(x < -2.0 for x in ends), "late_over_2s": sum(x > 2.0 for x in ends),
            "end_s_median": round(float(np.median(ends)), 2) if ends else None, "end_s_abs_median": round(float(np.median(np.abs(ends))), 2) if ends else None}
    print("summary:", json.dumps(summ))
    if a.out:
        Path(a.out).expanduser().write_text(json.dumps({"policy": a.policy_path, "dataset": str(root), "n_exec": a.n_exec, "hold": RO.DONE_HOLD,
                                                       "end_output": end_kind, "end_level": end_level,
                                                       "episodes": res, "summary": summ}, indent=1))


if __name__ == "__main__":
    main()
