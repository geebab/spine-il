# GR00T N1.7 training — dataset checks, command, evaluation

Fine-tuning uses LeRobot 0.6.1's GR00T policy (`--policy.type=groot`, base model `nvidia/GR00T-N1.7-3B`, backbone
`nvidia/Cosmos-Reason2-2B`, gated on Hugging Face). On Nebius see `nebius/README.md`.

## Before training

- **Quantile statistics.** GR00T scales state and absolute action groups to [-1, 1] with the dataset's q01 / q99 and clips
  values outside. LeRobot stores the average of per-episode quantiles, which is much narrower than the real range when
  demonstrations start from different places; run `python ../dataset/global_quantiles.py <dataset>` (exports do this already).
- **16-value state.** LeRobot's relative-action code subtracts state from action element by element, so state and action
  must have the same length; the 16th state value (`pad.zero`) stands in for progress, which the robot does not know.
- **Preflight** (no GPU, a few minutes): `python groot_preflight.py --dataset <dataset> --lerobot`. It checks lengths,
  groups, how much of the state would be clipped, the relative statistics, quaternion sign flips and an encode → decode
  round trip through LeRobot's own GR00T processor steps. The last line must be `no problem found`.

## Command

```bash
lerobot-train \
  --policy.type=groot \
  --policy.use_relative_actions=true \
  '--policy.relative_exclude_joints=["trigger","progress"]' \
  --dataset.repo_id=local/<dataset> --dataset.root=<path>/<dataset> \
  --policy.push_to_hub=false \
  --output_dir=<out> --steps=20000 --batch_size=32 --save_freq=2500
```

- Tool poses are relative actions; trigger and progress stay absolute.
- Chunk size: the default 40 (the server answers 40 rows; the robot loop executes the first part and asks again).
- Only the projector and action head are trained (LeRobot defaults), bf16, learning rate 1e-4.

## After training

1. Serve: `python ../inference/remote/policy_server.py --backend lerobot --policy-path <out>/checkpoints/<step>/pretrained_model --device cuda --token-file policy.token`
2. Open-loop check on held-out demonstrations: `python ../inference/end_level.py --dataset <dataset> --policy-url URL --policy-token-file F --episodes <ids> -o OUT.json`
   (direction and size of each predicted chunk vs the demonstration, trigger agreement, progress error, and the progress level at which to stop).
3. Simulated closed loop (`../inference/rollout.py`), then a dry run with simulated arms (`../inference/live/dry_step.sh`), then the robot.
