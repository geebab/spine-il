# robot/eval/judge — checks against the hosted endpoint

Small scripts (Python standard library + numpy) for the hackathon judges. They need no robot, no tracker and no local GPU:
the policy server and the render API run on one Nebius Serverless Endpoint during the judging period (2026-12-08 to 2026-12-23, Pacific Time). Its two URLs and the bearer
token are in the Devpost testing instructions.

```bash
export POLICY_URL=https://<policy endpoint>  RENDER_URL=https://<render endpoint>  ENDPOINT_TOKEN=<token>
python3 robot/eval/judge/fetch_robot_runs.py      # once: ~75 MB from huggingface.co/datasets/vincent-won/spine-il-hand90 (robot_runs/)
```

Files land in `$SPINE_IL_DATA` (default `<repo>/data/robot_runs`); `SPINE_IL_REPO` overrides the repository root.

| script | proves | needs | time | expected |
|---|---|---|---|---|
| `reproduce_runs.sh` (→ `reproduce_runs.py`) | the run-17 / run-19 numbers in the video and paper come from the recorded robot runs | `RENDER_URL` | ~17 + 12 min (real-time replay); `--video` ~45 + 31 min | run 17: 1069.6 mm³, 26.8 % past the L4 midline; run 19: 944.1 mm³, 0.5 %; voxel IoU 1.0 against `eval_cache/`. `--video`: also `data/videos/run{17,19}_{endo,scene}.mp4` (endoscope and 3D view, 512 px, 10 fps) |
| `instruction_probe.py` | the instruction sentence alone changes the policy's answer for the same observation | `POLICY_URL` | ~2.5 min (120 calls) | predicted progress lower with the contralateral sentence in 7–8 / 8 frames (0.11–0.12 ± 0.08 over our runs); tip position difference +0.4 to +1.2 ± ~1.8 mm (not distinguishable from repeat spread). Sampled answers: numbers vary slightly per run. `--examples` lists sentences not used in training; `--sentence "…"` adds yours (~4 min for two) |
| `sentence_loop.py "…"` | your sentence drives the policy in a simulated closed loop from run 17's drilling start | `POLICY_URL`, `RENDER_URL`, `websockets` | ~20–30 min per sentence | `data/sentence_loop/<time>/endo.mp4`, `scene.mp4`, `summary.json`. Limitation: the step session removes ~60 % of the robot page's bone, so the drill stops short of the midline whatever the sentence (see the main README) |
| `carve_share.py` | approximate share past the midline from any `live.json` (removed volume booked to the drill tip's x) | — | seconds | run 17's `live.json`: a proxy close to, not equal to, the voxel figure |
| `fetch_robot_runs.py` | downloads the files below | internet | ~1 min | — |

Data (`robot_runs/` in the dataset, CC BY-NC 4.0):

| path | content |
|---|---|
| `sessions/s_20261017T182312_altician_lumbar1_trk_1855.jsonl` | run 17 recorded session: L4, unseen contralateral sentence (the furthest of the 48 contralateral runs) |
| `sessions/s_20261017T184609_altician_lumbar1_trk_3854.jsonl` | run 19 recorded session: L4, same start pose, unseen ipsilateral sentence |
| `eval_cache/iou_cache_rofull_<session>.npz` | the evaluation's removed voxels for each (header + voxel indices) |
| `run17_live/live.json`, `run17_live/frames/*.png` | run 17's loop log and 8 of the endoscope frames the policy received |

**What the hosted endpoint cannot show.** On the same trajectory the render API's step session removes about 60 % of the
bone that the record page removed during the robot run (run 17: 641 or 658 mm³ against 1070 mm³), so a simulated closed
loop (policy endpoint + step session + simulated arms, `robot/inference/live/dry_step.sh`) does not reproduce the robot's
runs: in five such runs from run 17's start pose the drill stopped 6–8 mm before the midline under both sentences. It
demonstrates that the policy and the renderer work together; the closed-loop results come from the 105 recorded robot runs
(`robot/eval/deep_1018/`).
