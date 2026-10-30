# spine-il

**spine-il — imitation learning (IL) for endoscopic spine surgery:** a surgeon's tracked tool motion becomes the
training data of a robot policy. One surgeon performed left L3, L4 and L5 laminectomies
with a tracked replica endoscope and drill through a hollow skin phantom; bone, bone removal and the endoscope image are
simulated from the tracked tool poses. NVIDIA GR00T N1.7 is fine-tuned on Nebius to map the endoscope view, the tool
state and an instruction sentence to tool motion, and two SO-ARM101 arms holding the replica tools drill in closed loop
under the same tracker and simulation. The instruction sets the extent of the laminectomy (ipsilateral only, or with
contralateral decompression), including a wording never seen in training for each extent.

Built for the Nebius × NVIDIA Global AI Hackathon 2026, Physical AI track.

> Research and evaluation only. Not a medical device, not for clinical use or use on patients.

## Status (2026-10-23)

| part | status |
|---|---|
| hand demonstrations | 90 left-sided recordings used for training, 30 per level (L3, L4, L5); labelled by midline crossing: ipsilateral 31, contralateral 34, unspecified 25 |
| dataset | LeRobot v3.0, 417 episodes, 178,068 frames at 15 fps, 256 × 256 endoscope view, 16-value state and action, 131 instruction sentences |
| model | GR00T N1.7 (3B) fine-tuned with LeRobot 0.6.1, 20,000 steps, batch 32, on a Nebius RTX PRO 6000 VM |
| robot evaluation | 105 closed-loop runs on two arms (35 per level from four start poses) |
| public release | weights and data on Hugging Face; the policy server and the render API run on one Nebius Serverless Endpoint until 2026-12-15 |

## Result

Across L3, L4 and L5 (105 runs; per level 16 contralateral, 16 ipsilateral and 3 neutral instructions from four start
poses), the instruction changed where the robot drilled. Over the three levels, the 48 runs told to decompress
contralaterally removed on average 9.9 % of their bone across the midline, against 0.4 % for the 48 runs told to stay
ipsilateral (p < 0.001); with wordings never seen in training, 9.8 % vs 0.2 %. Per level, with unseen wordings: L4
18.1 % vs 0.6 % and L5 10.6 % vs 0.0 % (p < 0.001 each, every contralateral run above every ipsilateral run at each
start pose); L3 0.6 % vs 0.1 % (p = 0.053). Contralateral reach was at most 5.2 mm, against a demonstrated mean of
10 to 12 mm, and a supervisory timeout ended 14 of the 48 contralateral runs: where to drill and when to stop both
remain to be learned. During drilling every motion comes from the policy; the approach to the start pose is scripted.
Every run, demonstration and analysis output is in one SQLite file built by `robot/eval/deep_1018/build_master.py`.

## How it fits together

```
tracked hand recording (JSONL, tool poses + trigger)
  ─▶ render API: the virtual anatomy renders the endoscope view, removes bone where the drill was
  ─▶ LeRobot dataset (tool-space state, endoscope video, per-frame instruction)      robot/dataset/
  ─▶ GR00T N1.7 fine-tuning on Nebius                                                 robot/train/nebius/
  ─▶ closed loop: policy ─▶ safety gate ─▶ two SO-ARM101 arms ─▶ tracker ─▶ render API ─▶ policy
                                                                                      robot/inference/
  ─▶ evaluation: removed-bone voxels, share past the midline, comparison with the demonstrations
                                                                                      robot/eval/
```

The same renderer produces the training images and the images the policy sees on the robot, so the policy input matches.

| directory | content |
|---|---|
| `converter/` | recorder schema v2 JSONL → LeRobot v3 dataset ([README](converter/README.md)) |
| `robot/dataset/` | hand sessions → LeRobot dataset in tool space, rendered endoscope view, per-frame instructions, Open-H-Embodiment export (`export_lerobot.py --layout openh`) |
| `robot/train/nebius/` | GR00T N1.7 fine-tuning and serving on a Nebius VM (`vm.sh`, runbook README); `serverless/` = the policy server image for a Nebius Serverless Endpoint |
| `robot/inference/` | closed-loop rollout against the render API, robot safety gate, 5-panel video |
| `robot/eval/` | removed-bone metrics; `deep_1018/` is the 105-run study (pre-registration, analysis, master database) |
| `robot/lerobot_robot_spinearm/` | arm server, LeRobot robots `spine_arm` / `spine_bi_arm`, marker IK |
| `ros2_ws/` | ROS 2 bridge for the two arms, URDF and the tracker in Foxglove |
| `bridge/` | render API client (`spine_render`: input checks, job and step protocol, token, removed-bone voxel sets) ([README](bridge/README.md)); the server is the hosted render API |

## Release

| item | where | licence |
|---|---|---|
| code (this repository) | [github.com/geebab/spine-il](https://github.com/geebab/spine-il) | MPL-2.0 (SO-101 meshes in `ros2_ws/src/spinearm_description`: Apache-2.0, TheRobotStudio) |
| fine-tuned GR00T weights | [vincent-won/spine-il-gr00t-n1.7](https://huggingface.co/vincent-won/spine-il-gr00t-n1.7) | NVIDIA Open Model License, research and evaluation use only (see below) |
| dataset (LeRobot v3.0; `export_lerobot.py --layout openh` writes an Open-H-Embodiment copy) and raw hand recordings | [datasets/vincent-won/spine-il-hand90](https://huggingface.co/datasets/vincent-won/spine-il-hand90) | CC BY-NC 4.0 |
| level meshes (L3, L4, L5, sacrum) and calibration files | with the dataset | CC BY-NC 4.0 |
| robot runs 17 and 19 (recorded sessions, evaluation voxels, what the policy saw) for the judge checks | the dataset, `robot_runs/` | CC BY-NC 4.0 |
| policy server (`spine-policy/1`, the fine-tuned weights above) | Nebius Serverless Endpoint until 2026-12-15 | service; build it yourself from `robot/train/nebius/serverless/` |
| render API | the same Nebius Serverless Endpoint until 2026-12-15, limited concurrent sessions | service |

The virtual anatomy is rendered by Altician's viewer, which stays closed source; it is offered as the hosted render API
above, which this repository's code calls. The render server (the part of `bridge/` that runs the viewer in headless
Chrome) and the recorder-page launcher used on the rig (`robot_origin_rec.py`) are not in this repository; a simulated
closed loop (`robot/inference/live/dry_step.sh`: the policy endpoint, the render API and simulated arms) needs
no robot and no local GPU, but see the limitation under [Quick start for judges](#quick-start-for-judges). The skin model of the phantom is not released. Every recorded row keeps the
point where each tool's shaft crosses the skin, as the viewer computed it (left working area only); the robot harness
rebuilds its skin entry model from those points (`python robot/eval/skin_surface.py --build`), and during a run the
render API reports the live crossing.

The fine-tuned weights derive from `nvidia/GR00T-N1.7-3B` (backbone `nvidia/Cosmos-Reason2-2B`, from
`Qwen/Qwen3-VL-2B-Instruct`, Apache-2.0). Built on NVIDIA Cosmos. Licensed by NVIDIA Corporation under the NVIDIA Open
Model License; the GR00T N1.7 repository also carries the NVIDIA License, which limits use to research and evaluation,
and the release follows both.

## Quick start for judges

The robot and its tracker cannot be shipped, so the hosted Nebius Serverless Endpoint (policy server + render API, one
container on an RTX PRO 6000) lets you check the two claims of the video without them. The endpoint URLs and the token are
in the Devpost testing instructions:

```bash
export POLICY_URL=https://<policy endpoint>  RENDER_URL=https://<render endpoint>  ENDPOINT_TOKEN=<token>
python3 -m pip install numpy websockets                        # the judge scripts need only the standard library + these two
python3 robot/eval/judge/fetch_robot_runs.py                   # ~75 MB from the Hugging Face dataset → data/robot_runs/
```

**(a) The run-17 / run-19 numbers come from the recorded robot runs** — `bash robot/eval/judge/reproduce_runs.sh`
(about 30 min: the render API replays each recorded session in real time). Each recorded run's cuts are applied again on
the 0.8 mm voxel grid and the share of removed bone past the L4 midline is printed, with the voxel overlap against the
evaluation's own result. Add `--video` (`bash robot/eval/judge/reproduce_runs.sh --video`, about 75 min) to also get the
two viewer panes of each replay — the endoscope view and the 3D view — as `data/videos/run17_endo.mp4`, `run17_scene.mp4`,
`run19_endo.mp4`, `run19_scene.mp4` (512 × 512, 10 fps). Expected: run 17 (told to go contralateral; the furthest of the 48 contralateral runs) 1069.6 mm³,
**26.8 %**; run 19 (told to stay ipsilateral) 944.1 mm³, **0.5 %**; voxel IoU 1.0 for both.

**(b) The instruction alone changes the policy's answer** — `python3 robot/eval/judge/instruction_probe.py` (about 2.5 min:
120 policy calls). The 8 drilling observations the policy saw in run 17 are sent to the policy endpoint with the same image
and tool state but three sentences (contralateral, ipsilateral, neutral), 5 times each. The policy samples its answer, so
the numbers change a little from run to run. What we got in three runs: with the contralateral sentence the
predicted progress was lower in 8, 8 and 7 of the 8 frames (by 0.11–0.12 ± 0.08) — the policy judges itself less done and keeps
drilling — while the predicted tip position differed by +0.4 to +1.2 mm (± ~1.8) toward the midline, within the
spread of repeated answers.
One run and one start pose: an illustration, not a study; the study is the 105 recorded runs above.

**Your own sentence.** `python3 robot/eval/judge/instruction_probe.py --examples` lists sentences that were **not used in
training** (each checked against the dataset's 131 training sentences, `lerobot/meta/tasks.parquet`; the neutral "Left L4
laminectomy." is a training sentence). Add any of them, or your own, with `--sentence "…"` (repeatable; ~4 min for two).
In our run the new contralateral example "Go across the midline and decompress the right side as well." gave a mean
predicted progress of 0.52 (run 17's sentence 0.51) and the new ipsilateral "Stay on the left side and stop before you
reach the midline." 0.62 (run 19's 0.62): unseen wordings landed with the sentences of the same meaning.

Then let the sentence drive: `python3 robot/eval/judge/sentence_loop.py "…"` starts where run 17 began drilling, sends the
policy endpoint the rendered endoscope image, the tool state and your sentence, moves the virtual tools along 8 of each 40
predicted rows (no arms; each step clamped to 2 mm / 3°) and renders again, until the policy ends the drilling or 150 s.
Output in `data/sentence_loop/<time>/`: `endo.mp4` (endoscope view), `scene.mp4` (3D view), `summary.json` (bone removed,
share cut past the midline, how it ended). About 20–30 min per sentence through the endpoint.

**(c) Limitation — the simulated closed loop is not the robot's loop.** On the same trajectory the render API's step
session removes about 60 % of the bone the record page removed on the robot (run 17: 641 or 658 mm³ against 1070 mm³), so
a simulated drill meets more bone than the robot did. In `sentence_loop.py` runs (two new example sentences, run 17's and
run 19's sentences) the drill stopped 0.8–5.0 mm before the midline whatever the sentence, and the two extents did not
separate consistently. The simulated loop shows that the endpoint's policy and renderer work together; the closed-loop
results reported here come from the 105 recorded robot runs, whose camera videos (tracker IR, RGB, webcam) are in the
dataset (`robot_runs/videos/`).

## Quick start (robot, no hardware)

```bash
git clone https://github.com/geebab/spine-il.git && cd spine-il
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install "lerobot[feetech,kinematics,async,dataset]==0.6.1" -e "robot/lerobot_robot_spinearm[ee,dev]"
spine-arm-server --arm left --sim --cmd-port 6601 --pub-port 6602 &   # virtual arm, no hardware
spine-arm-ctl --cmd-port 6601 lease && spine-arm-ctl --cmd-port 6601 enable
pytest robot/lerobot_robot_spinearm
```

## Citation

```bibtex
@misc{spine_il_2026,
  title  = {Spine IL: a surgeon's hands teach two robot arms where to drill},
  author = {Won, Donghyun and the Spine IL team},
  year   = {2026},
  note   = {Nebius x NVIDIA Global AI Hackathon 2026, Physical AI track},
  url    = {https://github.com/geebab/spine-il},
  howpublished = {Code: https://github.com/geebab/spine-il; weights: https://huggingface.co/vincent-won/spine-il-gr00t-n1.7; data: https://huggingface.co/datasets/vincent-won/spine-il-hand90}
}
```

## Licence

Code: MPL-2.0 ([LICENSE](LICENSE)); the SO-101 meshes keep their Apache-2.0 licence
([ros2_ws/src/spinearm_description/LICENSE](ros2_ws/src/spinearm_description/LICENSE)). Model weights and data: see
[Release](#release).
