# robot/inference — policy rollout and the closed loop

Two ways to close the loop:

- **Simulated**: no arms, no tracker. The render API's step session (`WS /v1/step`) renders the endoscope view and removes
  bone; `live/dry_step.sh` runs `live_loop.py` with simulated arms and can ask a remote policy endpoint, `rollout.py` runs a
  local checkpoint and takes the commanded poses as the next state.
- **On the robot** (`live_loop.py`): the tracker measures the two tools held by the arms; the measured poses are the state,
  the view comes from the recorder page (`--obs-url`, `../page_obs.py`) or from a render-API step session; the policy's
  chunk goes through the safety gate to the two arms. `--continuous` executes while the next chunk is being computed.

## Simulated closed loop

With the hosted endpoints (no GPU needed locally): `live_loop.py` with simulated arms (`fake_arms.py --ik`) and the
render API's step session for the images.

```bash
export RENDER_API_TOKEN=<render token>
POL=https://<policy endpoint> POL_TOKEN_FILE=policy.token TAG=sim1 bash robot/inference/live/dry_step.sh
```

With a local checkpoint on your own GPU, `rollout.py` (the policy's commanded poses become the next state directly):

```bash
cd robot
python inference/rollout.py --policy-path <checkpoint>/pretrained_model --api wss://<render endpoint>/v1/step \
    --dataset <dataset> --episode <n> --cycle --phase-rules -o traj.json
python inference/verify_traj.py traj.json --radius-mm 120          # replays the trajectory through the arm safety gate
python inference/policy2rec.py --traj traj.json                     # trajectory → recorder v2 recording
python eval/policy_eval.py <recording.jsonl> --label <name>         # removed bone vs the demonstrations
```

`rollout.py --help` and the header of `live/dry_step.sh` list the options (start pose, gate pair, drill limits,
`--arm-model`).

## Constraints on the policy output (in order)

Step limit per frame (tip 2 mm, rotation 3°), rotation about the skin entry point (pivot), trigger off when the drill tip
is outside the endoscope view, the arm safety gate (`safety_gate.py`: halves or stops a step that would fail on the two
arms), and the end rules: the policy's progress output above the end level for 5 steps ends drilling; harness limits
(`stall`: < 30 mm³ in 25 s, `pause`, drill time, step budget) are only caps — a run ended by them counts as failed.

## Files

| file | does |
|---|---|
| `rollout.py` | simulated closed loop against the render API |
| `live_loop.py`, `live_robot.py` | closed loop on the robot; tracker tap → state, targets → arms (UDP), trigger heartbeat |
| `remote_policy.py`, `remote/` | protocol `spine-policy/1`, client, reference server ([remote/README.md](remote/README.md)) |
| `safety.py`, `safety_gate.py`, `verify_traj.py` | motion limits, two-arm safety gate, offline gate check |
| `stream_plan.py` | timetable for the continuous loop ([CONTINUOUS.md](CONTINUOUS.md)) |
| `end_level.py` | open-loop scores on held-out demonstrations and the progress level to stop at |
| `openloop_eval.py`, `sensitivity_eval.py` | chunk direction / size vs demonstrations; sensitivity to small input changes |
| `policy2rec.py`, `render_video.py`, `thin_session.py` | recordings from trajectories, replay video (5 panels with camera views), thinned sessions |
| `fake_arms.py` | simulated arms (lag, bias, noise, optional IK) for dry runs |
| `cam_capture.py` | saves the tracker's camera views during a robot run |
| `ready_sweep.py`, `trim_gate.py`, `phase_check.py`, `aim_check.py`, `outside_check.py` | start pose and phase checks |
| `live/` | run scripts: `live_run.sh` (one robot run), `run_remote_base.sh` (robot run with a remote policy), `dry_step.sh` / `dry_page.sh` / `dry_stream.sh` / `dry_remote.sh` (dry runs without the robot), `remote_intake.sh` (new model: check → evaluate → dry run), reports |
| `tests/` | unit tests (no robot, no render API) |

The scripts in `live/` and `run_cycle_ph.sh` use `SPINE_IL_REPO` (default `~/spine-il`) and `~/il-robot/{datasets,runs}`
as working folders.
