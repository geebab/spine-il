# robot/eval — removed-bone metrics, robot-run tools, the 105-run study

## Comparing removed bone

```bash
python policy_eval.py <recording.jsonl> [--robot <robot session.jsonl> ...] [--label name]   # vs every demonstration
python policy_eval.py --demos-only                                                           # baseline between demonstrations
python cut_agreement.py <run folder> [...] [--json out.json]                                 # run × run, demo × demo, run × demo IoU
```

Removed bone is a voxel set (from the render API's `removed_voxels`); metrics are IoU / Dice, 1-voxel tolerant overlap,
ASSD, HD95 (`metrics_ext.py`), cover, precision, bone removed outside the target, and the share of drilling time the drill
was visible in the endoscope. The demonstrations are compared with each other the same way, so a run is judged against the
spread between demonstrations (`run_criteria.py`).

## The 105-run study (`deep_1018/`)

Question: does the instruction sentence change where the robot drills? 105 closed-loop runs
on L3, L4, L5 (per level 16 contralateral, 16 ipsilateral, 3 neutral instructions from four start poses), including one
held-out wording per extent (`novel_*.txt`). `build_master.py` builds one SQLite file with every run, demonstration and
analysis output; `stage1.py` and the analysis scripts read it.

## Robot runs

| file | does |
|---|---|
| `robot_origin_run.sh` | one two-arm run recorded as a robot-origin session; presets and limits from environment variables (`RO_*`) |
| `guard_pairs.py` | run guard: stops both arms if a tip moves too far from its start or the two tips come closer than 5 mm |
| `new_recording.py`, `first_cut.py` | new hand recording → start-pose search → safety check → queued robot replay |
| `pairstart/` | two-arm start-pose search |
| `pair_reach.py`, `fit_registration.py`, `sitting_base.py` | reach of one start pair; fitting a recording's scene registration to the robot's tools; representative pose of a session |
| `skin_surface.py` | skin entry model from the entry points stored in the recordings (`--build`); where a tool axis crosses the skin |
| `scope_screen.py` | is the tip inside the endoscope view (shared by the harness and the metrics) |
| `run_criteria.py`, `run_trace.py` | pass criteria from the demonstrations' own spread; what a run did chunk by chunk |
| `render_gate.py` | pauses rendering work while a robot run needs the machine |
| `ro_queue2.sh`, `ro_iou.py`, `ro_summary.py`, `ro_paired.py` | robot replay queue and comparison with the hand |
| `iou_eval.py`, `vol_eval.py`, `metrics_ext.py`, `tip_motion.py`, `resim_fair.py`, `view_recount.py` | metrics and helpers |
| `judge/` | checks for the hackathon judges against the hosted endpoint: replay runs 17 / 19 (`reproduce_runs.sh`), the instruction probe, the data fetch — see [judge/README.md](judge/README.md) |
| `tests/` | unit tests |

Scripts use `SPINE_IL_REPO` (default `~/spine-il`) for this repository and `~/il-robot/{datasets,runs,calib}` as working
folders.
