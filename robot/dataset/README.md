# robot/dataset — hand recordings → LeRobot v3 dataset (GR00T N1.7 fine-tuning)

Turns tracked hand recordings (recorder schema v2 JSONL) into a LeRobot v3 dataset: the endoscope view rendered by the
render API, the two tools' tip poses as state, the next poses plus bone-removal progress as action, and one instruction
sentence per frame. An Open-H-Embodiment v2 layout can be written from the same sessions.

## Files

| file | does |
|---|---|
| `sources.py` | finds sessions, checks compatibility (records why one is skipped), robot-run metadata, arm-log clock alignment |
| `phases.py` | splits each session into six phases (skin → target → drilling …) |
| `instructions.py` | instruction sentences per level and extent (ipsilateral / contralateral), including held-out wordings |
| `render_cache.py` | renders the endoscope view of each session through the render API and caches it (background queue) |
| `render_step.py` | re-renders training frames with the same step renderer the closed loop uses |
| `export_lerobot.py` | writes the dataset and reads it back to validate; `--progress-output` adds `drill.progress`, `--layout openh` writes Open-H |
| `actions.py` | action representations (absolute, delta) |
| `trim.py` | keeps only the stretch of a demonstration the arms can follow |
| `global_quantiles.py` | rewrites `meta/stats.json` quantiles over all frames (LeRobot's per-episode average is too narrow for GR00T's clipping) |
| `openh.py`, `openh_readme.md` | Open-H-Embodiment v2 writer and its dataset README template |
| `public_dataset.py` | public copy of a dataset (operator renamed, local paths removed) |
| `phase_report.py`, `phase_video.py` | phase-label checks (JSON report, overlay video) |
| `end_gesture.py` | reports how each demonstration ends |
| `groot_dataset.sh` | one command for new recordings: render → export → preflight (`../train/groot_preflight.py`) |
| `tests/` | unit tests on synthetic sessions (no render API needed) |

## Features (the released dataset)

| key | shape | content |
|---|---|---|
| `observation.images.endo` | video 256 × 256 × 3 | the rendered endoscope view (water medium) |
| `observation.state` | 16 float32 | endoscope tip xyz (mm) + quaternion wxyz, drill tip xyz + quaternion wxyz, drill trigger, `pad.zero` (always 0) — scene coordinates |
| `action` | 16 float32 | the same 15 values at the next frame (absolute) + `drill.progress` (bone removed so far ÷ the demonstration's total, 0 → 1) |
| `task` | string | the instruction sentence of that frame |

15 fps. Other columns (`phase.index`, `track.valid`, `time.src_s`, robot joints for robot episodes) are not policy inputs.

## Run

```bash
cd robot/dataset
python render_cache.py --fps 15 --include hand          # needs the render API (RENDER_API_TOKEN, --server URL)
python export_lerobot.py --out ~/il-robot/datasets/<name> --mode full --include hand --fps 15 --progress-output --overwrite
python export_lerobot.py --out ~/il-robot/datasets/<name> --validate-only
python global_quantiles.py ~/il-robot/datasets/<name>
python -m pytest -q tests
```
