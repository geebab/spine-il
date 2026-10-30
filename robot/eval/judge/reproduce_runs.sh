#!/usr/bin/env bash
# Judge check (a): replay the recorded run-17 (contralateral) and run-19 (ipsilateral) robot sessions through the hosted
# render API → share of removed bone past the L4 midline (paper: 26.8 % / 0.5 %). About 30 min in total.
#   RENDER_URL=https://<render endpoint> ENDPOINT_TOKEN=<token> bash robot/eval/judge/reproduce_runs.sh
set -euo pipefail
D=$(cd "$(dirname "$0")" && pwd)
: "${RENDER_URL:?set RENDER_URL (Devpost testing instructions)}" "${ENDPOINT_TOKEN:?set ENDPOINT_TOKEN (Devpost testing instructions)}"
python3 "$D/fetch_robot_runs.py"
python3 "$D/reproduce_runs.py" "$@"   # --video: also save the endoscope and 3D views of both runs as mp4
