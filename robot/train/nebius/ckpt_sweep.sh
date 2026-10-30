#!/usr/bin/env bash
# The checkpoints of one training compared WITHOUT the robot (2026-10-12): for each step the policy server is started on
# the VM with that checkpoint, then
#   1 the held-out demonstrations   inference/end_level.py --policy-url: open-loop scores, progress error
#   2 the observations of robot runs asked again   inference/remote/replay_obs.py: the progress output by removed volume,
#     where a level would have ended the drilling, the spread of one observation asked REPEAT times, turns between rows
# and at the end one table (ckpt_sweep.py).
#   bash ckpt_sweep.sh NAME [STEP …]        no steps = every checkpoint of NAME on the VM
#   RUNS="dir …"  robot runs to ask again (default: the three runs of 10-12)   REPEAT=3  EVERY=2
#   OUT=~/il-robot/runs/groot_prep/sweep_<NAME>_<date>   DS=held-out dataset   EPISODES=2,18
# The VM must be up and set (vm.sh set USER@IP). About 10 minutes a checkpoint (model load 1–2, held-out 5, replay 3).
# The server of the LAST step is left running (vm.sh serve-stop NAME when done). Nothing here moves an arm.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-$HOME/il-robot/.venv/bin/python}; G="$PY $R/eval/render_gate.py"
NAME=${1:?run name}; shift
LV=$HOME/il-robot/runs/act_smoke/live
RUNS=${RUNS:-"$LV/R_G24a_C1006_r0 $LV/R_G24a_cont_C1006_r0 $LV/R_G24a_cont2_C1006_r0"}
DS=${DS:-$HOME/il-robot/datasets/spine_hand24_replay_trim_water_progress}; EPISODES=${EPISODES:-2,18}
OUT=${OUT:-$HOME/il-robot/runs/groot_prep/sweep_${NAME}_$(date +%m%d)}; mkdir -p "$OUT"
STEPS="$*"
[ -n "$STEPS" ] || STEPS=$(bash "$R/train/nebius/vm.sh" ssh "ls ~/il-robot/out/$NAME/run/checkpoints" 2>/dev/null | grep -E '^[0-9]+$' | tr '\n' ' ')
[ -n "$STEPS" ] || { echo "no checkpoints of $NAME found on the VM (is it up? bash vm.sh set USER@IP)"; exit 2; }
echo "$(date +%T) sweep of $NAME: steps $STEPS → $OUT" | tee -a "$OUT/log"
for STEP in $STEPS; do
  if [ -s "$OUT/${STEP}_heldout.json" ] && [ -s "$OUT/${STEP}_replay.json" ]; then echo "$(date +%T) $STEP: done before" | tee -a "$OUT/log"; continue; fi
  bash "$R/train/nebius/vm.sh" serve "$NAME" "$STEP" > "$OUT/${STEP}_serve.txt" 2>&1 || { echo "$(date +%T) $STEP: the server did not start — see ${STEP}_serve.txt" | tee -a "$OUT/log"; continue; }
  bash "$R/train/nebius/vm.sh" tunnel "$NAME" >> "$OUT/${STEP}_serve.txt" 2>&1
  eval "$($PY "$R/inference/remote/endpoint.py" env "$NAME")"          # EP_URL EP_TOKEN_FILE
  [ -n "${EP_URL:-}" ] || { echo "$(date +%T) $STEP: no endpoint $NAME" | tee -a "$OUT/log"; continue; }
  echo "$(date +%T) $STEP: held-out demonstrations" | tee -a "$OUT/log"
  (cd "$R" && $G run --on-pause kill --low --mem-gb 2 --name sweep_eval -- $PY inference/end_level.py --dataset "$DS" --policy-url "$EP_URL" \
     --policy-token-file "$EP_TOKEN_FILE" --episodes "$EPISODES" -o "$OUT/${STEP}_heldout.json") > "$OUT/${STEP}_heldout.txt" 2>&1 \
     || echo "$(date +%T) $STEP: held-out evaluation failed — see ${STEP}_heldout.txt" | tee -a "$OUT/log"
  echo "$(date +%T) $STEP: robot observations again" | tee -a "$OUT/log"
  (cd "$R" && $G run --on-pause kill --low --mem-gb 2 --name sweep_replay -- $PY inference/remote/replay_obs.py $RUNS --policy-url "$EP_URL" \
     --policy-token-file "$EP_TOKEN_FILE" --every "${EVERY:-2}" --repeat "${REPEAT:-3}" --json "$OUT/${STEP}_replay.json") > "$OUT/${STEP}_replay.txt" 2>&1 \
     || echo "$(date +%T) $STEP: replay failed — see ${STEP}_replay.txt" | tee -a "$OUT/log"
done
$PY "$R/train/nebius/ckpt_sweep.py" "$OUT" | tee "$OUT/TABLE.txt"
echo "$(date +%T) sweep done — the server of the last step is still up (bash vm.sh serve-stop $NAME)" | tee -a "$OUT/log"
