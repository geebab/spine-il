#!/usr/bin/env bash
# ON THE GPU MACHINE: one GR00T N1.7 fine-tuning with LeRobot (the command of train/GROOT.md section 3). Started by
# vm.sh train / smoke from the robot server (in tmux), or by hand:
#   NAME=run DS=~/il-robot/datasets/<dataset> STEPS=20000 BATCH=32 SAVE=5000 [EPISODES='[0,1,3,…]'] bash vm_train.sh
#   EPISODES  the episodes trained on (a JSON list). Empty = every episode — the last training, nothing held out.
#   BASE      another base model (default nvidia/GR00T-N1.7-3B) · CKPT=0 no checkpoint at all (the smoke test)
#   RESUME=1  go on from this run's last checkpoint
#   LLM_TOP=N train the top N layers of the language model too (default 0: LeRobot's default, the whole vision-language
#             backbone frozen — user 10-14: the language part must learn; L30b answered the extent words only weakly)
# Relative actions with the trigger and the progress kept absolute, chunk 40 (LeRobot's default = the model's own
# horizon; GROOT.md says why). Output: ~/il-robot/out/NAME/{run/ (LeRobot's folder: checkpoints/<step>/pretrained_model),
# train.log, gpu.csv (memory and use every 5 s), run.json, DONE}.
set -uo pipefail
WS=${IL_ROBOT_WS:-$HOME/il-robot}; NAME=${NAME:?run name}; DS=${DS:?dataset folder}; O=$WS/out/$NAME
STEPS=${STEPS:-20000}; BATCH=${BATCH:-32}; SAVE=${SAVE:-5000}
[ -s "$DS/meta/info.json" ] || { echo "no dataset at $DS"; exit 2; }
mkdir -p "$O" && cd "$WS" || exit 2
export HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false
if [ "${RESUME:-0}" = 1 ]; then
  CFG=$O/run/checkpoints/last/pretrained_model/train_config.json
  [ -s "$CFG" ] || { echo "nothing to resume: $CFG"; exit 2; }
  ARGS=(--config_path="$CFG" --resume=true)
else
  [ -e "$O/run" ] && { echo "$O/run exists — another NAME, or RESUME=1"; exit 2; }
  ARGS=(--policy.type=groot --policy.use_relative_actions=true '--policy.relative_exclude_joints=["trigger","progress"]'
        --dataset.repo_id="local/$(basename "$DS")" --dataset.root="$DS" --policy.push_to_hub=false
        --output_dir="$O/run" --job_name="$NAME" --steps="$STEPS" --batch_size="$BATCH" --save_freq="$SAVE"
        --log_freq="${LOG:-50}" --num_workers="${WORKERS:-8}" --wandb.enable=false)
  [ -n "${EPISODES:-}" ] && ARGS+=("--dataset.episodes=$EPISODES")
  [ -n "${BASE:-}" ] && ARGS+=("--policy.base_model_path=$BASE")
  [ "${LLM_TOP:-0}" != 0 ] && ARGS+=("--policy.tune_top_llm_layers=$LLM_TOP")
  [ "${CKPT:-1}" = 0 ] && ARGS+=(--save_checkpoint=false)
  printf '{"name": "%s", "dataset": "%s", "episodes": %s, "steps": %s, "batch": %s, "save_freq": %s, "log_freq": %s, "base": "%s", "llm_top": %s, "started": "%s", "gpu": "%s", "kit": "%s"}\n' \
    "$NAME" "$(basename "$DS")" "${EPISODES:-null}" "$STEPS" "$BATCH" "$SAVE" "${LOG:-50}" "${BASE:-nvidia/GR00T-N1.7-3B}" "${LLM_TOP:-0}" "$(date -Is)" \
    "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)" "$(cut -c1-60 "$WS/kit/VERSION" 2>/dev/null)" > "$O/run.json"
fi
rm -f "$O/DONE"
( while :; do nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits >> "$O/gpu.csv" 2>/dev/null; sleep 5; done ) &
SMI=$!
echo "$(date '+%m-%d %T') lerobot-train ${ARGS[*]}" >> "$O/train.log"
"$WS/.venv/bin/lerobot-train" "${ARGS[@]}" >> "$O/train.log" 2>&1
RC=$?
kill $SMI 2>/dev/null
echo "rc $RC $(date -Is)" > "$O/DONE"
exit $RC
