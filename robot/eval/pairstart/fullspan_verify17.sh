#!/usr/bin/env bash
# 10-03 (user: "방향 한계 완화"): gate check of the three long drillings over their whole kept span with the IK orientation
# limit at 17° instead of 10°, for the full-span pairs (anchored at the span start) and, for comparison, the old 30 s pairs.
# Cores 12-15, nice 19, paused while the arms move.
P=$HOME/il-robot/runs/jitter_iter/work/pairstart; PY=$HOME/il-robot/.venv/bin/python
cd ${SPINE_IL_REPO:-$HOME/spine-il}/robot
run() {   # pairs file, output name
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 taskset -c 12-15 $PY -u inference/trim_gate.py \
    --dataset $HOME/il-robot/datasets/spine_rev4_step_delta --trimmed $HOME/il-robot/datasets/spine_rev4_step_delta_trim \
    --pairs $1 --only bad1,d340,bef5 --stride 3 --oerr-deg 17 -o $P/$2.json > $P/$2.log 2>&1 &
  local G=$!
  while kill -0 $G 2>/dev/null; do
    if pgrep -f "[e]ndo_follow\.py" >/dev/null; then kill -STOP $G; else kill -CONT $G; fi
    sleep 5
  done
}
run $P/pairs_fullspan.json trim_gate_fullspan_17
run $P/pairs_by_recording_chosen.json trim_gate_old_17
echo done > $P/verify17.done
