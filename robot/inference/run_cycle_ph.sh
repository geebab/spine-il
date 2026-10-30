#!/usr/bin/env bash
# Phase-conditioned ACT (out_ph) → full-cycle rollout (ready pose → entry → policy → exit → ready) → verify → viewer video
# → robot playback (5 panels) → outcome metrics vs the 8 demos.   run_cycle_ph.sh [CKPT=last] [TAG=f] [--no-robot]
# Waits for the training to end when CKPT=last. Status in cycle_ph_<CKPT>.done (one line), logs beside it.
A=$HOME/il-robot/runs/act_smoke; PY=$HOME/il-robot/.venv/bin/python; W=$HOME/il-robot/runs/jitter_iter/work
L=$HOME/il-robot/runs/jitter_iter/robot_origin.log
CK=${1:-last}; TAG=${2:-f}; ROBOT=1; [ "$3" = "--no-robot" ] && ROBOT=0
N=cycle_ph_$CK; DONE=$A/$N.done; rm -f $DONE
if [ $CK = last ]; then until ! pgrep -f "[l]erobot-train.*out_ph" >/dev/null && [ -d $A/out_ph/checkpoints/030000 ]; do sleep 60; done; fi
POL=$A/out_ph/checkpoints/$CK/pretrained_model; [ -d $POL ] || { echo "no checkpoint $POL" > $DONE; exit 1; }
idle() { until (cd $W && $PY render_idle.py) && ! pgrep -f "[r]o_iou\.py|[p]olicy_eval\.py" >/dev/null && ! pgrep -f "[e]ndo_follow\.py" >/dev/null; do sleep 15; done; }
idle
cd ${SPINE_IL_REPO:-$HOME/spine-il}/robot
OMP_NUM_THREADS=2 taskset -c 8-11 $PY -u inference/rollout.py --policy-path $POL \
  --dataset $HOME/il-robot/datasets/spine_rev4_step_delta_ph --episode 6 --start-frame 255 --gate-pair 3868 --cycle --steps 900 \
  --drill-max-s 40 --drill-target-mm3 487 --phase-rules -o $A/traj_$N.json > $A/rollout_$N.log 2>&1 || { echo "rollout failed" > $DONE; exit 1; }
OMP_NUM_THREADS=1 taskset -c 8-11 $PY inference/verify_traj.py $A/traj_$N.json --radius-mm 120 > $A/verify_$N.json 2>&1 || { echo "verify failed" > $DONE; exit 1; }
$PY inference/policy2rec.py --traj $A/traj_$N.json > $A/p2r_$N.json 2>&1 || { echo "policy2rec failed" > $DONE; exit 1; }
REC=$(python3 -c "import json;print(json.load(open('$A/p2r_$N.json'))['recording'])"); RID=$(python3 -c "import json;print(json.load(open('$A/p2r_$N.json'))['rid'])")
LEN=$(python3 -c "import json;print(int(json.load(open('$A/p2r_$N.json'))['frames']/15)+1)")
idle
OMP_NUM_THREADS=2 $PY -u inference/render_video.py $REC -o $A/${N}_inference_replay.mp4 > $A/video_$N.log 2>&1
echo "$REC" > $A/${N}_video1.done
[ $ROBOT = 0 ] && { echo "viewer only: $REC" > $DONE; exit 0; }
robot_run() {   # TAG → 0 ok (O = run dir) · 1 skipped · 2 the arms stopped early (guard) or lost their lease (tracker outage)
  local T=$1
  (cd $W && $PY -u new_recording.py $RID --from-s 0 --len-s $LEN --reps 1 --start-from-traj $A/traj_$N.json --radius-mm 120 --tag $T > $A/new_recording_${N}_$T.log 2>&1)
  until grep -qE "pr_${RID}${T}1 done" $L || grep -q "SKIPPED" $A/new_recording_${N}_$T.log; do sleep 20; done
  grep -q "SKIPPED" $A/new_recording_${N}_$T.log && return 1
  O=$(ls -td $HOME/il-robot/runs/robot_origin/*_replay_pr_${RID}${T}1 2>/dev/null | head -1)
  grep -E "pr_${RID}${T}1 done" $L | tail -1 | grep -qE "LOST THEIR LEASE|STOPPED EARLY" && return 2
  return 0
}
robot_run $TAG; RC=$?
if [ $RC = 2 ]; then echo "robot run pr_${RID}${TAG}1 did not complete — one retry" >> $A/chain_note_$N.log; robot_run ${TAG}r; RC=$?; fi
[ $RC != 0 ] && { echo "robot run failed (rc $RC, see robot_origin.log)" > $DONE; exit 1; }
until [ -e $O/video.log ] && ! pgrep -f "[r]ender_video.py.*-o $O" >/dev/null && [ -s $O/session_5panel.mp4 ]; do sleep 20; done
SID=$(tail -1 $O/result.json | python3 -c "import json,sys;print(json.load(sys.stdin).get('sessionId') or '')")
SF=$(find $HOME/il-robot/viewer-data/sessions/robot -name "$SID.jsonl" | head -1)
idle
(cd $W && OPENBLAS_NUM_THREADS=2 nice $PY -u policy_eval.py $REC --robot $SF --label $N > $A/eval_$N.log 2>&1)
echo "$O" > $DONE
