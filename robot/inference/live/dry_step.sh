#!/usr/bin/env bash
# Dry run of the closed loop as a robot run executes it — live_loop.py itself, chunk by chunk — with simulated arms that
# move like the real ones (fake_arms.py --ik: the tip as asked, the tool turned only as far as the arms' IK turns it) and
# the render API's step session for the images (no recorder page, no robot). What a robot run would do in 30 min, here in
# about 15: does the policy get to the bone, does it drill, what does the safety gate stop and why.
#   TAG=name POL=<pretrained_model dir> [COMMON="--dataset … --episode … --gate-anchor-s … --gate-pair … --start-state …"]
#   [LOOP_X="--gate-turn-wait"] [FAKE_X="--r0 model"] [STEPS=900] [DMAX=120] bash dry_step.sh
#   FAKE_X="--r0 model" for a seed that does not stand on the 3868 pair: the fake arms then engage at the pose their own
#   start joints give (the default takes the engage rows of a real 3868-pair run)
# COMMON as live_run.sh builds it (default: the 3868 base and start 1). Output in ~/il-robot/runs/act_smoke/dry_step/<TAG>/:
# live.json (the run's own file: measured frames, commanded frames, safety_gate), live.log, fake.log, frames/.
A=$HOME/il-robot/runs/act_smoke; PY=$HOME/il-robot/.venv/bin/python; W=$HOME/il-robot/runs/jitter_iter/work
TAG=${TAG:-dry_step_$(date +%H%M%S)}; O=$A/dry_step/$TAG
[ -e $O ] && { echo "$O exists — another TAG"; exit 2; }
mkdir -p $O/lat
POL=${POL:-$A/out_rel_h24_replay_water/checkpoints/005000/pretrained_model}
# POL=http(s)://host:port: a policy server instead of a checkpoint (inference/remote_policy.py; POL_TOKEN_FILE, POL_CA_FILE, POL_OPTICS) —
# the same loop, the same simulated arms, the question "what next?" asked over the wire
POLARG="--policy-path $POL"
case "$POL" in http://*|https://*) POLARG="--policy-url $POL${POL_TOKEN_FILE:+ --policy-token-file $POL_TOKEN_FILE}${POL_CA_FILE:+ --policy-ca-file $POL_CA_FILE} --optics ${POL_OPTICS:-water}";; esac
COMMON=${COMMON:---dataset $HOME/il-robot/datasets/spine_rev4_step_delta_ph_trim --episode 6 --start-frame 0 --gate-anchor-s 17 --gate-pair 3868 --start-state $A/iter/starts/start_1.json}
cd ${SPINE_IL_REPO:-$HOME/spine-il}/robot || exit 2
echo "$(date +%T) $TAG: $POL | $COMMON | $LOOP_X" > $O/log
OMP_NUM_THREADS=2 nice taskset -c ${CORES:-4-7} $PY -u inference/live_loop.py --prepare-out $O/seed.json $COMMON --cycle > $O/prepare.log 2>&1 || { echo "prepare failed" >> $O/log; exit 1; }
until (cd $W && $PY render_idle.py) && ! pgrep -f "[r]ender_video\.py|[e]ndo_follow\.py|[p]olicy_eval\.py|[r]o_iou\.py" >/dev/null; do sleep 20; done
RUN=FAKE_$(date +%H%M%S)
nice taskset -c ${CORES:-4-7} $PY -u inference/fake_arms.py --run $RUN --seed-traj $O/seed.json --latency-dir $O/lat --max-s 3600 --ik $FAKE_X > $O/fake.log 2>&1 &
FA=$!
OMP_NUM_THREADS=4 nice taskset -c ${CORES:-4-7} $PY -u inference/live_loop.py --live-run $RUN --latency-dir $O/lat --tap-port 9894 \
  $POLARG $COMMON --cycle --steps ${STEPS:-900} --n-action-steps ${NEXEC:-20} --phase-rules --drill-max-s ${DMAX:-120} \
  --drill-pause-s ${PAUSE_S:-15} --drill-stall-s ${STALL_S:-25} --drill-stall-mm3 30 --deadline-s 0 --frames-dir $O/frames $LOOP_X \
  -o $O/live.json > $O/live.log 2>&1
echo "$(date +%T) live rc $?" >> $O/log
sleep 2; kill $FA 2>/dev/null
echo done > $O/DONE
