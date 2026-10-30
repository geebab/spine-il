#!/usr/bin/env bash
# Dry run of the closed loop with simulated arms (no robot), with faults: entry → policy chunks → exit.
#   TAG=name STEPS=100 FAKE_X="--stamp-lag 100:125:300" bash dry2.sh      (or FAKE_X="--lose-lease-s 110"; POL=<pretrained_model dir>)
# 10-05: checks that live_loop.py waits through stale tracker stamps and stops when a follower left its replay.
L=$HOME/il-robot/runs/act_smoke/live/t1; PY=$HOME/il-robot/.venv/bin/python; A=$HOME/il-robot/runs/act_smoke
TAG=${TAG:-dry}; O=$L/$TAG; mkdir -p $O; rm -rf $O/lat $O/frames $O/DONE; mkdir -p $O/lat
cd ${SPINE_IL_REPO:-$HOME/spine-il}/robot
RUN=FAKE_$(date +%H%M%S)
$PY -u inference/fake_arms.py --run $RUN --seed-traj $L/seed.json --latency-dir $O/lat --max-s 1500 $FAKE_X > $O/fake.log 2>&1 &
FA=$!
OMP_NUM_THREADS=4 taskset -c 8-11 $PY -u inference/live_loop.py --live-run $RUN --latency-dir $O/lat --tap-port 9894 \
  --policy-path ${POL:-$A/out_rel/checkpoints/015000/pretrained_model} --dataset $HOME/il-robot/datasets/spine_rev4_step_delta_ph_trim \
  --episode 6 --start-frame 0 --gate-anchor-s 17 --gate-pair 3868 --cycle --start-state $A/iter/starts/start_1.json \
  --steps ${STEPS:-100} --n-action-steps ${NEXEC:-20} --phase-rules --drill-max-s 40 --drill-target-mm3 487 $LOOP_X \
  -o $O/live.json > $O/live.log 2>&1
echo "live rc $?" >> $O/live.log
sleep 2; kill $FA 2>/dev/null
echo done > $O/DONE
