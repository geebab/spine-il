#!/usr/bin/env bash
# Dry run of the closed loop looking at a recorder page (live_loop.py --obs-url): simulated arms (fake_arms.py) and a
# stand-in page (page_obs.py --fake) — no robot, no viewer, no render API.
#   TAG=name POL=<pretrained_model dir> bash dry_page.sh      (PAUSE_S / STALL_S: the drilling's end rules, 3 / 5 s here)
# Checks the loop end to end: entry → policy chunks while the stand-in page removes bone (CUTS looks, after SKIP looks
# without) → the cutting stops → the drilling ends by the pause or the stall rule → exit. The stand-in's image is a flat
# colour and its bone does not depend on the tools: what the policy does with it means nothing; the loop's bookkeeping does.
L=$HOME/il-robot/runs/act_smoke/live/t1; PY=$HOME/il-robot/.venv/bin/python; A=$HOME/il-robot/runs/act_smoke
TAG=${TAG:-dry_page_$(date +%H%M%S)}; O=$L/$TAG
[ -e $O ] && { echo "$O exists — another TAG"; exit 2; }
mkdir -p $O/lat
POL=${POL:-$A/out_rel_sn_aug_water/checkpoints/002500/pretrained_model}; PORT=${PORT:-18792}
OPT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('optics') or 'air')" $POL/action_rep.json)
cd ${SPINE_IL_REPO:-$HOME/spine-il}/robot
RUN=FAKE_$(date +%H%M%S)
# the stand-in's skin entry = the demos' endoscope portal (rollout.demo_portal of the dataset below): "inside" from the start
$PY -u page_obs.py --fake $PORT --medium $OPT --entry 54.85 -1.7 1043.7 --skip ${SKIP:-6} --cuts ${CUTS:-5} > $O/page.log 2>&1 &
PG=$!
$PY -u inference/fake_arms.py --run $RUN --seed-traj $L/seed.json --latency-dir $O/lat --max-s 1500 $FAKE_X > $O/fake.log 2>&1 &
FA=$!
OMP_NUM_THREADS=4 nice taskset -c 12-15 $PY -u inference/live_loop.py --live-run $RUN --latency-dir $O/lat --tap-port 9894 \
  --policy-path $POL --dataset $HOME/il-robot/datasets/spine_rev4_step_delta_ph_trim \
  --episode 6 --start-frame 0 --gate-anchor-s 17 --gate-pair 3868 --cycle --start-state $A/iter/starts/start_1.json \
  --steps ${STEPS:-600} --n-action-steps ${NEXEC:-20} --phase-rules --drill-max-s 240 --drill-pause-s ${PAUSE_S:-3} \
  --drill-stall-s ${STALL_S:-5} --deadline-s 0 --frames-dir $O/frames --obs-url http://127.0.0.1:$PORT $LOOP_X \
  -o $O/live.json > $O/live.log 2>&1
echo "live rc $?" >> $O/live.log
sleep 2; kill $FA $PG 2>/dev/null
echo done > $O/DONE
