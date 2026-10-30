#!/usr/bin/env bash
# The evaluation of robot runs made with LIVE_EVAL=0 (live_run.sh), later: the same steps as live_run.sh's last ones — the
# measured poses as a policy recording, then policy_eval.py against the demos (a step replay of the measured poses and the
# replay of the robot session's own cuts). The replay of a robot session holds the render API for about 2.5 times the
# session's length, and no render may run beside a robot run: with runs of up to 48 min (240 s of drilling) the next run
# would wait two hours for the evaluation of the one before. What a run did is known without it (live.json: removed volume,
# drill_end, eval/run_trace.py); the overlap with the demos comes when the robot part of a series is over.
#   bash eval_later.sh NAME [NAME ...]        NAME = the run's folder under runs/act_smoke/live
# Waits while a robot run is queued or active and while the render API is busy — and while the robot queue itself waits for
# the renderer (work/RO_WANT, kept by ro_want.sh): the queue takes a job off ro_jobs.txt BEFORE it waits for the renderer, so
# for that time neither the job list nor a process shows that a run is coming. 10-08 01:01: the evaluation of the run
# before finished, this script and the queue both saw an idle renderer within seconds, and a replay render ran beside a
# robot run for its whole length. Hence RO_WANT, a second look a few seconds later, and one more right before the render.
# The robot cuts come first: the step re-simulation of the measured poses (policy_eval's "policy" row, 12–14 min a run,
# not used by run_criteria.py) is left out unless EVAL_LATER_ROBOT_ONLY=0 — run policy_eval.py again for it when the
# render API has time (the robot cut is cached then).
export POLICY_EVAL_ROBOT_ONLY=$([ "${EVAL_LATER_ROBOT_ONLY:-1}" = 0 ] || echo 1)
A=$HOME/il-robot/runs/act_smoke; LV=$A/live; PY=$HOME/il-robot/.venv/bin/python; W=$HOME/il-robot/runs/jitter_iter/work
busy() { [ -e $W/RO_WANT ] || ! (cd $W && $PY render_idle.py) || pgrep -f "[e]ndo_follow\\.py|[r]obot_origin_run\\.sh|[r]ender_video\\.py|[p]olicy_eval\\.py|[r]o_iou\\.py" >/dev/null \
         || [ -n "$(grep -v '^#' $W/ro_jobs.txt 2>/dev/null | grep -m1 .)" ] \
         || { $PY ${SPINE_IL_REPO:-$HOME/spine-il}/robot/eval/render_gate.py status >/dev/null 2>&1; [ $? = 1 ]; }; }   # 10-09: the gate has a pause in force (an arm in use from any account)
for NAME in "$@"; do
  O=$LV/$NAME; RUN=$(cat $O/run_dir 2>/dev/null); SF=$(cat $O/robot_session 2>/dev/null)
  if ! [ -s $O/live.json ] || [ -z "$RUN" ] || ! [ -s "$SF" ]; then echo "$(date +%T) $NAME: no finished run — skipped"; continue; fi
  if grep -q "^robot " $O/eval.log 2>/dev/null; then echo "$(date +%T) $NAME: already evaluated"; continue; fi
  POL=$(tr ' ' '\n' < $O/live.args | grep -A1 -m1 -- "--policy-path" | tail -1)
  OPT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('optics') or '')" $POL/action_rep.json 2>/dev/null)
  # a run of a remote policy has no checkpoint folder: live_run.sh wrote its medium into live.args (--policy-url … --optics water)
  [ -z "$POL" ] && OPT=$(tr ' ' '\n' < $O/live.args | grep -A1 -m1 -- "--optics" | tail -1)
  free_now() { while :; do busy && { sleep 20; continue; }; sleep 7; busy || break; done; }   # idle, and still idle a moment later
  # 10-08 16:05-16:13: the disk was full — twelve evaluations of this batch failed within seconds each (policy2rec wrote
  # empty recordings). With less than EVAL_MIN_FREE_GB free the batch stops instead (an evaluation's render job: 15-65 MB)
  FREE_GB=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  [ "${FREE_GB:-0}" -ge ${EVAL_MIN_FREE_GB:-3} ] || { echo "$(date +%T) $NAME: disk nearly full (${FREE_GB:-?} GB free) — batch stopped"; exit 3; }
  free_now
  echo "$(date +%T) evaluation (later)" >> $O/log
  if ! (cd ${SPINE_IL_REPO:-$HOME/spine-il}/robot && $PY inference/policy2rec.py --traj $O/live.json > $O/p2r_meas.json 2>&1); then
    echo "$(date +%T) FAILED: policy2rec (measured)" >> $O/log; continue; fi
  RECM=$(python3 -c "import json;print(json.load(open('$O/p2r_meas.json'))['recording'])")
  SFE=$RUN/thin/$(basename "$SF"); [ -s "$SFE" ] || SFE=$SF
  busy && free_now   # the seconds of policy2rec: a robot job may have come
  if (cd $W && OPENBLAS_NUM_THREADS=2 nice $PY -u policy_eval.py $RECM --robot $SFE --label live_$NAME ${OPT:+--optics $OPT} > $O/eval.log 2>&1); then
    echo "$(date +%T) evaluated" >> $O/log
  else echo "$(date +%T) FAILED: policy_eval" >> $O/log; fi
done
