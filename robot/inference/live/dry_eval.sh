#!/usr/bin/env bash
# The cut of a finished dry run (dry_step.sh) against the demos, as live_run.sh judges a robot run's measured trajectory:
# live.json → policy2rec.py (a recording) → eval/policy_eval.py (the trajectory's cut simulated in the viewer at the live
# page's 15 Hz: removed volume, mean IoU, coverage, precision, off-target, drill visible while cutting).
#   bash dry_eval.sh TAG        → ~/il-robot/runs/jitter_iter/policy_eval_dry_<TAG>.json, <dry run>/eval.log
# A dry run that removed nothing is not rendered. Waits for the render API and for a running dry run to end.
A=$HOME/il-robot/runs/act_smoke; PY=$HOME/il-robot/.venv/bin/python; W=$HOME/il-robot/runs/jitter_iter/work
TAG=$1; O=$A/dry_step/$TAG
[ -s $O/live.json ] || { echo "$O/live.json missing"; exit 2; }
read REM POL <<<$(python3 -c "
import json, sys
m = json.load(open(sys.argv[1]))['meta']; r = m.get('removed_mm3'); print(r[-1] if isinstance(r, list) else r, m.get('policy'))" $O/live.json)
python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > 1.0 else 1)" "$REM" || { echo "nothing removed ($REM mm3)" > $O/eval.log; exit 0; }
OPT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('optics') or '')" $POL/action_rep.json 2>/dev/null)
cd ${SPINE_IL_REPO:-$HOME/spine-il}/robot || exit 2
$PY inference/policy2rec.py --traj $O/live.json > $O/p2r_meas.json 2>&1 || { echo "policy2rec failed" > $O/eval.log; exit 1; }
REC=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['recording'])" $O/p2r_meas.json)
until (cd $W && $PY render_idle.py) && ! pgrep -f "[l]ive_loop\.py|[f]ake_arms\.py|[r]ender_video\.py|[e]ndo_follow\.py|[p]olicy_eval\.py" >/dev/null; do sleep 20; done
(cd $W && OPENBLAS_NUM_THREADS=2 nice taskset -c 4-7 $PY -u policy_eval.py $REC --label dry_$TAG ${OPT:+--optics $OPT} > $O/eval.log 2>&1)
echo "rc $?" >> $O/eval.log
