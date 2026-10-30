#!/usr/bin/env bash
# deep_1018 L3 / L5 series: the evaluations rendered on the Mac node (user 10-19: "레노보와 vm은 로봇을 돌리고 평가는 맥세션에서").
# One worker per Mac service port (default 18785 18786): each takes the next finished run of chain_lv_1019.sh (lv_runs_done.txt)
# that has no evaluation yet (a lock dir per run), checks its port's /v1/health, then runs eval_later.sh's own steps — policy2rec of
# the measured poses, policy_eval.py --robot (POLICY_EVAL_ROBOT_ONLY=1) — with RENDER_API_URL=<the port> RENDER_API_INLINE=1, so the
# render runs on the Mac and the Lenovo only sends the session and counts the result (nice 19, one thread each: the tracker is
# not starved while the robot runs). A port that does not answer is retried every 60 s; nothing falls back to the Lenovo's own
# renderer while the robot series runs. Ends when the chain has ended and every finished run is evaluated.
# Start only after ab_mac_replay.py passed on that port.   setsid nohup bash eval_mac_1019.sh [PORT ...] >> eval_mac_1019.log 2>&1 < /dev/null &
set -u
D=$HOME/il-robot/runs/deep_1018; LV=$HOME/il-robot/runs/act_smoke/live; W=$HOME/il-robot/runs/jitter_iter/work
R=${SPINE_IL_REPO:-$HOME/spine-il}/robot; PY=$HOME/il-robot/.venv/bin/python; C=$HOME/il-robot/runs/jitter_iter
PORTS=${*:-18785 18786}; mkdir -p $D/eval_locks
say() { echo "$(date '+%m-%d %T') $*"; }
evaluated() { local sid; sid=$(basename "$(cat $LV/$1/robot_session 2>/dev/null)" .jsonl); [ -n "$sid" ] && [ -s $C/iou_cache_rofull_$sid.npz ]; }
worker() {
  local P=$1 URL=http://127.0.0.1:$1 NAME O RUN SF SFE RECM
  while :; do
    NAME=""
    for c in $(cat $D/lv_runs_done.txt 2>/dev/null); do evaluated $c && continue; mkdir $D/eval_locks/$c 2>/dev/null && { NAME=$c; break; }; done
    if [ -z "$NAME" ]; then
      awk '/^---- /{buf=""} {buf=buf $0 "\n"} END{printf "%s", buf}' $D/chain_lv_1019.log 2>/dev/null | grep -qE " end$|stopped|no runs made" && { say "port $P: nothing left — worker done"; return; }
      sleep 60; continue
    fi
    until [ "$(curl -s -m 5 -o /dev/null -w "%{http_code}" $URL/v1/health)" != 000 ]; do say "port $P: Mac node not answering — waiting"; sleep 60; done
    O=$LV/$NAME; RUN=$(cat $O/run_dir); SF=$(cat $O/robot_session); SFE=$RUN/thin/$(basename "$SF"); [ -s "$SFE" ] || SFE=$SF
    say "port $P: $NAME evaluation"; echo "$(date +%T) evaluation (Mac :$P)" >> $O/log
    if ! (cd $R && nice -n 19 $PY inference/policy2rec.py --traj $O/live.json > $O/p2r_meas.json 2>&1); then
      echo "$(date +%T) FAILED: policy2rec (measured)" >> $O/log; say "port $P: $NAME policy2rec failed"; continue; fi
    RECM=$(python3 -c "import json;print(json.load(open('$O/p2r_meas.json'))['recording'])")
    if (cd $W && env POLICY_EVAL_ROBOT_ONLY=1 RENDER_API_URL=$URL RENDER_API_INLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 \
          $PY -u policy_eval.py $RECM --robot $SFE --label live_$NAME --optics water > $O/eval.log 2>&1); then
      echo "$(date +%T) evaluated (Mac :$P)" >> $O/log; say "port $P: $NAME evaluated"
    else echo "$(date +%T) FAILED: policy_eval (Mac :$P)" >> $O/log; say "port $P: $NAME policy_eval failed — lock kept, see $O/eval.log"; fi
  done
}
say "start, ports $PORTS"
for P in $PORTS; do worker $P & done
wait; say "all workers done"; say end
