#!/usr/bin/env bash
# The closed loop through the remote path with neither a robot nor a model: the reference server's demo backend
# (remote/policy_server.py --backend demo --progress-output) answers every request with a demonstration's own
# continuation and its progress, and dry_step.sh runs live_loop.py on it — simulated arms, the render API's step session.
# What it shows before a cloud server and the robot are spent on it: the loop asks and converts as the protocol says, the
# predicted progress ends the drilling at the level in use ("drill_end policy"), and a server that stops answering ends
# the run in an abort with the tools taken out.
#   TAG=name [EP=6] [KILL_AFTER=N] [END_LEVEL=x] bash dry_remote.sh
#   EP          the demonstration the server continues (default 6 = recording 3868, the base A0928 stands on: 40 s of drilling)
#   KILL_AFTER  stop the server after its N-th answer — the abort path
#   URL         a REAL policy server instead of the local demo backend (http(s)://host:port; TOK = its token file, default
#               ~/.config/ube/policy.token, TOK= for a server without one; POL_CA_FILE = its own certificate): the one dry
#               run a new model path gets before the robot (live/remote_intake.sh DRY=1 starts it). Same base, same start,
#               same loop options as the demo-backend run of 10-09 (dry_remote_1009b: 41 requests, 404 mm³, ended at 0.91).
# Start it through the gate, it cannot wait for the robot:
#   python eval/render_gate.py run --on-pause kill --low --mem-gb 3 -- bash inference/live/dry_remote.sh
A=$HOME/il-robot/runs/act_smoke; PY=$HOME/il-robot/.venv/bin/python; R=${SPINE_IL_REPO:-$HOME/spine-il}/robot
P=$HOME/il-robot/runs/jitter_iter/work/pairstart; D=$HOME/il-robot/datasets
TAG=${TAG:-dry_remote_$(date +%H%M%S)}; EP=${EP:-6}; PORT=${PORT:-8911}; URL=${URL:-}
[ -e $A/dry_step/$TAG ] && { echo "$A/dry_step/$TAG exists — another TAG"; exit 2; }
cd $R || exit 2
SP=; SL=
if [ -n "$URL" ]; then
  case "$URL" in http://*|https://*) ;; *) echo "URL must be http(s)://host:port"; exit 2;; esac
  TOK=${TOK-$HOME/.config/ube/policy.token}     # TOK= (set, empty): a server without a token
  [ -z "$TOK" ] || [ -s "$TOK" ] || { echo "no token file $TOK"; exit 2; }
  POLURL=$URL
else
  TOK=${TOK:-$HOME/.config/ube/policy_local_test.token}
  [ -s "$TOK" ] || { echo "no token file $TOK (any private file with one line: chmod 600)"; exit 2; }
  SL=$A/dry_step/${TAG}.server.log
  nice taskset -c ${CORES:-12-15} $PY -u inference/remote/policy_server.py --backend demo --dataset $D/spine_hand24_replay_trim_water_progress \
    --episodes $EP --progress-output --token-file $TOK --port $PORT > $SL 2>&1 &
  SP=$!
  trap 'kill $SP 2>/dev/null' EXIT
  for i in $(seq 1 60); do ss -ltn | grep -q ":$PORT " && break; kill -0 $SP 2>/dev/null || { echo "the server did not start: $(tail -n 2 $SL)"; exit 3; }; sleep 0.5; done
  if [ -n "$KILL_AFTER" ]; then
    ( until [ "$(grep -c '/act seq' $SL)" -ge "$KILL_AFTER" ] || ! kill -0 $SP 2>/dev/null; do sleep 1; done
      kill $SP 2>/dev/null && echo "[dry_remote] $(date +%T) server stopped after answer $KILL_AFTER" >> $SL ) &
  fi
  POLURL=http://127.0.0.1:$PORT
fi
TAG=$TAG POL=$POLURL POL_TOKEN_FILE=$TOK POL_CA_FILE=${POL_CA_FILE:-} NEXEC=${NEXEC:-16} STEPS=${STEPS:-900} DMAX=${DMAX:-120} CORES=${CORES:-12-15} FAKE_X="--r0 model" \
  LOOP_X="--gate-turn-wait --no-aim${END_LEVEL:+ --end-level $END_LEVEL}" \
  COMMON="--dataset $D/spine_hand24_replay_ph_trim_water --episode 6 --start-frame 0 --gate-anchor-s 17 --gate-pair A0928 --gate-pairs-file $P/pair_clear_3b3b.json --gate-registration $P/registration_A0928.json --gate-anchor-state $P/base_A0928.json --portal-from run --start-state $A/iter/starts/start_A0928rand_0.json" \
  bash inference/live/dry_step.sh
RC=$?
[ -n "$SP" ] && { kill $SP 2>/dev/null; mv $SL $A/dry_step/$TAG/server.log 2>/dev/null; }
$PY - $A/dry_step/$TAG/live.json <<'PYEOF'
import json, sys
try:
    j = json.load(open(sys.argv[1]))
except Exception as e:
    print("no live.json:", e); sys.exit(1)
m = j.get("meta", j)
rem = m.get("removed_mm3")
keep = {k: m.get(k) for k in ("drill_end", "drill_s", "abort", "end_output", "end_level", "policy_done", "policy", "remote_policy", "wall_s") if k in m}
keep["removed_mm3"] = rem[-1] if isinstance(rem, list) and rem else rem
print(json.dumps(keep, ensure_ascii=False)[:1200])
PYEOF
exit $RC
