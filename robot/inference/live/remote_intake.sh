#!/usr/bin/env bash
# From "here is the address" to "ready for the robot runs" — nothing here moves an arm.
#   bash inference/live/remote_intake.sh [NAME | URL]   NAME = an endpoint of remote/endpoint.py (nothing: the default one)
#   env: TAG       the model's name in run names and folders (default: NAME, or "groot")
#        DRY=1     also the one dry run of this model path (15–20 min; user 10-07: a new path gets one dry run, not more)
#        STEPS     "1 2 3" by default; e.g. STEPS=3 DRY=1 OUT=<folder of the first pass> for the dry run alone
#        EPISODES  2,18 = the demonstrations no model was trained on · DS = the dataset with the progress column
# 1 remote/connect_check.py    address · reach · auth · model card · real answers · timing          ~1 min, ~15 requests
# 2 end_level.py --policy-url  the model on the held-out demonstrations: open-loop scores, progress error, end level
#                                                                                                 ~5 min, ~190 requests
# 3 live/dry_remote.sh URL=…   the closed loop on simulated arms through the real server (only with DRY=1)
# Each step runs through the render gate at low priority and is ENDED, not paused, if somebody takes the robot — a
# request cut in half is worth nothing; run the intake again then. A step that fails stops the chain.
# Output: ~/il-robot/runs/groot_prep/intake_<TAG>_<MMDD_HHMM>/ — check.json, model_eval.json, dry.json and SUMMARY.txt:
# what was found, the end level for the robot runs, and the robot-run commands (printed — this script starts no run).
# The server has to be up for all of it: about 10 min without the dry run, 30 min with it.
set -u
R=${SPINE_IL_REPO:-$HOME/spine-il}/robot; PY=$HOME/il-robot/.venv/bin/python; G="$PY $R/eval/render_gate.py"
ARG=${1:-}
cd $R || exit 2
EV=$($PY inference/remote/endpoint.py env ${ARG:+"$ARG"}) || exit 2
eval "$EV"
case "$ARG" in http://*|https://*|"") DEF=groot;; *) DEF=$ARG;; esac
TAG=${TAG:-$DEF}; DS=${DS:-$HOME/il-robot/datasets/spine_hand24_replay_trim_water_progress}; EPISODES=${EPISODES:-2,18}; STEPS=${STEPS:-1 2 3}
case "$TAG" in *[!A-Za-z0-9_-]*) echo "TAG: letters, digits, _ and - only"; exit 2;; esac
O=${OUT:-$HOME/il-robot/runs/groot_prep/intake_${TAG}_$(date +%m%d_%H%M)}; mkdir -p "$O" || exit 2
[ -n "$EP_CA_FILE" ] && export SPINE_POLICY_CA_FILE=$EP_CA_FILE
say() { echo "$(date '+%m-%d %T') $*" | tee -a "$O/intake.log"; }
summary() { $PY inference/remote/intake_summary.py "$O" --tag "$TAG" --url "$EP_URL" --token-file "$EP_TOKEN_FILE" --ca-file "$EP_CA_FILE"; }
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
why() { case $1 in 1) echo "something to fix";; 2) echo "not reachable";; 3) echo "another API";; 75) echo "the robot was taken — run the intake again later";; *) echo "exit $1";; esac; }
say "intake: endpoint $EP_NAME ($EP_URL), tag $TAG, steps $STEPS${DRY:+, with the dry run} → $O"

if has 1; then
  $G run --on-pause kill --low --mem-gb 2 --name intake_check -- $PY inference/remote/connect_check.py ${ARG:+"$ARG"} \
    --dataset "$DS" --episodes "$EPISODES" -o "$O/check.json" 2>&1 | tee -a "$O/intake.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { say "step 1 (connection) stopped the intake: $(why $rc)"; summary; exit "$rc"; }
fi

if has 2; then
  say "step 2: the model on the held-out demonstrations $EPISODES (a question every 16 frames)"
  $G run --on-pause kill --low --mem-gb 2 --name intake_eval -- $PY inference/end_level.py --dataset "$DS" --policy-url "$EP_URL" \
    ${EP_TOKEN_FILE:+--policy-token-file "$EP_TOKEN_FILE"} --episodes "$EPISODES" -o "$O/model_eval.json" 2>&1 | tee -a "$O/intake.log"
  rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || { say "step 2 (model evaluation) stopped the intake: $(why $rc)"; summary; exit "$rc"; }
fi
summary > /dev/null      # SUMMARY.json: the end level the dry run and the robot runs use

if has 3 && [ "${DRY:-0}" = 1 ]; then
  LV=$($PY -c "import json,sys; d=json.load(open(sys.argv[1])); print((d.get('end_level') or {}).get('level') or '')" "$O/SUMMARY.json" 2>/dev/null)
  DTAG=dry_${TAG}_$(date +%m%d_%H%M)
  say "step 3: one dry run through the server ($DTAG${LV:+, end level $LV}) — 15 to 20 min"
  # the dry run needs the render service to itself and waits for it: say so, the server's time is paid for
  (cd $HOME/il-robot/runs/jitter_iter/work && $PY render_idle.py) > /dev/null 2>&1 \
    || say "the render service is busy with other work ($($G status 2>/dev/null | head -n 1)) — the dry run starts when it is idle; stop that work first if the server is waiting"
  $G run --on-pause kill --low --mem-gb 2.5 --name intake_dry -- env TAG="$DTAG" URL="$EP_URL" TOK="$EP_TOKEN_FILE" POL_CA_FILE="$EP_CA_FILE" ${LV:+END_LEVEL=$LV} \
    bash inference/live/dry_remote.sh > "$O/dry.log" 2>&1
  rc=$?
  grep '^{' "$O/dry.log" | tail -n 1 > "$O/dry.json"; [ -s "$O/dry.json" ] || rm -f "$O/dry.json"
  echo "$HOME/il-robot/runs/act_smoke/dry_step/$DTAG" > "$O/dry_dir.txt"
  [ "$rc" -eq 0 ] || say "step 3 (dry run) ended with $(why $rc) — see $O/dry.log and $(cat "$O/dry_dir.txt")/live.log"
fi
summary | tee -a "$O/intake.log"
say "intake done → $O/SUMMARY.txt"
