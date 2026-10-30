#!/usr/bin/env bash
# One closed-loop robot run of a REMOTE policy (a GR00T model on a cloud GPU, inference/remote_policy.py) on a sitting's
# base — the harness of the ACT final test of 10-08, only the question "what next?" goes to the server:
#   no endoscope aim correction, the run's own pivot points (--portal-from run), the trigger blocked only while the drill
#   tip is off the screen, turn_wait on, drilling bound 240 s, run guard's tip separation 5 mm, random start r0 … r4.
#   bash run_remote_base.sh BASE URL TAG START [DRILL_MAX_S=240]
#   BASE   C1006 = the 10-06 sitting (pair of the synthetic recording 2ca6) · A0928 = the 09-28 sitting (3b3b)
#   URL    http(s)://host:port of the policy server — through an SSH tunnel: http://127.0.0.1:18900
#   TAG    a name for the model, e.g. GR00T_gen_s1 or GR00TH_med_s1 (run name: R_<TAG>_<BASE>_<START>)
#   START  r0 … r4 = iter/starts/start_<BASE>rand_<k>.json (the starts of the ACT final test: the same five)
# Environment: LIVE_POLICY_TOKEN_FILE (default ~/.config/ube/policy.token, chmod 600), LIVE_POLICY_CA_FILE (https with the
#   server's own certificate), LIVE_OPTICS (default water),
#   LIVE_NEXEC = executed steps of each answer (default 16 = a GR00T answer's 16 rows; 8 = ask twice as often),
#   LIVE_VIDEO=now | later (default), LIVE_EVAL=0 (evaluate later: live/eval_later.sh NAME …), LIVE_EXTRA,
#   LIVE_CONTINUOUS=1 = the policy part without stops (inference/CONTINUOUS.md: looks while the arms move, windows of
#   LIVE_STREAM_PERIOD (8) frames starting LIVE_STREAM_LEAD (16) frames after their look; the recorder page's screencast is
#   switched on for it). LIVE_STREAM_ANCHOR=command (default; measured = the first continuous run of 10-12, which swung —
#   only to reproduce it) with LIVE_STREAM_GAMMA (0.5); LIVE_STREAM_BLEND (4) is the fade of the measured anchor.
#   The lead must cover look → targets ready: 2.5 s median with the VM in eu-north1 (16 frames = 3.2 s; 10-12).
#   Give such a run its own TAG (e.g. GR00T_gen_s1_cont).
#   LIVE_GOAL = LEVEL[:EXTENT] — the goal said to a model trained with the lang1 instructions (L3|L4|L5 : ipsilateral|contralateral)
#   LIVE_END_LEVEL = the level above which the model's progress ends the drilling (default rollout.END_LEVEL: 0.91, computed
#   without a model — compute it on this model first: inference/end_level.py --policy-url URL --episodes 2,18).
# Before the first run of a server: live/remote_intake.sh (connection check, the model on the held-out demonstrations with
# its end level, one dry run — a new path gets one dry run, not more: user 10-07); its SUMMARY.txt ends with this command.
# Before every run the server is asked twice here (a grey image): the first answer may be slow (a model that has to wake
# up), the second must come inside the loop's 6 s — a run is not started on a server that would time out in it.
# A run whose server stops answering is aborted by the loop (the tools are withdrawn by the scripted exit).
LV=$HOME/il-robot/runs/act_smoke/live; A=$HOME/il-robot/runs/act_smoke; D=$HOME/il-robot/datasets
PSD=$HOME/il-robot/runs/jitter_iter/work/pairstart; PY=$HOME/il-robot/.venv/bin/python
BASE=${1:?base: C1006 | A0928}; URL=${2:?policy server url}; TAG=${3:?model tag}; ST=${4:?start: r0 … r4}; DMAX=${5:-240}
case "$BASE" in
  C1006) RID=2ca6; EP=18; AS=65.4; SEP=5; READY=15;;   # ready pose 15 mm outside the skin: 20 is on the edge of the gate's sphere here
  A0928) RID=3b3b; EP=6; AS=17; SEP=5; READY=;;
  *) echo "unknown base $BASE" >&2; exit 2;;
esac
case "$URL" in http://*|https://*) ;; *) echo "URL must be http(s)://host:port" >&2; exit 2;; esac
TOK=${LIVE_POLICY_TOKEN_FILE:-$HOME/.config/ube/policy.token}
log() { echo "$(date '+%m-%d %T') $*" >> $LV/page_test.log; }
START=$A/iter/starts/start_${BASE}rand_${ST#r}.json; N=R_${TAG}_${BASE}_$ST
[ -s $PSD/pair_clear_$RID.json ] && [ -s $PSD/base_$BASE.json ] && [ -s $PSD/registration_$BASE.json ] || { log "$N: base $BASE is not complete (pair / anchor / registration) — not started"; exit 0; }
[ -s $START ] || { log "$N: no start file $START — not started"; exit 0; }
[ -e $LV/$N ] && { log "$LV/$N exists — not started"; exit 0; }
[ -s $TOK ] || { log "$N: no token file $TOK — not started"; exit 0; }
# the server must be there and be what this run expects before anything moves (its model card: /info)
INFO=$(cd ${SPINE_IL_REPO:-$HOME/spine-il}/robot && $PY - "$URL" "$TOK" "${LIVE_POLICY_CA_FILE:-}" <<'PYEOF' 2>&1
import sys, time
sys.path.insert(0, "inference")
import numpy as np
import remote_policy as RP
try:
    rp = RP.RemotePolicy(sys.argv[1], RP.read_token(sys.argv[2]), 60, 0, run="warm-up", ca_file=sys.argv[3] or None)
    i = rp.info()
    st = np.asarray(RP.CHECK_STATE, float); st[3:7] /= np.linalg.norm(st[3:7]); st[10:14] /= np.linalg.norm(st[10:14])
    img = np.full((int(i["image"]["size"]), int(i["image"]["size"]), 3), 127, np.uint8); t = []
    for _ in range(2):
        t0 = time.monotonic(); rp.act(img, st, rp.task(4)); t.append(time.monotonic() - t0)
    if t[1] >= 6.0:
        print(f"FAILED the server needs {t[1]:.1f} s an answer (first {t[0]:.1f} s) — over the loop's 6 s")
    else:
        print(f"ok {i.get('model')} horizon {i['action_horizon']} done_output {i['done_output']} image {i.get('image')} answers in {t[1]:.2f} s (first {t[0]:.2f} s)")
except Exception as e:
    print(f"FAILED {e}")
PYEOF
)
case "$INFO" in ok*) ;; *) log "$N: policy server $URL: $INFO — not started"; exit 0;; esac
VID=${LIVE_VIDEO:-later}
CONT=""; [ "${LIVE_CONTINUOUS:-0}" = 1 ] && CONT=" --continuous --stream-period ${LIVE_STREAM_PERIOD:-8} --stream-lead ${LIVE_STREAM_LEAD:-16} --stream-blend ${LIVE_STREAM_BLEND:-4} --stream-anchor ${LIVE_STREAM_ANCHOR:-command} --stream-gamma ${LIVE_STREAM_GAMMA:-0.5} RO_ENV=RO_OBS_STREAM=1"
log "start: $N (remote policy $URL: $INFO; base $BASE = the representative pose of its sitting, pair of $RID, start $ST, ${LIVE_NEXEC:-16} steps per answer, turn_wait, guard sep ${SEP} mm, LIVE_DRILL_MAX_S=$DMAX, video $VID, aim off, pivot run${CONT:+, CONTINUOUS:$CONT}${LIVE_EXTRA:+, extra: $LIVE_EXTRA})"
env LIVE_DS=$D/spine_hand24_replay_ph_trim_water LIVE_EPISODE=$EP LIVE_ANCHOR_S=$AS LIVE_ANCHOR_STATE=$PSD/base_$BASE.json LIVE_PAIR=$BASE LIVE_PAIRS_FILE=$PSD/pair_clear_$RID.json \
  LIVE_REG=$PSD/registration_$BASE.json LIVE_PORTAL=run LIVE_GUARD_SEP_MM=$SEP ${READY:+LIVE_READY_OUT_MM=$READY} LIVE_DRILL_MAX_S=$DMAX LIVE_VIDEO=$VID \
  LIVE_POLICY_TOKEN_FILE=$TOK ${LIVE_POLICY_CA_FILE:+LIVE_POLICY_CA_FILE=$LIVE_POLICY_CA_FILE} LIVE_OPTICS=${LIVE_OPTICS:-water} \
  bash $LV/live_run.sh $N "$URL" $START "--n-action-steps ${LIVE_NEXEC:-16} --gate-turn-wait --no-aim${LIVE_END_LEVEL:+ --end-level $LIVE_END_LEVEL}${LIVE_GOAL:+ --goal $LIVE_GOAL}$CONT${LIVE_EXTRA:+ $LIVE_EXTRA}" > $LV/$N.out 2>&1
log "live_run rc $?: $(cat $LV/$N/DONE 2>/dev/null | cut -c1-160)"
