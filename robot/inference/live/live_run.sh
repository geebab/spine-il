#!/usr/bin/env bash
# One closed-loop policy run on the two arms (inference/live_loop.py): plan the ready pose → seed recording → robot start
# cfg → queue the run (ro_queue2 → eval/robot_origin_run.sh RO_LIVE: relay, recorder page, both arms on the policy's stream,
# 5-panel video) → the measured trajectory as a recording → cut vs the 8 demos (what the policy saw + the robot session).
#   live_run.sh NAME CKPT_DIR START_JSON ["LIVE_LOOP EXTRA ARGS"]        CKPT_DIR = …/pretrained_model (chunk-relative)
#   LIVE_ENV="RO_NO_VIDEO=1" : link tests — no 5-panel video, no evaluation
#   LIVE_VIDEO=later : the 5-panel video is not rendered now (it takes five times the run and keeps the render API from the
#   next run): robot_origin_run.sh leaves its command in <run>/video_cmd.sh, eval/video_later.sh renders the list. The
#   evaluation is made as always. For a series of runs.
#   words of the 4th argument that are not for live_loop.py: ARM_X=--opt=v;--opt=v (more endo_follow options for both arms,
#   e.g. ARM_X=--vis-dead-mm=0.8;--sag-dead-deg=0.1) and RO_ENV=NAME=V (as LIVE_ENV, e.g. RO_ENV=RO_NO_VIDEO=1)
# Limits (10-06, user: "드릴링 시간 제한 풀어"). Until then the drilling was cut at 40 s of policy time, 487 mm³ or 575 s of wall
# time, because the FE recorder closes a session at 600 s. Now the recorder's limit is raised for the run's own page
# (robot_origin_rec.py --rec-max-s) and the policy ends the drilling itself: no cut for LIVE_PAUSE_S with the trigger off
# (15 s; the 90 training demos pause for at most 11.2 s) or less than LIVE_STALL_MM3 removed in the last LIVE_STALL_S of drilling,
# trigger on or not (30 mm³ in 25 s; the least any of the 90 demos removes in 25 s of drilling is 17.9 mm³, 3 of them stay
# under 30 — eval/deep_1018/stall_basis.py). What is left
# are safety bounds: LIVE_DRILL_MAX_S of policy time (240 s — the longest demo drills for 218 s) and the wall time that
# follows from it (stop-and-go at 0.33x: about 13 s of wall time per second of policy time → LIVE_MAX_S, default 64 min).
# LIVE_DRILL_TARGET (mm³) brings a volume stop back; LIVE_DRILL_MAX_S=40 LIVE_MAX_S=900 the old short runs.
# What the policy looks at (10-06, user: "정책이 기록 페이지의 화면을 직접 보게 해"): LIVE_OBS=page (default) — the run's own
# recorder page, the simulation the run is recorded and judged in (robot_origin_rec.py --obs-port, page_obs.py); nothing is
# fed to a second viewer, so a chunk takes about 10 s of wall time instead of 17. LIVE_OBS=step — a render-API step session
# fed with the measured poses, as every run before (a second simulation: it removed 165–200 mm³ where the page recorded
# 265–327). The recorder page shows the water medium on this build (FE recordDefaults.ts), so a policy trained on air
# renders gets the step session whatever LIVE_OBS says.
# The demo the run stands on (10-07): the arms start at ONE recording's paired start pose, and the safety gate lets the tools
# go only where the arms can follow from there (IK within 10°) — eval/pair_reach.py shows which demos' drilling that is.
# Default: 3868 (09-28). A policy trained mostly on other recordings needs a pair among those (the 24-demo policy on the
# 3868 pair asked for the 10-06 demos' endoscope orientation and was held for 30 min):
#   LIVE_DS=<dataset with that recording> LIVE_EPISODE=<its episode> LIVE_ANCHOR_S=<the pair's window start, session clock>
#   LIVE_PAIR=<recording id> [LIVE_PAIRS_FILE=<pairstart/pair_clear_<rid>.json>]      START_JSON "-": the episode's frame
#   LIVE_START_FRAME (0) instead of a start file.   LIVE_REG=<pairstart/registration_<rid>.json> (eval/fit_registration.py):
#   LIVE_ANCHOR_STATE=<pairstart/base_<NAME>.json> (eval/sitting_base.py): a base that is not one recording — the gate is anchored at
#   the representative pose of a sitting; LIVE_EPISODE then only names the sitting, LIVE_REG / LIVE_PAIRS_FILE are that base's.
#   a recording made on another rig — the scene as this rig's camera sees it, fitted to the arms' tools.
#   LIVE_PORTAL=batch: the skin portals of the base episode's own sitting (default: the median of the whole dataset).
#   LIVE_READY_OUT_MM=15: the ready pose that far outside the skin (default 20). On a base anchored deep in the bone (C1006:
#   the drill tip 61 mm inside the skin) 20 mm puts the ready pose on the edge of the gate's 80 mm sphere — the preparation
#   then takes the last frame the gate passes, and the scripted return ends in a refused step (10-08, first run there).
#   LIVE_GUARD_SEP_MM=5: the run guard's tip-separation stop for this run (default 10 mm; user 10-07: 5 mm on the 10-06
#   sitting, where the demonstrations hold the two tips 17 mm apart and come within 10 mm).
# The arms run preset j at 0.33x with the target deadband off: live targets are anchored at the measured pose, and the
# shaper's 1.5 mm / 1° deadband (a held point trailing the target) would swallow that much of every chunk.
A=$HOME/il-robot/runs/act_smoke; LV=$A/live; PY=$HOME/il-robot/.venv/bin/python; W=$HOME/il-robot/runs/jitter_iter/work
L=$HOME/il-robot/runs/jitter_iter/robot_origin.log; D=$HOME/il-robot/datasets
NAME=$1; POL=$2; START=$3; ARMX=""; EXTRA=""
for w in $4; do case "$w" in ARM_X=*) ARMX="${w#ARM_X=}";; RO_ENV=*) LIVE_ENV="${LIVE_ENV:+$LIVE_ENV,}${w#RO_ENV=}";; *) EXTRA="${EXTRA:+$EXTRA }$w";; esac; done
# the endoscope medium the policy was trained on (action_rep.json "optics"; none = the recordings' flag, air): live_loop.py
# opens its viewer session with it by itself; the 5-panel video and the evaluation renders get it from here
OPT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('optics') or '')" $POL/action_rep.json 2>/dev/null)
# A policy on another machine (user 10-09, inference/remote_policy.py): the 2nd argument is its URL instead of a checkpoint
# folder — http(s)://host:port, e.g. a GR00T model on a cloud GPU behind an SSH tunnel. LIVE_POLICY_TOKEN_FILE = the file
# with its token (chmod 600; never the token itself), LIVE_OPTICS = the medium its images were rendered in (default water:
# the shared GR00T datasets are water renders, and the recorder page the policy looks at shows water).
POLARG="--policy-path $POL"
case "$POL" in http://*|https://*)
  OPT=${LIVE_OPTICS:-water}
  POLARG="--policy-url $POL${LIVE_POLICY_TOKEN_FILE:+ --policy-token-file $LIVE_POLICY_TOKEN_FILE}${LIVE_POLICY_CA_FILE:+ --policy-ca-file $LIVE_POLICY_CA_FILE} --optics $OPT";;
esac
OBS=${LIVE_OBS:-page}; [ "$OBS" = page ] && [ "$OPT" != water ] && OBS="step (the recorder page shows water, this policy saw ${OPT:-air} renders)"
O=$LV/$NAME; mkdir -p $O; rm -f $O/DONE; echo "$(date +%T) start: $POL | $START | $EXTRA${ARMX:+ | arms $ARMX}${LIVE_ENV:+ | $LIVE_ENV}${OPT:+ | optics $OPT} | observation $OBS" > $O/log
fail() { echo "$1" > $O/DONE; echo "$(date +%T) FAILED: $1" >> $O/log; exit 1; }
# 10-08 16:05: the root filesystem ran full during a run — the arm followers died writing their logs (the run aborted with
# "the endo arm is not following"), the next two runs died at their start, and the run script's end step could not park the
# right arm (three hours in HOLD with the torque on). A run writes 0.4-0.6 GB (camera frames, follower logs, session):
# nothing is started with less than LIVE_MIN_FREE_GB free.
FREE_GB=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
[ "${FREE_GB:-0}" -ge ${LIVE_MIN_FREE_GB:-6} ] || fail "disk nearly full (${FREE_GB:-?} GB free, ${LIVE_MIN_FREE_GB:-6} needed) — not started"
busy() { ! (cd $W && $PY render_idle.py) || pgrep -f "[r]o_iou\.py|[p]olicy_eval\.py|[r]ender_video\.py|[e]ndo_follow\.py|[r]obot_origin_run\.sh" >/dev/null || [ -n "$(grep -v '^#' $W/ro_jobs.txt 2>/dev/null | grep -m1 .)" ] || { $PY ${SPINE_IL_REPO:-$HOME/spine-il}/robot/eval/render_gate.py status >/dev/null 2>&1; [ $? = 1 ]; }; }
idle() { while :; do busy && { sleep 15; continue; }; sleep 20; busy || break; done; }
COMMON="--dataset ${LIVE_DS:-$D/spine_rev4_step_delta_ph_trim} --episode ${LIVE_EPISODE:-6} --start-frame ${LIVE_START_FRAME:-0} --gate-anchor-s ${LIVE_ANCHOR_S:-17} --gate-pair ${LIVE_PAIR:-3868} ${LIVE_PAIRS_FILE:+--gate-pairs-file $LIVE_PAIRS_FILE} ${LIVE_REG:+--gate-registration $LIVE_REG} ${LIVE_ANCHOR_STATE:+--gate-anchor-state $LIVE_ANCHOR_STATE} ${LIVE_PORTAL:+--portal-from $LIVE_PORTAL} ${LIVE_READY_OUT_MM:+--ready-out-mm $LIVE_READY_OUT_MM} --cycle"
[ "$START" = "-" ] || COMMON="$COMMON --start-state $START"
cd ${SPINE_IL_REPO:-$HOME/spine-il}/robot
OMP_NUM_THREADS=2 taskset -c 8-11 $PY -u inference/live_loop.py --prepare-out $O/seed.json $COMMON > $O/prepare.log 2>&1 || fail "prepare failed"
$PY inference/policy2rec.py --traj $O/seed.json > $O/p2r.json 2>&1 || fail "policy2rec (seed) failed"
REC=$(python3 -c "import json;print(json.load(open('$O/p2r.json'))['recording'])"); RID=$(python3 -c "import json;print(json.load(open('$O/p2r.json'))['rid'])")
(cd $W && $PY -u new_recording.py $RID --from-s 0 --len-s 1 --reps 1 --start-from-traj $O/seed.json --radius-mm ${LIVE_RADIUS_MM:-130} --tag v --no-iou --no-queue > $O/new_recording.log 2>&1)
grep -q "SKIPPED" $O/new_recording.log && fail "robot start cfg failed"
# 10-15 (user: "내시경 및 드릴 반경은 최대치로 변경해"): follower sphere and guard travel 130 mm for both tools (guard_pairs.py
# MAX_RUN_LIMIT_MM; was 120 and 95:115) — with the gate sphere and the loop's travel limit at 130 too, the safe-zone checks bind
DMAX=${LIVE_DRILL_MAX_S:-240}; WALL=${LIVE_MAX_S:-$(( DMAX * $([ "$OBS" = page ] && echo 10 || echo 14) + 480 ))}
LIMITS="--phase-rules --drill-max-s $DMAX --drill-pause-s ${LIVE_PAUSE_S:-15} --drill-stall-s ${LIVE_STALL_S:-25} --drill-stall-mm3 ${LIVE_STALL_MM3:-30} ${LIVE_DRILL_TARGET:+--drill-target-mm3 $LIVE_DRILL_TARGET} --steps $(( (DMAX + 20) * 15 )) --deadline-s 0"
printf '%s\n' "$POLARG" "$COMMON" "$LIMITS" "--frames-dir $O/frames" "$EXTRA" > $O/live.args
DB="--deadband-mm=0;--deadband-deg=0${ARMX:+;$ARMX}"
# the camera's 32-bit µs stamp counter wraps every 71.6 min and the tracker's stamps may then drift for a minute (the arms hold):
# a run that would cross a wrap starts 90 s after it (live/wrap_guard.py, stored fit only) — a run of up to WALL seconds
# (over an hour at the default) therefore nearly always waits for the next wrap and starts right behind it
$PY inference/live/wrap_guard.py --wait $(( WALL + 180 )) >> $O/log 2>&1
echo "pr_${RID}v1 $(basename $REC) 0 $WALL pj_$RID drill RO_RADIUS_MM=${LIVE_RADIUS_MM:-130},RO_GUARD_MM=${LIVE_GUARD_MM:-130:130}${LIVE_GUARD_SEP_MM:+,RO_GUARD_SEP_MM=$LIVE_GUARD_SEP_MM},RO_LEFT_X=$DB,RO_RIGHT_X=$DB,RO_LIVE=$O/live.args,RO_REC_MAX_S=$(( WALL + 300 ))$([ "$OBS" = page ] && echo ",RO_OBS_PORT=${LIVE_OBS_PORT:-18790}")${OPT:+,RO_VIDEO_OPTICS=$OPT}$([ "$LIVE_VIDEO" = later ] && echo ",RO_VIDEO_LATER=1")${LIVE_ENV:+,$LIVE_ENV}" >> $W/ro_jobs.txt
echo "$(date +%T) seed $RID queued (pr_${RID}v1)" >> $O/log
# the queue's "done" line can be lost (10-08 16:11, disk full: the run folder had its DONE, the line was never written and
# this loop waited for three hours) — a run whose folder has DONE and whose run script is gone is over as well
until grep -qE "pr_${RID}v1 done" $L; do
  RUNX=$(ls -td $HOME/il-robot/runs/robot_origin/*_replay_pr_${RID}v1 2>/dev/null | head -1)
  if [ -n "$RUNX" ] && [ -e $RUNX/DONE ] && [ $(( $(date +%s) - $(stat -c %Y $RUNX/DONE) )) -gt 180 ] && ! pgrep -f "[r]obot_origin_run\.sh pr_${RID}v1 " >/dev/null; then
    echo "$(date +%T) the run's folder has DONE but the log has no \"done\" line — taken as over" >> $O/log; break
  fi
  sleep 15
done
RUN=$(ls -td $HOME/il-robot/runs/robot_origin/*_replay_pr_${RID}v1 2>/dev/null | head -1); echo "$RUN" > $O/run_dir
cp $RUN/live.json $RUN/live.log $O/ 2>/dev/null
echo "$(date +%T) run $(basename $RUN): $(grep -E "pr_${RID}v1 done" $L | tail -1 | cut -c1-200)" >> $O/log
[ -s $O/live.json ] || fail "no live.json (the live loop did not start?)"
SID=$(tail -1 $RUN/result.json | python3 -c "import json,sys;print(json.load(sys.stdin).get('sessionId') or '')" 2>/dev/null)
SF=$(find $HOME/il-robot/viewer-data/sessions/robot -name "$SID.jsonl" 2>/dev/null | head -1)
echo "$SF" > $O/robot_session; echo "$(date +%T) robot session $SID" >> $O/log
[ -n "$SF" ] || fail "no robot session"
case "$LIVE_ENV" in *RO_NO_VIDEO*) echo "$RUN" > $O/DONE; echo "$(date +%T) done (no video)" >> $O/log; exit 0;; esac
if [ "$LIVE_VIDEO" = later ]; then echo "$(date +%T) 5-panel video left for later ($RUN/video_cmd.sh)" >> $O/log
else
until [ -e $RUN/video.log ] && ! pgrep -f "[r]ender_video.py.*-o $RUN" >/dev/null && [ -s $RUN/session_5panel.mp4 ]; do sleep 15; done
echo "$(date +%T) 5-panel video done" >> $O/log
fi
[ "$LIVE_EVAL" = 0 ] && { echo "$RUN" > $O/DONE; exit 0; }
$PY inference/policy2rec.py --traj $O/live.json > $O/p2r_meas.json 2>&1 || fail "policy2rec (measured) failed"
RECM=$(python3 -c "import json;print(json.load(open('$O/p2r_meas.json'))['recording'])")
idle
# the robot session's cut is replayed from the thinned copy when the run script made one (a long session does not load in the
# render service; the copy has every cut row and the same file name, so the result and its cache key are the session's)
SFE=$RUN/thin/$(basename "$SF"); [ -s "$SFE" ] || SFE=$SF
(cd $W && OPENBLAS_NUM_THREADS=2 nice $PY -u policy_eval.py $RECM --robot $SFE --label live_$NAME ${OPT:+--optics $OPT} > $O/eval.log 2>&1) || fail "policy_eval failed"
echo "$RUN" > $O/DONE; echo "$(date +%T) done" >> $O/log
