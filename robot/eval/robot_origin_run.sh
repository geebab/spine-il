#!/usr/bin/env bash
# One two-arm replay recorded as an FE robot-origin session (ROBOT_ORIGIN.md): preset d + the recording's paired start,
# a robot relay on :18767 for this run, the headless recorder page (robot_origin_rec.py), trigger from the drill arm.
#   robot_origin_run.sh TAG REC_FILE FROM_S LEN_S CFG [ANCHOR]      e.g.  pr_3868001 s_..._3868.jsonl 17 30 pd_3868
# RO_LIVE=<file with live_loop.py arguments>: closed-loop policy run (robot/inference/live_loop.py). REC_FILE is then the
#   seed recording (the ready pose), the arms take their targets from the policy as it runs (endo_follow
#   --replay-stream-udp 9891 / 9892), the trigger comes from the live loop, LEN_S is the longest run in wall seconds.
# RO_OBS_STREAM=1 (with RO_OBS_PORT): the page's screencast runs too, so the continuous loop can look without a screenshot
#   while the arms move (robot_origin_rec.py --obs-stream; live_loop.py --continuous; inference/CONTINUOUS.md).
# RO_OBS_PORT=<port> (with RO_LIVE): the policy looks at this run's recorder page — the page serves its endoscope view and
#   bridge metrics there (robot_origin_rec.py --obs-port, page_obs.py) and the live loop reads them (--obs-url) instead of
#   opening a render-API step session of its own.
S=$HOME/il-robot/runs/jitter_iter/work
PY=$HOME/il-robot/.venv/bin/python
RD=$HOME/il-robot/worktrees/joycon-relay/_workspace/ube-tracking
M=$HOME/il-robot/calib/tracker_models
SESS=$HOME/il-robot/viewer-data/sessions/robot
LOG=${RUNLOG:-$HOME/il-robot/runs/jitter_iter/robot_origin.log}
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
TAG=$1; F=$2; FROM=$3; LEN=$4; CFG=$5; ANCHOR=${6:-drill}
FP=$(find $HOME/il-robot/viewer-data/sessions -name "$F" | head -1); [ -z "$FP" ] && { echo "[ro] no recording $F"; exit 2; }
SPEED=""; LEFT_EXTRA=""; RIGHT_EXTRA=""; source $S/cfg/$CFG.env || exit 2
SPEED=${RO_SPEED:-$SPEED}   # RO_SPEED / RO_LEFT_X / RO_RIGHT_X: per-job overrides from the queue (setting search)
LSTART=$(grep -o -- "--start=[^ ]*" <<<"$LEFT_EXTRA"); RSTART=$(grep -o -- "--start=[^ ]*" <<<"$RIGHT_EXTRA")
# the hand tool calibration the recording names in its header (hand_calib.py): the presets carry the 09-28 files, and the
# 10-06 demos' drill tip is 5.1 mm from those — a policy recording has the header of the demo it started from
LCAL=""; RCAL=""
if HC=$($PY ${SPINE_IL_REPO:-$HOME/spine-il}/robot/hand_calib.py "$FP" 2>/dev/null); then
  read LCAL RCAL <<<$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d['endo'], d['drill'])" "$HC")
else echo "[ro] $F: hand tool calibration not found — the presets' (09-28) files are used"; fi
SIZE=$(cd $S && $PY -c "import vol_eval as V; print(V.drill_size('$FP'))" 2>/dev/null)
STAMP=$(date +%Y%m%dT%H%M%S); RUN=${STAMP}_replay_${TAG}; SID=$(basename $F .jsonl)
OUTD=$HOME/il-robot/runs/robot_origin/$RUN; mkdir -p $OUTD $SESS
READY=$OUTD/READY; STARTED=$OUTD/STARTED; DONE=$OUTD/DONE; GO=$OUTD/GO
echo "[ro] $TAG: $F from $FROM s, $LEN s @${SPEED:-0.2}x, drill $SIZE, anchor $ANCHOR, run $RUN${RO_LEFT_X:+ · left+ $RO_LEFT_X}${RO_RIGHT_X:+ · right+ $RO_RIGHT_X}"
echo "{\"preset\": \"${RO_PRESET:-j}\", \"speed\": \"${SPEED:-0.2}\", \"left_x\": \"$RO_LEFT_X\", \"right_x\": \"$RO_RIGHT_X\", \"cfg\": \"$CFG\"}" > $OUTD/settings.json
# RO_GUARD_MM=ENDO:DRILL — guard_pairs.py travel limits for a run whose trajectory was verified over a longer travel
# RO_GUARD_SEP_MM=N — this run's tip-separation stop (guard_pairs.py: only lower than its 10 mm, floor 3 mm; user 10-07: 5 mm
# for runs on the 10-06 sitting, whose demonstrations work with the tips closer than 10 mm)
if [ -n "$RO_GUARD_MM$RO_GUARD_SEP_MM" ]; then
  python3 -c "
import json, sys
mm, sep = sys.argv[1], sys.argv[2]; g = {}
if mm: g['endo_mm'], g['drill_mm'] = (float(x) for x in mm.split(':'))
if sep: g['min_sep_mm'] = float(sep)
print(json.dumps(g))" "$RO_GUARD_MM" "$RO_GUARD_SEP_MM" > $OUTD/guard.json
fi
cd ${SPINE_IL_REPO:-$HOME/spine-il}/robot

cleanup() { for p in $DRV $FAN $RELAY $LIVE $WATCH; do [ -n "$p" ] && kill $p 2>/dev/null; done; [ -n "$TRAIN" ] && kill -CONT $TRAIN 2>/dev/null; }
trap cleanup EXIT
# CPU: a CPU training beside the run starves the tracker (ACT on cores 12-15, 10-01: 90 → 54 fps idle, 33-44 fps during a
# run; paused → 89) → trainers are paused for the run (resumed when the arms finish / on exit), the tracker keeps cores 0-11
TRAIN=$(pgrep -f "[l]erobot-train" | tr '\n' ' ')
[ -n "$TRAIN" ] && kill -STOP $TRAIN 2>/dev/null && echo "[ro] training paused for the run (pid $TRAIN)"
for p in $(pgrep -f "[t]racker_service\.py track"); do taskset -a -cp 0-11 $p >/dev/null 2>&1; done
python3 $RD/ws_relay.py --udp 127.0.0.1:9873 --ws-port 18767 --rumble-udp off --source robot --run-id $RUN \
  --replay-of $SID --calib $M/tool_calib_endo_robotL_tip.json --drill-calib $M/tool_calib_drill_robotR_tip.json \
  --trig-udp 127.0.0.1:9872 --sessions-dir $SESS > $OUTD/relay.log 2>&1 &
RELAY=$!
for i in $(seq 1 40); do ss -ltn | grep -q ":18767 " && break; sleep 0.25; done
grep -m1 "origin source=robot" $OUTD/relay.log | sed 's/^/[ro] /' || { echo "[ro] robot relay did not start"; cat $OUTD/relay.log; exit 3; }
python3 $S/tap_fanout.py 9881 9882 9883 9884 & FAN=$!
LIVE=""; WATCH=""
if [ -n "$RO_LIVE" ]; then   # the policy loads while the arms go to their start; its trigger heartbeat must run before the recorder starts
  OMP_NUM_THREADS=4 taskset -c 12-15 $PY -u inference/live_loop.py --live-run $RUN --speed ${SPEED:-0.33} --ready-flag $OUTD/LIVE_READY --windows-out $OUTD/motion_windows.json \
    $(grep -v '^#' "$RO_LIVE" | tr '\n' ' ') ${RO_OBS_PORT:+--obs-url http://127.0.0.1:$RO_OBS_PORT} -o $OUTD/live.json > $OUTD/live.log 2>&1 &
  LIVE=$!
fi
$PY -u robot_origin_rec.py --recording $FP --from-s $FROM --run-id $RUN --anchor $ANCHOR --drill-size ${SIZE:-medium} \
  --ready-file $READY --started-file $STARTED --done-file $DONE --pin-baseline --arm-on-entry ${RO_LIVE:+--max-s $((LEN + 120))} \
  ${RO_REC_MAX_S:+--rec-max-s $RO_REC_MAX_S} ${RO_LIVE:+${RO_OBS_PORT:+--obs-port $RO_OBS_PORT ${RO_OBS_STREAM:+--obs-stream}}} > $OUTD/result.json 2>> $OUTD/driver.err &   # RO_REC_MAX_S: a session longer than the recorder's 600 s
DRV=$!

one() {  # arm tool out port
  local ARGS
  if [ $1 = left ]; then ARGS="@presets/replay_left_${RO_PRESET:-j}.args $LSTART ${LCAL:+--tool-calib=$LCAL} ${RO_LEFT_X//;/ }"
  elif [ -n "$RO_LIVE" ]; then ARGS="@presets/replay_right_${RO_PRESET:-j}.args $RSTART ${RCAL:+--tool-calib=$RCAL} ${RO_RIGHT_X//;/ } --run-id $RUN"
  else ARGS="@presets/replay_right_${RO_PRESET:-j}.args $RSTART ${RCAL:+--tool-calib=$RCAL} ${RO_RIGHT_X//;/ } --trig-udp 127.0.0.1:9872 --run-id $RUN"; fi
  [ -n "$RO_LIVE" ] && ARGS="$ARGS --replay-stream-udp $([ $1 = left ] && echo 9891 || echo 9892)"
  $PY -u endo_follow.py $ARGS --mode replay --replay-run $FP --replay-from-s $FROM --replay-max-s $LEN \
    --replay-speed ${SPEED:-0.2} --replay-interp linear --radius-mm ${RO_RADIUS_MM:-80} --out $3 --go-file $GO --tap-port $4 \
    2>&1 | grep --line-buffered -v Warning | grep --line-buffered -E "replay\]|follow\]|error|Error|Traceback|disengage|준비|시작" \
    | sed -u "s/^/[$1] /" | tee -a $OUTD/arms.log
  return ${PIPESTATUS[0]}
}
PIDS=""; OUTS=""; ABORT=0
for x in right left; do   # paired start poses were path-checked right first, then left
  TOOL=$([ $x = left ] && echo endo || echo drill); PORT=$([ $x = left ] && echo 9881 || echo 9882)
  OUT=$HOME/il-robot/runs/latency/${RUN}_${TOOL}; OUTS="$OUTS $OUT"
  one $x $TOOL $OUT $PORT & PIDS="$PIDS $!"
  for i in $(seq 1 120); do grep -q "\[$x\] \[replay\] 준비됨" $OUTD/arms.log 2>/dev/null && break; sleep 0.5; done
  grep -q "\[$x\] \[replay\] 준비됨" $OUTD/arms.log 2>/dev/null || { echo "[ro] $x did not reach its start — abort"; ABORT=1; break; }
  echo "[ro] $x at start"
done
if [ $ABORT = 0 ] && [ -n "$RO_LIVE" ]; then
  for i in $(seq 1 600); do [ -e $OUTD/LIVE_READY ] && break; kill -0 $LIVE 2>/dev/null || break; sleep 0.5; done
  [ -e $OUTD/LIVE_READY ] || { echo "[ro] the live policy loop did not get ready — abort"; tail -n 5 $OUTD/live.log; ABORT=1; }
fi
if [ $ABORT = 0 ]; then
  sleep 3; touch $READY
  for i in $(seq 1 240); do [ -e $STARTED ] && break; kill -0 $DRV 2>/dev/null || break; sleep 0.5; done
  if [ -e $STARTED ]; then echo "[ro] session $(cat $STARTED) recording — GO"
    # the tracker live view's IR / RGB / webcam during the run (5-panel video with the viewer replay afterwards)
    $PY inference/cam_capture.py $OUTD/cams --stop-file $DONE > $OUTD/cams.log 2>&1 &
    touch $GO
    if [ -n "$LIVE" ]; then   # the live loop gone (done, aborted, crashed): trigger off, and no arm may wait for targets that never come
      ( while kill -0 $LIVE 2>/dev/null; do sleep 1; done
        python3 -c "import socket,json,time; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.sendto(json.dumps({'t':time.time()*1000,'tool':'drill','btn_trig':0,'src':'robot','run':'$RUN','end':1}).encode(),('127.0.0.1',9872))"
        sleep 6; for p in $PIDS; do pkill -TERM -P $p 2>/dev/null; kill -TERM $p 2>/dev/null; done ) &
      WATCH=$!
    fi
  else echo "[ro] the recorder page did not start — abort"; ABORT=1; fi
fi
if [ $ABORT = 1 ]; then for p in $PIDS; do pkill -TERM -P $p 2>/dev/null; kill -TERM $p 2>/dev/null; done; fi
RC=0; for p in $PIDS; do wait $p || RC=1; done
if [ -n "$LIVE" ]; then
  [ $ABORT = 1 ] && kill $LIVE 2>/dev/null
  wait $LIVE; LRC=$?; LIVE=""; echo "[ro] live loop rc $LRC: $(tail -n 1 $OUTD/live.log | cut -c1-300)"
fi
touch $DONE; wait $DRV; DRC=$?; DRV=""
[ -n "$TRAIN" ] && kill -CONT $TRAIN 2>/dev/null
for x in right left; do $PY $S/home_off.py $x; done
# the arms are parked: render work may go on (the queue held it for the run — render_gate.py; an arm that is not OFF keeps
# it paused by itself). What follows in this script is render work of its own: the session's video.
[ -n "$RO_RENDER_HOLD" ] && $PY ${SPINE_IL_REPO:-$HOME/spine-il}/robot/eval/render_gate.py release $RO_RENDER_HOLD
cat $OUTD/result.json
tail -n 12 $OUTD/driver.err
SIDR=$(tail -1 $OUTD/result.json | python3 -c "import json,sys;print(json.load(sys.stdin).get('sessionId') or '')" 2>/dev/null)
if [ -n "$SIDR" ]; then
  SF=$(find $SESS -name "$SIDR.jsonl" | head -1); echo "[ro] session file $SF"
  [ -n "$SF" ] && python3 $RD/qc_v2.py "$SF" > $OUTD/qc.txt 2>&1; tail -n 15 $OUTD/qc.txt | sed 's/^/[qc] /'
fi
if [ $ABORT = 0 ] && [ $RC = 0 ] && [ -z "$RO_LIVE" ]; then
  EVO=$(for o in $OUTS; do echo $o; done | awk '/_endo$/{e=$0} /_drill$/{d=$0} END{print e, d}')
  for o in $OUTS; do python3 $S/add_tip.py $o ${o}_tip >/dev/null; done
  OPENBLAS_NUM_THREADS=2 nice $PY $S/iter_eval.py $TAG $CFG $FP $FROM $LEN $EVO >> $HOME/il-robot/runs/jitter_iter/eval.log 2>&1
fi
if [ -n "$SF" ] && [ $ABORT = 0 ] && [ -z "$RO_NO_VIDEO" ]; then   # viewer replay of the session + the three cameras → one mp4 at the recording's pace
  # one frame per 1/15 s of hand time, played at 15 fps — unless the session is long (a run without a drilling time limit): at
  # most RO_VIDEO_MAX_FRAMES frames (about 1 s of rendering each), played so that the video still runs at the recording's pace
  read VF PF <<<$(tail -1 $OUTD/result.json | python3 -c "
import json, sys
sp = float('${SPEED:-0.2}'); el = float(json.load(sys.stdin).get('elapsed_s') or 0)
vf = min(15 * sp, ${RO_VIDEO_MAX_FRAMES:-4500} / el) if el > 0 else 15 * sp
print(round(vf, 3), round(vf / sp, 3))" 2>/dev/null || echo "$(python3 -c "print(round(15*float('${SPEED:-0.2}'),3))") 15")
  # a long session (a run without a drilling time limit: 190 MB for 30 min) does not load in the render service — the video
  # is rendered from a thinned copy (inference/thin_session.py: frame rows at 15 Hz, everything around cuts and trigger kept)
  SFV=$($PY inference/thin_session.py "$SF" -o $OUTD/thin 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['path'])" 2>/dev/null)
  [ -s "$SFV" ] || SFV=$SF
  [ "$SFV" != "$SF" ] && echo "[ro] video from the thinned session copy $SFV"
  if [ -n "$RO_VIDEO_LATER" ]; then
    # RO_VIDEO_LATER=1: the video takes five times the run (the renderer draws every 16 ms of the session) and holds the render
    # API, which the next robot run needs free. A series of runs leaves its videos for later: the command is kept in
    # video_cmd.sh and the run is listed in runs/robot_origin/video_later.txt (eval/video_later.sh renders them one by one).
    { echo '#!/usr/bin/env bash'
      echo "cd ${SPINE_IL_REPO:-$HOME/spine-il}/robot && OMP_NUM_THREADS=2 $PY -u inference/render_video.py \"$SFV\" -o $OUTD/session_5panel.mp4 --fps $VF --play-fps $PF --cams $OUTD/cams ${RO_LIVE:+--windows $OUTD/motion_windows.json} ${RO_VIDEO_OPTICS:+--optics $RO_VIDEO_OPTICS} > $OUTD/video.log 2>&1"
    } > $OUTD/video_cmd.sh
    echo "$OUTD" >> $HOME/il-robot/runs/robot_origin/video_later.txt
    echo "[ro] 5-panel video deferred → $OUTD/video_cmd.sh (video_later.txt)"
  else
  (cd ${SPINE_IL_REPO:-$HOME/spine-il}/robot && OMP_NUM_THREADS=2 nohup $PY -u inference/render_video.py "$SFV" -o $OUTD/session_5panel.mp4 \
     --fps $VF --play-fps $PF --cams $OUTD/cams ${RO_LIVE:+--windows $OUTD/motion_windows.json} \
     ${RO_VIDEO_OPTICS:+--optics $RO_VIDEO_OPTICS} > $OUTD/video.log 2>&1 &)   # RO_VIDEO_OPTICS=water: the medium a water-trained policy saw
  echo "[ro] 5-panel video → $OUTD/session_5panel.mp4 (rendering)"
  fi
fi
STOPPED=""; grep -q "(sigterm)" $OUTD/arms.log 2>/dev/null && STOPPED=", STOPPED EARLY (sigterm — see guard.log)"
grep -q "goal_no_lease" $OUTD/arms.log 2>/dev/null && STOPPED="$STOPPED, ARMS LOST THEIR LEASE (no goals for 15 s — tracker outage?)"
echo "[ro] $TAG done: arms rc $RC, recorder rc $DRC, abort $ABORT$STOPPED"
