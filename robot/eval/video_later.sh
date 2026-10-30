#!/usr/bin/env bash
# Render the 5-panel videos that robot runs left for later (robot_origin_run.sh RO_VIDEO_LATER=1 / live_run.sh
# LIVE_VIDEO=later), oldest first, one at a time.
#   bash video_later.sh
# A video holds the render API for about five times its run, and a robot run needs the API free: start this only when no
# robot run is due soon. It waits while a run is queued or active and while the render API is busy, and it stops before the
# next video when ~/il-robot/runs/robot_origin/video_later.HOLD exists. Log: runs/robot_origin/video_later.log.
R=$HOME/il-robot/runs/robot_origin; L=$R/video_later.txt; W=$HOME/il-robot/runs/jitter_iter/work; PY=$HOME/il-robot/.venv/bin/python
busy() { [ -e $W/RO_WANT ] || ! (cd $W && $PY render_idle.py) || pgrep -f "[e]ndo_follow\\.py|[r]obot_origin_run\\.sh|[r]ender_video\\.py|[p]olicy_eval\\.py|[r]o_iou\\.py" >/dev/null \
         || [ -n "$(grep -v '^#' $W/ro_jobs.txt 2>/dev/null | grep -m1 .)" ] \
         || { $PY ${SPINE_IL_REPO:-$HOME/spine-il}/robot/eval/render_gate.py status >/dev/null 2>&1; [ $? = 1 ]; }; }   # 10-09: the gate has a pause in force (an arm in use from any account)
[ -s $L ] || { echo "nothing listed in $L"; exit 0; }
while read -r D; do
  [ -d "$D" ] && [ -s "$D/video_cmd.sh" ] || continue
  [ -s "$D/session_5panel.mp4" ] && continue
  [ -e $R/video_later.HOLD ] && { echo "$(date '+%m-%d %T') HOLD — stopped before $(basename $D)" >> $R/video_later.log; exit 0; }
  # a video's render job leaves up to 5.5 GB under render-jobs (frames of two views and their zip); 10-08 16:05 the root
  # filesystem ran full and took three robot runs with it — no video is started with less than VIDEO_MIN_FREE_GB free
  FREE_GB=$(df --output=avail -BG / | tail -1 | tr -dc 0-9)
  [ "${FREE_GB:-0}" -ge ${VIDEO_MIN_FREE_GB:-14} ] || { echo "$(date '+%m-%d %T') disk: ${FREE_GB:-?} GB free (${VIDEO_MIN_FREE_GB:-14} needed) — stopped before $(basename $D)" >> $R/video_later.log; exit 0; }
  while :; do busy && { sleep 20; continue; }; sleep 7; busy || break; done   # idle, and still idle a moment later (10-08: a render started beside a robot run)
  echo "$(date '+%m-%d %T') $(basename $D): rendering" >> $R/video_later.log
  bash "$D/video_cmd.sh"; echo "$(date '+%m-%d %T') $(basename $D): rc $? $(ls -la $D/session_5panel_motion.mp4 2>/dev/null | awk '{print $5}') bytes (motion cut)" >> $R/video_later.log
done < <(grep -v '^#' $L | awk '!seen[$0]++')
