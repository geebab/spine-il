#!/usr/bin/env bash
# The GR00T training dataset from everything that is in the session store, in one go (user 10-10: "그루트 학습을 위한
# 준비를 해라") — for the day the new hand recordings are in (target: 100 by 10-16):
#   1. replay render (water, the recording's own cut rows) of every usable hand session that has no cached frames yet
#      → datasets/render_cache_replay_water        (0.5 s a frame: a 2-minute recording ≈ 17 min; pauses for the robot)
#   2. export: 15 Hz · 256 px · --trim robot · absolute actions + drill.progress (state 16 with pad.zero)
#      → datasets/NAME, meta/stats.json quantiles rewritten over all frames (global_quantiles.py)
#   3. checks: train/groot_preflight.py (stops here when it finds a problem), dataset/end_gesture.py (a report, no gate)
#   4. a team-readable copy in /srv/il-robot/datasets/ (runs/open_h/share_dataset.sh) unless SHARE=0
#   bash groot_dataset.sh NAME [QC_EXCLUDE.json]        e.g. NAME=spine_hand100_replay_trim_water_progress
#   CHUNK=40 (the chunk size the pre-flight is run for), SHARE=0, RENDER=0 (the frames are all there),
#   EXPORT=0 (the dataset is there: checks and sharing only)
# New recordings first: stage and check them (memory "hand recordings intake"), copy them into
# ~/il-robot/viewer-data/sessions/<date>/ — the export takes whatever hand session is there.
# Everything heavy goes through eval/render_gate.py run --low: low priority, a memory ceiling, stopped while the robot is used.
D=$HOME/il-robot/datasets; PY=$HOME/il-robot/.venv/bin/python; R=${SPINE_IL_REPO:-$HOME/spine-il}/robot; W=$HOME/il-robot/runs/open_h
NAME=${1:?dataset name}; QC=${2:-$HOME/il-robot/runs/hand_incoming_20261006/qc_trim.json}; C=$D/render_cache_replay_water
G="$PY $R/eval/render_gate.py run --low"; O=$HOME/il-robot/runs/groot_prep; mkdir -p $O
log() { echo "$(date '+%m-%d %T') groot_dataset $NAME: $*" | tee -a $W/chain.log; }
cd $R/dataset || exit 2
if [ "${RENDER:-1}" != 0 ]; then
  log "replay render of the sessions without cached frames ($($PY render_cache.py --fps 15 --include hand --cache $C --optics water --chunk 0 --removed --dry-run 2>/dev/null | tail -n 1))"
  $G --mem-gb 2 --name replay_render -- $PY -u render_cache.py --fps 15 --include hand --cache $C --optics water --chunk 0 --removed >> $D/render_replay_water_groot.log 2>&1 \
    || { log "render stopped (rc $?) — see $D/render_replay_water_groot.log; run again to go on"; exit 3; }
fi
if [ "${EXPORT:-1}" != 0 ]; then
  log "export"
  $G --mem-gb 3 --name groot_export -- $PY export_lerobot.py --out $D/$NAME --mode full --include hand --fps 15 --action-mode absolute --progress-output \
      --cache $C --trim robot --exclude $QC --repo-id local/$NAME --overwrite > $W/export_$NAME.log 2>&1 || { log "export failed — $W/export_$NAME.log"; exit 4; }
  log "exported: $(grep -c '\[export\] ep' $W/export_$NAME.log) episodes; $(grep 'quantiles' $W/export_$NAME.log | tail -n 1 | cut -c1-120)"
else
  [ -e $D/$NAME/meta/info.json ] || { log "EXPORT=0 but there is no dataset $D/$NAME"; exit 4; }
  $PY global_quantiles.py $D/$NAME --check > /dev/null 2>&1 || { $PY global_quantiles.py $D/$NAME > /dev/null 2>&1; log "quantiles of the existing dataset rewritten over all frames"; }
fi
cd $R || exit 2
$G --mem-gb 2.5 --name groot_preflight -- $PY train/groot_preflight.py --dataset $D/$NAME --chunk-size ${CHUNK:-40} --held-out "${HELD_OUT-2,18}" --lerobot -o $O/preflight_$NAME.json > $O/preflight_$NAME.txt 2>&1
RC=$?; log "pre-flight rc $RC: $(tail -n 1 $O/preflight_$NAME.txt)"
[ $RC = 0 ] || { log "NOT shared — $O/preflight_$NAME.txt says what is wrong"; exit 5; }
$G --mem-gb 1 --name end_gesture -- $PY dataset/end_gesture.py --dataset $D/$NAME -o $O/end_gesture_$NAME.json > $O/end_gesture_$NAME.txt 2>&1
log "end gesture: $(tail -n 1 $O/end_gesture_$NAME.txt)"
if [ "${SHARE:-1}" != 0 ]; then log "shared: $(bash $W/share_dataset.sh $NAME 2>&1 | tail -n 1)"; fi
HOLD=$(python3 -c "
import json
ep = [json.loads(l) for l in open('$D/$NAME/meta/spine_il/episodes.jsonl') if l.strip()]
print(','.join(str(i) for i, m in enumerate(ep) if m['session_id'].rsplit('_', 1)[-1] in ('6639', '92c5')))" 2>/dev/null)
log "done — the two demonstrations held out so far (6639, 92c5) are episodes $HOLD of this dataset"
