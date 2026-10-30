#!/usr/bin/env bash
# Closed-loop runs one after another: each line of live_jobs.txt = NAME|CKPT_DIR|START_JSON|EXTRA (live_run.sh arguments).
# A run includes its 5-panel video and evaluation (about 1.5 h). Stops on live/STOP; idles when the list is empty.
A=$HOME/il-robot/runs/act_smoke; LV=$A/live
while [ ! -e $LV/STOP ]; do
  L=$(grep -v '^#' $LV/live_jobs.txt 2>/dev/null | grep -m1 .)
  [ -z "$L" ] && { sleep 30; continue; }
  python3 - "$L" <<'PY'
import sys; from pathlib import Path
p = (Path.home() / "il-robot/runs/act_smoke/live/live_jobs.txt"); ls = p.read_text().splitlines()
ls.remove(sys.argv[1]); p.write_text("\n".join(ls) + ("\n" if ls else ""))
PY
  IFS='|' read -r NAME CK START EXTRA <<< "$L"
  echo "[liveq] $(date +%T) $NAME | $CK | $START | $EXTRA"
  bash $LV/live_run.sh "$NAME" "$CK" "$START" "$EXTRA" > $LV/$NAME.out 2>&1
  echo "[liveq] $(date +%T) $NAME → $(cat $LV/$NAME/DONE 2>/dev/null | cut -c1-120)"
done
