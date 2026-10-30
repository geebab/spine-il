#!/usr/bin/env bash
# Robot priority over long render batches: RO_WANT exists while the robot queue waits for the renderer (other render
# clients must not submit new jobs while it exists); removed once a recording runs.
W=$HOME/il-robot/runs/jitter_iter/work; L=$HOME/il-robot/runs/jitter_iter/robot_origin.log
while :; do
  if pgrep -f "[r]obot_origin_run\.sh" >/dev/null; then rm -f $W/RO_WANT
  elif tail -n 1 $L | grep -q "waiting for renders"; then touch $W/RO_WANT
  else rm -f $W/RO_WANT; fi
  sleep 3
done
