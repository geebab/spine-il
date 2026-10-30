#!/usr/bin/env bash
# keep the full-span pair search off the CPU while a robot run is moving the arms (the tracker needs the cores)
while pgrep -f "fullspan_searc[h]\.sh" >/dev/null; do
  S=$(pgrep -f "^$HOME/il-robot/.venv/bin/python -u (search|postpass)_clear\.py")
  if pgrep -f "[e]ndo_follow\.py" >/dev/null; then [ -n "$S" ] && kill -STOP $S 2>/dev/null; else [ -n "$S" ] && kill -CONT $S 2>/dev/null; fi
  sleep 5
done
