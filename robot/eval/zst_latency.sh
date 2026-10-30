#!/usr/bin/env bash
# 10-08: the root filesystem (233 GB) ran full at about 16:05 and took three robot runs and an evaluation batch with it.
# The largest single item is the arm followers' event logs, runs/latency/<run>_<tool>/events.jsonl: 54.5 GB of JSON lines
# in 2,591 files, 36 GB of it from September (the replay and jitter experiments, 2,338 folders). JSON lines compress 6-7x.
# This script stores the logs of the folders whose name starts with PREFIX zstd-compressed — LOSSLESS, nothing is dropped:
#     events.jsonl  ->  events.jsonl.zst        (the source is removed only after `zstd -t` has verified the copy)
# Reading one again:   zstd -dc events.jsonl.zst | head        or        zstd -d --keep events.jsonl.zst   (restores the file)
# October's logs are left as they are (run_trace.py, run_criteria.py and the 5-panel videos of the pending runs read them).
#     bash zst_latency.sh 202609          stop: touch runs/latency/ZST_STOP
L=$HOME/il-robot/runs/latency; P=${1:?prefix of the run folders, e.g. 202609}; LOG=$L/zst_$P.log
n=0; saved=0; failed=0
echo "$(date '+%m-%d %T') $P: start ($(df --output=avail -BM / | tail -1 | tr -d ' ') free)" >> $LOG
for f in $L/${P}*/events.jsonl; do
  [ -e $L/ZST_STOP ] && { echo "$(date '+%m-%d %T') stopped by ZST_STOP" >> $LOG; break; }
  [ -f "$f" ] || continue
  s=$(stat -c %s "$f"); [ "$s" -lt 4096 ] && continue
  if taskset -c 12-15 nice zstd -10 -T4 -q -f "$f" -o "$f.zst" && zstd -tq "$f.zst"; then
    z=$(stat -c %s "$f.zst"); rm "$f"; n=$((n + 1)); saved=$((saved + s - z))
  else
    failed=$((failed + 1)); echo "FAILED $f" >> $LOG; [ -f "$f" ] && [ -f "$f.zst" ] && rm "$f.zst"
  fi
done
echo "$(date '+%m-%d %T') $P: $n files compressed, $((saved / 1000000)) MB freed, $failed failed ($(df --output=avail -BM / | tail -1 | tr -d ' ') free)" >> $LOG
