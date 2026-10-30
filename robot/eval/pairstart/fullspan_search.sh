#!/usr/bin/env bash
# 10-03: robot start pairs for the three long drillings over their whole kept span (--trim robot), not the 30 s window.
# Results: pair_full_<rid>.json (+ search_full_<rid>.log). Afterwards the three windows.json entries are removed and the
# old pair_clear_bef5.json is restored (backup_20261003/). Cores 12-15, nice; fullspan_pause.sh stops it during robot runs.
P=$HOME/il-robot/runs/jitter_iter/work/pairstart; PY=$HOME/il-robot/.venv/bin/python
cd $P
for rid in bad1 d340 bef5; do
  echo "$(date +%T) $rid start" >> fullspan.log
  PS_RADIUS_M=0.08 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 taskset -c 12-15 $PY -u search_clear.py $rid > search_full_$rid.log 2>&1
  PS_RADIUS_M=0.08 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 nice -n 19 taskset -c 12-15 $PY -u postpass_clear.py $rid >> search_full_$rid.log 2>&1
  cp pair_clear_$rid.json pair_full_$rid.json
  echo "$(date +%T) $rid done: $(tail -1 search_full_$rid.log | cut -c1-200)" >> fullspan.log
done
cp backup_20261003/pair_clear_bef5.json pair_clear_bef5.json
python3 - <<'PY'
import json
from pathlib import Path
p=(Path.home() / "il-robot/runs/jitter_iter/work/pairstart/windows.json"); w=json.loads(p.read_text())
for k in ('bad1','d340','bef5'): w.pop(k, None)
p.write_text(json.dumps(w, indent=1))
PY
echo "$(date +%T) all done" >> fullspan.log
