#!/usr/bin/env bash
# What the GR00T training side needs from this repository, in one flat folder on the team share (no checkout of the
# private repository on the GPU machine):   bash make_groot_kit.sh [OUT_DIR=/srv/il-robot]
# README.md (= train/GROOT.md: the dataset, the statistics defect and its fix, the training command, the checks),
# groot_preflight.py (what LeRobot's GR00T code makes of a dataset — run it before a training), global_quantiles.py (the
# fix for a dataset copy that still has LeRobot's averaged quantiles), the pre-flight reports made here (the progress dataset at chunk 40 and 16, and the 15-value dataset of the trial
# trainings as it was shared), VERSION.
# Readable by the team group, not writable (owner owner1, group team, files 0640, folders 2750).
set -e
R=$(cd "$(dirname "$0")/.." && pwd); OUT=${1:-/srv/il-robot}; B=$OUT/groot_train; P=$HOME/il-robot/runs/groot_prep
rm -rf $B.new; install -d -m 2750 -g team $B.new
cp $R/train/GROOT.md $B.new/README.md
cp $R/train/groot_preflight.py $R/dataset/global_quantiles.py $B.new/
for f in $P/preflight_progress_chunk*_lerobot.json $P/preflight_progress_chunk*_lerobot.txt $P/preflight_trial_dataset_15values_as_shared.json; do [ -s "$f" ] && cp "$f" $B.new/; done
(cd $R/.. && git log -1 --format='%h %ci %s' 2>/dev/null) > $B.new/VERSION
(cd $B.new && python3 -m py_compile groot_preflight.py global_quantiles.py && rm -rf __pycache__)
chgrp -R team $B.new; find $B.new -type f -exec chmod 0640 {} +
if [ -e $B ]; then rm -rf $OUT/.groot_train.prev; mv $B $OUT/.groot_train.prev; fi      # the one before stays, hidden
mv $B.new $B
ls -la $B
