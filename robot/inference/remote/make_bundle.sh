#!/usr/bin/env bash
# The files the server side needs, in one flat folder and one .tar.gz on the team share — no checkout of the (private)
# repository on the GPU machine:   bash make_bundle.sh [OUT_DIR=/srv/il-robot/remote_policy]
# policy_server.py (reference server), remote_policy.py (protocol + client + connection check), actions.py, phases.py
# (pure numpy, from robot/dataset), groot_map.example.json, README.md (what to run, the contract, how to test), VERSION.
# Readable by the team group, not writable (as the shared datasets: owner owner1, group team, files 0640, folders 2750).
set -e
R=$(cd "$(dirname "$0")/../.." && pwd); OUT=${1:-/srv/il-robot/remote_policy}; B=$OUT/spine_policy_server
install -d -m 2750 -g team $OUT; rm -rf $B.new; install -d -m 2750 -g team $B.new
cp $R/inference/remote/policy_server.py $R/inference/remote_policy.py $R/dataset/actions.py $R/dataset/phases.py $R/dataset/instructions.py \
   $R/inference/remote/groot_map.example.json $R/inference/remote/README.md $B.new/
(cd $R/.. && git log -1 --format='%h %ci %s' 2>/dev/null) > $B.new/VERSION
(cd $B.new && python3 -m py_compile policy_server.py remote_policy.py actions.py phases.py instructions.py && rm -rf __pycache__)
chgrp -R team $B.new; find $B.new -type f -exec chmod 0640 {} +
if [ -e $B ]; then rm -rf $OUT/.spine_policy_server.prev; mv $B $OUT/.spine_policy_server.prev; fi   # the one before stays, hidden
mv $B.new $B
(cd $OUT && tar czf spine_policy_server.tar.gz spine_policy_server && chgrp team spine_policy_server.tar.gz && chmod 0640 spine_policy_server.tar.gz)
ls -la $OUT $B
