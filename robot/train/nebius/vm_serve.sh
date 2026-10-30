#!/usr/bin/env bash
# ON THE GPU MACHINE: the trained model behind the robot loop's protocol (server/policy_server.py, spine-policy/1),
# listening on this machine's own 127.0.0.1 only — the robot server reaches it through its SSH tunnel.
#   NAME=run [STEP=last] [PORT=8900] [GUIDANCE=W] [GUIDANCE_NEG=other|neutral] bash vm_serve.sh
#   (GUIDANCE: policy_server.py --guidance, lang1 / lang2 GR00T models; GUIDANCE_NEG: --guidance-negative, neutral = lang2 only)
# The token is made here once (~/il-robot/policy.token, 600) and copied to the robot server file to file (vm.sh serve).
set -uo pipefail
WS=${IL_ROBOT_WS:-$HOME/il-robot}; NAME=${NAME:?run name}; STEP=${STEP:-last}; PORT=${PORT:-8900}
P=$WS/out/$NAME/run/checkpoints/$STEP/pretrained_model
[ -d "$P" ] || { echo "no checkpoint $P — have: $(ls "$WS/out/$NAME/run/checkpoints" 2>/dev/null | tr '\n' ' ')"; exit 2; }
T=$WS/policy.token
[ -s "$T" ] || ( umask 077; "$WS/.venv/bin/python" -c "import secrets; print(secrets.token_urlsafe(32))" > "$T" )
chmod 600 "$T"
cd "$WS/server" || exit 2
# the instruction set of the training's dataset (run.json "dataset"; its meta/spine_il/export.json) — the loop asks in its words
DSN=$("$WS/.venv/bin/python" -c "import json,sys; print(json.load(open(sys.argv[1])).get('dataset') or '')" "$WS/out/$NAME/run.json" 2>/dev/null)
IF=${INSTRUCTIONS_FILE:-$WS/datasets/$(basename "${DSN:-none}")/meta/spine_il/export.json}
[ -s "$IF" ] && grep -q '"instructions"' "$IF" && set -- --instructions-file "$IF" || set --
exec "$WS/.venv/bin/python" -u policy_server.py --backend lerobot --policy-path "$P" --device cuda --token-file "$T" --host 127.0.0.1 --port "$PORT" ${GUIDANCE:+--guidance "$GUIDANCE"} ${GUIDANCE_NEG:+--guidance-negative "$GUIDANCE_NEG"} "$@"
