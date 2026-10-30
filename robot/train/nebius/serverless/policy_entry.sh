#!/usr/bin/env bash
# Start policy_server.py on 0.0.0.0:$POLICY_PORT. The token comes from the environment (POLICY_TOKEN, e.g. an endpoint
# --env-secret) or a file (POLICY_TOKEN_FILE); policy_server.py wants a 0600 file, so it is written to one here.
set -euo pipefail
[ -d "$POLICY_PATH" ] || { echo "[policy] no checkpoint at $POLICY_PATH (mount the bucket at /ckpt)"; exit 2; }
# GrootPolicy reads the base model's config from the Hugging Face hub at load time — nvidia/Cosmos-Reason2-2B is GATED,
# so a container without an HF login fails (401). The endpoint carries the hub cache in the bucket (/ckpt/hf-hub, the
# GPU VM's ~/.cache/huggingface/hub) and runs offline; with no cache, HF_TOKEN (a secret) must be set instead.
if [ -d /ckpt/hf-hub ]; then export HF_HUB_CACHE=/ckpt/hf-hub HF_HUB_OFFLINE=1
elif [ -z "${HF_TOKEN:-}" ]; then echo "[policy] no /ckpt/hf-hub and no HF_TOKEN: the gated base model cannot load"; fi
T=/run/spine/policy.token
mkdir -p /run/spine && chmod 700 /run/spine
if [ -n "${POLICY_TOKEN:-}" ]; then (umask 077; printf '%s\n' "$POLICY_TOKEN" > "$T")
elif [ -n "${POLICY_TOKEN_FILE:-}" ] && [ -s "$POLICY_TOKEN_FILE" ]; then (umask 077; cat "$POLICY_TOKEN_FILE" > "$T")
else echo "[policy] POLICY_TOKEN or POLICY_TOKEN_FILE is required"; exit 2; fi
set --
[ -s "${POLICY_INSTRUCTIONS:-}" ] && grep -q '"instructions"' "$POLICY_INSTRUCTIONS" && set -- --instructions-file "$POLICY_INSTRUCTIONS"
cd /opt/server
exec /opt/venv-policy/bin/python -u policy_server.py --backend lerobot --policy-path "$POLICY_PATH" --device cuda \
  --token-file "$T" --host 0.0.0.0 --port "$POLICY_PORT" "$@"
