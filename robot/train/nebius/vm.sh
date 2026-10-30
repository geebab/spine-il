#!/usr/bin/env bash
# FROM THE ROBOT SERVER: drive a GPU machine of our own over SSH — environment, data, training, serving.
# The machine is a Nebius VM the user makes in
# the console (GPU, an Ubuntu image with the NVIDIA driver, a public address, the public half of ~/.ssh/id_ed25519_policy
# as the user's SSH key). Creating, stopping, deleting and paying for it stay with the user — nothing here does that.
#
#   bash vm.sh set USER@HOST [SSH_PORT]    remember the machine, try the connection, show its GPU, memory and disk
#   bash vm.sh push [DATASET_DIR]          kit + server files + dataset → ~/il-robot/ there (rsync: again = only what changed)
#   bash vm.sh setup                       the Python environment there (LeRobot 0.6.1 with its GR00T extra)
#   bash vm.sh hf                          log in to Hugging Face THERE — run it in your own terminal: the token is typed on
#                                          the VM and never passes through this machine's logs
#   bash vm.sh hfcheck                     can the VM read the gated repository the tokenizer comes from?
#   bash vm.sh preflight [DATASET]         train/groot_preflight.py there, with LeRobot's own functions
#   bash vm.sh smoke [DATASET] [BATCH]     60 steps, no checkpoint: seconds a step and GPU memory → how long a training takes
#   bash vm.sh train NAME [STEPS] [BATCH] [DATASET]   a training in tmux there (default 20000 steps, batch 32, a checkpoint
#                                          every 5000). Held out by default: the demonstrations 6639 and 92c5 (HELD="" = none,
#                                          the last training; EPISODES='[…]' = exactly these). RESUME=1 goes on.
#   bash vm.sh status [NAME]               step, loss, speed, time left, GPU, disk, checkpoints
#   bash vm.sh serve NAME [STEP]           the model behind the loop's protocol there (127.0.0.1:8900, tmux), its token here
#   bash vm.sh tunnel NAME                 the SSH tunnel and the endpoint entry here (port 18900 + 10 for each team
#                                          account after the first: LPORT=N for another)
#                                          → then: TAG=NAME DRY=1 bash inference/live/remote_intake.sh NAME
#   bash vm.sh fetch NAME [STEP]           keep a checkpoint (its pretrained_model, several GB) on this machine:
#                                          ~/il-robot/runs/groot_ckpt/NAME_STEP — for the day the VM and its disk are gone
#   bash vm.sh restore NAME STEP           put a kept checkpoint on the (new) machine, then: serve NAME STEP
#   bash vm.sh prune NAME                  during a training: remove the optimizer state (13 GB) of every checkpoint but the
#                                          newest complete one, now and every minute until the training ends — the models stay
#   bash vm.sh stop NAME | serve-stop NAME | ssh [COMMAND…]
#
# DATASET = a folder name under ~/il-robot/datasets (default spine_hand24_replay_trim_water_progress). PRICE=1.8 (USD an
# hour) makes smoke and status print what the time costs.
set -u
R=$(cd "$(dirname "$0")/../.." && pwd); PY=${PY:-$HOME/il-robot/.venv/bin/python}; D=$HOME/il-robot/datasets
# where a dataset is: the caller's own ~/il-robot/datasets, else the team share (/srv/il-robot/datasets — every team
# account reads it, none writes it); UBE_DATASETS=DIR before both. A team member has no datasets of their own.
dsdir() { local b; for b in "${UBE_DATASETS:-}" "$HOME/il-robot/datasets" /srv/il-robot/datasets; do
            [ -n "$b" ] && [ -s "$b/$1/meta/info.json" ] && { echo "$b"; return 0; }; done; echo "$HOME/il-robot/datasets"; }
# the tunnel's port on this machine: one a team account, so that two members' tunnels do not meet (owner1 18900, the next
# account 18910, …); LPORT=N for another
lport() { echo "${LPORT:-$((18900 + ($(id -u) - 1000) * 10))}"; }
KEY=${KEY:-$HOME/.ssh/id_ed25519_policy}; CF=${UBE_CONFIG_DIR:-$HOME/.config/ube}/nebius_vm
DSN_DEFAULT=spine_hand24_replay_trim_water_progress
CMD=${1:-help}; shift || true
vm() { [ -s "$CF" ] || { echo "no machine yet — bash $0 set USER@HOST"; exit 2; }; read -r DEST PORT < "$CF"; PORT=${PORT:-22}
       [ -s "$KEY" ] || { echo "no key at $KEY"; exit 2; }
       SSHO=(-i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o ConnectTimeout=10); }
vssh() { ssh "${SSHO[@]}" -o BatchMode=yes -p "$PORT" "$DEST" "$@"; }
vrsync() { rsync -az --info=stats1 -e "ssh ${SSHO[*]} -o BatchMode=yes -p $PORT" "$@"; }
dsn() { basename "${1:-$DSN_DEFAULT}"; }
# the episodes of the two demonstrations every comparison so far left out (sessions 6639 and 92c5), by session id
held() { $PY - "$D/$1" "${HELD-6639,92c5}" <<'PYE'
import json, sys
from pathlib import Path
ds, ids = Path(sys.argv[1]), [x for x in sys.argv[2].split(",") if x]
n = json.loads((ds / "meta/info.json").read_text())["total_episodes"]
ep = [json.loads(l) for l in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()]
out = [i for i, m in enumerate(ep) if any(str(m.get("session_id", "")).endswith(x) for x in ids)]
print(json.dumps([i for i in range(n) if i not in out]).replace(" ", ""), ",".join(map(str, out)) or "none")
PYE
}

case "$CMD" in
  set)
    DESTN=${1:?USER@HOST}; case "$DESTN" in *@*) ;; *) echo "USER@HOST"; exit 2;; esac
    case "${2:-22}" in *[!0-9]*) echo "the SSH port is a number"; exit 2;; esac
    mkdir -p "$(dirname "$CF")" && chmod 700 "$(dirname "$CF")" && echo "$DESTN ${2:-22}" > "$CF"; vm
    vssh 'echo "connected: $(whoami)@$(hostname), $(. /etc/os-release; echo $PRETTY_NAME)"
          nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | sed "s/^/GPU: /" || echo "GPU: nvidia-smi does not answer (no driver in this image?)"
          echo "CPU $(nproc) · RAM $(free -g | awk "NR==2{print \$2}") GiB · disk free $(df -h --output=avail ~ | tail -n 1 | tr -d " ")"
          sudo -n true 2>/dev/null && echo "sudo: yes" || echo "sudo: needs a password (vm_setup.sh installs packages with sudo)"' \
      || { echo "no connection to $DESTN — is the public key registered for that user? (cat $KEY.pub)"; exit 3; };;
  push)
    vm; DSN=$(dsn "${1:-}"); D=$(dsdir "$DSN"); [ -s "$D/$DSN/meta/info.json" ] || { echo "no dataset $DSN in ~/il-robot/datasets or /srv/il-robot/datasets"; exit 2; }
    $PY "$R/dataset/global_quantiles.py" "$D/$DSN" --check > /dev/null 2>&1 \
      || { echo "$DSN still has LeRobot's averaged quantiles (half of the endoscope's state would be clipped) — python dataset/global_quantiles.py $D/$DSN"; exit 2; }
    T=$(mktemp -d) && mkdir -p "$T/kit" "$T/server" || exit 2
    cp "$R/train/GROOT.md" "$T/kit/README.md"; cp "$R/train/groot_preflight.py" "$R/dataset/global_quantiles.py" "$R"/train/nebius/vm_*.sh "$T/kit/"
    (cd "$R/.." && git log -1 --format='%h %ci %s') > "$T/kit/VERSION"
    cp "$R/inference/remote/policy_server.py" "$R/inference/remote_policy.py" "$R/dataset/actions.py" "$R/dataset/phases.py" "$R/dataset/instructions.py" "$R/inference/remote/groot_map.example.json" "$T/server/"
    vssh "mkdir -p ~/il-robot/kit ~/il-robot/server ~/il-robot/datasets ~/il-robot/out" || exit 3
    vrsync --delete "$T/kit/" "$DEST:il-robot/kit/" && vrsync "$T/server/" "$DEST:il-robot/server/" && echo "kit and server files: $(cat "$T/kit/VERSION" | cut -c1-60)"
    rm -rf "$T"
    echo "dataset $DSN ($(du -sh "$D/$DSN" | cut -f1)) …"
    vrsync --delete --exclude '.cache' "$D/$DSN/" "$DEST:il-robot/datasets/$DSN/" || exit 3
    vssh "echo \"there: \$(du -sh ~/il-robot/datasets/$DSN | cut -f1), meta: \$(ls ~/il-robot/datasets/$DSN/meta | tr '\n' ' ')\"";;
  setup) vm; vssh "bash ~/il-robot/kit/vm_setup.sh";;
  hf)
    vm; echo "Hugging Face login on $DEST — paste a READ token of the account that accepted the terms of nvidia/Cosmos-Reason2-2B:"
    ssh "${SSHO[@]}" -t -p "$PORT" "$DEST" 'cd ~/il-robot && (.venv/bin/hf auth login || .venv/bin/huggingface-cli login)';;
  hfcheck)
    vm; vssh '~/il-robot/.venv/bin/python - <<"PYE"
from huggingface_hub import hf_hub_download
for repo, f in (("nvidia/Cosmos-Reason2-2B", "tokenizer_config.json"), ("nvidia/GR00T-N1.7-3B", "config.json")):
    try:
        hf_hub_download(repo, f); print("ok  ", repo)
    except Exception as e:
        print("FAIL", repo, "—", type(e).__name__, str(e).splitlines()[0][:140])
PYE';;
  preflight)
    vm; DSN=$(dsn "${1:-}"); read -r _ HO <<< "$(held "$DSN")"
    vssh "cd ~/il-robot/kit && ../.venv/bin/python groot_preflight.py --dataset ~/il-robot/datasets/$DSN --chunk-size 40 --held-out '${HO/none/}' --lerobot -o ~/il-robot/out/preflight_$DSN.json 2>&1 | tail -n 25";;
  smoke)
    vm; DSN=$(dsn "${1:-}"); B=${2:-32}; N=_smoke_$(date +%m%d_%H%M%S); read -r EP HO <<< "$(held "$DSN")"
    echo "smoke test $N: 60 steps, batch $B, dataset $DSN (held out: episodes $HO) — the first minutes are downloads and the relative-action statistics"
    vssh "NAME=$N DS=\$HOME/il-robot/datasets/$DSN STEPS=60 BATCH=$B SAVE=1000000 CKPT=0 LOG=10 LLM_TOP=${LLM_TOP:-0} EPISODES='$EP' bash ~/il-robot/kit/vm_train.sh; echo \"rc \$?\"; grep -E 'step:|Error|error|Traceback|CUDA out of memory' ~/il-robot/out/$N/train.log | tail -n 12; echo GPUMAX \$(sort -t, -k2 -n ~/il-robot/out/$N/gpu.csv | tail -n 1)" \
      | tee /dev/stderr | $PY -c "
import re, sys, statistics as st
t = sys.stdin.read(); u = [float(x) for x in re.findall(r'updt_s:([0-9.]+)', t)]; d = [float(x) for x in re.findall(r'data_s:([0-9.]+)', t)]
g = re.search(r'GPUMAX [^,]*, *([0-9]+)', t)
if len(u) < 3:
    print('no speed measured — see the log lines above'); sys.exit(1)
s = st.median(u[1:]) + st.median(d[1:]); price = float('${PRICE:-0}')
print(f'→ {s:.2f} s a step at batch $B (update {st.median(u[1:]):.2f} + data {st.median(d[1:]):.2f}); GPU memory peak {int(g.group(1)) / 1024:.1f} GiB' if g else f'→ {s:.2f} s a step')
for n in (5000, 10000, 20000):
    h = n * s / 3600; print(f'   {n:>6} steps: {h:4.1f} h' + (f'  ≈ \${h * price:.0f}' if price else '') + f'  ({n * $B / 1000:.0f}k samples)')
";;
  train)
    vm; NAME=${1:?run name}; STEPS=${2:-20000}; B=${3:-32}; DSN=$(dsn "${4:-}")
    case "$NAME" in *[!A-Za-z0-9_-]*) echo "NAME: letters, digits, _ and - only"; exit 2;; esac
    D=$(dsdir "$DSN")
    if [ -n "${EPISODES:-}" ]; then EP=$EPISODES; HO="(EPISODES given)"; else read -r EP HO <<< "$(held "$DSN")"; fi
    vssh "mkdir -p ~/il-robot/out/$NAME && cat > ~/il-robot/out/$NAME/env.sh" <<EOF || exit 3
NAME=$NAME
DS=\$HOME/il-robot/datasets/$DSN
STEPS=$STEPS
BATCH=$B
SAVE=${SAVE:-5000}
EPISODES='$EP'
BASE='${BASE:-}'
LLM_TOP=${LLM_TOP:-0}
RESUME=${RESUME:-0}
EOF
    vssh "tmux has-session -t groot_$NAME 2>/dev/null && { echo 'a training named $NAME is already running'; exit 4; }
          [ -e ~/il-robot/out/$NAME/run ] && [ '${RESUME:-0}' != 1 ] && { echo 'a run named $NAME exists there — another NAME, or RESUME=1 to go on from its last checkpoint'; exit 4; }
          tmux new-session -d -s groot_$NAME \"bash -c 'set -a; . ~/il-robot/out/$NAME/env.sh; set +a; bash ~/il-robot/kit/vm_train.sh'\" && echo 'started: $NAME — $STEPS steps, batch $B, dataset $DSN, held out: episodes $HO (bash vm.sh status $NAME)'";;
  status)
    vm; NAME=${1:-}
    [ -n "$NAME" ] || { vssh "ls -t ~/il-robot/out 2>/dev/null | head -n 12; tmux ls 2>/dev/null"; exit 0; }
    vssh "O=~/il-robot/out/$NAME; cat \$O/run.json 2>/dev/null; echo; tmux has-session -t groot_$NAME 2>/dev/null && echo 'tmux: running' || echo 'tmux: not running'
          [ -e \$O/DONE ] && echo \"DONE: \$(cat \$O/DONE)\"; echo \"LINES \$(grep -c 'step:' \$O/train.log 2>/dev/null)\"; grep 'step:' \$O/train.log 2>/dev/null | tail -n 3 | cut -c1-200
          grep -E 'Traceback|Error|out of memory' \$O/train.log 2>/dev/null | tail -n 3 | cut -c1-200
          echo \"checkpoints: \$(ls \$O/run/checkpoints 2>/dev/null | tr '\n' ' ')\"; echo \"GPU now: \$(tail -n 1 \$O/gpu.csv 2>/dev/null) (MiB, %)\"; echo \"disk free: \$(df -h --output=avail ~ | tail -n 1 | tr -d ' ')\"" \
      | tee /dev/stderr | $PY -c "
import json, re, sys, statistics as st
t = sys.stdin.read(); m = re.search(r'^\{.*\}$', t, re.M); n = re.search(r'LINES (\d+)', t)
u = [float(x) for x in re.findall(r'updt_s:([0-9.]+)', t)]; d = [float(x) for x in re.findall(r'data_s:([0-9.]+)', t)]
if m and n and u:
    r = json.loads(m.group(0)); done = int(n.group(1)) * int(r.get('log_freq') or 50); left = max(0, int(r['steps']) - done); s = st.median(u) + st.median(d); price = float('${PRICE:-0}')
    print(f'→ about step {done} of {r[\"steps\"]}, {s:.2f} s a step: {left * s / 3600:.1f} h left' + (f' (≈ \${left * s / 3600 * price:.0f})' if price else ''))
" 2>/dev/null;;
  serve)
    vm; NAME=${1:?run name}; STEP=${2:-last}; RPORT=${RPORT:-8900}
    vssh "tmux kill-session -t serve_$NAME 2>/dev/null; tmux new-session -d -s serve_$NAME \"NAME=$NAME STEP=$STEP PORT=$RPORT GUIDANCE=${GUIDANCE:-} GUIDANCE_NEG=${GUIDANCE_NEG:-} bash ~/il-robot/kit/vm_serve.sh > ~/il-robot/out/$NAME/serve.log 2>&1\"
          for i in \$(seq 1 120); do ss -ltn | grep -q '127.0.0.1:$RPORT ' && break; tmux has-session -t serve_$NAME 2>/dev/null || break; sleep 2; done
          ss -ltn | grep -q '127.0.0.1:$RPORT ' && { echo 'server up:'; grep '\[server\]' ~/il-robot/out/$NAME/serve.log | tail -n 3 | cut -c1-260; } || { echo 'the server did not come up:'; tail -n 15 ~/il-robot/out/$NAME/serve.log; exit 5; }" || exit $?
    TF=${UBE_CONFIG_DIR:-$HOME/.config/ube}/policy.$NAME.token
    ( umask 077; scp -q "${SSHO[@]}" -o BatchMode=yes -P "$PORT" "$DEST:il-robot/policy.token" "$TF.tmp" ) && chmod 600 "$TF.tmp" && mv "$TF.tmp" "$TF" && echo "token → $TF (600)"
    echo "next: bash $0 tunnel $NAME";;
  tunnel)
    vm; NAME=${1:?run name}; LPORT=$(lport)
    NAME=$NAME KEY=$KEY SSH_PORT=$PORT bash "$R/inference/remote/tunnel.sh" up "$DEST" "${RPORT:-8900}" "$LPORT" || exit $?
    $PY "$R/inference/remote/endpoint.py" set "http://127.0.0.1:$LPORT" --name "$NAME"
    echo "next: cd $R && TAG=$NAME DRY=1 bash inference/live/remote_intake.sh $NAME";;
  fetch)
    vm; NAME=${1:?run name}; STEP=${2:-last}; L=${FETCH_TO:-$HOME/il-robot/runs/groot_ckpt}
    S=$(vssh "basename \$(readlink -f ~/il-robot/out/$NAME/run/checkpoints/$STEP)" 2>/dev/null)
    case "$S" in ""|*[!0-9]*) echo "no checkpoint '$STEP' of $NAME there"; exit 2;; esac
    RD=il-robot/out/$NAME/run/checkpoints/$S/pretrained_model; LD=$L/${NAME}_$S/pretrained_model
    NEED=$(vssh "du -sm ~/$RD | cut -f1"); HAVE=$(du -sm "$LD" 2>/dev/null | cut -f1); FREE=$(df -m --output=avail "$HOME/il-robot" | tail -n 1 | tr -d ' ')
    [ "$FREE" -gt $((NEED - ${HAVE:-0} + 15000)) ] || { echo "the checkpoint is $NEED MB (${HAVE:-0} MB here), this disk has $FREE MB free and 15 GB stay free — not fetched"; exit 2; }
    mkdir -p "$LD" || exit 2
    echo "fetching $NAME step $S ($NEED MB) → $L/${NAME}_$S"
    # the big files first, in parallel ranges (fetch_ranges.sh: one ssh stream brings 6 MB/s from eu-north1, ten bring 17),
    # each compared with the VM's sha256 and given the VM's time so that the rsync below passes over it; a file whose
    # checksum differs is left to that rsync, which repairs it (one stream)
    while read -r BIG BSZ BMT <&3; do
      [ -n "$BIG" ] || continue
      if [ "$(stat -c '%s %Y' "$LD/$BIG" 2>/dev/null)" = "$BSZ $BMT" ] && ! [ -e "$LD/$BIG.ranges" ]; then continue; fi
      bash "$R/train/nebius/fetch_ranges.sh" "$RD/$BIG" "$LD/$BIG" "${FETCH_STREAMS:-10}" || { echo "$BIG is incomplete — again: bash $0 fetch $NAME $STEP"; exit 3; }
      if [ "$(vssh "sha256sum ~/$RD/$BIG | cut -c1-64")" = "$(sha256sum "$LD/$BIG" | cut -c1-64)" ]; then touch -d "@$BMT" "$LD/$BIG"; echo "$BIG: sha256 as on the VM"
      else echo "$BIG: sha256 differs from the VM's — rsync repairs it (one stream)"; fi
    done 3< <(vssh "cd ~/$RD && find . -maxdepth 1 -type f -size +${FETCH_BIG_MB:-1024}M -printf '%f\n' | xargs -r stat -c '%n %s %Y'")
    rsync -a --partial --info=stats1 -e "ssh ${SSHO[*]} -o BatchMode=yes -p $PORT" "$DEST:$RD" "$L/${NAME}_$S/" || exit 3
    for f in run.json train.log serve.log gpu.csv; do vrsync "$DEST:il-robot/out/$NAME/$f" "$L/${NAME}_$S/" > /dev/null 2>&1; done
    echo "kept: $L/${NAME}_$S ($(du -sh "$L/${NAME}_$S" | cut -f1)) — on a new machine: bash $0 restore $NAME $S, then serve $NAME $S";;
  restore)
    vm; NAME=${1:?run name}; STEP=${2:?step, as in ~/il-robot/runs/groot_ckpt/NAME_STEP}; L=$HOME/il-robot/runs/groot_ckpt/${NAME}_$STEP
    [ -d "$L/pretrained_model" ] || { echo "no $L/pretrained_model"; exit 2; }
    vssh "mkdir -p ~/il-robot/out/$NAME/run/checkpoints/$STEP" || exit 3
    echo "sending $L ($(du -sh "$L/pretrained_model" | cut -f1)) …"
    rsync -a --partial --info=stats1 -e "ssh ${SSHO[*]} -o BatchMode=yes -p $PORT" "$L/pretrained_model" "$DEST:il-robot/out/$NAME/run/checkpoints/$STEP/" || exit 3
    [ -s "$L/run.json" ] && vrsync "$L/run.json" "$DEST:il-robot/out/$NAME/" > /dev/null 2>&1
    vssh "cd ~/il-robot/out/$NAME/run/checkpoints && ln -sfn $STEP last && echo \"there: \$(du -sh $STEP | cut -f1) — next: bash vm.sh serve $NAME $STEP\"";;
  prune)
    vm; NAME=${1:?run name}
    vrsync "$R/train/nebius/vm_prune.py" "$DEST:il-robot/kit/" > /dev/null || exit 3
    vssh "tmux kill-session -t prune_$NAME 2>/dev/null; ~/il-robot/.venv/bin/python ~/il-robot/kit/vm_prune.py $NAME --once
          tmux new-session -d -s prune_$NAME \"~/il-robot/.venv/bin/python ~/il-robot/kit/vm_prune.py $NAME >> ~/il-robot/out/$NAME/prune.log 2>&1\"
          du -sh ~/il-robot/out/$NAME/run/checkpoints/0* 2>/dev/null | tr '\n' ' '; echo; echo \"disk free: \$(df -h --output=avail ~ | tail -n 1 | tr -d ' ')\"";;
  stop) vm; vssh "tmux kill-session -t groot_${1:?run name} && echo stopped || echo 'not running'";;
  serve-stop) vm; vssh "tmux kill-session -t serve_${1:?run name} && echo stopped || echo 'not running'"; NAME=$1 bash "$R/inference/remote/tunnel.sh" down;;
  ssh) vm; if [ $# -gt 0 ]; then vssh "$@"; else ssh "${SSHO[@]}" -t -p "$PORT" "$DEST"; fi;;
  *) sed -n '2,34p' "$0";;
esac
