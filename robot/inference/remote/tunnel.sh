#!/usr/bin/env bash
# The SSH tunnel to a policy server that listens on its own 127.0.0.1 only — the recommended way to reach it: nothing
# is opened to the internet and the wire is encrypted. Runs as a user unit: it outlives this shell and comes back by
# itself after a network drop (a request that falls into the gap times out; the loop asks once more).
#   bash tunnel.sh up USER@HOST [REMOTE_PORT=8900] [LOCAL_PORT]    → http://127.0.0.1:LOCAL_PORT
#        LOCAL_PORT defaults to 18900 for the first account of this machine and +10 for each one after it (uid 1001 → 18910)
#   bash tunnel.sh status | down
#   env: KEY=~/.ssh/id_ed25519_policy   the private key whose .pub the server's owner put into authorized_keys
#        NAME=<endpoint name>           a second tunnel for a second server (unit ube-policy-tunnel-NAME; give another LOCAL_PORT)
#        SSH_PORT=22
# Without a systemd user session (a team account that came in over Tailscale SSH and has no lingering) the tunnel runs as
# a detached loop instead (pid in ~/.config/ube/tunnel[.NAME].pid, log beside it): same commands, same behaviour.
# The key is NOT made here (user, 10-09: only when the teammate asks for it). When that is asked:
#   bash tunnel.sh key          prints the one command that makes it and where the public half is
# After "up": python inference/remote/endpoint.py set http://127.0.0.1:LOCAL_PORT [--name NAME], then live/remote_intake.sh.
set -u
CMD=${1:-status}; KEY=${KEY:-$HOME/.ssh/id_ed25519_policy}; UNIT=ube-policy-tunnel${NAME:+-$NAME}
CFD=${UBE_CONFIG_DIR:-$HOME/.config/ube}; PIDF=$CFD/tunnel${NAME:+.$NAME}.pid; LOGF=$CFD/tunnel${NAME:+.$NAME}.log
DEF_LPORT=$((18900 + ($(id -u) - 1000) * 10))      # one port a team account (vm.sh lport)
has_sd() { [ "${TUNNEL_NO_SYSTEMD:-0}" != 1 ] && systemctl --user show-environment > /dev/null 2>&1; }
loop_up() { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }
case "$CMD" in
  key)
    if [ -s "$KEY.pub" ]; then echo "the key exists — the public half to hand over:"; cat "$KEY.pub"
    else echo "no key yet. To make one (no passphrase: the tunnel has to come back unattended; it is used for this tunnel only):"
         echo "  ssh-keygen -t ed25519 -N \"\" -C \"lenovo-robot policy tunnel\" -f $KEY"
         echo "then send $KEY.pub (the PUBLIC half only) to the server's owner."; fi;;
  up)
    DEST=${2:?USER@HOST}; RPORT=${3:-8900}; LPORT=${4:-$DEF_LPORT}
    case "$DEST" in *@*) ;; *) echo "USER@HOST"; exit 2;; esac
    case "$RPORT$LPORT" in *[!0-9]*) echo "ports are numbers"; exit 2;; esac
    [ -s "$KEY" ] || { echo "no key at $KEY — bash $0 key"; exit 2; }
    SSHCMD=(ssh -N -p "${SSH_PORT:-22}" -i "$KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o ExitOnForwardFailure=yes
            -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o StrictHostKeyChecking=accept-new
            -L "127.0.0.1:$LPORT:127.0.0.1:$RPORT" "$DEST")
    ss -ltn | grep -q "127.0.0.1:$LPORT " && { echo "port $LPORT on this machine is in use — another LOCAL_PORT"; exit 2; }
    if has_sd; then
      systemctl --user is-active --quiet "$UNIT" && { echo "$UNIT is already up — bash $0 down first"; exit 2; }
      systemd-run --user --unit "$UNIT" --collect --quiet -p Restart=always -p RestartSec=5 "${SSHCMD[@]}" || exit 3
      HOW="unit $UNIT"
    else
      loop_up && { echo "the tunnel loop is already up (pid $(cat "$PIDF")) — bash $0 down first"; exit 2; }
      mkdir -p "$CFD" && chmod 700 "$CFD"
      # detached, in its own session: it outlives this shell and the login; ssh again 5 s after it ends
      setsid nohup bash -c 'while :; do "$@"; sleep 5; done' tunnel-loop "${SSHCMD[@]}" > "$LOGF" 2>&1 < /dev/null &
      echo $! > "$PIDF"; HOW="detached loop, pid $(cat "$PIDF")"
    fi
    for _ in $(seq 1 30); do ss -ltn | grep -q "127.0.0.1:$LPORT " && break; sleep 0.5; done
    if ss -ltn | grep -q "127.0.0.1:$LPORT "; then
      echo "tunnel up: http://127.0.0.1:$LPORT → $DEST 127.0.0.1:$RPORT ($HOW)"
      echo "next: python inference/remote/endpoint.py set http://127.0.0.1:$LPORT${NAME:+ --name $NAME}"
    else
      echo "the tunnel did not come up:"
      if has_sd; then journalctl --user -u "$UNIT" -n 5 --no-pager 2>/dev/null | cut -c1-200; systemctl --user stop "$UNIT" 2>/dev/null
      else tail -n 5 "$LOGF" 2>/dev/null | cut -c1-200; loop_up && kill -- "-$(ps -o pgid= -p "$(cat "$PIDF")" | tr -d ' ')" 2>/dev/null; rm -f "$PIDF"; fi
      exit 3
    fi;;
  down)
    if has_sd && systemctl --user is-active --quiet "$UNIT"; then systemctl --user stop "$UNIT" && echo "$UNIT stopped"
    elif loop_up; then kill -- "-$(ps -o pgid= -p "$(cat "$PIDF")" | tr -d ' ')" 2>/dev/null; rm -f "$PIDF"; echo "tunnel loop stopped"
    else echo "the tunnel was not running"; fi;;
  status)
    if has_sd && systemctl --user is-active --quiet "$UNIT"; then
      echo "$UNIT: up since $(systemctl --user show "$UNIT" -p ActiveEnterTimestamp --value), restarts $(systemctl --user show "$UNIT" -p NRestarts --value)"
      systemctl --user show "$UNIT" -p ExecStart --value | grep -o "\-L [^ ]*" | head -n 1
    elif loop_up; then echo "tunnel loop: up (pid $(cat "$PIDF")); $(ps -o args= --ppid "$(cat "$PIDF")" 2>/dev/null | grep -o "\-L [^ ]*" | head -n 1)"
    else echo "the tunnel is not running"; fi;;
  *) echo "usage: bash $0 up USER@HOST [REMOTE_PORT] [LOCAL_PORT] | status | down | key"; exit 2;;
esac
