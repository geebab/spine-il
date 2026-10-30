#!/usr/bin/env bash
# Register arm-server@left/right as systemd --user services (no sudo). Re-run to update.
#   bash systemd/install-arm-server.sh            # install + start both arms
#   bash systemd/install-arm-server.sh uninstall
# Logs: /srv/il-robot/logs/arm-<arm>.log when that directory exists (team-readable), else journal only.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"                 # …/spine-il
VENV="${VENV:-$HOME/il-robot/.venv}"
UNIT_DIR="$HOME/.config/systemd/user"
UNIT="arm-server@.service"

if [ "${1:-}" = "uninstall" ]; then
  systemctl --user disable --now arm-server@left arm-server@right 2>/dev/null || true
  rm -f "$UNIT_DIR/$UNIT"; systemctl --user daemon-reload; echo "removed"; exit 0
fi

[ -x "$VENV/bin/spine-arm-server" ] || { echo "spine-arm-server not found in $VENV — uv pip install -e robot/lerobot_robot_spinearm"; exit 1; }
if [ -d /srv/il-robot/logs ] && [ -w /srv/il-robot/logs ]; then
  LOGARG='--log /srv/il-robot/logs/arm-%i.log'; RW='ReadWritePaths=/srv/il-robot/logs'
else
  LOGARG=''; RW='# (no /srv/il-robot/logs yet — journal only: journalctl --user -u arm-server@left)'
fi

mkdir -p "$UNIT_DIR"
sed -e "s|__VENV__|$VENV|g" -e "s|__REPO__|$REPO|g" -e "s|__LOGARG__|$LOGARG|g" -e "s|__READWRITE__|$RW|g" \
  "$HERE/$UNIT" > "$UNIT_DIR/$UNIT"
systemctl --user daemon-reload
systemctl --user enable --now arm-server@left arm-server@right
sleep 1
systemctl --user --no-pager status arm-server@left arm-server@right | grep -E "arm-server@|Active:" || true
echo "linger: $(loginctl show-user "$USER" -p Linger 2>/dev/null) — 'sudo loginctl enable-linger $USER' keeps them running without a login"
