#!/usr/bin/env bash
# One big file from the VM in parallel byte ranges (2026-10-12).
#   bash fetch_ranges.sh REMOTE_FILE LOCAL_FILE [STREAMS=10]        the machine and the key as vm.sh has them
# A single ssh stream over this distance (Seoul ↔ eu-north1) brought 6 MB/s — 35 minutes for the 12.6 GB of a GR00T
# checkpoint, with the VM billed meanwhile; ten streams brought 17 MB/s (10-12: 11 GB in 12 minutes). Every range is written
# into its place in the local file. What is already there is kept: a partial file rsync left (its length, less 16 MiB), or
# the ranges an earlier call of this script finished (LOCAL_FILE.ranges — it is there only while the file is incomplete).
# This script does not check the content: the caller compares a checksum of the whole file (vm.sh fetch: sha256).
# Exit: 0 all ranges there, 2 cannot start, 3 a stream failed (call again: only the missing ranges are fetched).
set -u
RF=${1:?remote file, relative to the home of the VM user or absolute}; LF=${2:?local file}; N=${3:-10}
CF=${UBE_CONFIG_DIR:-$HOME/.config/ube}/nebius_vm; KEY=${KEY:-$HOME/.ssh/id_ed25519_policy}
[ -s "$CF" ] || { echo "no machine yet — bash vm.sh set USER@HOST"; exit 2; }
read -r DEST PORT < "$CF"; PORT=${PORT:-22}
SSHO=(-i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p "$PORT")
SIZE=$(ssh "${SSHO[@]}" "$DEST" "stat -c %s '$RF'" 2>/dev/null)
case "$SIZE" in ""|*[!0-9]*) echo "no file $RF on $DEST"; exit 2;; esac
TOTAL_MB=$(( (SIZE + 1048575) / 1048576 )); PF=$LF.ranges
mkdir -p "$(dirname "$LF")" || exit 2
if [ -s "$PF" ] && [ "$(head -n 1 "$PF" | cut -d' ' -f5)" = "$SIZE" ] && [ -f "$LF" ]; then
  read -r _ START_MB PER N _ < "$PF"                         # an earlier call's plan: go on with it
else
  HAVE=0; [ -f "$LF" ] && HAVE=$(stat -c %s "$LF")
  [ "$HAVE" -gt "$SIZE" ] && { echo "$LF is longer ($HAVE) than the remote file ($SIZE) — another file? not touched"; exit 2; }
  [ "$HAVE" = "$SIZE" ] && { echo "$LF has the full length already — nothing fetched"; exit 0; }
  START_MB=$(( HAVE / 1048576 - 16 )); [ "$START_MB" -lt 0 ] && START_MB=0
  PER=$(( (TOTAL_MB - START_MB + N - 1) / N )); [ "$PER" -lt 1 ] && PER=1
  echo "plan $START_MB $PER $N $SIZE" > "$PF"
fi
truncate -s "$SIZE" "$LF" || exit 2
echo "$(date +%T) $(basename "$LF"): $SIZE bytes, MiB $START_MB … $TOTAL_MB in $N streams of $PER MiB"
for i in $(seq 0 $((N - 1))); do
  S=$(( START_MB + i * PER )); C=$PER; [ $(( S + C )) -gt "$TOTAL_MB" ] && C=$(( TOTAL_MB - S )); [ "$C" -le 0 ] && continue
  grep -qx "done $i" "$PF" && continue
  ( ssh "${SSHO[@]}" "$DEST" "dd if='$RF' bs=1M skip=$S count=$C status=none" | dd of="$LF" bs=1M seek="$S" conv=notrunc status=none
    rc="${PIPESTATUS[*]}"; [ "$rc" = "0 0" ] && echo "done $i" >> "$PF" || echo "$(date +%T) stream $i (MiB $S + $C) failed: rc $rc" ) &
done
wait
MISSING=0
for i in $(seq 0 $((N - 1))); do
  [ $(( START_MB + i * PER )) -ge "$TOTAL_MB" ] && continue
  grep -qx "done $i" "$PF" || MISSING=$((MISSING + 1))
done
[ "$MISSING" = 0 ] || { echo "$(date +%T) $MISSING of $N ranges missing — call again"; exit 3; }
rm -f "$PF"; echo "$(date +%T) all ranges there ($(stat -c %s "$LF") bytes)"
