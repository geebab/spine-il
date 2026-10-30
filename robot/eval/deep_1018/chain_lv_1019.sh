#!/usr/bin/env bash
# deep_1018 L3 / L5 series (PREREG v4; user 10-19 "l3,5도 4와 동일한 조건, 횟수만큼 추론시켜서 다 포함시킨 결과가 필요하다",
# "레노보와 vm은 로봇을 돌리고 평가는 맥세션에서" — runs only after the user's go): wait for the VM, serve H90g 020000, tunnel, then
# 59 robot runs so that L3 and L5 each match L4's 35 (the 10-17 L3 / L5 runs count as start A's trained runs):
#   L3: start rL3_0 trained contra/ipsi to 5 each (+2/+2), novel contra/ipsi ×5; starts rL3_1 · rL3_2 · rL3_3 × (novel contra ×2,
#       novel ipsi ×2, neutral ×1)
#   L5: start rL5_2 trained contra/ipsi to 5 each (+3/+2), novel ×5 each; starts rL5_3 · rL5_6 · rL5_7 × the same five (rL5_0 · rL5_1 failed prepare, PREREG v4.1)
# end level 0.85, rcm-first, every other setting as R_H90g_pf_* / R_H90g_L3* (10-17). Fixed random order (seed 101902). No run starts
# in the sunny-morning tracker windows. After the last run: "VM no longer needed". No evaluation here (the Mac evaluates:
# eval_mac_1019.sh); this chain only writes stream_report per run.
#   setsid nohup bash chain_lv_1019.sh >> chain_lv_1019.log 2>&1 < /dev/null &
set -u
R=${SPINE_IL_REPO:-$HOME/spine-il}/robot; V=$R/train/nebius/vm.sh; PY=$HOME/il-robot/.venv/bin/python
LV=$HOME/il-robot/runs/act_smoke/live; D=$HOME/il-robot/runs/deep_1018; M=H90g
# a repeat after an abort (user's go) runs with LV_ORDER="<start>:<L><group><k>b …" (PREREG v4)
ORDER=${LV_ORDER:-"rL5_6:L5ncontra2 rL3_1:L3nipsi1 rL5_3:L5ncontra2 rL3_1:L3ncontra2 rL3_0:L3nipsi1 rL5_2:L5ncontra3 rL3_0:L3contra4 rL5_7:L5ncontra2 rL5_2:L5ncontra5 rL3_0:L3ncontra4 rL5_6:L5nipsi2 rL3_1:L3nipsi2 rL3_0:L3nipsi2 rL5_6:L5nipsi1 rL3_2:L3neutral1 rL3_0:L3ipsi4 rL5_2:L5contra3 rL5_2:L5contra4 rL5_2:L5nipsi3 rL3_1:L3ncontra1 rL3_3:L3neutral1 rL3_0:L3ipsi5 rL3_2:L3nipsi1 rL5_2:L5nipsi5 rL5_6:L5neutral1 rL3_0:L3ncontra2 rL5_2:L5ipsi4 rL3_3:L3nipsi1 rL5_3:L5ncontra1 rL3_0:L3ncontra3 rL3_0:L3nipsi3 rL5_2:L5nipsi2 rL5_7:L5nipsi1 rL5_2:L5contra5 rL3_1:L3neutral1 rL5_2:L5ncontra1 rL5_2:L5ncontra2 rL5_7:L5nipsi2 rL3_0:L3nipsi4 rL3_2:L3nipsi2 rL5_2:L5ncontra4 rL3_3:L3ncontra1 rL5_3:L5nipsi2 rL5_7:L5neutral1 rL5_3:L5neutral1 rL3_0:L3contra5 rL3_3:L3ncontra2 rL5_2:L5ipsi5 rL3_0:L3nipsi5 rL3_3:L3nipsi2 rL3_0:L3ncontra1 rL5_2:L5nipsi1 rL5_3:L5nipsi1 rL5_6:L5ncontra1 rL3_0:L3ncontra5 rL3_2:L3ncontra2 rL5_2:L5nipsi4 rL5_7:L5ncontra1 rL3_2:L3ncontra1"}
say() { echo "$(date '+%m-%d %T') $*"; }
while :; do
  read -r DEST PORT < $HOME/.config/ube/nebius_vm; H=${DEST#*@}
  timeout 8 bash -c "</dev/tcp/$H/${PORT:-22}" 2>/dev/null && break; sleep 120
done
say "VM $H answers"; sleep 60
ok=0
for try in 1 2 3 4 5; do
  timeout 8 bash -c "</dev/tcp/$H/${PORT:-22}" 2>/dev/null || { say "serve try $try: VM not answering"; sleep 90; continue; }
  bash $V serve $M 020000 2>&1 | tail -n 2
  NAME=$M bash $R/inference/remote/tunnel.sh down > /dev/null 2>&1
  bash $V tunnel $M 2>&1 | tail -n 1
  curl -s -m 10 "$(cat $HOME/.config/ube/policy.$M.url)/health" | grep -q ok && { ok=1; break; }
  say "serve try $try: no /health answer"; sleep 60
done
[ $ok = 1 ] || { say "policy server not reachable after 5 tries — no runs made, stopped (VM state unchanged)"; exit 4; }
say "policy server up: $(cat $HOME/.config/ube/policy.$M.url)"
cd $LV; n=0
for IT in $ORDER; do
  # user 10-20: "절기가 바뀌어서 햇빛이 안드는데, 햇빛 때문에 멈추지 말고 계속해" -> SUN_WAIT=0 skips the sunny-morning windows
  while [ "${SUN_WAIT:-1}" = 1 ]; do h=$(date +%H%M); { [ $h -ge 0825 ] && [ $h -lt 0920 ]; } || { [ $h -ge 1010 ] && [ $h -lt 1050 ]; } || break; sleep 60; done
  ST=${IT%%:*}; T=${IT#*:}; L=${T:0:2}; G=${T:2}; A=${G%%[0-9]*}; X=""
  case $A in ncontra) GO=contralateral; X=" --goal-text-file $D/novel_contra_$L.txt";;
             nipsi) GO=ipsilateral; X=" --goal-text-file $D/novel_ipsi_$L.txt";;
             contra) GO=contralateral;; ipsi) GO=ipsilateral;; neutral) GO=unspecified;;
             *) say "$IT: unknown group — skipped"; continue;; esac
  NR=R_${M}_${T}_C1006_$ST
  [ -e $NR/live.json ] && { say "$NR exists — skipped"; continue; }
  [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 8 ] || { say "disk low — protocol stopped before $NR"; break; }
  LIVE_GOAL=$L:$GO LIVE_END_LEVEL=0.85 LIVE_EXTRA="--stream-rcm-first$X" LIVE_POLICY_TOKEN_FILE=$HOME/.config/ube/policy.$M.token LIVE_EVAL=0 LIVE_CONTINUOUS=1 bash run_remote_base.sh C1006 "$(cat $HOME/.config/ube/policy.$M.url)" ${M}_$T $ST
  [ -s $NR/live.json ] || { say "$NR: no record — protocol stopped"; break; }
  WANT=$(case $A in nipsi) cat $D/novel_ipsi_$L.txt;; ncontra) cat $D/novel_contra_$L.txt;; *) $PY -c "import sys;sys.path.insert(0,'$R/dataset');import instructions as I;print(I.text('lang3',4,'$L','$GO'))";; esac)
  grep -qF "drilling '$WANT'" $NR/live.log || { say "$NR: the sentence '$WANT' was not sent — protocol stopped"; break; }
  $PY -c 'import json,sys; m=json.load(open(sys.argv[1]))["meta"]; print(sys.argv[2], "abort", m.get("abort"), "| end", m.get("drill_end"), "| level", m.get("end_level"), "| mm3", (m.get("removed_mm3") or [0])[-1], "| drill_s", m.get("drill_s"), "| clamped", m.get("clamped_frames")); sys.exit(3 if m.get("abort") else 0)' $NR/live.json $NR || { say "$NR aborted — protocol stopped"; break; }
  $PY $R/inference/live/stream_report.py $NR/live.json > $NR/stream_report.txt 2>&1; tail -n 3 $NR/stream_report.txt | sed "s/^/  /"; n=$((n+1))
  echo $NR >> $D/lv_runs_done.txt
done
[ $n -gt 0 ] && say "protocol done ($n runs) — VM no longer needed" || { say "protocol made no run — stopped (VM state unchanged)"; exit 5; }
say end
