#!/usr/bin/env bash
# deep_1018 P3 (PREREG v3; user 10-19 "p3 준비하고, vm 필요할때 이야기해" — runs only after the user's go): wait for the VM, serve
# H90g 020000, tunnel, then 15 robot runs: C1006 starts r1 · r2 · r4 × (novel contra ×2, novel ipsi ×2, neutral ×1), end level
# 0.85, otherwise as R_H90g_pf_* (10-17). Fixed random order (seed 101901). No run starts in the sunny-morning tracker windows.
# After the last run: "VM no longer needed". Then stream_report per run, evaluations (one at a time, render gate), p3_analysis.py.
#   setsid nohup bash chain_p3_1019.sh >> chain_p3_1019.log 2>&1 < /dev/null &
set -u
R=${SPINE_IL_REPO:-$HOME/spine-il}/robot; V=$R/train/nebius/vm.sh; PY=$HOME/il-robot/.venv/bin/python; G=$R/eval/render_gate.py
LV=$HOME/il-robot/runs/act_smoke/live; D=$HOME/il-robot/runs/deep_1018; PF=$HOME/il-robot/runs/groot_prep/proof_1016; M=H90g
# a repeat after an abort (user's go) runs with P3_ORDER="<start>:<group><k>b …" (PREREG v3)
ORDER=${P3_ORDER:-"r4:neutral1 r2:nipsi1 r2:neutral1 r1:ncontra1 r1:neutral1 r2:ncontra1 r4:ncontra2 r1:nipsi1 r4:nipsi1 r4:nipsi2 r1:nipsi2 r2:ncontra2 r4:ncontra1 r2:nipsi2 r1:ncontra2"}
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
  while :; do h=$(date +%H%M); { [ $h -ge 0825 ] && [ $h -lt 0920 ]; } || { [ $h -ge 1010 ] && [ $h -lt 1050 ]; } || break; sleep 60; done
  ST=${IT%%:*}; T=${IT#*:}; A=${T%%[0-9]*}; X=""
  case $A in nipsi) GO=ipsilateral; X=" --goal-text-file $PF/novel_ipsi.txt";;
             ncontra) GO=contralateral; X=" --goal-text-file $PF/novel_contra.txt";;
             neutral) GO=unspecified;; esac
  NR=R_${M}_p3_${T}_C1006_$ST
  [ -e $NR/live.json ] && { say "$NR exists — skipped"; continue; }
  [ "$(df --output=avail -BG / | tail -1 | tr -dc 0-9)" -ge 8 ] || { say "disk low — protocol stopped before $NR"; break; }
  LIVE_GOAL=L4:$GO LIVE_END_LEVEL=0.85 LIVE_EXTRA="--stream-rcm-first$X" LIVE_POLICY_TOKEN_FILE=$HOME/.config/ube/policy.$M.token LIVE_EVAL=0 LIVE_CONTINUOUS=1 bash run_remote_base.sh C1006 "$(cat $HOME/.config/ube/policy.$M.url)" ${M}_p3_$T $ST
  [ -s $NR/live.json ] || { say "$NR: no record — protocol stopped"; break; }
  WANT=$(case $A in nipsi) cat $PF/novel_ipsi.txt;; ncontra) cat $PF/novel_contra.txt;; *) $PY -c "import sys;sys.path.insert(0,'$R/dataset');import instructions as I;print(I.text('lang3',4,'L4','$GO'))";; esac)
  grep -qF "drilling '$WANT'" $NR/live.log || { say "$NR: the sentence '$WANT' was not sent — protocol stopped"; break; }
  $PY -c 'import json,sys; m=json.load(open(sys.argv[1]))["meta"]; print(sys.argv[2], "abort", m.get("abort"), "| end", m.get("drill_end"), "| level", m.get("end_level"), "| mm3", (m.get("removed_mm3") or [0])[-1], "| drill_s", m.get("drill_s"), "| clamped", m.get("clamped_frames")); sys.exit(3 if m.get("abort") else 0)' $NR/live.json $NR || { say "$NR aborted — protocol stopped"; break; }
  $PY $R/inference/live/stream_report.py $NR/live.json > $NR/stream_report.txt 2>&1; tail -n 3 $NR/stream_report.txt | sed "s/^/  /"; n=$((n+1))
done
[ $n -gt 0 ] && say "protocol done ($n runs) — VM no longer needed" || { say "protocol made no run — stopped (VM state unchanged)"; exit 5; }
for NR in $(ls -d R_${M}_p3_*_C1006_r[0-9] 2>/dev/null); do
  [ -s $NR/live.json ] || continue
  sid=$(basename "$(cat $NR/robot_session)" .jsonl); [ -s $HOME/il-robot/runs/jitter_iter/iou_cache_rofull_$sid.npz ] && continue
  say "$NR: evaluation"; $PY $G run --low --mem-gb 2.5 -- bash $LV/eval_later.sh $NR > /dev/null 2>&1
done
cd $R && $PY eval/deep_1018/p3_analysis.py $D > /dev/null 2>&1; say "p3 analysis rc=$?"
say end
