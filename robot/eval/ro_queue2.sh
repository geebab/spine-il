#!/usr/bin/env bash
# Robot-origin session queue (v2): one line of ro_jobs.txt at a time ("TAG REC FROM LEN CFG [ANCHOR] [ENV=V,...]"), removed
# when taken. A recording never overlaps a render (the headless recorder shares the GPU — concurrent renders stalled its
# drill loop 0.5–1.1 s): until 10-09 the recording waited for the render to end, since then render work is paused for it
# (render_gate.py, see below). Each session's IoU vs the hand (ro_iou.py) runs synchronously after it. Stops on RO_STOP2;
# idles when the list is empty.
# 10-08: it also waits for render_video.py and policy_eval.py themselves — a video is composed and encoded (ffmpeg, minutes
# of CPU) after its render job has left the API, and a run started in that gap (08:24, five minutes of encoding beside it).
# 10-08 (2): the root filesystem ran full during a run (16:05). The arm followers died writing their logs, the next runs
# died at their start, the run script's end step (home_off.py) could not even start and the right arm stood for three hours
# in HOLD with the torque on. Hence: no job is taken with less than RO_MIN_FREE_GB free (default 6; a run writes 0.4-0.6 GB),
# and after every run the queue itself looks whether both arms are parked (state OFF) — it parks an arm that is not, once,
# and stops (RO_STOP2) if that does not help.
S=$HOME/il-robot/runs/jitter_iter/work; J=$HOME/il-robot/runs/jitter_iter; PY=$HOME/il-robot/.venv/bin/python
MIN_FREE_GB=${RO_MIN_FREE_GB:-6}
G="$PY ${SPINE_IL_REPO:-$HOME/spine-il}/robot/eval/render_gate.py"
free_gb() { df --output=avail -BG / | tail -1 | tr -dc 0-9; }
arm_state() { $PY - "$1" <<'PY' 2>/dev/null
import sys
from lerobot_robot_spinearm.client import ArmRpc
from lerobot_robot_spinearm.protocol import DEFAULT_PORTS
print(ArmRpc("127.0.0.1",DEFAULT_PORTS[sys.argv[1]][0]).call(cmd="state").get("state"))
PY
}
temp() { $PY - "$1" <<'PY' 2>/dev/null
import sys
from lerobot_robot_spinearm.client import ArmRpc
from lerobot_robot_spinearm.protocol import DEFAULT_PORTS
st=ArmRpc("127.0.0.1",DEFAULT_PORTS[sys.argv[1]][0]).call(cmd="state"); print(int(max((st.get("temp") or {}).values() or [0])))
PY
}
while [ ! -e $S/RO_STOP2 ]; do
  L=$(grep -v '^#' $S/ro_jobs.txt 2>/dev/null | grep -m1 .)
  [ -z "$L" ] && { sleep 30; continue; }
  n=0
  until [ "$(free_gb)" -ge $MIN_FREE_GB ] || [ -e $S/RO_STOP2 ]; do
    [ $n = 0 ] && echo "[roq] $(date +%T) disk nearly full ($(free_gb) GB free, $MIN_FREE_GB needed) — the job stays listed, waiting"; n=1; sleep 60
  done
  [ -e $S/RO_STOP2 ] && break
  python3 - "$L" <<'PY'
import sys; from pathlib import Path
p = (Path.home() / "il-robot/runs/jitter_iter/work/ro_jobs.txt"); ls = p.read_text().splitlines()
ls.remove(sys.argv[1]); p.write_text("\n".join(ls) + ("\n" if ls else ""))
PY
  set -- $L; TAG=$1; REC=$2; FROM=$3; LEN=$4; CFG=$5; ANCHOR=${6:-drill}; ENVX=${7:-}
  for x in right left; do while :; do T=$(temp $x); [ "${T:-99}" -le 50 ] && break; echo "[roq] $x ${T} °C — cooling"; sleep 30; done; done
  # 10-09 (user: "외부에서 로봇 런 시도가 들어오면 렌더 pause하도록"): the run no longer waits for render work to end — render
  # work pauses for the run (render_gate.py: the service's job waits between two frames, the clients and their encoders
  # are stopped) and goes on when the arms are parked: robot_origin_run.sh lifts the hold there (RO_RENDER_HOLD), and the
  # queue lifts it again below, whatever became of the run. "At rest" takes about a second; a job in its setup or its
  # packaging gets there within a minute. Without the watcher nothing pauses, and the old rule holds: wait until the
  # render API is idle and no client runs.
  $G hold roq --pid $$ --reason "robot job $TAG" --wait 5 > $S/.gate_settled 2>&1
  n=0
  until $G settled > $S/.gate_settled 2>&1 || { (cd $S && $PY render_idle.py) && ! pgrep -f "[r]o_iou\.py|[i]ou_eval\.py|[r]ender_video\.py|[p]olicy_eval\.py" >/dev/null; }; do
    [ $n = 0 ] && echo "[roq] $(date +%T) waiting for render work to come to rest ($(cat $S/.gate_settled))"; n=1; sleep 2
  done
  echo "[roq] $(date +%T) job: $L   (render: $(cat $S/.gate_settled))"
  env ${ENVX//,/ } RO_RENDER_HOLD=roq bash $S/robot_origin_run.sh $TAG $REC $FROM $LEN $CFG $ANCHOR
  $G release roq
  for x in right left; do
    st=$(arm_state $x); [ "$st" = OFF ] && continue
    echo "[roq] $(date +%T) the $x arm is '$st' after the run, not parked — parking it once more"
    $PY $S/home_off.py $x 2>&1 | tail -n 1 | sed 's/^/[roq] /'
    st=$(arm_state $x)
    [ "$st" = OFF ] || { echo "[roq] $(date +%T) the $x arm is still '$st' — the queue stops here (RO_STOP2). By hand: $PY $S/home_off.py $x"; touch $S/RO_STOP2; }
  done
  RUN=$(ls -td $HOME/il-robot/runs/robot_origin/*_replay_${TAG} 2>/dev/null | head -1)
  SID=$(tail -1 $RUN/result.json 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('sessionId') or '')" 2>/dev/null)
  SF=$(find $HOME/il-robot/viewer-data/sessions/robot -name "$SID.jsonl" 2>/dev/null | head -1)
  REC4=$(echo $REC | grep -o '_trk_[0-9a-f]*' | cut -c6-)
  HT=$(ls $J/pd_${REC4}*_hand.json 2>/dev/null | head -1 | xargs -r basename | sed 's/_hand.json//')
  if [ -n "$SF" ] && [ -n "$HT" ]; then
    (cd $S && OPENBLAS_NUM_THREADS=2 nice $PY -u ro_iou.py $SF $HT --tag $TAG --resim-tick-hz $(python3 -c "import json;print(round(15*float(json.load(open('$RUN/settings.json'))['speed']),3))" 2>/dev/null || echo 4.95) 2>&1 | grep -v Warning | tail -1 | sed "s/^/[ro-iou] $TAG /")
  fi
done
echo "[roq] RO_STOP2 — ending"
