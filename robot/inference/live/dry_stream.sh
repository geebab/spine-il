#!/usr/bin/env bash
# Dry run of the CONTINUOUS loop (live_loop.py --continuous, CONTINUOUS.md): simulated arms (fake_arms.py --ik), a stand-in
# page (page_obs.py --fake) and the reference server's demo backend as the policy — no robot, no viewer, no render API, no GPU.
#   TAG=name [CONT=1] [PERIOD=8] [LEAD=10] [BLEND=4] [DELAY_MS=600] [EP=6] bash dry_stream.sh
#   CONT=0     the stop-and-go loop on the same stand-ins (16 steps a chunk) — the comparison
#   DELAY_MS   the stand-in server answers this much later (network + model of a far-away GPU)
#   URL        a REAL policy server instead of the stand-in (TOK = its token file): the loop's timing with the real model
#              and the real network — look → targets ready against the lead. The stand-in page shows a flat colour, so
#              what the model answers means nothing here; whether its answers arrive, pass and are sent in time does.
#   REL=1      the stand-in policy answers as a relative-action model does (the state it was given + the demonstration's
#              displacement) — without it a loop that echoes the arms' error cannot be seen (10-12: the first continuous
#              robot run swung, every dry run before it had been smooth)
#   BUMP=0.3:0.2:1.2 WANDER=1.0   the stand-in arms answer as the real ones were measured to (fake_arms.py --bump / --wander-mm)
#   ANCHOR=command|measured [GAMMA=0.5]   live_loop.py --stream-anchor / --stream-gamma (measured = the run of 10-12)
#   FAKE_EXTRA="--stamp-lag 70:100:900"   more fake_arms.py options (here: the tracker's stamps 0.9 s behind the clock from
#              70 to 100 s after the arms' start — the stream must end and the run go on stop-and-go)
#   EXTRA="…"  more live_loop.py options, after the script's own (the last one of a kind counts). To compare how two settings
#              MOVE over the same length: EXTRA="--end-level 2 --drill-pause-s 600 --drill-stall-s 600" (nothing ends the
#              drilling but STEPS) — with REL=1 the stand-in's progress runs ahead of the tools and ends a run early
# What it shows: windows follow each other (no hold between them), late answers cost frames and not the run, the progress
# ends the drilling, the exit is scripted. What it cannot show: the image (a flat colour) and the bone (the stand-in page
# removes a fixed amount per look) — those need the recorder page, i.e. the robot.
# Start it through the gate:  python eval/render_gate.py run --on-pause kill --low --mem-gb 2.5 -- bash inference/live/dry_stream.sh
A=$HOME/il-robot/runs/act_smoke; PY=$HOME/il-robot/.venv/bin/python; R=${SPINE_IL_REPO:-$HOME/spine-il}/robot
P=$HOME/il-robot/runs/jitter_iter/work/pairstart; D=$HOME/il-robot/datasets
TAG=${TAG:-dry_stream_$(date +%H%M%S)}; O=$A/dry_stream/$TAG; EP=${EP:-6}; PORT=${PORT:-8913}; OPORT=${OPORT:-18793}
TOK=${TOK:-$HOME/.config/ube/policy_local_test.token}
[ -e $O ] && { echo "$O exists — another TAG"; exit 2; }
[ -n "${URL:-}" ] && TOK=${TOK_REAL:-${TOK}}
[ -s "$TOK" ] || { echo "no token file $TOK"; exit 2; }
mkdir -p $O/lat && cd $R || exit 2
COMMON="--dataset $D/spine_hand24_replay_ph_trim_water --episode 6 --start-frame 0 --gate-anchor-s 17 --gate-pair A0928 --gate-pairs-file $P/pair_clear_3b3b.json --gate-registration $P/registration_A0928.json --gate-anchor-state $P/base_A0928.json --portal-from run --start-state $A/iter/starts/start_A0928rand_0.json"
echo "$(date +%T) $TAG: continuous ${CONT:-1}, period ${PERIOD:-8}, lead ${LEAD:-10}, anchor ${ANCHOR:-command}, gamma ${GAMMA:-0.5}, blend ${BLEND:-4}, server delay ${DELAY_MS:-600} ms, relative answers ${REL:-0}, arms bump ${BUMP:--} wander ${WANDER:--}" > $O/log
OMP_NUM_THREADS=2 nice taskset -c ${CORES:-12-15} $PY -u inference/live_loop.py --prepare-out $O/seed.json $COMMON --cycle > $O/prepare.log 2>&1 || { echo "prepare failed" >> $O/log; exit 1; }
ENTRY=$($PY -c "import re,sys; t=open(sys.argv[1]).read(); m=re.search(r'endo \[([-0-9., ]+)\]', t); print(m.group(1).replace(',', ' ') if m else '54.85 -1.7 1043.7')" $O/prepare.log)
SP=
if [ -z "${URL:-}" ]; then
  nice taskset -c ${CORES:-12-15} $PY -u inference/remote/policy_server.py --backend demo --dataset $D/spine_hand24_replay_trim_water_progress --episodes $EP \
    --progress-output --horizon 40 --token-file $TOK --port $PORT --delay-ms ${DELAY_MS:-600} ${REL:+--relative} > $O/server.log 2>&1 &
  SP=$!; URL=http://127.0.0.1:$PORT
  for i in $(seq 1 60); do ss -ltn | grep -q ":$PORT " && break; sleep 0.5; done
fi
$PY -u page_obs.py --fake $OPORT --medium water --entry $ENTRY --skip ${SKIP:-14} --cuts ${CUTS:-400} --cut-mm3 2 > $O/page.log 2>&1 &
PG=$!
RUN=FAKE_$(date +%H%M%S)
nice taskset -c ${CORES:-12-15} $PY -u inference/fake_arms.py --run $RUN --seed-traj $O/seed.json --latency-dir $O/lat --max-s 1800 --ik --r0 model ${BUMP:+--bump $BUMP} ${WANDER:+--wander-mm $WANDER} ${FAKE_EXTRA:-} > $O/fake.log 2>&1 &
FA=$!
trap 'kill $SP $PG $FA 2>/dev/null' EXIT
MODE="--n-action-steps 16"; [ "${CONT:-1}" = 1 ] && MODE="--continuous --stream-period ${PERIOD:-8} --stream-lead ${LEAD:-10} --stream-blend ${BLEND:-4} --stream-anchor ${ANCHOR:-command} --stream-gamma ${GAMMA:-0.5}"
OMP_NUM_THREADS=4 nice taskset -c ${CORES:-12-15} $PY -u inference/live_loop.py --live-run $RUN --latency-dir $O/lat --tap-port 9894 \
  --policy-url $URL --policy-token-file $TOK --optics water $COMMON --cycle --steps ${STEPS:-1200} $MODE --phase-rules \
  --drill-max-s 240 --drill-pause-s 15 --drill-stall-s 25 --drill-stall-mm3 30 --deadline-s 0 --obs-url http://127.0.0.1:$OPORT \
  --gate-turn-wait --no-aim ${EXTRA:-} -o $O/live.json > $O/live.log 2>&1
RC=$?; echo "$(date +%T) live rc $RC" >> $O/log
sleep 2; kill $SP $PG $FA 2>/dev/null
$PY - $O/live.json <<'PYEOF'
import json, sys, statistics as st
try:
    j = json.load(open(sys.argv[1]))
except Exception as e:
    print("no live.json:", e); sys.exit(1)
m = j["meta"]; ch = [c for c in m["chunks"] if not c.get("scripted")]
t = [c["t_s"] for c in ch]; fr = sum(c["frames"] for c in ch)
keep = {k: m.get(k) for k in ("drill_end", "drill_s", "abort", "end_level", "policy_done", "wall_s", "continuous")}
keep.update(policy_frames=fr, policy_wall_s=round(t[-1] - t[0], 1) if len(t) > 1 else None,
            wall_s_per_frame=round((t[-1] - t[0]) / max(1, fr - ch[0]["frames"]), 3) if len(t) > 1 else None,
            dev_mm_median=[round(st.median(c["dev_mm"][k] for c in ch), 2) for k in (0, 1)] if ch else None,
            gate={k: m["safety_gate"].get(k) for k in ("checked", "shrunk", "held", "turn_cut")}, gate_s_per_check=m["timing"]["gate_s_per_check_median"],
            inference_s=m["timing"]["inference_s_median"], requests=(m.get("remote_policy") or {}).get("requests"))
print(json.dumps(keep, ensure_ascii=False))
PYEOF
$PY inference/live/stream_report.py $O/live.json --json $O/stream_report.json
echo done > $O/DONE
exit $RC
