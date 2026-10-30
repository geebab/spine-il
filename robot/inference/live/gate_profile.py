#!/usr/bin/env python3
"""Where a safety-gate check spends its time: the gate of a past robot run rebuilt offline (rollout.prepare with the run's
own arguments) and the run's commanded policy frames passed through it as live_loop.execute does (a check when a tip moved
1 mm or a tool turned 1° since the last checked frame). No robot, no renderer."""
import cProfile, io, json, pstats, shlex, sys, time
from pathlib import Path
import numpy as np
R = Path(__import__("os").environ.get("SPINE_IL_REPO", str(Path.home() / "spine-il"))) / "robot"
sys.path.insert(0, str(R / "inference")); sys.path.insert(0, str(R))
import rollout as RO

run = Path(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
argv = shlex.split((run / "live.args").read_text())
ap = RO.build_parser("")
a, _ = ap.parse_known_args(argv)
a.gate_turn_wait = "--gate-turn-wait" in argv
t0 = time.monotonic(); P = RO.prepare(a); gate = P["gate"]
print(f"prepare {time.monotonic() - t0:.1f} s | gate {type(gate).__name__} turn_wait {gate.turn_wait}", flush=True)
j = json.loads((run / "live.json").read_text()); kind = j["meta"]["frame_kind"]
cmd = [np.asarray(c, float) for c, k in zip(j["frames_cmd"], kind) if k == "policy"]
safe = cmd[0].copy(); gate.filter(safe, cmd[0])            # warm-up
times, n = [], 0
pr = cProfile.Profile()
for vv in cmd[1:]:
    far = any(np.linalg.norm(vv[o:o + 3] - safe[o:o + 3]) >= 1.0 for o in (0, 7)) or any(
        2 * np.degrees(np.arccos(np.clip(abs(float(np.dot(vv[o:o + 4], safe[o:o + 4]))), 0, 1))) >= 1.0 for o in (3, 10))
    if not far:
        continue
    t = time.perf_counter(); pr.enable(); g, gi = gate.filter(safe, vv); pr.disable(); times.append(time.perf_counter() - t)
    safe = g.copy(); n += 1
    if n >= N:
        break
print(f"{n} checks: median {np.median(times) * 1000:.0f} ms, p90 {np.percentile(times, 90) * 1000:.0f} ms, max {max(times) * 1000:.0f} ms | shrunk {gate.n_shrunk} held {gate.n_held} turn_cut {gate.n_turn_cut}")
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(22); print(s.getvalue()[:5200])
