#!/usr/bin/env python3
"""The gate with the arm-distance cutoff at its threshold (10 mm) against the gate as it was (70 mm): every check of past
robot runs' commanded frames must give the same verdict, reason, scale and filtered frame. Plus frames pushed toward each
other until the arms-apart rule fires (the rule this change touches)."""
import copy, json, shlex, sys, time
from pathlib import Path
import numpy as np
R = Path(__import__("os").environ.get("SPINE_IL_REPO", str(Path.home() / "spine-il"))) / "robot"
sys.path.insert(0, str(R / "inference")); sys.path.insert(0, str(R))
import rollout as RO

def far(vv, safe):
    return any(np.linalg.norm(vv[o:o + 3] - safe[o:o + 3]) >= 1.0 for o in (0, 7)) or any(
        2 * np.degrees(np.arccos(np.clip(abs(float(np.dot(vv[o:o + 4], safe[o:o + 4]))), 0, 1))) >= 1.0 for o in (3, 10))

tot = {"checks": 0, "same": 0, "fail_old": 0, "t_old": [], "t_new": []}
for run in sys.argv[1:]:
    run = Path(run); argv = shlex.split((run / "live.args").read_text())
    a, _ = RO.build_parser("").parse_known_args(argv); a.gate_turn_wait = "--gate-turn-wait" in argv
    g_new = RO.prepare(a)["gate"]; g_old = RO.prepare(a)["gate"]; g_old.arms_cutoff_m = 0.07
    j = json.loads((run / "live.json").read_text()); kind = j["meta"]["frame_kind"]
    cmd = [np.asarray(c, float) for c in j["frames_cmd"]]
    safe = cmd[0].copy(); n = same = 0
    for vv in cmd[1:]:
        if not far(vv, safe):
            continue
        t = time.perf_counter(); go, io_ = g_old.filter(safe, vv); t1 = time.perf_counter(); gn, in_ = g_new.filter(safe, vv); t2 = time.perf_counter()
        tot["t_old"].append(t1 - t); tot["t_new"].append(t2 - t1)
        eq = np.allclose(go, gn, atol=1e-12) and io_ == in_
        same += eq; n += 1; tot["fail_old"] += not io_["ok"] or io_["scale"] < 1 or io_.get("turn") is not None
        if not eq:
            print("DIFFERENT", run.name, n, io_, in_)
        safe = go.copy()
    # the arms-apart rule itself: walk the drill toward the endoscope from the last frame, 1 mm a step, until a check fails
    v = cmd[len(cmd) // 2].copy(); dvec = v[0:3] - v[7:10]; dvec /= np.linalg.norm(dvec); k_fail = None; why_o = why_n = None
    for k in range(1, 80):
        c = v.copy(); c[7:10] = v[7:10] + dvec * k
        oo, wo, _ = g_old.check(c); on, wn, _ = g_new.check(c)
        if (oo, wo) != (on, wn):
            print("DIFFERENT walk", run.name, k, wo, "|", wn)
        if not oo and k_fail is None:
            k_fail, why_o = k, wo
        if "apart" in wo:
            why_n = wn; break
    print(f"{run.name}: {n} checks, {same} identical | walking the drill toward the endoscope: first refusal after {k_fail} mm ({why_o}); arms-apart seen: {why_n}")
    tot["checks"] += n; tot["same"] += same
print(f"total {tot['checks']} checks, {tot['same']} identical, {tot['fail_old']} of them shrunk/held/turn-cut | per check: 70 mm cutoff {np.median(tot['t_old']) * 1000:.0f} ms, 10 mm cutoff {np.median(tot['t_new']) * 1000:.0f} ms (median)")
