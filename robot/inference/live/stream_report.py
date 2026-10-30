#!/usr/bin/env python3
"""How a run MOVED, from its live.json — the numbers the two robot runs of 10-12 were compared by (CONTINUOUS.md 8).

    python robot/inference/live/stream_report.py LIVE_JSON [LIVE_JSON …] [--json OUT]

Per run, over the policy frames: wall time a frame, the share of the time the arms moved, waits; then per tool the path of
the commands against the path of their 5 s mean (how much of the motion was to and fro), the swing around that mean
(commands and measured tips, rms), the frames at the per-frame limit, the correlation of the commands' steps one hand-over
cycle apart (lead + period frames: an echo of the loop shows there), and how far the measured tip was from its command.
The first continuous robot run swung (command path 2.0 m against 0.2-0.4 m stop-and-go, swing 4 mm, correlation 0.8) and
its summary line said only that it never stopped: this is the check that was missing."""
import argparse
import json

import numpy as np


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    k = np.ones(w) / w
    return np.stack([np.convolve(np.pad(x[:, j], (w // 2, w - 1 - w // 2), mode="edge"), k, mode="valid") for j in range(x.shape[1])], 1)


def report(path: str) -> dict:
    j = json.load(open(path)); m = j["meta"]; dt = 1.0 / (float(j.get("fps") or 15.0) * float(m.get("speed") or 0.33))
    kinds = np.array(m["frame_kind"]); cmd = np.array(j["frames_cmd"], float); meas = np.array(j["frames"], float)
    pol = np.where(kinds == "policy")[0]; pol = pol[pol > 0]
    ch = m["chunks"]; i0 = next((i for i, c in enumerate(ch) if not c.get("scripted")), None)
    cont = m.get("continuous") or {}
    out = {"run": path, "mode": "continuous" if cont else "stop-and-go", "anchor": cont.get("anchor") or ("measured" if cont else None),
           "gamma": cont.get("gamma"), "lead": cont.get("lead"), "policy_frames": int(len(pol)), "abort": m.get("abort"), "drill_end": m.get("drill_end"),
           "removed_mm3": (lambda r: round(float(r[-1]), 1) if isinstance(r, list) and r else r)(m.get("removed_mm3")), "drill_s": m.get("drill_s"),
           "clamped_frames": m.get("clamped_frames"),
           "gate": {k: (m.get("safety_gate") or {}).get(k) for k in ("checked", "shrunk", "held", "turn_cut")},
           "waited_s": cont.get("waited_s"), "look_to_ready_s": cont.get("look_to_ready_s"), "budget_s": cont.get("budget_s")}
    if i0 is not None and len(pol) > 30:
        pc = [c for c in ch if not c.get("scripted")]
        wall = pc[-1]["t_s"] - (ch[i0 - 1]["t_s"] if i0 else 0.0); fr = sum(c["frames"] for c in pc)
        out.update(policy_wall_s=round(wall, 1), wall_s_per_frame=round(wall / max(fr, 1), 3), moving_share=round(min(1.0, fr * dt / max(wall, 1e-9)), 3))
        lag = int((cont.get("lead") or 0) + (cont.get("period") or 0)) or 16
        for tool, o in (("endo", 0), ("drill", 7)):
            c, z = cmd[pol, o:o + 3], meas[pol, o:o + 3]
            w = max(5, int(round(5.0 / dt)))
            cs = smooth(c, w)
            st = np.linalg.norm(np.diff(c, axis=0), axis=1); s = np.diff(c, axis=0); s = s - s.mean(axis=0)
            ac = float((s[:-lag] * s[lag:]).sum() / max(np.sqrt((s[:-lag] ** 2).sum() * (s[lag:] ** 2).sum()), 1e-12)) if len(s) > 2 * lag else None
            e = np.linalg.norm(z - c, axis=1)
            out[tool] = {"command_path_mm": round(float(st.sum()), 1), "path_of_5s_mean_mm": round(float(np.linalg.norm(np.diff(cs, axis=0), axis=1).sum()), 1),
                         "to_and_fro": round(float(st.sum() / max(np.linalg.norm(np.diff(cs, axis=0), axis=1).sum(), 1e-9)), 2),
                         "swing_command_mm": round(float(np.sqrt((np.linalg.norm(c - cs, axis=1) ** 2).mean())), 2),
                         "swing_measured_mm": round(float(np.sqrt((np.linalg.norm(z - smooth(z, w), axis=1) ** 2).mean())), 2),
                         "frames_at_limit": int((st >= 1.99).sum()), "step_mm_median": round(float(np.median(st)), 3), "step_mm_p90": round(float(np.percentile(st, 90)), 3),
                         f"step_corr_{lag}_frames": None if ac is None else round(ac, 2),
                         "measured_from_command_mm": [round(float(np.median(e)), 2), round(float(np.percentile(e, 90)), 2)]}
    return out


def line(r: dict) -> str:
    s = (f"{r['mode']}" + (f" (anchor {r['anchor']}" + (f", gamma {r['gamma']:g}" if r.get("gamma") is not None else "") + f", lead {r['lead']})" if r["mode"] == "continuous" else "")
         + f": {r['policy_frames']} policy frames, {r.get('wall_s_per_frame')} s a frame, moving {r.get('moving_share')}, waited {r.get('waited_s')} s, "
           f"limit {r.get('clamped_frames')} frames, drilling ended by {r.get('drill_end')}, {r.get('removed_mm3')} mm3, abort {r.get('abort')}")
    for tool in ("endo", "drill"):
        d = r.get(tool)
        if d:
            lagk = next(k for k in d if k.startswith("step_corr_"))
            s += (f"\n   {tool}: command path {d['command_path_mm']:.0f} mm = {d['to_and_fro']}x the path of its 5 s mean; swing {d['swing_command_mm']} mm (command) "
                  f"{d['swing_measured_mm']} mm (measured); {d['frames_at_limit']} frames at the limit; {lagk.replace('_', ' ')} {d[lagk]}; measured {d['measured_from_command_mm'][0]} mm from the command (median)")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("live_json", nargs="+"); ap.add_argument("--json", default=None)
    a = ap.parse_args()
    rs = [report(f) for f in a.live_json]
    for r in rs:
        print(r["run"]); print("  " + line(r))
    if a.json:
        json.dump(rs, open(a.json, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
