#!/usr/bin/env python3
"""What the policy did with the drill in a run, chunk by chunk, against the demos.

    python run_trace.py RUN [RUN ...] [--dataset DS] [--chunk 20] [-o trace.json] [--quiet]

RUN = a live.json (live_loop.py: robot or dry run) or the folder that holds one. The removed volume of a run says how
much, not why. This reads the loop's own log — measured frames, commanded frames, phases, removed volume per frame, gate
events — and the training dataset, and prints for every executed chunk of the drilling phase:

  depth, lateral   the drill tip against the first cut, along the drill portal → first cut line (+ = deeper) and across it
  cmd depth        the same for the commanded tip at the chunk's end: the arms followed if it matches the measured one
  demo cut         distance to the nearest drill tip of a demo with the trigger on — the base demo / its sitting
  ori gap          the tools' orientation against the sitting's drilling frames nearest in position (both tips): how far
                   the run's pose is from what the demos had at that place (the arms hold the tools' orientation)
  gate             the safety gate's events in the chunk, by arm: turn (turn_wait cut the turn), shrink, hold

and one summary per run: trigger bursts with their volume, the retreat, the idle time before the drilling ended and where
the tip was by then. Written 10-07 for the first run on the 92c5 base (146 mm3, drill_end = pause).
The end of the drilling (10-07 evening): `pause` (15 s since the last cut and the trigger off at that moment) and `stall`
(under 30 mm3 in 25 s) are both harness timers that notice the cutting has stopped — which one fires first says little
about the policy. The summary therefore gives what the policy did at the end: how long nothing had been cut, how much of
that it held the trigger (pulled with nothing to cut), how long the trigger had been off, how far the pause rule was;
and how much the endoscope was moved (as commanded): in the demos the trigger is released when the endoscope moves.
Nothing here moves the robot.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

W = Path(__file__).resolve().parent
DS = Path.home() / "il-robot/datasets/spine_hand24_replay_ph_trim_water"
DRILL, ENDO, TRIG = slice(7, 10), slice(0, 3), 14


def bursts(trig: np.ndarray, vol: np.ndarray, fps: float, gap_s: float = 1.0) -> list[dict]:
    """Trigger bursts (pauses shorter than gap_s join their neighbours): start, length and removed volume of each."""
    on = np.flatnonzero(np.asarray(trig) > 0.5)
    if not len(on):
        return []
    cut = np.flatnonzero(np.diff(on) > gap_s * fps)
    out = []
    for a, b in zip(np.r_[on[0], on[cut + 1]], np.r_[on[cut], on[-1]]):
        v0 = float(vol[a - 1]) if a > 0 else 0.0
        out.append({"start_s": round(float(a) / fps, 1), "len_s": round(float(b - a + 1) / fps, 1), "frames": int((np.asarray(trig)[a:b + 1] > 0.5).sum()),
                    "removed_mm3": round(float(vol[b]) - v0, 1)})
    return out


def split(p: np.ndarray, p0: np.ndarray, u: np.ndarray) -> tuple[float, float]:
    """(depth along u, distance across it) of p against p0."""
    r = np.asarray(p, float) - p0
    d = float(r @ u)
    return d, float(np.linalg.norm(r - d * u))


def qang(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0, 1)))


def ending(trig, vol, cut, fps: float, pause_s: float = 15.0, stall_s: float = 25.0) -> dict:
    """The end of the drilling phase as the phase tracker saw it (the phase's frames only): seconds since the last cut,
    how much of that the trigger was held (pulled with nothing cut), how long the trigger had been off at the end, how far
    the pause rule was (pause_s since the last cut AND the trigger off at that moment) and what the last stall_s seconds
    removed (the stall rule's window). cut = the per-frame cut flags of the run's log; None → frames where the volume rose."""
    trig = np.asarray(trig, float) > 0.5; vol = np.asarray(vol, float); n = len(trig)
    c = np.asarray(cut, float) > 0 if cut is not None else np.r_[False, np.diff(vol) > 1e-6]
    lc = int(np.flatnonzero(c)[-1]) if c.any() else -1
    on = np.flatnonzero(trig)
    return {"no_cut_s": round((n - 1 - lc) / fps, 1), "trigger_held_without_cut_s": round(float(trig[lc + 1:].sum()) / fps, 1),
            "trigger_off_at_end_s": round((n - 1 - int(on[-1])) / fps, 1) if len(on) else round(n / fps, 1),
            "pause_rule_in_s": round(max(0.0, pause_s - (n - 1 - lc) / fps), 1),
            "removed_last_window_mm3": round(float(vol[-1] - vol[max(0, n - 1 - int(round(stall_s * fps)))]), 1)}


def tool_motion(S: np.ndarray, off: int, fps: float, h: int = 5) -> dict:
    """Tip path (mm), turn path (deg) and the median turn speed (deg/s over ±h frames) of one tool over the frames S."""
    S = np.asarray(S, float); p, q = S[:, off:off + 3], S[:, off + 3:off + 7]; n = len(S)
    i0, i1 = np.clip(np.arange(n) - h, 0, n - 1), np.clip(np.arange(n) + h, 0, n - 1)
    return {"tip_path_mm": round(float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()), 1),
            "turn_path_deg": round(float(qang(q[1:], q[:-1]).sum()), 1),
            "turn_speed_deg_s_median": round(float(np.median(qang(q[i1], q[i0]) / (np.maximum(i1 - i0, 1) / fps))), 2)}


def gate_kind(ev: list) -> tuple[str, str]:
    """(arm(s), turn | shrink | hold) of one gate event [frame, scale, why, note]."""
    why, note = str(ev[2]), str(ev[3]) if len(ev) > 3 else ""
    arm = "+".join(a for a in ("left", "right") if a in why) or "?"
    return arm, "turn" if note.startswith("turn") else "hold" if float(ev[1]) == 0.0 else "shrink"


def demos(ds: Path) -> dict:
    import pandas as pd
    df = pd.concat([pd.read_parquet(f, columns=["observation.state", "phase.index", "episode_index"])
                    for f in sorted(glob.glob(str(ds / "data/*/*.parquet")))])
    return {"state": np.stack(df["observation.state"].to_numpy()), "ep": df["episode_index"].to_numpy().ravel(),
            "phase": df["phase.index"].to_numpy().astype(int).ravel()}


def trace(path: Path, ds: Path, D: dict, k: int) -> dict:
    sys.path.insert(0, str(W.parent / "inference"))
    import rollout as RO
    d = json.loads(path.read_text()); m = d["meta"]; fps = float(d["fps"])
    F, C, ph = np.asarray(d["frames"], float), np.asarray(d["frames_cmd"], float), np.asarray(m["phases_seen"], int)
    vol = np.asarray(m.get("removed_mm3") or np.zeros(len(F)), float)
    name = path.parent.name
    i4 = np.flatnonzero(ph == 4)
    if not len(i4):
        return {"run": name, "first_cut_s": None, "removed_mm3": float(vol[-1]) if len(vol) else 0.0, "drill_end": m.get("drill_end"), "chunks": []}
    a4, b4 = int(i4[0]), int(i4[-1])
    ep = int(m["episode"]) if not isinstance(m.get("episode"), dict) else int(m["episode"].get("index", 0))
    sitting = RO.batch_episodes(ds, ep) if m.get("portal_from") == "batch" else sorted(set(D["ep"].tolist()))
    portal = np.asarray(RO.demo_portals(ds, sitting)["drill"], float)
    p0 = F[a4, DRILL]; u = (p0 - portal) / np.linalg.norm(p0 - portal)
    S = D["state"]; cutting = (D["phase"] == 4) & (S[:, TRIG] > 0.5)
    base, sit = S[cutting & (D["ep"] == ep)][:, DRILL], S[cutting & np.isin(D["ep"], sitting)][:, DRILL]
    near = lambda x, Q: float(np.min(np.linalg.norm(Q - x, axis=1))) if len(Q) else None
    Sp = S[(D["phase"] == 4) & np.isin(D["ep"], sitting)][::2]           # the sitting's drilling frames, for the pose gap
    ev: dict = {}
    for e in (m.get("safety_gate") or {}).get("events") or []:
        ev.setdefault(int(e[0]), []).append(gate_kind(e))
    rows = []
    for s in range((a4 // k) * k, b4 + 1, k):
        e = min(len(F), s + k); x, c = F[e - 1], C[e - 1]
        dp, lt = split(x[DRILL], p0, u); cd, _ = split(c[DRILL], p0, u)
        X = F[s:e:4]
        dd = np.linalg.norm(X[:, None, [0, 1, 2, 7, 8, 9]] - Sp[None][:, :, [0, 1, 2, 7, 8, 9]], axis=2)
        idx = np.argsort(dd, axis=1)[:, :5]
        g: dict = {}
        for i in range(s, e):
            for arm, kind in ev.get(i, []):
                g[f"{arm} {kind}"] = g.get(f"{arm} {kind}", 0) + 1
        rows.append({"frames": [s, e - 1], "t_s": round((s - a4) / fps, 1), "trig": int((F[s:e, TRIG] > 0.5).sum()),
                     "removed_mm3": round(float(vol[e - 1]), 1), "depth_mm": round(dp, 1), "lateral_mm": round(lt, 1), "cmd_depth_mm": round(cd, 1),
                     "demo_cut_mm": {"base": round(near(x[DRILL], base), 1) if len(base) else None, "sitting": round(near(x[DRILL], sit), 1)},
                     "pose_gap": {"mm": round(float(np.median(np.take_along_axis(dd, idx[:, :1], 1))), 1),
                                  "endo_deg": round(float(np.median(qang(X[:, None, 3:7], Sp[idx][:, :, 3:7]).min(1))), 1),
                                  "drill_deg": round(float(np.median(qang(X[:, None, 10:14], Sp[idx][:, :, 10:14]).min(1))), 1)},
                     "endo_moved_mm": round(float(np.linalg.norm(x[ENDO] - F[a4, ENDO])), 1), "gate": g})
    T = F[a4:b4 + 1, TRIG] > 0.5
    B = bursts(F[a4:b4 + 1, TRIG], vol[a4:b4 + 1] - (vol[a4 - 1] if a4 else 0.0), fps)
    last = int(np.flatnonzero(T)[-1]) if T.any() else 0
    dlast, _ = split(F[a4 + last, DRILL], p0, u); dend, lend = split(F[b4, DRILL], p0, u)
    gate: dict = {}
    for i, L in ev.items():
        if a4 <= i <= b4:
            for arm, kind in L:
                gate[f"{arm} {kind}"] = gate.get(f"{arm} {kind}", 0) + 1
    depths = np.array([split(p, p0, u)[0] for p in F[a4:b4 + 1, DRILL]])
    dl = m.get("drill_log"); dl = json.loads(dl) if isinstance(dl, str) else dl
    cut = np.array([int(r[4] or 0) for r in dl], int)[a4:b4 + 1] if dl and len(dl) > b4 else None
    E = ending(F[a4:b4 + 1, TRIG], vol[a4:b4 + 1], cut, fps, float((m.get("filters") or {}).get("drill_pause_s") or 15.0))
    return {"run": name, "episode": ep, "sitting": [int(i) for i in sitting], "first_cut_s": round(a4 / fps, 1), "drilling_s": round((b4 - a4 + 1) / fps, 1),
            "removed_mm3": round(float(vol[b4]), 1), "trigger_frames": int(T.sum()), "bursts": B, "drill_end": m.get("drill_end"),
            "trigger_on_share": round(float(T.mean()), 2), "ending": E, "endo_cmd": tool_motion(C[a4:b4 + 1], 0, fps),
            "aim": m.get("aim"), "aimed_frames": m.get("aimed_frames"),
            "depth_mm": {"min": round(float(depths.min()), 1), "max": round(float(depths.max()), 1), "at_last_trigger": round(dlast, 1), "at_end": round(dend, 1)},
            "idle_before_end_s": round((b4 - a4 - last) / fps, 1), "lateral_at_end_mm": round(lend, 1),
            "demo_cut_at_end_mm": rows[-1]["demo_cut_mm"], "cmd_vs_measured_depth_mm_max": round(max(abs(r["depth_mm"] - r["cmd_depth_mm"]) for r in rows), 1),
            "gate": gate, "chunks": rows}


def show(t: dict, quiet: bool) -> None:
    if t["first_cut_s"] is None:
        print(f"{t['run']}: no bone cut (drill_end {t['drill_end']})"); return
    b = " | ".join(f"{x['start_s']:.1f} s +{x['len_s']:.1f} s {x['removed_mm3']:.1f} mm3" for x in t["bursts"])
    print(f"{t['run']}: first cut {t['first_cut_s']} s, drilling {t['drilling_s']} s, {t['removed_mm3']} mm3, end = {t['drill_end']}\n"
          f"  trigger {t['trigger_frames']} fr, bursts (after the first cut): {b}\n"
          f"  depth {t['depth_mm']['min']:+.1f} … {t['depth_mm']['max']:+.1f} mm, at the last trigger {t['depth_mm']['at_last_trigger']:+.1f}, at the end {t['depth_mm']['at_end']:+.1f} "
          f"(lateral {t['lateral_at_end_mm']}); idle before the end {t['idle_before_end_s']} s; nearest demo cut at the end: base {t['demo_cut_at_end_mm']['base']} / "
          f"sitting {t['demo_cut_at_end_mm']['sitting']} mm\n"
          f"  commanded vs measured depth ≤ {t['cmd_vs_measured_depth_mm_max']} mm; gate in the drilling phase: {t['gate'] or 'none'}")
    e, em = t.get("ending"), t.get("endo_cmd")
    if e and em:
        print(f"  end of the drilling: nothing cut for {e['no_cut_s']} s, the trigger held {e['trigger_held_without_cut_s']} s of that, off for the last "
              f"{e['trigger_off_at_end_s']} s; the pause rule was {e['pause_rule_in_s']} s away, the last 25 s removed {e['removed_last_window_mm3']} mm3\n"
              f"  trigger on {t['trigger_on_share']:.2f} of the drilling; endoscope as commanded: tip path {em['tip_path_mm']:.0f} mm, turn path "
              f"{em['turn_path_deg']:.0f}° (median {em['turn_speed_deg_s_median']}°/s); aim correction {'on' if t.get('aim') else 'off' if t.get('aim') is False else 'on (run before 10-07 21:45)'}"
              + (f", {t['aimed_frames']} frames" if t.get('aimed_frames') else ""))
    if quiet:
        return
    print("   t_s trig  mm3   depth  lat | cmd  | demo cut base/sitting | pose gap mm endo° drill° | endo mm | gate")
    for r in t["chunks"]:
        g = ", ".join(f"{k}×{v}" for k, v in r["gate"].items())
        print(f"  {r['t_s']:5.1f} {r['trig']:3d} {r['removed_mm3']:6.1f} {r['depth_mm']:+6.1f} {r['lateral_mm']:4.1f} | {r['cmd_depth_mm']:+5.1f} | "
              f"{r['demo_cut_mm']['base'] if r['demo_cut_mm']['base'] is not None else float('nan'):5.1f} {r['demo_cut_mm']['sitting']:5.1f}       | "
              f"{r['pose_gap']['mm']:5.1f} {r['pose_gap']['endo_deg']:5.1f} {r['pose_gap']['drill_deg']:5.1f}     | {r['endo_moved_mm']:5.1f}   | {g}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dataset", default=str(DS))
    ap.add_argument("--chunk", type=int, default=20, help="frames per row (the executed chunk)")
    ap.add_argument("--quiet", action="store_true", help="summaries only")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    ds = Path(a.dataset).expanduser(); D = demos(ds); out = []
    for r in a.runs:
        p = Path(r).expanduser(); p = p / "live.json" if p.is_dir() else p
        t = trace(p, ds, D, a.chunk); out.append(t); show(t, a.quiet)
    if a.out:
        Path(a.out).expanduser().write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
