#!/usr/bin/env python3
"""Random start poses inside the range of the demonstrations' start poses (for rollouts that do not start at a demo frame).

    python robot/inference/make_starts.py --dataset ~/il-robot/datasets/spine_rev4_step_ph_trim -n 5 --seed 1 -o starts/

Each demo's start = frame 0 of its trimmed episode (export_lerobot.py --trim robot: first cut − 7 s, both tools inside the
skin). Per tool, a new tip position is drawn uniformly inside the axis-aligned box of the 8 demo start tips; the tool is
then aimed through its portal (the shaft from the portal to the new tip — a tool inside the body pivots about its skin
entry), keeping the mean demo roll about the shaft. A draw is kept only if the robot safety gate accepts it: reachable
from the 3868 start pair's anchor in a straight move, the axial retraction to the ready pose (--ready-out-mm outside the
skin) and the tips at least --min-sep-mm apart. Writes start_<k>.json {"state": 15 values, ...} and starts.json.

A base on another sitting (10-07, the 92c5 base of the 24-demo model; user: the start should be a random value inside the
range found in the recordings, not a demonstration's own):
    --sitting              the box and the portals come from the recordings of the anchor episode's sitting only
    --frame 82             the moment that counts as a recording's start (82 = 1.5 s before its first cut)
    --orientation anchor   the tools keep the anchor frame's orientations (the arms hold a tool's orientation; a start
                           aimed through the portal would ask for a turn they do not make)
    --gate-registration F  the registration fitted to the arms' tools (eval/fit_registration.py), as the run uses it
    --pairs F              a search's pair_clear_<rid>.json ({"best": …}) is accepted
    --sep-from-demos       the two tips as far apart as in the recordings' own starts (their min … max) instead of --min-sep-mm
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout as RO           # noqa: E402
import safety_gate as G        # noqa: E402
from safety import _slerp      # noqa: E402

TOOLS = (("endo", 0), ("drill", 7))


def mean_quat(qs: np.ndarray) -> np.ndarray:
    M = sum(np.outer(q, q) for q in qs)
    w, v = np.linalg.eigh(M)
    q = v[:, -1]
    return q if q @ qs[0] >= 0 else -q


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="trimmed dataset (frame 0 = the start of the kept part)")
    ap.add_argument("-n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--anchor-episode", type=int, default=6, help="episode whose robot start pair anchors the gate (3868)")
    ap.add_argument("--anchor-s", type=float, default=17.0)
    ap.add_argument("--gate-pair", default="3868")
    ap.add_argument("--pairs", default="~/il-robot/runs/jitter_iter/work/pairstart/pairs_by_recording_chosen.json")
    ap.add_argument("--ready-out-mm", type=float, default=20.0)
    ap.add_argument("--min-sep-mm", type=float, default=15.0)
    ap.add_argument("--max-tries", type=int, default=60)
    ap.add_argument("--sitting", action="store_true", help="only the recordings of the anchor episode's sitting")
    ap.add_argument("--frame", type=int, default=0, help="the frame of each trimmed episode taken as its start")
    ap.add_argument("--orientation", choices=("portal", "anchor"), default="portal")
    ap.add_argument("--gate-registration", default=None)
    ap.add_argument("--sep-from-demos", action="store_true")
    ap.add_argument("--prefix", default="start_", help="file name prefix of the starts written")
    ap.add_argument("--anchor-state", default=None, help="a base that is not one recording (eval/sitting_base.py base_<NAME>.json): the gate "
                    "is anchored at this state and --orientation anchor keeps ITS tool orientations; --anchor-episode then only names the sitting")
    ap.add_argument("--level", default=None, help="only the recordings of this level (a multi-level dataset's goal.level; 10-15: L3 / L5 starts — "
                    "without it the box spans the three levels' skin entries, 22–40 mm apart)")
    ap.add_argument("--before-cut-s", type=float, default=None, help="each recording's start = this many seconds before its own first cut "
                    "(meta trim t_cut0 / t_a) instead of one --frame for all (a from-skin-entry dataset starts at different distances from the cut)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    ds = Path(a.dataset).expanduser(); out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    eps = [json.loads(l) for l in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()]
    st, ep, fi, ts = [], [], [], []
    for f in sorted(glob.glob(str(ds / "data/*/*.parquet"))):
        t = pq.read_table(f, columns=["observation.state", "episode_index", "frame_index", "time.src_s"])
        st += t.column("observation.state").to_pylist(); ep += t.column("episode_index").to_pylist()
        fi += t.column("frame_index").to_pylist(); ts += [x[0] if isinstance(x, list) else x for x in t.column("time.src_s").to_pylist()]
    st, ep, fi, ts = np.asarray(st, float)[:, :15], np.asarray(ep), np.asarray(fi), np.asarray(ts, float)
    use = RO.batch_episodes(ds, a.anchor_episode) if a.sitting else list(range(len(eps)))
    use = [e for e in use if not eps[e].get("decision_copy")]   # export_lerobot.py --decision-copies: not recordings
    if a.level:
        use = [e for e in use if (eps[e].get("goal") or {}).get("level", "L4") == a.level]
        if not use:
            raise SystemExit(f"no recording of level {a.level} in {ds}")
    def start_frame(e: int) -> int:
        if a.before_cut_s is None:
            return a.frame
        tr = eps[e]["trim"]
        return max(0, int(round((tr["t_cut0"] - a.before_cut_s - tr["t_a"]) * 15.0)))
    S0 = np.array([st[(ep == e) & (fi == start_frame(e))][0] for e in use])
    full = Path(str(ds).replace("_ph_trim", "_delta").replace("_trim", "_delta"))
    portals = RO.demo_portals(ds, use) if (a.sitting or a.level) else RO.demo_portals(full if full.exists() else ds)
    shafts = RO.shaft_axes(eps[a.anchor_episode]["recording"])
    seps = np.linalg.norm(S0[:, 0:3] - S0[:, 7:10], axis=1)
    lo = {t: S0[:, o:o + 3].min(0) for t, o in TOOLS}; hi = {t: S0[:, o:o + 3].max(0) for t, o in TOOLS}
    qm = {t: mean_quat(S0[:, o + 3:o + 7]) for t, o in TOOLS}
    # gate anchored at the 3868 pair's window start, as the rollout does
    w = (ep == a.anchor_episode)
    anchor = st[w][int(np.argmin(np.abs(ts[w] - a.anchor_s)))]
    if a.anchor_state:
        anchor = np.asarray(json.loads(Path(a.anchor_state).expanduser().read_text())["state"], float)[:15]
    with open(eps[a.anchor_episode]["recording"]) as fh:
        hdr = next(d for d in map(json.loads, fh) if d.get("k") == "hdr")
    pj = json.loads(Path(a.pairs).expanduser().read_text())
    pj = pj["best"] if "best" in pj else pj[a.gate_pair]
    T0 = hdr["registration"]["T0"]
    if a.gate_registration:
        reg = json.loads(Path(a.gate_registration).expanduser().read_text())
        T0 = {"R": reg["R"], "t": reg["t"]}
    i_anchor = use.index(a.anchor_episode) if a.anchor_episode in use else 0
    rng = np.random.default_rng(a.seed)
    kept, tried = [], 0
    while len(kept) < a.n and tried < a.max_tries:
        tried += 1
        s = anchor.copy(); s[14] = 0.0
        for t, o in TOOLS:
            tip = rng.uniform(lo[t], hi[t])
            s[o:o + 3] = tip
            if a.orientation == "portal":
                R0 = RO.q2R(qm[t]); u0 = R0 @ shafts[t]                        # mean shaft direction (portal → tip)
                R = rot_between(u0, tip - portals[t]) @ R0                    # aimed through the portal, mean roll
                s[o + 3:o + 7] = RO.R2q(R, qm[t])
            # (anchor: s already carries the anchor frame's orientations)
        sep = float(np.linalg.norm(s[0:3] - s[7:10]))
        why = None
        if a.sep_from_demos and not seps.min() <= sep <= seps.max():
            why = f"tips {sep:.1f} mm apart (the recordings' starts: {seps.min():.1f}–{seps.max():.1f})"
        elif not a.sep_from_demos and sep < a.min_sep_mm:
            why = f"tips {sep:.0f} mm apart"
        else:
            g = G.RobotGate(anchor, T0, pj["q_left"], pj["q_right"])
            n = max(2, int(np.ceil(max(np.linalg.norm(s[o:o + 3] - anchor[o:o + 3]) for o in (0, 7)) / 0.5)))
            for j in range(1, n + 1):
                f = anchor + (s - anchor) * (j / n)
                for o in (3, 10):
                    f[o:o + 4] = _slerp(anchor[o:o + 4], s[o:o + 4], j / n)
                ok, w_, q = g.check(f)
                if not ok:
                    why = f"from the anchor: {w_}"; break
                g.q.update(q)
            if why is None:
                ready = RO.retract(s, portals, shafts, a.ready_out_mm)
                m = max(2, int(np.ceil(max(np.linalg.norm(ready[o:o + 3] - s[o:o + 3]) for o in (0, 7)) / 1.0)))
                for j in range(1, m + 1):
                    f = s + (ready - s) * (j / m); f[3:7], f[10:14] = s[3:7], s[10:14]
                    ok, w_, q = g.check(f)
                    if not ok:
                        why = f"ready path: {w_}"; break
                    g.q.update(q)
        if why:
            print(f"[starts] try {tried}: rejected — {why}", flush=True)
            continue
        k = len(kept)
        rec = {"state": [round(float(x), 5) for x in s], "seed": a.seed, "try": tried, "tips_sep_mm": round(sep, 1),
               "tip_box": {t: [lo[t].round(1).tolist(), hi[t].round(1).tolist()] for t, _ in TOOLS},
               f"offset_from_{a.gate_pair}_start_mm": {t: round(float(np.linalg.norm(s[o:o + 3] - S0[i_anchor, o:o + 3])), 1) for t, o in TOOLS},
               "nearest_demo_start_mm": {t: round(float(np.min(np.linalg.norm(S0[:, o:o + 3] - s[o:o + 3], axis=1))), 1) for t, o in TOOLS},
               "from": f"uniform draw inside the box of the start tips ({f'{a.before_cut_s} s before each first cut' if a.before_cut_s is not None else f'frame {a.frame}'}) of {len(use)} recordings"
                       + (f" of level {a.level}" if a.level else "")
                       + (f" of the sitting of episode {a.anchor_episode}" if a.sitting else "") + f"; orientation: {a.orientation}; seed {a.seed}, try {tried}",
               "episodes": [int(e) for e in use], "tips_sep_demos_mm": [round(float(seps.min()), 1), round(float(seps.max()), 1)],
               "anchor_state": a.anchor_state}
        (out / f"{a.prefix}{k}.json").write_text(json.dumps(rec, indent=1))
        kept.append(rec)
        print(f"[starts] {a.prefix}{k}: endo {np.round(s[0:3], 1)} drill {np.round(s[7:10], 1)} | tips {sep:.1f} mm apart | from the {a.gate_pair} start "
              f"{rec[f'offset_from_{a.gate_pair}_start_mm']} mm, nearest demo {rec['nearest_demo_start_mm']} mm", flush=True)
    (out / f"{a.prefix}starts.json" if a.prefix != "start_" else out / "starts.json").write_text(
        json.dumps({"kept": kept, "tried": tried, "box": kept[0]["tip_box"] if kept else None}, indent=1))
    print(f"[starts] {len(kept)} kept of {tried} tried")


if __name__ == "__main__":
    main()
