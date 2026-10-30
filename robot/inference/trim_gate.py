#!/usr/bin/env python3
"""Check the trimmed training episodes against the robot safety gate, and find each recording's gate-safe span.

    python robot/inference/trim_gate.py --dataset ~/il-robot/datasets/<untrimmed, full sessions> \
        --trimmed ~/il-robot/datasets/<the --trim robot export> [--stride 3] [--only 3868,91cb] [-o trim_gate.json] \
        [--spans-out spans.json]

export_lerobot.py --trim robot cuts every hand episode with one rule (dataset/trim.py: first cut − pre s with both tools
inside the skin … both tools out). This tool asks whether the arms could play that part back, for every recording that
has a verified robot start pair (eval/pairstart/pairs_by_recording_chosen.json, anchored at the recording's window start):
  span      the contiguous run of frames around the anchor that pass the gate (safety_gate.RobotGate.check, the inference
            gate: safe zone with both markers seen, Clearance, arm distance, IK ≤ 5 mm / 10°, joint limits)
  rule_ok   the rule's span lies inside it
  entry_ok  the scripted axial entry — from the ready pose (--ready-out-mm outside the skin along the shafts, at least
            --ready-min-out-mm when the arms do not reach that: entry_out_mm) to the first kept frame — passes
            (rollout.py --cycle; the cut front of the episode is replaced by it on the robot)
  exit_ok   the scripted exit from the last kept frame to that ready pose passes
--spans-out writes {id: [t_a, t_b]} (s, session clock) for export_lerobot.py --trim-spans. The gate needs ~0.3–0.7 s per
frame (two IKs + the safe-zone model), so frames are checked every --stride and the span is reported on that grid; run it
when no robot job is queued (it competes with the tracker for CPU).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rollout as RO                     # noqa: E402
import safety_gate as G                  # noqa: E402

WIN = {"3868": 17, "91cb": 15, "bef5": 17, "d340": 27, "bad1": 12, "0338": 18, "919d": 10, "6639": 14}   # pairstart/recs.py


def episode_full(ds: Path, ep: int):
    """states (N, 15), session time (N,), phase (N,) of one episode, frame order."""
    import glob
    import pyarrow.parquet as pq
    st, ts, ph, fi = [], [], [], []
    for f in sorted(glob.glob(str(ds / "data/*/*.parquet"))):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state", "time.src_s", "phase.index"])
        sel = np.nonzero(np.asarray(t.column("episode_index").to_pylist()) == ep)[0]
        if len(sel):
            a, b, c, d = (t.column(k).to_pylist() for k in ("observation.state", "time.src_s", "phase.index", "frame_index"))
            st += [a[i] for i in sel]; ts += [b[i] for i in sel]; ph += [c[i] for i in sel]; fi += [d[i] for i in sel]
    o = np.argsort(fi)
    return (np.asarray(st, float)[o][:, :15], np.asarray(ts, float).reshape(-1)[o], np.asarray(ph).reshape(-1)[o])


def walk(gate, states, ks) -> tuple[int, str]:
    """Step the gate through frames ks (the first one must be safe). (last safe frame, why it stopped)."""
    last, why = ks[0], ""
    for k in ks:
        ok, w, q = gate.check(states[k])
        if not ok:
            return last, w
        gate.q.update(q); last = k
    return last, why


def path_ok(gate, a: np.ndarray, b: np.ndarray, step_mm: float = 2.0) -> tuple[bool, str, float]:
    """Straight move a → b (positions linear, a's orientations, as the rollout's scripted entry/exit checks it).
    (all safe, why not, the fraction of the way that is safe)."""
    n = max(2, int(np.ceil(max(np.linalg.norm(b[o:o + 3] - a[o:o + 3]) for o in (0, 7)) / step_mm)))
    for j in range(1, n + 1):
        f = a + (b - a) * (j / n); f[3:7], f[10:14] = a[3:7], a[10:14]; f[14] = 0.0
        ok, w, q = gate.check(f)
        if not ok:
            return False, f"{j}/{n}: {w}", (j - 1) / n
        gate.q.update(q)
    return True, "", 1.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="an UNTRIMMED full-session dataset (time.src_s, phase.index per frame)")
    ap.add_argument("--pairs", default="~/il-robot/runs/jitter_iter/work/pairstart/pairs_by_recording_chosen.json")
    ap.add_argument("--windows", default="~/il-robot/runs/jitter_iter/work/pairstart/windows.json",
                    help="new recordings' window starts (new_recording.py); the rev.4 eight are built in")
    ap.add_argument("--trimmed", required=True, help="the --trim robot export of the same sessions (its sidecar carries "
                    "each episode's kept span; --images none is enough)")
    ap.add_argument("--ready-out-mm", type=float, default=20.0)
    ap.add_argument("--ready-min-out-mm", type=float, default=10.0)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--radius-mm", type=float, default=150.0)
    ap.add_argument("--oerr-deg", type=float, default=None, help="IK orientation limit of the gate (default safety_gate.OERR_DEG, "
                    "10°; 10-03: 17° for the long drillings bad1, d340, bef5 — the hand turns the tools 12–16° beyond what the arm follows)")
    ap.add_argument("--only", default="")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--spans-out", default=None)
    a = ap.parse_args()
    G.RADIUS_M = a.radius_mm / 1000.0
    if a.oerr_deg is not None:
        G.OERR_DEG = a.oerr_deg
    ds = Path(a.dataset).expanduser()
    pairs = json.loads(Path(a.pairs).expanduser().read_text())
    win = dict(WIN)
    wf = Path(a.windows).expanduser()
    if wf.exists():
        win.update({k: v["from_s"] for k, v in json.loads(wf.read_text()).items()})
    only = {x for x in a.only.split(",") if x}
    eps = [json.loads(l) for l in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()]
    portals = RO.demo_portals(ds)
    kept = {m["session_id"]: m["trim"] for m in map(json.loads, (Path(a.trimmed).expanduser() / "meta/spine_il/episodes.jsonl")
                                                    .read_text().splitlines()) if m.get("trim")}
    res, spans = {}, {}
    for ep, e in enumerate(eps):
        rid = Path(e["recording"]).stem.split("_trk_")[-1]
        if e.get("source") != "hand" or rid not in pairs or (rid not in win and "anchor_s" not in pairs.get(rid, {})) or (only and rid not in only) \
                or e["session_id"] not in kept:
            continue
        if e.get("trim"):
            raise SystemExit(f"{ds} is already trimmed — pass the untrimmed full-session dataset")
        st, ts, ph = episode_full(ds, ep)
        with open(e["recording"]) as fh:
            hdr = next(d for d in map(json.loads, fh) if d.get("k") == "hdr")
        shafts = RO.shaft_axes(e["recording"])
        n = len(st)
        anchor_s = float(pairs[rid].get("anchor_s", win[rid]))   # a pair searched for another window carries its own start
        k_an = int(np.argmin(np.abs(ts - anchor_s)))
        tk = kept[e["session_id"]]            # the exporter's own span (dataset/trim.py) on the session clock
        ia, ib = int(np.argmin(np.abs(ts - tk["t_a"]))), int(np.argmin(np.abs(ts - tk["t_b"])))
        ib = min(ib, n - 1)
        mk = lambda: G.RobotGate(st[k_an], hdr["registration"]["T0"], pairs[rid]["q_left"], pairs[rid]["q_right"])
        back = sorted(set(list(range(k_an, -1, -a.stride)) + [ia]), reverse=True)
        back = [k for k in back if k <= k_an]
        fwd = sorted(set(list(range(k_an, n, a.stride)) + [ib]))
        g = mk(); ka, why_a = walk(g, st, back)
        q_a = {arm: g.q[arm].copy() for arm in g.q}
        g2 = mk(); kb, why_b = walk(g2, st, fwd)
        r = {"episode": ep, "anchor_s": anchor_s, "frames": n, "span_s": [round(float(ts[ka]), 2), round(float(ts[kb]), 2)],
             "stops": {"start": why_a[:80], "end": why_b[:80]}, "rule_s": [round(float(ts[ia]), 2), round(float(ts[ib]), 2)],
             "rule_ok": bool(ka <= ia and kb >= ib)}
        # scripted entry/exit at the rule's ends (only meaningful when those ends are inside the span)
        if ka <= ia:
            g = mk(); walk(g, st, [k for k in back if k >= ia])
            first = st[ia].copy(); first[14] = 0.0
            ready = RO.retract(first, portals, shafts, a.ready_out_mm)
            ok, w, x = path_ok(g, first, ready)
            # how far outside the skin the retraction stays safe (the rollout accepts ≥ --ready-min-out-mm, default 10)
            out = min(a.ready_out_mm - (1 - x) * float(np.linalg.norm(ready[o:o + 3] - first[o:o + 3])) for o in (0, 7))
            r["entry_out_mm"], r["entry_why"] = round(out, 1), w[:80]
            r["entry_ok"] = ok = bool(out >= a.ready_min_out_mm)
            if x < 1.0:
                ready = first + (ready - first) * x
            if kb >= ib and ok:
                g = mk(); walk(g, st, [k for k in fwd if k <= ib])
                last = st[ib].copy(); last[14] = 0.0
                out = last + (RO.retract(last, portals, shafts, a.ready_out_mm) - last) * x
                ok1, w1, _ = path_ok(g, last, out)
                ok2, w2, _ = path_ok(g, out, ready) if ok1 else (False, "", 0.0)
                r["exit_ok"], r["exit_why"] = bool(ok1 and ok2), (w1 or w2)[:80]
        res[rid] = r; spans[rid] = r["span_s"]
        print(rid, json.dumps(r, ensure_ascii=False), flush=True)
    if a.out:
        Path(a.out).expanduser().write_text(json.dumps(res, indent=1, ensure_ascii=False))
    if a.spans_out:
        Path(a.spans_out).expanduser().write_text(json.dumps(spans, indent=1))
    bad = [k for k, r in res.items() if not (r["rule_ok"] and r.get("entry_ok") and r.get("exit_ok"))]
    print(f"[trim_gate] {len(res)} recordings, rule + scripted entry/exit safe on {len(res) - len(bad)}" + (f"; not on {bad}" if bad else ""))


if __name__ == "__main__":
    main()
