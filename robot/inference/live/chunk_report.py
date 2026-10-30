#!/usr/bin/env python3
"""What the policy commands per chunk in closed-loop runs, and how far repeated runs are from each other (live.json).

    python robot/inference/live/chunk_report.py RUN_DIR [RUN_DIR ...] [--dataset DS] [--together] [--json OUT]

RUN_DIR = a folder of live_run.sh (it has live.json). A run's record holds one extra "settle" frame after every chunk
(meta.frame_kind): the measured pose the next chunk starts from — count frames by kind, not by the chunks' frame numbers.

per run, policy chunks of 20 or more frames, drill tip, medians (phase 3 approach / 4 drilling):
    cmd_net_mm      |last target − the measured pose the chunk started from|     cmd_path_mm   path through the targets
    first_step_mm   |first target − that measured pose|                           step_mm       median of the later steps
    meas_net_mm, meas_path_mm    the same of the measured frames (the path includes the arms' hunting)
    endo_cmd_net_mm              cmd_net_mm of the endoscope tip
    cut_chunks      drilling chunks in which the policy's viewer session removed more than 1 mm³
The demos' values over 20 steps of the dataset (absolute actions) are printed first.
--together: the runs are repeats of one condition — the distance between their drill tips at the same policy time
(policy frames only), per 5 s, next to how far a run is from its own position 5 s earlier.
"""
import argparse
import glob
import itertools
import json
from pathlib import Path

import numpy as np


def load(run: Path) -> dict:
    d = json.loads((run / "live.json").read_text()); m = d["meta"]
    F, C, fk = np.array(d["frames"], float), np.array(d["frames_cmd"], float), np.array(m["frame_kind"])
    ch, s = [], 0
    for k, e in enumerate(np.flatnonzero(fk == "settle")):
        assert m["chunks"][k]["frames"] == e - s, f"{run.name}: chunk {k} has {e - s} frames in the record, {m['chunks'][k]['frames']} in its summary"
        ch.append((s, int(e), m["chunks"][k])); s = int(e) + 1
    pol = np.flatnonzero(fk == "policy")
    return {"name": run.name, "F": F, "C": C, "chunks": ch, "P": F[pol], "ph": np.array(m["phases_seen"])[pol], "vol": np.array(m["removed_mm3"], float)[pol]}


def path(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(np.vstack([a, x]), axis=0), axis=1)


def chunk_rows(r: dict, min_frames: int) -> dict:
    rows, prev, cut, tot = {3: [], 4: []}, 0.0, 0, 0
    F, C = r["F"], r["C"]
    for s, e, c in r["chunks"]:
        if not c["scripted"] and c["frames"] >= min_frames and c["phase"] in rows and s > 0:
            a = F[s - 1, 7:10]; st = path(a, C[s:e, 7:10])
            rows[c["phase"]].append([np.linalg.norm(C[e - 1, 7:10] - a), st.sum(), st[0], np.median(st[1:]), np.linalg.norm(F[e, 7:10] - a),
                                     path(a, F[s:e, 7:10]).sum(), np.linalg.norm(C[e - 1, 0:3] - F[s - 1, 0:3])])
            if c["phase"] == 4:
                tot += 1; cut += c["removed_mm3"] - prev > 1.0
        prev = c["removed_mm3"]
    return {"rows": rows, "cut_chunks": [cut, tot]}


KEYS = ("cmd_net_mm", "cmd_path_mm", "first_step_mm", "step_mm", "meas_net_mm", "meas_path_mm", "endo_cmd_net_mm")


def med(R: list) -> dict:
    R = np.array(R)
    return {"n": len(R), **{k: round(float(np.median(R[:, i])), 2) for i, k in enumerate(KEYS)},
            "cmd_net_p25_p75": [round(float(np.percentile(R[:, 0], q)), 2) for q in (25, 75)]}


def demos(root: Path, n: int = 20) -> dict:
    import pandas as pd
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(root / "data/**/*.parquet"), recursive=True))], ignore_index=True)
    act, st = np.stack(df["action"].to_numpy()).astype(float), np.stack(df["observation.state"].to_numpy()).astype(float)
    ep, ph = df["episode_index"].to_numpy().reshape(-1), np.asarray(df["phase.index"].to_list()).reshape(-1).astype(int)
    out = {}
    for p in (3, 4):
        ok = np.flatnonzero((ph == p) & (np.r_[ep[n:], -np.ones(n)] == ep))   # the next n frames stay in the episode
        pos = np.concatenate([st[ok, None, 7:10], np.stack([act[ok + k, 7:10] for k in range(n)], 1)], 1)
        pe = np.concatenate([st[ok, None, 0:3], np.stack([act[ok + k, 0:3] for k in range(n)], 1)], 1)
        stp, net = np.linalg.norm(np.diff(pos, axis=1), axis=2), np.linalg.norm(pos[:, -1] - pos[:, 0], axis=1)
        out[p] = {"net_mm": round(float(np.median(net)), 2), "net_p25_p75": [round(float(np.percentile(net, q)), 2) for q in (25, 75)],
                  "path_mm": round(float(np.median(stp.sum(1))), 2), "step_mm": round(float(np.median(stp)), 2),
                  "endo_net_mm": round(float(np.median(np.linalg.norm(pe[:, -1] - pe[:, 0], axis=1))), 2)}
    return out


def together(R: list) -> dict:
    N = min(len(r["P"]) for r in R); D, out = [], {"pairs": {}}
    for a, b in itertools.combinations(range(len(R)), 2):
        d = np.linalg.norm(R[a]["P"][:N, 7:10] - R[b]["P"][:N, 7:10], axis=1); D.append(d)
        out["pairs"][f"{R[a]['name']} × {R[b]['name']}"] = {"median_mm": round(float(np.median(d)), 1), "p10_p90": [round(float(np.percentile(d, q)), 1) for q in (10, 90)],
                                                            "max_mm": round(float(d.max()), 1), "first_20_frames_mm": round(float(d[:20].mean()), 1)}
    D = np.array(D)
    out["per_5s_mm"] = [round(float(D[:, k:k + 75].mean()), 1) for k in range(0, N, 75)]
    out["self_5s_earlier_mm"] = [round(float(np.median(np.linalg.norm(r["P"][75:, 7:10] - r["P"][:-75, 7:10], axis=1)[r["ph"][75:] == 4])), 1) for r in R]
    out["first_cut_s"] = [round(float(np.argmax(r["vol"] > 1.0)) / 15.0, 1) for r in R]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--dataset", default="~/il-robot/datasets/spine_rev4_step_ph_trim")
    ap.add_argument("--min-frames", type=int, default=20)
    ap.add_argument("--together", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    out = {"demos": demos(Path(a.dataset).expanduser()), "runs": {}}
    for p, d in out["demos"].items():
        print(f"demos, phase {p}, over 20 steps: drill net {d['net_mm']} mm ({d['net_p25_p75'][0]}-{d['net_p25_p75'][1]}), path {d['path_mm']}, step {d['step_mm']} | endo net {d['endo_net_mm']}")
    R, every = [load(Path(r).expanduser()) for r in a.runs], {3: [], 4: []}
    for r in R:
        c = chunk_rows(r, a.min_frames); out["runs"][r["name"]] = {"cut_chunks": c["cut_chunks"]}
        for p in (3, 4):
            if c["rows"][p]:
                m = out["runs"][r["name"]][p] = med(c["rows"][p]); every[p] += c["rows"][p]
                print(f"{r['name']:24s} ph {p} n {m['n']:2d} | cmd net {m['cmd_net_mm']:.2f} ({m['cmd_net_p25_p75'][0]:.2f}-{m['cmd_net_p25_p75'][1]:.2f}) path {m['cmd_path_mm']:.2f} "
                      f"first step {m['first_step_mm']:.2f} later {m['step_mm']:.2f} | meas net {m['meas_net_mm']:.2f} path {m['meas_path_mm']:.2f} | endo cmd net {m['endo_cmd_net_mm']:.2f}"
                      + (f" | chunks that cut {c['cut_chunks'][0]}/{c['cut_chunks'][1]}" if p == 4 else ""))
    out["all"] = {p: med(v) for p, v in every.items() if v}
    for p, m in out["all"].items():
        print(f"all {len(R)} runs, phase {p}, {m['n']} chunks | cmd net {m['cmd_net_mm']:.2f} ({m['cmd_net_p25_p75'][0]:.2f}-{m['cmd_net_p25_p75'][1]:.2f}) path {m['cmd_path_mm']:.2f} "
              f"first step {m['first_step_mm']:.2f} later {m['step_mm']:.2f} | meas net {m['meas_net_mm']:.2f} path {m['meas_path_mm']:.2f} | endo cmd net {m['endo_cmd_net_mm']:.2f}")
    if a.together and len(R) > 1:
        t = out["together"] = together(R)
        for k, v in t["pairs"].items():
            print(f"  {k}: drill tips {v['median_mm']} mm apart (p10-p90 {v['p10_p90'][0]}-{v['p10_p90'][1]}, max {v['max_mm']}), first 20 frames {v['first_20_frames_mm']}")
        print(f"  per 5 s of policy time (mean of the pairs): {t['per_5s_mm']} | a run vs itself 5 s earlier while drilling: {t['self_5s_earlier_mm']} | first cut at {t['first_cut_s']} s")
    if a.json:
        Path(a.json).expanduser().write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
