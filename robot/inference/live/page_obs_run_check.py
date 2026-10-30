#!/usr/bin/env python3
"""After a closed-loop run that looked at the recorder page (live_loop.py --obs-url): was what the policy saw the
session the run recorded?

    python robot/inference/live/page_obs_run_check.py --live RUN/live.json --frames LIVE/NAME/frames --job JOB_DIR --out DIR

The policy's images are in --frames (one per look, named by frame index; live.json meta.page_observations has the wall
time of each look). The run's session, replayed by the render API with its own cut rows for the 5-panel video, is in
--job (frames/endo, metrics.jsonl with the camera time shown per frame; the job id is in the run's video.log). For
every look the replay frame nearest in time is compared with the image the policy got: the arms were holding, so the two
should show the same tools and the same bone. Reported: the pixel difference per look (and its spread), the removed
volume the loop noted against the replay's at that time, and the looks without a replay frame within --max-gap-s.
A sheet of the first, middle, last and worst looks (policy | replay | difference × 4) goes to --out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    from PIL import Image
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--job", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--view", default="endo")
    ap.add_argument("--max-gap-s", type=float, default=0.6)
    a = ap.parse_args()
    meta = json.loads(Path(a.live).expanduser().read_text())["meta"]
    if meta.get("observation") != "page":
        raise SystemExit(f"this run looked at {meta.get('observation') or 'a step session'}, not at the recorder page")
    job, fdir = Path(a.job).expanduser(), Path(a.frames).expanduser()
    with open(job / "recording.jsonl") as fh:
        hdr = json.loads(fh.readline())
    rows = [json.loads(l) for l in (job / "metrics.jsonl").read_text().splitlines() if l.strip()]
    # the wall time a replay frame stands for: the camera time on screen (tExp grid) or screen time since the session's start
    wall = np.array([r["tExpShown"] if r.get("tExpShown") is not None else hdr["t0Wall"] + (r.get("seek") or {}).get("tMs", r["t_ms"])
                     for r in rows], float)
    vol_job = np.array([float((r.get("bridge") or {}).get("removedMm3") or (r.get("seek") or {}).get("removedMm3") or 0.0) for r in rows])
    vols = meta.get("removed_mm3") or []
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    res, far = [], 0
    for i, t_ms, has in meta.get("page_observations") or []:
        f = fdir / f"{i:05d}.png"
        if not has or not f.exists():
            continue
        k = int(np.argmin(np.abs(wall - t_ms)))
        gap = (wall[k] - t_ms) / 1000.0
        if abs(gap) > a.max_gap_s:
            far += 1
            continue
        img = np.asarray(Image.open(f).convert("RGB"))
        ref = Image.open(job / "frames" / a.view / f"{rows[k]['i']:06d}.png").convert("RGB")
        if ref.size != img.shape[1::-1]:
            ref = ref.resize(img.shape[1::-1], Image.LANCZOS)
        ref = np.asarray(ref)
        d = np.abs(img.astype(np.int16) - ref.astype(np.int16))
        res.append({"frame": i, "job_frame": rows[k]["i"], "gap_s": round(gap, 3), "mean_abs_diff": round(float(d.mean()), 2),
                    "pixels_over_32": round(float((d.max(axis=2) > 32).mean()), 4),
                    "brightness": [round(float(img.mean()), 1), round(float(ref.mean()), 1)],
                    "removed_mm3": [round(float(vols[i]), 1) if i < len(vols) and vols[i] is not None else None, round(float(vol_job[k]), 1)],
                    "_img": img, "_ref": ref, "_d": d})
    if not res:
        raise SystemExit(f"no look could be compared ({far} without a replay frame within {a.max_gap_s} s)")
    md = np.array([r["mean_abs_diff"] for r in res])
    dv = np.array([r["removed_mm3"][0] - r["removed_mm3"][1] for r in res if r["removed_mm3"][0] is not None])
    worst = int(np.argmax(md))
    gapcol = np.full((res[0]["_img"].shape[0], 8, 3), 255, np.uint8)
    pick = sorted({0, len(res) // 4, len(res) // 2, 3 * len(res) // 4, len(res) - 1, worst})
    sheet = np.concatenate([np.concatenate([res[j]["_img"], gapcol, res[j]["_ref"], gapcol, np.clip(res[j]["_d"] * 4, 0, 255).astype(np.uint8)], axis=1)
                            for j in pick], axis=0)
    Image.fromarray(sheet).save(out / "policy_vs_session.png")   # rows = looks; policy's image | replay of the session | difference × 4
    q = lambda x, p: round(float(np.percentile(x, p)), 2)
    summary = {"looks": len(res), "not_compared": far, "mean_abs_diff": {"p10": q(md, 10), "median": q(md, 50), "p90": q(md, 90), "max": q(md, 100)},
               "worst_look": {k: v for k, v in res[worst].items() if not k.startswith("_")},
               "removed_mm3_loop_minus_replay": {"median": q(dv, 50), "min": q(dv, 0), "max": q(dv, 100)} if len(dv) else None,
               "sheet_rows": [res[j]["frame"] for j in pick], "sheet": str(out / "policy_vs_session.png")}
    (out / "policy_vs_session.json").write_text(json.dumps({"summary": summary, "looks": [{k: v for k, v in r.items() if not k.startswith("_")} for r in res]}, indent=1))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
