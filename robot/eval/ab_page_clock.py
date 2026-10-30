#!/usr/bin/env python3
"""A/B of the render bridge's page_clock="jump" on a real session, against the cached default replay.

    python ab_page_clock.py ROBOT_SESSION.jsonl [--optics water] [--fps 5]            # the evaluation's replay (no images)
    python ab_page_clock.py SESSION.jsonl --images --t0 60 --t1 90 [--views endo scene]   # pictures of a window, both ways

Evaluation mode: policy_eval.py's robot replay (removed voxels every 3 s + bridge metrics, no views) is run once more with
page_clock="jump" and compared with the cache the default run left (iou_cache_rofull_<session>.npz, view_rofull_….json):
the final removed set, every 3 s snapshot, the view statistics, the replay's own pass. The option may replace the default
for evaluations only if all of them are identical — it changes how often the page redraws, not what the replay applies.
Image mode: the same window rendered both ways; per view the share of pixels that differ and the largest and mean
difference. Needs the render API idle (no robot run beside it).
"""
from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np

import iou_eval as E
import policy_eval as PE


def result(z) -> dict:
    return json.loads(z.read("manifest.json")) if "manifest.json" in z.namelist() else {}


def evaluation(sf: Path, optics: str | None, fps: float) -> dict:
    key = "rofull_" + sf.stem
    ref = E.cached(key, None)
    ref_view = json.loads((E.OUT / f"view_{key}.json").read_text())
    opts = {**PE.REPLAY, "fps": fps, "metrics": True, "page_clock": "jump", **({"optics": optics} if optics else {})}
    t0 = time.time()
    z = E.render(str(sf), opts)
    dt = time.time() - t0
    vox = E.voxels(z)
    rows = [json.loads(l) for l in z.read("metrics.jsonl").decode().splitlines() if l.strip()]
    a, b = np.sort(ref["final"].astype(np.int64)), np.sort(vox["final"].astype(np.int64))
    snaps_ref, snaps = ref.get("snaps") or {}, vox["snaps"]
    common = sorted(set(map(float, snaps_ref)) & set(map(float, snaps)))
    get = lambda S, t: np.sort(np.asarray(S[t] if t in S else S[str(t)]).astype(np.int64))
    snap_diff = [t for t in common if not np.array_equal(get(snaps_ref, t), get(snaps, t))]
    rp = result(z).get("replay") or {}     # the replay's own summary (replay.py _summary) — not the recording's gate
    return {"session": sf.name, "seconds_jump": round(dt), "frames": len(rows), "page_clock": rp.get("pageClock"), "replay_pass": rp.get("pass"),
            "fail_reasons": rp.get("failReasons"), "cut_rows": [rp.get("cutRowsApplied"), rp.get("cutRowsExpected")],
            "voxels_ref": int(len(a)), "voxels_jump": int(len(b)),
            "final_identical": bool(np.array_equal(a, b)), "iou": round(float(E.iou(a, b)[0]), 6),
            "snapshots_compared": len(common), "snapshots_ref": len(snaps_ref), "snapshots_jump": len(snaps), "snapshots_differ_at_s": snap_diff[:10],
            "view_ref": ref_view, "view_jump": PE.view_stats(rows, optics),
            "view_identical": all(ref_view.get(k) == v for k, v in PE.view_stats(rows, optics).items() if k in ref_view)}


def images(sf: Path, optics: str | None, fps: float, t0: float, t1: float, views: list[str], size: int) -> dict:
    from PIL import Image
    out, shots = {}, {}
    for clock in ("screen", "jump"):
        opts = {"fps": fps, "views": views, "size": size, "status": False, "grid": "tMono", "t_start_s": t0, "t_end_s": t1,   # as render_video.py
                "removed_voxels": True, "page_clock": clock, **({"optics": optics} if optics else {})}
        t = time.time()
        z = E.render(str(sf), opts)
        out[f"seconds_{clock}"] = round(time.time() - t)
        shots[clock] = {n: np.asarray(Image.open(io.BytesIO(z.read(n))).convert("RGB"), np.int16)
                        for n in sorted(z.namelist()) if n.lower().endswith((".png", ".jpg"))}
    names = sorted(set(shots["screen"]) & set(shots["jump"]))
    out["frames_compared"] = len(names)
    for v in views:
        ds = [np.abs(shots["screen"][n] - shots["jump"][n]) for n in names if f"/{v}" in n or n.startswith(v) or f"_{v}" in n]
        if ds:
            out[v] = {"frames": len(ds), "max_abs": int(max(d.max() for d in ds)), "mean_abs": round(float(np.mean([d.mean() for d in ds])), 4),
                      "share_px_over_2": round(float(np.mean([(d.max(axis=2) > 2).mean() for d in ds])), 5),
                      "worst_frame_mean_abs": round(float(max(d.mean() for d in ds)), 4)}
    out["names_example"] = names[:3]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--optics", default="water")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--t0", type=float, default=60.0)
    ap.add_argument("--t1", type=float, default=90.0)
    ap.add_argument("--views", nargs="+", default=["endo", "scene"])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()
    sf = Path(a.session).expanduser()
    r = images(sf, a.optics or None, a.fps, a.t0, a.t1, a.views, a.size) if a.images else evaluation(sf, a.optics or None, a.fps)
    print(json.dumps(r, ensure_ascii=False))
    if a.out:
        Path(a.out).expanduser().write_text(json.dumps(r, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
