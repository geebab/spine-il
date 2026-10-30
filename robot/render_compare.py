#!/usr/bin/env python3
"""Render a hand trajectory and a robot trajectory (replay_to_render.py) through the render API with the same
options and compare what the viewer did with each: endoscope camera, drill tip, removed bone volume, endoscope image.

    python robot/render_compare.py hand.json robot.json -o out_dir [--fps 10] [--api http://127.0.0.1:18780]

Both run as `simulate` (the engine drives and cuts) with the rig look (ae on, rawabs off), so the only difference
between the two is the pose stream — the recorded hand vs what the arms did. Writes compare.json + compare.md.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np


def auth() -> dict:
    """Bearer token of the render API: RENDER_API_TOKEN, else the token file the service writes."""
    f = Path.home() / ".config/spinearm-render-api/token"
    tok = os.environ.get("RENDER_API_TOKEN") or (f.read_text().strip() if f.exists() else "")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def call(api: str, path: str, body=None, timeout=120):
    req = urllib.request.Request(api + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **auth()})
    return urllib.request.urlopen(req, timeout=timeout).read()


def render(api: str, traj: dict, opts: dict) -> zipfile.ZipFile:
    job = json.loads(call(api, "/v1/jobs", {"input": traj, "options": opts}))
    jid = job["id"]
    while True:
        s = json.loads(call(api, f"/v1/jobs/{jid}"))
        if s["status"] in ("done", "failed", "error"):
            break
        time.sleep(5)
    if s["status"] != "done":
        raise SystemExit(f"job {jid}: {s['status']} {s.get('error')}")
    return zipfile.ZipFile(io.BytesIO(call(api, f"/v1/jobs/{jid}/result.zip", timeout=600)))


def metrics(z: zipfile.ZipFile) -> list[dict]:
    return [json.loads(x) for x in z.read("metrics.jsonl").decode().splitlines() if x.strip()]


def stats(d) -> dict:
    d = np.asarray([x for x in d if np.isfinite(x)], float)
    if not len(d):
        return {}
    return {"n": int(len(d)), "rms": round(float(np.sqrt((d ** 2).mean())), 3), "p50": round(float(np.median(d)), 3),
            "p95": round(float(np.percentile(d, 95)), 3), "max": round(float(d.max()), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hand")
    ap.add_argument("robot")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--api", default="http://127.0.0.1:18780")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    opts = {"fps": a.fps, "views": ["endo", "scene"], "ae": "on", "flags_override": {"rawabs": False}}
    M, Z = {}, {}
    for k in ("hand", "robot"):
        traj = json.loads(Path(getattr(a, k)).read_text())
        t0 = time.time()
        Z[k] = render(a.api, traj, opts)
        with open(out / f"{k}.zip", "wb") as f:
            with zipfile.ZipFile(f, "w", zipfile.ZIP_STORED) as w:
                for n in Z[k].namelist():
                    w.writestr(n, Z[k].read(n))
        M[k] = metrics(Z[k])
        print(f"[render] {k}: {len(M[k])} frames in {time.time() - t0:.0f} s", flush=True)

    n = min(len(M["hand"]), len(M["robot"]))
    rows = []
    for i in range(n):
        h, r = M["hand"][i]["bridge"], M["robot"][i]["bridge"]
        row = {"t": M["hand"][i]["t_ms"] / 1000.0}
        ch, cr = h.get("endoCamera"), r.get("endoCamera")
        if ch and cr:
            row["endo_cam_mm"] = float(np.linalg.norm(np.subtract(ch["position"], cr["position"])))
            vh = np.subtract(ch["focalPoint"], ch["position"]); vr = np.subtract(cr["focalPoint"], cr["position"])
            row["endo_dir_deg"] = float(np.degrees(np.arccos(np.clip(vh @ vr / np.linalg.norm(vh) / np.linalg.norm(vr), -1, 1))))
        dh, dr = h.get("drill"), r.get("drill")
        if dh and dr and dh.get("B") and dr.get("B"):
            row["drill_tip_mm"] = float(np.linalg.norm(np.subtract(dh["B"], dr["B"])))
        row["removed_hand"], row["removed_robot"] = h.get("removedMm3"), r.get("removedMm3")
        rows.append(row)
    # endoscope images: mean absolute difference, 0–255
    img = []
    try:
        from PIL import Image
        names = sorted(x for x in Z["hand"].namelist() if x.startswith("frames/endo/"))
        for nm in names[:n]:
            if nm in Z["robot"].namelist():
                A = np.asarray(Image.open(io.BytesIO(Z["hand"].read(nm))).convert("L"), float)
                B = np.asarray(Image.open(io.BytesIO(Z["robot"].read(nm))).convert("L"), float)
                img.append(float(np.abs(A - B).mean()))
    except ImportError:
        pass
    res = {"frames": n, "fps": a.fps, "options": opts,
           "endo_camera_position_mm": stats([r_.get("endo_cam_mm", np.nan) for r_ in rows]),
           "endo_view_direction_deg": stats([r_.get("endo_dir_deg", np.nan) for r_ in rows]),
           "drill_tip_mm": stats([r_.get("drill_tip_mm", np.nan) for r_ in rows]),
           "removed_mm3_final": {"hand": rows[-1]["removed_hand"] if rows else None,
                                 "robot": rows[-1]["removed_robot"] if rows else None},
           "endo_image_mean_abs_diff": stats(img)}
    (out / "compare.json").write_text(json.dumps({"summary": res, "per_frame": rows}, indent=1))
    md = [f"# 렌더 비교 — 손(녹화) vs 로봇 팔", "", f"{n} 프레임 @ {a.fps:g} fps, simulate, ae on, rawabs off", "",
          "| 항목 | RMS | p50 | p95 | 최대 |", "|---|---|---|---|---|"]
    for k, lab in (("endo_camera_position_mm", "내시경 카메라 위치 (mm)"), ("endo_view_direction_deg", "내시경 시선 방향 (°)"),
                   ("drill_tip_mm", "드릴 팁 (mm)"), ("endo_image_mean_abs_diff", "내시경 영상 평균 밝기차 (0–255)")):
        s = res[k]
        if s:
            md.append(f"| {lab} | {s['rms']} | {s['p50']} | {s['p95']} | {s['max']} |")
    md += ["", f"제거된 뼈 부피 (끝): 손 {res['removed_mm3_final']['hand']} mm³ · 로봇 {res['removed_mm3_final']['robot']} mm³"]
    (out / "compare.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
