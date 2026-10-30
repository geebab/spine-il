#!/usr/bin/env python3
"""Judge check (a): reproduce the video's and the paper's run-17 / run-19 numbers from the RECORDED robot sessions — no model,
no robot. Each session goes to the hosted render API as a replay (its recorded cut rows are applied again on the 0.8 mm voxel
grid); the result is the volume of bone removed and the share of it past the L4 midline (x < 21.9 mm, scene frame) — the
paper's "contralateral fraction". Expected:

    run 17 (told to go contralateral, the furthest contralateral run)   1069.6 mm³   26.8 %
    run 19 (told to stay ipsilateral)                                    944.1 mm³    0.5 %

The service replays in real time: about 17 min for run 17 and 12 min for run 19.

--video also asks the renderer for the two viewer panes of the replay — the endoscope view and the 3D view, as displayed —
and saves them as $SPINE_IL_DATA/../videos/run17_endo.mp4, run17_scene.mp4, run19_… (512 × 512, 10 fps). The numbers come
from the same job. Capturing the frames makes the replay slower — measured on the endpoint: run 19 31 min instead of 12
(3,623 frames, 0.5 s each), so about 45 min for run 17 and 75 min for both; the four videos are ~100 MB together.

    RENDER_URL=https://<render endpoint> ENDPOINT_TOKEN=<token> python robot/eval/judge/reproduce_runs.py [--video] [SESSION.jsonl …]
    (no arguments: run 17 and run 19 from $SPINE_IL_DATA, fetched by fetch_robot_runs.py; each is compared with the
    evaluation's own voxel set in eval_cache/ when present)"""
import io
import json
import os
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, RUN17, RUN19, auth  # noqa: E402

API = (os.environ.get("RENDER_URL") or os.environ.get("RENDER_API_URL") or "http://127.0.0.1:18780").rstrip("/")
# the evaluation's replay options: no images, 5 fps bookkeeping, water optics, exact removed voxels
REPLAY = {"views": [], "status": False, "fps": 5.0, "optics": "water", "removed_voxels": True, "removed_every_s": 3}
MID = 21.9   # L4 midline x (mm, scene frame) — robot/dataset/instructions.py MIDLINE_X_MM["L4"]


def call(path, body=None, timeout=120):
    r = urllib.request.Request(API + path, data=None if body is None else json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", **auth()})
    return urllib.request.urlopen(r, timeout=timeout).read()


VIDEO = {"views": ["endo", "scene"], "video": True, "fps": 10.0, "size": 512}
LABEL = {RUN17: "run17", RUN19: "run19"}


def replay(session: Path, video_dir: Path | None = None):
    """The recording's text goes inline; a recording with cut rows is replayed (its recorded cuts), not re-simulated.
    With video_dir, the same job also renders the endoscope and 3D panes and their two mp4s are saved there."""
    options = {**REPLAY, **VIDEO} if video_dir else REPLAY
    j = json.loads(call("/v1/jobs", {"input": session.read_text(), "options": options}, 600))
    while True:
        s = json.loads(call(f"/v1/jobs/{j['id']}"))
        if s["status"] in ("done", "failed", "error"):
            break
        time.sleep(3)
    if s["status"] != "done":
        raise RuntimeError(f"render job {j['id']}: {s['status']} {s.get('error')}")
    z = zipfile.ZipFile(io.BytesIO(call(f"/v1/jobs/{j['id']}/result.zip", timeout=1200)))
    if video_dir:
        video_dir.mkdir(parents=True, exist_ok=True)
        for view in VIDEO["views"]:
            dst = video_dir / f"{LABEL.get(session.stem, session.stem)}_{view}.mp4"
            dst.write_bytes(z.read(f"frames/{view}.mp4"))
            print("saved", dst, flush=True)
    return {"hdr": json.loads(z.read("removed/header.json")), "final": np.frombuffer(z.read("removed/final.bin"), "<u4"),
            "replayMode": s.get("summary", {}).get("replayMode")}


def share(final, h):
    idx = np.asarray(final, np.int64)
    x = h["origin"][0] + (idx % h["dims"][0]) * h["spacing"][0]
    return {"removed_mm3": round(float(len(idx) * h["voxelMm3"]), 1), "past_midline_pct": round(float(100 * np.mean(x < MID)), 1),
            "reach_mm": round(float(MID - np.percentile(x, 5)), 1)}


def main():
    args = sys.argv[1:]
    video_dir = DATA.parent / "videos" if "--video" in args else None
    sessions = [Path(p) for p in args if p != "--video"] or [DATA / "sessions" / f"{s}.jsonl" for s in (RUN17, RUN19)]
    for s in sessions:
        if not s.exists():
            sys.exit(f"missing {s} — run robot/eval/judge/fetch_robot_runs.py first")
        t0 = time.time(); v = replay(s, video_dir)
        row = {"session": s.stem, "replayMode": v["replayMode"], **share(v["final"], v["hdr"]), "replay_s": round(time.time() - t0, 1)}
        ref = DATA / "eval_cache" / f"iou_cache_rofull_{s.stem}.npz"
        if ref.exists():
            r = np.load(ref); a, b = set(v["final"].tolist()), set(r["final"].tolist())
            row["evaluation"] = share(r["final"], json.loads(str(r["hdr"])))
            row["voxel_iou_vs_evaluation"] = round(len(a & b) / max(1, len(a | b)), 4)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
