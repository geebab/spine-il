#!/usr/bin/env python3
"""Judge check (b), part 2: YOUR sentence drives the policy in a simulated closed loop → two videos.

Starts at the pose where robot run 17 began drilling (L4, left), sends the policy endpoint the rendered endoscope image, the
tool state and your sentence, moves the virtual tools along the first 8 of the 40 predicted rows (as the robot did), renders
again … until the policy reports the drilling done (progress ≥ 0.85) or --max-s seconds of policy time. No robot arms: the
tools go exactly where the policy says (each step clamped to 2 mm / 3°, the robot's safety limit). Output in --out:
endo.mp4 (the endoscope view) and scene.mp4 (the 3D view) at 7.5 fps, frames/, and summary.json. About 20–30 min per sentence
through the hosted endpoint (each answer is one policy call plus 8 render steps over the internet).

    POLICY_URL=… RENDER_URL=… ENDPOINT_TOKEN=… python robot/eval/judge/sentence_loop.py "Your sentence." [--out DIR] [--max-s 150]
    python robot/eval/judge/sentence_loop.py --examples        # example sentences, none of them used in training

Needs numpy and websockets (pip install numpy websockets); ffmpeg on PATH for the mp4s (otherwise the PNG frames stay).

Read the result with the limitation in mind: the render API's step session removes about 60 % of the bone that the
recording page removed on the robot for the same tool path, so the simulated drill meets more bone than the robot did and
can stop before the midline whatever the sentence. The robot results are the 105 recorded runs (reproduce_runs.py).
"""
import argparse
import asyncio
import base64
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA, RUN17, auth, token  # noqa: E402

POLICY = (os.environ.get("POLICY_URL") or "http://127.0.0.1:8900").rstrip("/")
RENDER = (os.environ.get("RENDER_URL") or "http://127.0.0.1:18780").rstrip("/")
MID = 21.9          # L4 midline x (mm, scene frame); the drill works from the left at x > MID
START = 116         # run 17's first drilling observation answered by the policy (run17_live/frames/00116.png)
N_EXEC, EVERY = 8, 2  # rows executed per answer (the robot's 8 of 40); a video frame every 2nd row (15 fps → 7.5 fps)
END_LEVEL, END_HOLD = 0.85, 5
# None of these is a training sentence (checked against the dataset's 131 tasks, lerobot/meta/tasks.parquet) or one of the
# held-out sentences of the 105 robot runs. The robot run's own two sentences are unseen in training too.
EXAMPLES = {
    "contralateral": ["Go across the midline and decompress the right side as well.",
                      "Free up both sides at L4, working from the left past the spinous process.",
                      "Keep going after the left lamina until the opposite side is decompressed too."],
    "ipsilateral": ["Stay on the left side and stop before you reach the midline.",
                    "Only the left half of L4, leave everything on the right alone."],
    "robot runs 17 / 19 (unseen in training, used on the robot)": [
        "Take down the left L4 lamina and carry on over to the opposite side.",
        "Take down the left L4 lamina only, leaving the right side untouched."],
}


def act(png: bytes, state, task: str, seq: int) -> np.ndarray:
    body = {"protocol": "spine-policy/1", "seq": seq, "run": "sentence_loop", "image_png_b64": base64.b64encode(png).decode(),
            "state": [float(v) for v in state], "task": task}
    r = urllib.request.Request(POLICY + "/act", data=json.dumps(body).encode(), headers={"Content-Type": "application/json", **auth()})
    return np.asarray(json.loads(urllib.request.urlopen(r, timeout=120).read())["action"], float)


def pose(v, off):
    q = np.asarray(v[off + 3:off + 7], float); q = q / np.linalg.norm(q)
    return {"tip": np.round(v[off:off + 3], 4).tolist(), "quat": np.round(q, 6).tolist()}


def clamp(prev, nxt, max_mm=2.0, max_deg=3.0):
    """robot/inference/safety.py limit_step: each tool's tip move and rotation per row, relative to the last row."""
    out = np.array(nxt, float)
    for off in (0, 7):
        d = out[off:off + 3] - prev[off:off + 3]; n = float(np.linalg.norm(d))
        if n > max_mm:
            out[off:off + 3] = prev[off:off + 3] + d * (max_mm / n)
        q0 = prev[off + 3:off + 7] / np.linalg.norm(prev[off + 3:off + 7]); q1 = out[off + 3:off + 7] / np.linalg.norm(out[off + 3:off + 7])
        if np.dot(q0, q1) < 0:
            q1 = -q1
        ang = 2 * np.degrees(np.arccos(np.clip(float(np.dot(q0, q1)), -1, 1)))
        if ang > max_deg:  # slerp toward q1 by max_deg
            w = np.radians(ang / 2); t = max_deg / ang
            q1 = (np.sin((1 - t) * w) * q0 + np.sin(t * w) * q1) / np.sin(w)
        out[off + 3:off + 7] = q1 / np.linalg.norm(q1)
    return out


def removed_of(step_reply: dict):
    """The viewer's removed-bone volume (mm³) in a step reply's meta (bridge metrics), None if it is not there."""
    m = step_reply.get("meta") or {}
    r = m.get("removedMm3", (m.get("bridge") or {}).get("removedMm3"))
    return float(r) if r is not None else None


def header(session: Path) -> dict:
    with open(session) as fh:
        return next(d for d in map(json.loads, fh) if d.get("k") == "hdr")


async def run(a):
    import websockets
    live = json.loads((DATA / "run17_live" / "live.json").read_text())
    hdr = header(DATA / "sessions" / f"{RUN17}.jsonl")
    v = np.asarray(live["frames"][START][:15], float)
    fe = hdr.get("fe") or {}
    flags = [f"{k}={fe[k]}" for k in ("bite", "meltk", "meltv", "touchoff", "meltmax") if fe.get(k) is not None]
    T0 = hdr["registration"]["T0"]
    ws_url = RENDER.replace("https://", "wss://").replace("http://", "ws://") + "/v1/step"
    out = Path(a.out); shutil.rmtree(out, ignore_errors=True)
    for view in ("endo", "scene"):
        (out / "frames" / view).mkdir(parents=True)
    async with websockets.connect(ws_url, additional_headers={"Authorization": f"Bearer {token()}"}, max_size=64 << 20,
                                  open_timeout=120, ping_interval=None) as ws:
        reset = {"op": "reset", "case": hdr.get("meshId") or "altician_lumbar1", "registration": {"R": T0["R"], "t": T0["t"]},
                 "tools": {t: {"tipLocal": hdr["toolCalib"][t]["tipLocal"]} for t in ("endo", "drill")},
                 "options": {"views": ["endo", "scene"], "size": 256, "drill_rule": "r4",
                             "drill_size": (hdr.get("drill") or {}).get("sizeKey", "medium"), "optics": "water",
                             "extra_flags": flags, "ae": "on", "flags_override": {"rawabs": False}},
                 "start": {"endo": pose(v, 0), "drill": pose(v, 7)}}
        await ws.send(json.dumps(reset)); r = json.loads(await ws.recv())
        if not r.get("ok"):
            sys.exit(f"reset refused: {r}")
        dt = 1000.0 / 15
        n_img = 0

        async def step(vv, images):
            nonlocal n_img
            msg = {"op": "step", "dt_ms": dt, "endo": pose(vv, 0), "drill": {**pose(vv, 7), "trig": bool(vv[14] > 0.5)}}
            if not images:
                msg["images"] = False
            await ws.send(json.dumps(msg)); meta = json.loads(await ws.recv())
            if not meta.get("ok"):
                sys.exit(f"step refused: {meta}")
            imgs = {}
            for im in meta.get("images", []):
                imgs[im["view"]] = await ws.recv()
            if imgs:
                n_img += 1
                for view, b in imgs.items():
                    (out / "frames" / view / f"{n_img:06d}.png").write_bytes(b)
            return imgs, meta

        imgs, meta = await step(v, True)
        if a.check:
            (out / "start_endo.png").write_bytes(imgs["endo"])
            print(json.dumps({"start_frame": START, "meta_keys": sorted(meta.get("meta", {}).keys()), "views": list(imgs)}))
            return
        rows_done, seq, hold, t0, x_min = 0, 0, 0, time.time(), float(v[7])
        log, removed, cut_x, cut_mm3 = [], removed_of(meta), [], []
        while rows_done < a.max_s * 15:
            seq += 1
            A = act(imgs["endo"], v, a.sentence, seq)
            prog = float(A[min(N_EXEC, len(A)) - 1, 15]) if A.shape[1] > 15 else float("nan")
            for i in range(N_EXEC):
                v = clamp(v, A[i, :15])
                imgs_i, meta = await step(v, (i % EVERY == EVERY - 1))
                if imgs_i:
                    imgs = imgs_i
                rows_done += 1; x_min = min(x_min, float(v[7]))
                r_now = removed_of(meta)
                if r_now is not None and removed is not None and r_now > removed:   # book the new cut to the drill tip's x
                    cut_x.append(float(v[7])); cut_mm3.append(r_now - removed)
                removed = r_now if r_now is not None else removed
            log.append({"answer": seq, "t_s": round(rows_done / 15, 1), "drill_x": round(float(v[7]), 2),
                        "progress": round(prog, 3), "trig": bool(v[14] > 0.5), "removed_mm3": removed})
            print(json.dumps(log[-1]), flush=True)
            hold = hold + 1 if prog >= END_LEVEL else 0
            if hold >= 1 and rows_done >= END_HOLD:   # the progress of the last executed row is past the end level
                break
        await ws.send(json.dumps({"op": "close"}))
    summary = {"sentence": a.sentence, "answers": seq, "policy_s": round(rows_done / 15, 1), "wall_s": round(time.time() - t0, 1),
               "ended": "policy (progress ≥ 0.85)" if hold else f"time limit ({a.max_s} s)",
               "removed_mm3": round(float(sum(cut_mm3)), 1),
               "removed_past_midline_pct": round(100 * float(sum(m for x, m in zip(cut_x, cut_mm3) if x < MID)) / max(1e-9, sum(cut_mm3)), 1),
               "deepest_cut_past_midline_mm": round(MID - min(cut_x), 1) if cut_x else None,
               "drill_x_start": round(float(live["frames"][START][7]), 2), "drill_x_min": round(x_min, 2), "midline_x": MID,
               "share_note": "removed volume booked to the drill tip's x when it was cut (a proxy for the paper's voxel share)",
               "note": "simulated loop; the step session removes ~60 % of the bone the robot's page did — see the docstring"}
    (out / "summary.json").write_text(json.dumps({"summary": summary, "log": log}, indent=1))
    if shutil.which("ffmpeg"):
        for view in ("endo", "scene"):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(15 / EVERY), "-i",
                            str(out / "frames" / view / "%06d.png"), "-vf", "scale=512:512:flags=neighbor",
                            "-pix_fmt", "yuv420p", str(out / f"{view}.mp4")], check=False)
    print(json.dumps(summary, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sentence", nargs="?")
    ap.add_argument("--examples", action="store_true", help="print example sentences (none used in training) and exit")
    ap.add_argument("--out", default=None, help="output folder (default: data/sentence_loop/<time>)")
    ap.add_argument("--max-s", type=float, default=150, help="policy seconds before the loop stops (run 17 drilled 140 s)")
    ap.add_argument("--check", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.examples or not (a.sentence or a.check):
        for k, ss in EXAMPLES.items():
            print(f"{k}:"); [print(f'  "{s}"') for s in ss]
        print("\nNone of these was a training sentence. Pass one (or your own) in quotes.")
        return
    a.out = a.out or str(DATA.parent / "sentence_loop" / time.strftime("%H%M%S"))
    asyncio.run(run(a))


if __name__ == "__main__":
    main()
