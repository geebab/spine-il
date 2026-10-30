#!/usr/bin/env python3
"""A recording (policy2rec output, hand or robot session) replayed in the FE viewer → mp4 (endoscope | 3D side by side;
with --cams also the run's IR | RGB | webcam recordings underneath — 5 panels).

    python robot/inference/render_video.py REC.jsonl -o out.mp4 [--fps 15] [--size 512] [--views endo,scene]

Policy recordings have no cut rows, so they run in `simulate` (drill rule r4) at the live page's 15 Hz tick — the cut you
see is what the viewer does with the policy's poses; hand/robot recordings replay their own recorded cuts. Frames come
from the render API (frames/<view>/NNNNNN.png) and are joined with ffmpeg (hstack, H.264, yuv420p).

The render job is removed once its video is verified (user, 10-09: "영상 렌더 중간 프레임 삭제 및 규칙 허용"): a 5-panel
video's job leaves 3-5.5 GB in the service's jobs folder (the frames of two views and the same again as result.zip), eleven of
them were 33 GB when the 233 GB disk ran full on 10-08. Verified = the mp4 (and the _motion.mp4) holds exactly the frames
that were written to its encoder, counted with ffprobe. Kept instead with --keep-job or RENDER_VIDEO_KEEP_JOB=1 (e.g. for
live/page_obs_run_check.py, which reads the job's frames) and always after --job (composing again from a kept job).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from rollout import token  # noqa: E402

API = os.environ.get("RENDER_API_URL", "http://127.0.0.1:18780")   # another service: ube-render-b :18781, a Mac node's tunnel
JOBS = Path(os.environ.get("UBE_RENDER_JOBS_DIR", Path.home() / "il-robot/render-jobs"))   # the service's job folders (same machine)


def job_input(src):
    """A job's input for the service in API: the path as it is (the service on this machine reads the file itself), or —
    RENDER_API_INLINE=1, a service on another machine such as a Mac render node behind an SSH tunnel — the file's text."""
    if os.environ.get("RENDER_API_INLINE") == "1" and isinstance(src, (str, Path)) and "\n" not in str(src) and Path(src).is_file():
        return Path(src).read_text()
    return src


def call(path: str, body=None, t: int = 120) -> bytes:
    h = {"Content-Type": "application/json"}
    tok = token()
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(API + path, data=None if body is None else json.dumps(body).encode(), headers=h,
                                 method="GET" if body is None else "POST")
    with urllib.request.urlopen(req, timeout=t) as r:
        return r.read()


def video_frames(path: str) -> int:
    """Frames in an mp4's video stream (packets counted by ffprobe, nothing decoded); -1 = not readable."""
    try:
        o = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                            "stream=nb_read_packets", "-of", "csv=p=0", path], capture_output=True, text=True, timeout=900).stdout
        o = o.strip().split(",")[0]
        return int(o) if o.isdigit() else -1
    except Exception:
        return -1


def drop_job(jid: str) -> str | None:
    """Remove a finished render job: through the service (DELETE /v1/jobs/<id> — it removes the folder and forgets the job),
    and the folder itself if it is still there (a service restarted since the render no longer knows the job)."""
    how = None
    try:
        h = {"Authorization": f"Bearer {token()}"} if token() else {}
        with urllib.request.urlopen(urllib.request.Request(API + f"/v1/jobs/{jid}", headers=h, method="DELETE"), timeout=120) as r:
            how = "service" if r.status == 200 else None
    except Exception:
        pass
    jd = JOBS / jid
    if re.fullmatch(r"[0-9a-f]{32}", jid) and jd.is_dir() and jd.parent == JOBS:
        shutil.rmtree(jd, ignore_errors=True)
        how = how or "folder"
    return how


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--views", default="endo,scene")
    ap.add_argument("--optics", choices=("air", "water"), default=None,
                    help="endoscope medium of the replay (default: the recording's scopecal flag) — water for the runs of a "
                         "water-trained policy, whose session header still says air")
    ap.add_argument("--cams", default=None, help="cam_capture.py output dir: adds IR | RGB | webcam under the two viewer "
                    "panels (5 panels in one mp4), each camera frame the nearest in time to the viewer frame")
    ap.add_argument("--windows", default=None, help="JSON [[t0_ms, t1_ms], …] (wall clock): with --cams, also write "
                    "<out>_motion.mp4 with only the frames inside these windows — a closed-loop run (live_loop.py) without the "
                    "holds between chunks")
    ap.add_argument("--job", default=None, help="compose again from this finished render job instead of rendering (its result is "
                    "still under the service's jobs folder)")
    ap.add_argument("--keep-job", action="store_true", help="keep the render job (its frames) after the video is verified; "
                    "also RENDER_VIDEO_KEEP_JOB=1. Default: the job is removed — always kept after --job")
    ap.add_argument("--no-strip", action="store_true", help="5 panels without the phase / trigger / bone-cut strip")
    ap.add_argument("--play-fps", type=float, default=None,
                    help="encode at this fps (default = --fps); a 0.33× robot session rendered at 5 fps and played at 15 "
                         "runs at the recording's own speed")
    a = ap.parse_args()
    with open(a.recording) as fh:
        hdr = next(d for d in map(json.loads, fh) if d.get("k") == "hdr")
    policy = (hdr.get("origin") or {}).get("source") == "policy"
    views = [v for v in a.views.split(",") if v]
    opts = {"views": views, "fps": a.fps, "size": a.size, "status": False, "removed_voxels": True,
            **({"optics": a.optics} if a.optics else {})}
    if policy:
        opts.update({"drill_rule": "r4", "tick_hz": 15.0})
    if os.environ.get("RENDER_VIDEO_PAGE_CLOCK"):   # jump: one redraw per output frame (bridge render.py page_clock) —
        opts["page_clock"] = os.environ["RENDER_VIDEO_PAGE_CLOCK"]   # set only after the pictures were compared (eval/ab_page_clock.py --images)
    j = {"id": a.job} if a.job else json.loads(call("/v1/jobs", {"input": job_input(a.recording), "options": opts}, 600))
    print(json.dumps({"job": j["id"]}), flush=True)   # --job <id> composes again; live/page_obs_run_check.py reads its frames
    t0 = time.time()
    while True:
        s = json.loads(call(f"/v1/jobs/{j['id']}"))
        if s["status"] in ("done", "failed", "error"):
            break
        time.sleep(5)
    if s["status"] != "done":
        raise SystemExit(f"render job {j['id']}: {s['status']} {s.get('error')}")
    removed = None
    with tempfile.TemporaryDirectory(dir=str(Path(a.out).resolve().parent)) as td:   # beside the output, on disk
        td = Path(td)
        jd = JOBS / j["id"]
        if (jd / "frames" / views[0]).is_dir():   # same machine: the job's own files (a long session's zip is gigabytes)
            fdir, mtext = jd / "frames", (jd / "metrics.jsonl").read_text() if (jd / "metrics.jsonl").exists() else ""
            if (jd / "removed" / "header.json").exists():
                removed = json.loads((jd / "removed" / "header.json").read_text()).get("final", {}).get("mm3")
        else:
            zf = td / "result.zip"
            with urllib.request.urlopen(urllib.request.Request(API + f"/v1/jobs/{j['id']}/result.zip", headers={"Authorization": f"Bearer {token()}"} if token() else {}),
                                        timeout=900) as r, open(zf, "wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 20)
            z = zipfile.ZipFile(zf)
            z.extractall(td, [n for n in z.namelist() if n.startswith("frames/") and n.endswith(".png")])
            fdir, mtext = td / "frames", z.read("metrics.jsonl").decode() if "metrics.jsonl" in z.namelist() else ""
            if "removed/header.json" in z.namelist():
                removed = json.loads(z.read("removed/header.json")).get("final", {}).get("mm3")
        n = len(list((fdir / views[0]).glob("*.png")))
        pf = a.play_fps or a.fps
        wrote = {a.out: n}   # mp4 → the frames written to it (checked before the job is removed)
        if a.cams:   # 5 panels: endoscope | 3D on top, IR | RGB | webcam below, each camera frame nearest in time
            from PIL import Image, ImageDraw
            rows = [json.loads(x) for x in mtext.splitlines() if x.strip()]
            t_frame = [r.get("tExpShown") for r in rows] if rows and rows[0].get("tExpShown") else None
            cams = {}
            for c in ("ir", "rgb", "webcam"):
                fs = sorted((Path(a.cams) / c).glob("*.jpg"), key=lambda f: int(f.stem))
                cams[c] = ([int(f.stem) for f in fs], fs)
            if t_frame is None:   # fall back: spread the session over the capture span
                c0 = min(ts[0] for ts, _ in cams.values() if ts); c1 = max(ts[-1] for ts, _ in cams.values() if ts)
                t_frame = [c0 + (c1 - c0) * i / max(1, n - 1) for i in range(n)]
            W = a.size * len(views)
            wsum = 848 / 480 + 4 / 3 + 4 / 3
            hb = int(W / wsum) // 2 * 2
            # under the panels: the session's phases, trigger spans and bone-cut rows with the play position (10-06, user: the
            # robot videos must show when the trigger is on and when bone is cut, like the phase check videos)
            rep_, sbg = None, None
            if not a.no_strip:
                try:
                    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dataset"))
                    import phase_report as PR, phase_video as PV
                    rep_ = PR.analyse_path(a.recording, expect_full=False)
                    rep_["phases"] = {int(k): v for k, v in rep_["phases"].items()}
                    sbg = PV.strip_background(rep_, W)
                except Exception as e:   # a session without bone cuts has no phase 4: the video is still made
                    print(json.dumps({"strip": f"skipped: {type(e).__name__}: {e}"}), flush=True)
            hs = PV.STRIP_H if sbg is not None else 0
            H = (a.size + hb + hs + 1) // 2 * 2
            enc = lambda out: subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", f"{pf:g}",
                                                "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out], stdin=subprocess.PIPE)
            win = json.loads(Path(a.windows).read_text()) if a.windows and Path(a.windows).exists() else None
            out_m = str(Path(a.out).with_name(Path(a.out).stem + "_motion.mp4"))
            full, mot, n_mot = enc(a.out), (enc(out_m) if win else None), 0   # frames go straight to the encoders: no files in between
            import bisect
            for i in range(n):
                tf = t_frame[min(i, len(t_frame) - 1)]
                fr = Image.new("RGB", (W, H)); x = 0
                for jv, v in enumerate(views):
                    fr.paste(Image.open(fdir / v / f"{i:06d}.png").convert("RGB").resize((a.size, a.size)), (jv * a.size, 0))
                if sbg is not None:
                    fr.paste(PV.strip(sbg, rep_, (tf - rep_["t_exp_first"]) / 1000.0), (0, a.size + hb))
                for c, label in (("ir", "IR"), ("rgb", "RGB"), ("webcam", "webcam")):
                    ts, fs = cams[c]
                    wpan = int(round(hb * (848 / 480 if c == "ir" else 4 / 3)))
                    if ts:
                        k = min(len(ts) - 1, max(0, bisect.bisect_left(ts, tf)))
                        if k > 0 and abs(ts[k - 1] - tf) < abs(ts[k] - tf):
                            k -= 1
                        fr.paste(Image.open(fs[k]).convert("RGB").resize((wpan, hb)), (x, a.size))
                    ImageDraw.Draw(fr).text((x + 6, a.size + 4), label, fill=(0, 204, 255))
                    x += wpan
                raw = fr.tobytes()
                full.stdin.write(raw)
                if mot is not None and any(w0 <= tf <= w1 for w0, w1 in win):   # the same frames without the holds between chunks
                    mot.stdin.write(raw); n_mot += 1
            for e in (full, mot):
                if e is not None:
                    e.stdin.close()
                    if e.wait() != 0:
                        raise SystemExit("ffmpeg failed")
            if mot is not None:
                wrote[out_m] = n_mot
                print(json.dumps({"motion": out_m, "frames": n_mot, "of": n}), flush=True)
        else:
            ins = []
            for v in views:
                ins += ["-framerate", f"{pf:g}", "-i", str(fdir / v / "%06d.png")]
            filt = (["-filter_complex", "".join(f"[{i}:v]" for i in range(len(views))) + f"hstack=inputs={len(views)}"]
                    if len(views) > 1 else [])
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *ins, *filt, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-crf", "20", a.out], check=True)
    print(json.dumps({"out": a.out, "frames": n, "seconds": round(n / (a.play_fps or a.fps), 1), "mode": "simulate r4 @15 Hz" if policy else "replay",
                      "removed_mm3": removed, "render_s": round(time.time() - t0)}), flush=True)
    # the job's frames are needed no longer once every mp4 holds what was written to it
    got = {o: video_frames(o) for o in wrote}
    ok = n > 0 and all(got[o] == wrote[o] for o in wrote)
    if a.keep_job or a.job or os.environ.get("RENDER_VIDEO_KEEP_JOB") == "1":
        print(json.dumps({"job": j["id"], "kept": "asked to" if not a.job else "--job", "verified": ok}))
    elif ok:
        print(json.dumps({"job": j["id"], "removed": drop_job(j["id"]), "verified": got}))
    else:
        print(json.dumps({"job": j["id"], "kept": "the video does not hold the frames written", "written": wrote, "found": got}))
        raise SystemExit(4)


if __name__ == "__main__":
    main()
