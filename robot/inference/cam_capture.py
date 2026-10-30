#!/usr/bin/env python3
"""Record the tracker live view's three cameras during a robot run: IR (tracking), RGB (D435 colour) and the laptop webcam.

    python robot/inference/cam_capture.py OUT_DIR --stop-file FILE [--hz 5]

Polls the tracker's own snapshot endpoints (127.0.0.1:8089 /frame, /rgb, /webcam — what https://…:8444 shows) and writes
OUT_DIR/{ir,rgb,webcam}/<epoch_ms>.jpg until --stop-file exists (or --max-s). File names are the request times in epoch ms,
the same clock as the session's tExp, so render_video.py can lay them next to the viewer frames. 5 Hz per camera keeps the
tracker's capture thread free (a JPEG encode is ~3 ms).
"""
from __future__ import annotations

import argparse
import threading
import time
import urllib.request
from pathlib import Path

SRC = {"ir": "/frame", "rgb": "/rgb", "webcam": "/webcam"}


def poll(name: str, path: str, out: Path, hz: float, stop: Path, t_end: float, base: str) -> None:
    d = out / name; d.mkdir(parents=True, exist_ok=True)
    period = 1.0 / hz
    while not stop.exists() and time.time() < t_end:
        t = time.time()
        try:
            with urllib.request.urlopen(f"{base}{path}?{int(t * 1000)}", timeout=2) as r:
                if r.status == 200:
                    (d / f"{int(t * 1000)}.jpg").write_bytes(r.read())
        except Exception:   # a busy tracker / the webcam warming up (503): skip this tick
            pass
        time.sleep(max(0.0, period - (time.time() - t)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--stop-file", required=True)
    ap.add_argument("--hz", type=float, default=5.0)
    ap.add_argument("--max-s", type=float, default=900.0)
    ap.add_argument("--base", default="http://127.0.0.1:8089")
    a = ap.parse_args()
    out, stop, t_end = Path(a.out), Path(a.stop_file), time.time() + a.max_s
    th = [threading.Thread(target=poll, args=(n, p, out, a.hz, stop, t_end, a.base), daemon=True) for n, p in SRC.items()]
    for t in th:
        t.start()
    for t in th:
        t.join()
    print({n: len(list((out / n).glob("*.jpg"))) for n in SRC})


if __name__ == "__main__":
    main()
