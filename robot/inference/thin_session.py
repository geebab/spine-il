#!/usr/bin/env python3
"""A lighter copy of a long robot session for rendering and evaluation.

    python robot/inference/thin_session.py SESSION.jsonl -o OUT_DIR [--hz 15] [--keep-s 0.5] [--min-mb 40]

A closed-loop run without a drilling time limit lasts half an hour, and the recorder writes both tools at the tracker's
rate the whole time: 190 MB for 30 min (10-07, session 947d: 99,203 rows a tool at 55 Hz). The viewer has to load the whole
file to replay it, and the render service's page (1.5 GB of JavaScript heap, 2 GB for the service) does not get through
that — the 5-panel video job of that run sat at 0 frames until it was cancelled. Most of those rows say nothing: the arms
hold still between chunks.

The copy keeps every row that is not a tool frame (header, events, cut rows, footer) and, per tool, one frame row every
1/--hz s — but every frame row within --keep-s of a bone-cut row or of a trigger edge, so the drilling keeps its full
detail. The removed bone is the cut rows' alone, so a replay of the copy removes exactly what the session did; a video
frame is at most 1/(2·hz) s from the pose the full file would show. The copy is written to OUT_DIR under the session's
own file name (same stem: the evaluation caches keyed by it stay valid) — never into viewer-data/sessions, where it would
be taken for a second session. A session under --min-mb is not copied: its own path is printed. The limit was 80 MB until
10-08: a session of 79.8 MB (85,878 rows, a 14-min run) went to the renderer as it was and killed the render page before
its first frame, twice ("Target page, context or browser has been closed") — its thinned copy rendered. Now 40 MB.
Prints one JSON line: {"path", "thinned", "mb": [in, out], "rows": {tool: [in, out]}}.
"""
from __future__ import annotations

import argparse
import bisect
import json
from pathlib import Path


def thin(src: Path, out_dir: Path, hz: float = 15.0, keep_s: float = 0.5, min_mb: float = 40.0) -> dict:
    mb = src.stat().st_size / 1e6
    if mb < min_mb:
        return {"path": str(src), "thinned": False, "mb": [round(mb, 1), round(mb, 1)], "rows": {}}
    if "viewer-data/sessions" in str(out_dir.resolve()):
        raise SystemExit("the copy must not go under viewer-data/sessions (it would be found as a session)")
    marks, btn, last_row, n = [], None, {}, 0   # tMono (ms) of cut rows and trigger edges: full detail around them
    is_frame = lambda head: '"k":"f"' in head or '"k": "f"' in head
    with open(src) as fh:
        for n, line in enumerate(fh):
            head = line[:40]
            if '"cut"' in head:
                marks.append(json.loads(line)["tMono"])
            elif is_frame(head):
                d = json.loads(line)
                tool = d.get("tool") or "endo"
                last_row[tool] = n   # a tool's last frame row is kept: the replay's end time
                if tool == "drill":
                    b = bool(d.get("btn"))
                    if btn is not None and b != btn:
                        marks.append(d["tMono"])
                    btn = b
    marks.sort()

    def near(t: float) -> bool:
        i = bisect.bisect_left(marks, t)
        return (i < len(marks) and marks[i] - t <= keep_s * 1000.0) or (i > 0 and t - marks[i - 1] <= keep_s * 1000.0)
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / src.name
    step, last, rows = 1000.0 / hz, {}, {}
    with open(src) as fh, open(dst, "w") as fo:
        for n, line in enumerate(fh):
            if not is_frame(line[:40]):
                fo.write(line)
                continue
            d = json.loads(line)
            tool, t = d.get("tool") or "endo", d["tMono"]
            rows.setdefault(tool, [0, 0])[0] += 1
            if tool not in last or t - last[tool] >= step - 1e-6 or n == last_row[tool] or near(t):
                fo.write(line)
                rows[tool][1] += 1
                last[tool] = t
    return {"path": str(dst), "thinned": True, "mb": [round(mb, 1), round(dst.stat().st_size / 1e6, 1)], "rows": rows,
            "kept_around": len(marks)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("-o", "--out-dir", required=True)
    ap.add_argument("--hz", type=float, default=15.0)
    ap.add_argument("--keep-s", type=float, default=0.5)
    ap.add_argument("--min-mb", type=float, default=40.0)
    a = ap.parse_args()
    print(json.dumps(thin(Path(a.session).expanduser(), Path(a.out_dir).expanduser(), a.hz, a.keep_s, a.min_mb)))


if __name__ == "__main__":
    main()
