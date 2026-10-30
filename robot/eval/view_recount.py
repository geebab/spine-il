#!/usr/bin/env python3
"""The drill-on-screen share of cached evaluations, counted again from the render jobs' own per-frame metrics.

    python view_recount.py [SESSION_STEM ...] [--all] [--dry]

policy_eval.py keeps the view statistics of a replay next to its removed set (runs/jitter_iter/view_<key>.json). Until
2026-10-07 16:00 they held only the stricter share (drill tip in view and not hidden). The run criteria now ask whether the
tip is on the endoscope's screen, hidden behind bone or not (user, 10-07; scope_screen.tip_on_screen). The frames are still
on disk: every render job keeps metrics.jsonl and manifest.json (render-jobs/<id>/). This finds, for each view file, the
newest replay job of that session with metrics and writes the statistics again — the old numbers stay in the file, the
screen share and the counts (hidden on screen, off screen) are added. Nothing is rendered.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import iou_eval as E
import policy_eval as PE

JOBS = Path(os.environ.get("UBE_RENDER_JOBS_DIR", Path.home() / "il-robot/render-jobs"))


def jobs_of(session: str) -> list[Path]:
    """Job folders that replayed this session with metrics, newest first."""
    out = []
    for d in JOBS.iterdir():
        mf, mt = d / "manifest.json", d / "metrics.jsonl"
        if not (mf.exists() and mt.exists() and mt.stat().st_size):
            continue
        try:
            m = json.loads(mf.read_text())
        except ValueError:
            continue
        if (m.get("source") or {}).get("sessionId") == session and (m.get("options") or {}).get("mode", "replay") == "replay" \
                and not (m.get("options") or {}).get("views"):
            out.append(d)
    return sorted(out, key=lambda d: d.stat().st_mtime, reverse=True)


def recount(vf: Path, dry: bool) -> dict | None:
    key = vf.stem[len("view_"):]
    session = key.split("_", 1)[1] if key.startswith(("rofull_", "psimfull15_", "hfull_")) else key
    old = json.loads(vf.read_text())
    js = jobs_of(session)
    if not js:
        return {"view": vf.name, "job": None, "old": old}
    d = js[0]
    opt = (json.loads((d / "manifest.json").read_text()).get("options") or {}).get("optics")
    rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if l.strip()]
    new = PE.view_stats(rows, opt if opt in ("air", "water") else None)
    same = all(old.get(k) == new.get(k) for k in ("cutting_frames", "visible_while_cutting"))   # the same frames as the cached count?
    if same and not dry:
        tmp = vf.with_name("." + vf.name + ".new"); tmp.write_text(json.dumps(new)); os.replace(tmp, vf)
    return {"view": vf.name, "job": d.name[:8], "frames": len(rows), "same_frames_as_cached": same, "old_seen": old.get("visible_while_cutting"),
            "on_screen": new["on_screen_while_cutting"], "hidden_on_screen_frames": new["hidden_on_screen_frames"],
            "off_screen_frames": new["off_screen_frames"], "cutting_frames": new["cutting_frames"], "written": bool(same and not dry)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="session stems (s_…_trk_xxxx); default with --all: every view_rofull_*.json")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    files = sorted(E.OUT.glob("view_rofull_*.json")) if a.all else [E.OUT / f"view_rofull_{s}.json" for s in a.sessions]
    for vf in files:
        if not vf.exists():
            print(json.dumps({"view": vf.name, "missing": True})); continue
        print(json.dumps(recount(vf, a.dry), ensure_ascii=False))


if __name__ == "__main__":
    main()
