#!/usr/bin/env python3
"""Training images through the SAME renderer the closed-loop inference uses (render API /v1/step), so the policy's
image input is identical in training and at inference.

    python render_step.py --dataset ~/il-robot/datasets/spine_rev4_20260929 [--cache ~/il-robot/datasets/render_cache_step]
    python export_lerobot.py … --cache ~/il-robot/datasets/render_cache_step      # then export with these images

For each hand episode: open a step session set up like robot/inference/rollout.py (recording flags, ae on, rawabs off,
drill rule r4), feed the episode's own 15 fps states (observation.state; the last grid frame = the last action), and keep
the endoscope image of every step. The viewer cuts from those poses, so the bone state in the images is the step
renderer's own (431 mm³ vs the live 434 on 3868). Cache layout = render_cache.py's (one chunk c<k0>-<k1> with DONE), so
export_lerobot.py reads it unchanged. Yields to robot recordings (RO_WANT / robot_origin_run.sh) between episodes.

Several workers at once (10-06): a render service renders one session at a time and a step costs about 0.5 s of mostly
waiting, so the way to go faster is a second service on another port (memory allowing: about 1.3 GB each) and one
worker per service — `--api ws://127.0.0.1:18781/v1/step --reverse` for the second. A worker leaves an episode alone
while another one is writing its frames (the chunk's .tmp folder changed within the last 2 minutes).
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
import rollout as R  # noqa: E402

RO_WANT = Path.home() / "il-robot/runs/jitter_iter/work/RO_WANT"


def robot_busy() -> bool:
    return RO_WANT.exists() or subprocess.run(["pgrep", "-f", "[r]obot_origin_run\\.sh"], capture_output=True).returncode == 0


def episode_frames(ds: Path, ep: int) -> np.ndarray:
    """States of every grid frame k0..k1: the dataset keeps k0..k1−1 as frames and the last as that frame's action."""
    import pyarrow.parquet as pq
    st, ac, fi = [], [], []
    for f in sorted(glob.glob(str(ds / "data/*/*.parquet"))):
        t = pq.read_table(f)
        e = np.asarray(t.column("episode_index").to_pylist())
        sel = np.nonzero(e == ep)[0]
        if len(sel):
            s = t.column("observation.state").to_pylist(); a = t.column("action").to_pylist(); k = t.column("frame_index").to_pylist()
            st += [s[i] for i in sel]; ac += [a[i] for i in sel]; fi += [k[i] for i in sel]
    o = np.argsort(fi)
    st, ac = np.asarray(st, float)[o], np.asarray(ac, float)[o]
    return np.vstack([st, ac[-1:]])


async def render_episode(ds: Path, ep: int, meta: dict, cache: Path, size: int, log, optics: str | None = None,
                         api: str = "ws://127.0.0.1:18780/v1/step") -> None:
    from PIL import Image
    S = episode_frames(ds, ep)
    g = meta["grid"]; k0, k1 = int(g["k0"]), int(g["k1"])
    if len(S) != k1 - k0 + 1:
        raise SystemExit(f"episode {ep}: {len(S)} states for grid {k0}..{k1}")
    spec = f"f{float(g['render_fps']):.4f}".rstrip("0").rstrip(".") + f"_endo_{size}"
    d = cache / meta["session_id"] / spec / f"c{k0:06d}-{k1:06d}"
    if (d / "DONE").exists():
        log(f"ep {ep} {meta['session_id'][-4:]}: cached"); return
    tmp = d.with_name(d.name + ".tmp")
    if tmp.exists() and time.time() - max((f.stat().st_mtime for f in tmp.rglob("*")), default=tmp.stat().st_mtime) < 120:
        log(f"ep {ep} {meta['session_id'][-4:]}: another worker is rendering it — skipped"); return
    (tmp / "frames/endo").mkdir(parents=True, exist_ok=True)
    with open(meta["recording"]) as fh:
        hdr = next(x for x in map(json.loads, fh) if x.get("k") == "hdr")
    fe = hdr.get("fe") or {}
    flags = [f"{k}={fe[k]}" for k in ("bite", "meltk", "meltv", "touchoff", "meltmax", "scopecal") if fe.get(k) is not None]
    dt_ms = 1000.0 / float(g["render_fps"])
    t0 = time.time()
    # the step session opens only with the endoscope through the skin: warm it up on the episode's first inside pose
    # (phase 3 start + 0.3 s), then feed frames from 0 — the drive's entry is already the working portal, like the
    # render API's warm-up of replayed recordings
    ph3 = (meta.get("phases") or {}).get("3")
    k_warm = min(len(S) - 1, int(round((ph3[0] + 0.3) * float(g["render_fps"])))) if ph3 else 0
    ws, _ = await R.session(api, hdr, S[k_warm], size, flags, optics=optics)
    rows = []
    try:
        for i, v in enumerate(S):
            img, m = await R.step(ws, v, dt_ms)
            Image.fromarray(img).save(tmp / "frames/endo" / f"{i:06d}.png")
            b = (m.get("meta") or {}).get("bridge") or {}
            rows.append({"i": i, "t_ms": m.get("t_ms"), "removedMm3": b.get("removedMm3"), "endoPose": b.get("endoPose"),
                         "drill": {k: (b.get("drill") or {}).get(k) for k in ("B", "proxy", "clearanceMm")}})
    finally:
        await ws.close()
    (tmp / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (tmp / "manifest.json").write_text(json.dumps({"renderer": "render API /v1/step (render_step.py)", "flags": flags, "warm_frame": k_warm,
                                                   "optics": optics or "recording",
                                                   "ae": "on", "rawabs": False, "drill_rule": "r4", "size": size,
                                                   "frames": len(S), "k0": k0, "k1": k1}))
    (tmp / "DONE").write_text("")
    tmp.rename(d)
    log(f"ep {ep} {meta['session_id'][-4:]}: {len(S)} frames in {time.time() - t0:.0f} s, removed {rows[-1]['removedMm3']} mm³")


async def main_async(a) -> None:
    ds = Path(a.dataset).expanduser()
    eps = [json.loads(l) for l in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines() if l.strip()]
    log = lambda *x: print("[step-render]", time.strftime("%T"), *x, flush=True)
    for ep, meta in (list(enumerate(eps))[::-1] if a.reverse else enumerate(eps)):
        if meta.get("source") != "hand" or (a.episodes and ep not in a.episodes):
            continue
        while robot_busy():
            log("robot has priority — waiting"); await asyncio.sleep(30)
        await render_episode(ds, ep, meta, Path(a.cache).expanduser(), a.size, log, a.optics, a.api)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cache", default=str(Path.home() / "il-robot/datasets/render_cache_step"))
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--api", default="ws://127.0.0.1:18780/v1/step", help="the render service's step endpoint")
    ap.add_argument("--reverse", action="store_true", help="last episode first (a second worker meeting the first in the middle)")
    ap.add_argument("--optics", choices=("air", "water"), default=None,
                    help="endoscope medium (default: the recording's scopecal flag = air). Use a cache of its own per medium: "
                         "the chunk folders are not keyed by it. The hand recordings' operator saw water (their &record page)")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
