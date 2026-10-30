"""Endoscope images for the dataset, rendered by the FE viewer through the render API and cached on disk.

A recording is rendered in the viewer's replay mode (cuts = the recording's own cut rows, screen = the recorded drawn
poses) on the render API's tExp grid: frame k shows the moment the endoscope screen first displayed camera frame
tExp ≥ t_exp_first + k·step (step = 1000 / render_fps ms, t_exp_first = the file's first f.tExp). The dataset samples
the tracker at exactly those tExp, so image k and state k are the same camera frame.

  hand session   render_fps = dataset fps
  robot session  render_fps = dataset fps × replay speed (0.33): one frame per 1/fps s of *hand* time, i.e. the
                 robot episode is put back on the recording's time scale (see export_lerobot --robot-time)

Long recordings are rendered in chunks (CHUNK frames each, one job at a time) so a job never holds the renderer for
long; each chunk seeks from the file start, so the carve state at its first frame is the recording's own.
Before every job the queue waits while a robot-origin run is in progress (robot_origin_run.sh / endo_follow.py).

Cache layout: <cache>/<sessionId>/<spec>/c<k0>-<k1>/{frames/<view>/NNNNNN.png, metrics.jsonl, manifest.json, DONE}
  spec = f<render_fps>_<views>_<size>. Frame i of a chunk is grid frame k0 + i (checked against metrics t_ms).

CLI (background queue):
  python render_cache.py --fps 15 --include hand,robot [--views endo] [--size 256] [--limit N] [--dry-run]

Training images (10-06): the demos' images must show the bone as the operator saw it — the recording's own cut rows,
i.e. this replay render — not a re-simulation of the cutting from the 15 Hz poses (render_step.py removes 0.55–1.07 of
the recorded volume, in other places, and then shows the burr inside bone that was long gone). For that:
  RENDER_API_URL=http://127.0.0.1:18781 python render_cache.py --include hand --only IDS --cache <cache> --optics water \
      --chunk 0 --removed
--optics air|water = the endoscope medium (default: the recording's flag; its own cache per medium), --chunk 0 = one job
per session, --removed (with --chunk 0) also stores the session's removed-voxel set as eval's iou_cache_hfull_<session>.npz.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import sources as S  # noqa: E402
else:
    from . import sources as S

API = os.environ.get("RENDER_API_URL", "http://127.0.0.1:18780")
CACHE_ROOT = Path.home() / "il-robot/datasets/render_cache"
CHUNK = 225            # frames per render job (15 s of hand time at 15 fps)
BUSY_ARGV = ("robot_origin_run.sh", "endo_follow.py")


# ---------------------------------------------------------------- grid
def timeline(path) -> dict:
    """The render API's grid facts of a recording (same rules as spine_render.recref): t_exp_first = min f.tExp,
    shown_last = per tool the last drawn.tExp."""
    t_first, shown = None, {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if '"k":"f"' not in line[:12] and '"k": "f"' not in line[:12]:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            te = r.get("tExp")
            if isinstance(te, (int, float)) and math.isfinite(te):
                t_first = te if t_first is None else min(t_first, te)
            d = r.get("drawn")
            if isinstance(d, dict) and isinstance(d.get("tExp"), (int, float)):
                tool = r.get("tool") or "endo"
                shown[tool] = max(shown.get(tool, -math.inf), d["tExp"])
    if t_first is None or "endo" not in shown:
        raise ValueError(f"{path}: no tExp / endo drawn frames — cannot render on the tExp grid")
    return {"t_exp_first": t_first, "shown_last_endo": shown["endo"]}


@dataclass
class RenderPlan:
    session_id: str
    path: str
    render_fps: float
    k0: int          # first grid frame of the episode
    k1: int          # last grid frame (inclusive)
    t_exp_first: float

    @property
    def step_ms(self) -> float:
        return 1000.0 / self.render_fps

    def t_of(self, k: int) -> float:
        """Epoch ms (tExp axis) of grid frame k."""
        return self.t_exp_first + k * self.step_ms


def grid_range(tl: dict, render_fps: float, t_lo: float | None = None, t_hi: float | None = None) -> tuple[int, int]:
    """Grid frames k with t_lo ≤ t_exp_first + k·step ≤ min(t_hi, shown_last_endo) (epoch ms)."""
    step = 1000.0 / render_fps
    t0 = tl["t_exp_first"]
    hi = tl["shown_last_endo"] if t_hi is None else min(t_hi, tl["shown_last_endo"])
    k0 = 0 if t_lo is None else max(0, math.ceil((t_lo - t0) / step - 1e-9))
    k1 = math.floor((hi - t0) / step + 1e-9)
    return k0, k1


def spec_tag(render_fps: float, views, size: int) -> str:
    return f"f{render_fps:.4f}".rstrip("0").rstrip(".") + f"_{'+'.join(views)}_{size}"


def chunks(k0: int, k1: int, n: int = CHUNK) -> list[tuple[int, int]]:
    return [(a, min(a + n - 1, k1)) for a in range(k0, k1 + 1, n)]


def chunk_dir(cache: Path, plan: RenderPlan, views, size, a: int, b: int) -> Path:
    return cache / plan.session_id / spec_tag(plan.render_fps, views, size) / f"c{a:06d}-{b:06d}"


def cached_frames(cache: Path, plan: RenderPlan, views, size) -> dict[int, dict[str, Path]] | None:
    """{grid k: {view: png}} for every frame of the plan, or None if any chunk is missing."""
    out = {}
    base = cache / plan.session_id / spec_tag(plan.render_fps, views, size)
    done = sorted(p for p in base.glob("c*-*") if (p / "DONE").exists()) if base.exists() else []
    for d in done:
        a, b = (int(x) for x in d.name[1:].split("-"))
        for k in range(a, b + 1):
            out[k] = {v: d / "frames" / v / f"{k - a:06d}.png" for v in views}
    if all(k in out for k in range(plan.k0, plan.k1 + 1)):
        return {k: out[k] for k in range(plan.k0, plan.k1 + 1)}
    return None


def cached_metrics(cache: Path, plan: RenderPlan, views, size) -> dict[int, dict]:
    """{grid k: metrics row} over the cached chunks (the viewer's own per-frame state: drawn poses, drill proxy,
    removedMm3, endo camera)."""
    out = {}
    base = cache / plan.session_id / spec_tag(plan.render_fps, views, size)
    for d in sorted(base.glob("c*-*")) if base.exists() else []:
        if not (d / "DONE").exists():
            continue
        a = int(d.name[1:].split("-")[0])
        for line in (d / "metrics.jsonl").read_text().splitlines():
            if line.strip():
                m = json.loads(line)
                out[a + int(m["i"])] = m
    return out


# ---------------------------------------------------------------- API
def _auth() -> dict:
    tok = os.environ.get("RENDER_API_TOKEN") or ""
    f = Path.home() / ".config/spinearm-render-api/token"
    if not tok and f.exists():
        tok = f.read_text().strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def job_input(src):
    """A job's input for the service in API: the path as it is (the service on this machine reads the file itself), or —
    RENDER_API_INLINE=1, a service on another machine such as a Mac render node behind an SSH tunnel — the file's text."""
    if os.environ.get("RENDER_API_INLINE") == "1" and isinstance(src, (str, Path)) and "\n" not in str(src) and Path(src).is_file():
        return Path(src).read_text()
    return src


def call(p: str, body=None, timeout: float = 120) -> bytes:
    r = urllib.request.Request(API + p, data=None if body is None else json.dumps(body).encode(),
                               headers={"Content-Type": "application/json", **_auth()})
    return urllib.request.urlopen(r, timeout=timeout).read()


RO_WANT = Path.home() / "il-robot/runs/jitter_iter/work/RO_WANT"  # the robot queue waits for an idle renderer


def gate_pause() -> str | None:
    """The reason of the pause eval/render_gate.py has in force (an arm in use from any account, a held robot job) — the
    pause file of the render service's switch (bridge spine_render/pause.py), read here without either module."""
    f = Path(os.environ.get("UBE_RENDER_PAUSE_FILE") or Path(os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}") / "ube-render" / "pause")
    try:
        if time.time() - f.stat().st_mtime > 15.0:
            return None
        return str(json.loads(f.read_text()).get("reason") or "pause file")
    except (OSError, ValueError, AttributeError):
        return None


def robot_busy() -> str | None:
    """Why the renderer must be left to the robot now, else None: the robot queue's RO_WANT flag file exists
    (it waits for an idle renderer), `pgrep -f "[r]obot_origin_run\\.sh"` finds a run, or an arm replay
    (endo_follow.py as a whole argv element) is running."""
    if RO_WANT.exists():
        return f"{RO_WANT.name} exists"
    why = gate_pause()
    if why:
        return f"render gate: {why}"
    r = subprocess.run(["pgrep", "-f", r"[r]obot_origin_run\.sh"], capture_output=True, text=True)
    if r.stdout.strip():
        return f"robot_origin_run.sh pid {r.stdout.split()[0]}"
    me = os.getpid()
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        for a in argv[:4]:
            s = a.decode(errors="replace")
            if any(s.endswith(x) for x in BUSY_ARGV):
                return f"pid {p.name}: {s}"
    return None


def wait_robot_idle(log=print, poll_s: float = 10.0) -> None:
    """Block (polling every poll_s) while robot_busy(). Called before every new job; running jobs are not touched."""
    said = False
    while (why := robot_busy()) is not None:
        if not said:
            log(f"[render] robot has priority ({why}) — waiting")
            said = True
        time.sleep(poll_s)
    if said:
        log("[render] robot idle — continuing")


def inflight_job(session_id: str, n_frames: int) -> str | None:
    """A queued/running job of ours for this session and chunk size (left by a restarted queue) — adopted instead of
    submitting the same chunk again."""
    try:
        for j in json.loads(call("/v1/jobs")):
            src = (j.get("summary") or {}).get("source")
            prog = j.get("progress") or [0, 0]
            if j.get("status") in ("queued", "running") and isinstance(src, dict) \
                    and src.get("sessionId") == session_id and (prog[1] in (0, n_frames)):
                return j["id"]
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return None


def render_chunk(plan: RenderPlan, a: int, b: int, views, size: int, out: Path, log=print, optics: str | None = None,
                 removed: bool = False) -> dict:
    """One job: grid frames a..b of the plan → out (atomic: tmp dir then rename). Returns the job's stats."""
    step = plan.step_ms
    opts = {"views": list(views), "fps": plan.render_fps, "size": size, "status": False,
            "t_start_s": (a * step - 0.5) / 1000.0, "t_end_s": (b * step + 0.5) / 1000.0,
            **({"optics": optics} if optics else {}), **({"removed_voxels": True, "removed_every_s": 3} if removed else {})}
    jid = inflight_job(plan.session_id, b - a + 1)
    if jid:
        log(f"[render] adopting in-flight job {jid[:8]} for {plan.session_id} c{a}-{b}")
        j = {"id": jid}
    else:
        wait_robot_idle(log)
        j = json.loads(call("/v1/jobs", {"input": job_input(plan.path), "options": opts}, 600))
    while True:
        s = json.loads(call(f"/v1/jobs/{j['id']}"))
        if s["status"] in ("done", "failed", "error"):
            break
        time.sleep(3)
    if s["status"] != "done":
        raise RuntimeError(f"render job {j['id']}: {s['status']} {str(s.get('error'))[:300]}")
    z = zipfile.ZipFile(io.BytesIO(call(f"/v1/jobs/{j['id']}/result.zip", timeout=600)))
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for n in z.namelist():
        if n.startswith("frames/") and n.endswith(".png") or n in ("metrics.jsonl", "manifest.json"):
            z.extract(n, tmp)
    rows = [json.loads(x) for x in (tmp / "metrics.jsonl").read_text().splitlines() if x.strip()]
    if len(rows) != b - a + 1:
        raise RuntimeError(f"{plan.session_id} c{a}-{b}: {len(rows)} frames rendered, {b - a + 1} asked")
    bad = [r["i"] for r in rows if abs(r["t_ms"] - (a + r["i"]) * step) > 0.01]
    if bad:
        raise RuntimeError(f"{plan.session_id} c{a}-{b}: frame times off the grid at i={bad[:5]}")
    man = json.loads((tmp / "manifest.json").read_text())
    if optics:   # export_lerobot.image_optics reads it: the medium travels with the dataset and the policy
        man["optics"] = optics; (tmp / "manifest.json").write_text(json.dumps(man))
    if removed:  # the session's removed-voxel set, as eval/iou_eval.cached keeps it (the reference of run_criteria.py)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
        import numpy as np
        import iou_eval as E
        v = E.voxels(z)
        np.savez_compressed(E.OUT / f"iou_cache_hfull_{Path(plan.path).stem}.npz", hdr=json.dumps(v["hdr"]), final=v["final"],
                            snaps=np.array(v["snaps"], dtype=object))
    rep = man.get("replay") or {}
    stats = {"job": j["id"], "frames": len(rows), "pass": rep.get("pass"), "failReasons": rep.get("failReasons"),
             "msPerFrame": (s.get("summary") or {}).get("stats", {}).get("msPerFrame")}
    (tmp / "DONE").write_text(json.dumps(stats))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return stats


# ---------------------------------------------------------------- plans
def plan_for(sess: "S.Session", fps: float, tl: dict | None = None) -> RenderPlan:
    """The frames an episode needs. Hand: the whole recording (windows are sub-ranges of it). Robot: the replay span
    both arms executed, on hand time (render_fps = fps × speed)."""
    tl = tl or timeline(sess.path)
    if sess.source == "hand":
        rf = float(fps)
        k0, k1 = grid_range(tl, rf)
    else:
        rf = float(fps) * float(sess.meta["speed"])
        t_go, t_end = S.replay_span(sess.meta["arm_dirs"])
        k0, k1 = grid_range(tl, rf, t_go, t_end)
    return RenderPlan(sess.session_id, str(sess.path), round(rf, 6), k0, k1, tl["t_exp_first"])


def usable_sessions(include=("hand", "robot"), root: Path | None = None, only: set | None = None
                    ) -> tuple[list, list]:
    """(used, skipped) sessions; robot ones get their run meta."""
    cont = S.contaminated_tags()
    used, skipped = [], []
    for s in S.discover(root, only):
        if s.skip is None and s.source == "robot":
            S.run_meta(s, contaminated=cont)
        (skipped if s.skip is not None or s.source not in include else used).append(s)
    return used, skipped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--include", default="hand,robot")
    ap.add_argument("--views", default="endo")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--cache", type=Path, default=None, help="default ~/il-robot/datasets/render_cache")
    ap.add_argument("--sessions", type=Path, default=None, help="default ~/il-robot/viewer-data/sessions")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many jobs (0 = all)")
    ap.add_argument("--only", default="", help="comma list of session short ids / run tags to render")
    ap.add_argument("--optics", choices=("air", "water"), default=None, help="endoscope medium (default: the recording's flag)")
    ap.add_argument("--chunk", type=int, default=CHUNK, help="frames per job; 0 = one job per session")
    ap.add_argument("--removed", action="store_true", help="with --chunk 0: also keep the session's removed-voxel set (eval cache)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    views = tuple(v for v in a.views.split(",") if v)
    a.cache = a.cache or CACHE_ROOT

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    used, _ = usable_sessions(tuple(a.include.split(",")), a.sessions, set(a.only.split(",")) if a.only else None)
    used.sort(key=lambda s: (s.source != "hand", s.session_id))  # hand first (the primary episodes)
    todo = []
    for s in used:
        try:
            plan = plan_for(s, a.fps)
        except (ValueError, KeyError) as e:
            log(f"[render] {s.session_id}: skip ({e})")
            continue
        for c0, c1 in chunks(plan.k0, plan.k1, a.chunk or plan.k1 - plan.k0 + 1):
            d = chunk_dir(a.cache, plan, views, a.size, c0, c1)
            if not (d / "DONE").exists():
                todo.append((s, plan, c0, c1, d))
    n_frames = sum(c1 - c0 + 1 for _, _, c0, c1, _ in todo)
    log(f"[render] {len(todo)} jobs, {n_frames} frames to render (views {views}, {a.size}px)")
    if a.dry_run:
        for s, plan, c0, c1, d in todo[:20]:
            log(f"  {s.source} {s.sid} {s.meta.get('tag', '')} fps {plan.render_fps} frames {c0}-{c1}")
        return 0
    done = 0
    for s, plan, c0, c1, d in todo:
        if a.limit and done >= a.limit:
            break
        for attempt in range(3):
            try:
                t = time.time()
                st = render_chunk(plan, c0, c1, views, a.size, d, log, a.optics, a.removed and not a.chunk)
                log(f"[render] {s.source} {s.sid} {s.meta.get('tag', '')} c{c0}-{c1} ok {time.time() - t:.0f}s "
                    f"({st['msPerFrame']} ms/frame, pass={st['pass']})")
                break
            except (RuntimeError, urllib.error.URLError, OSError, ValueError) as e:
                log(f"[render] {s.session_id} c{c0}-{c1} attempt {attempt + 1}: {type(e).__name__}: {e}")
                time.sleep(30)
        done += 1
    log("[render] queue finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
