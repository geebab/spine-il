#!/usr/bin/env python3
"""The recorder page as the policy's observation (2026-10-06, user: "정책이 기록 페이지의 화면을 직접 보게 해").

A closed-loop run has had two viewers: the recorder page (robot_origin_rec.py — the live engine, driven by the tracker and
by the robot's trigger; its session is what a run is judged by) and a render-API step session the measured poses were fed
into after every chunk, whose image the policy saw. The two simulate the cutting separately and disagree (10-06: the step
session removed 165–200 mm³ where the page recorded 265–327), so the policy looked at bone the run had already removed.
Here the policy looks at the page itself — one simulation:

    robot_origin_rec.py --obs-port P            the page is opened with &render (window.__ubeRender) at VIEWPORT; serve()
    live_loop.py --obs-url http://127.0.0.1:P   get() after every chunk, while the arms hold with the trigger off

    GET /obs?size=256[&image=0] → {"ok", "t_ms", "meta": {"bridge": __ubeRender.metrics()}, "view": clip | null,
                                   "png": base64 | null, "s": seconds it took}
    GET /health                 → {"ok", "bridge": the page has window.__ubeRender, "requests": n}

The image is made the way the render API makes a training frame (bridge worker/browser.py Session.capture + render._encode):
a screenshot of the endoscope view's main canvas (info().views, id "endoscope") at device scale 1, resized to size × size
with Lanczos. "meta" has the shape of a step session's reply, so rollout.py's phase rules, visibility test and endoscope
aiming read it unchanged. "png" is null while the page shows no endoscope view.

Nothing here drives the page: the tools come from the tracker, the trigger from the relay. An observation is asked for only
while the arms hold — a screenshot and metrics() (a pass over the bone volume) take the page's main thread for a moment, and
the live engine removes bone per tick.

    python robot/page_obs.py --fake 18790 [--cut-mm3 4 --cuts 12]   a stand-in server for a dry run (inference/fake_arms.py)

The stream (10-12, for the continuous loop — inference/CONTINUOUS.md): GET /obs?stream=1 takes the image from Chrome's
screencast instead of a screenshot (Stream below: the compositor's frames, encoded outside the page's main thread, the
newest one kept) and asks the page for metrics() only — 0.05 s instead of 0.23 s on a quiet page, and nothing that stops
the page, so it may be asked while the arms move. The 256 px image is the screenshot's (mean difference 0.81 of 255, no
pixel over 32: live/screencast_probe.py) as long as the frames come at FULL size; every 6th frame is enough and costs the
page nothing measurable (every frame does: a seek went from 54 to 97 ms). "t_ms" is then the frame's own time.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

VIEWPORT = {"width": 1600, "height": 1000}   # the render API's page (bridge worker/browser.py SessionConfig.viewport)
VIEW_ID = "endoscope"
INFO_JS = "() => window.__ubeRender ? window.__ubeRender.info() : null"
METRICS_JS = "() => window.__ubeRender ? window.__ubeRender.metrics() : null"
# the endoscope camera's view angle per medium (deg): what a render-API session reports for the viewer's built-in profile
# (rollout.SCOPE: 2 × (theta_max + 2°)); the recorder page starts in water whatever its scopecal flag says (FE recordDefaults.ts)
VIEW_ANGLE = {"air": 92.93, "water": 67.40}


def view_clip(info: dict | None) -> dict | None:
    """The endoscope view's main canvas rect from info().views (as the render API's Session._views_from_info), or None."""
    for v in (info or {}).get("views") or []:
        if isinstance(v, dict) and v.get("id") == VIEW_ID:
            for r in ((v.get("canvas") or {}).get("rect"), v.get("rect")):
                if isinstance(r, dict) and all(isinstance(r.get(k), (int, float)) for k in ("x", "y", "width", "height")) \
                        and r["width"] > 0 and r["height"] > 0:
                    return {k: float(r[k]) for k in ("x", "y", "width", "height")}
    return None


def encode(png: bytes, size: int) -> bytes:
    """= the render API's render._encode for a PNG frame: RGB, size × size, Lanczos."""
    from PIL import Image
    im = Image.open(io.BytesIO(png)).convert("RGB").resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


async def observe(page, size: int = 256, image: bool = True) -> dict:
    """One observation of the page: the bridge metrics and (image) the endoscope view as the render API captures it."""
    t0 = time.monotonic()
    info = await page.evaluate(INFO_JS)
    if info is None:
        return {"ok": False, "error": "the page has no window.__ubeRender (a dev build opened with &render?)"}
    m = await page.evaluate(METRICS_JS)
    t_ms = time.time() * 1000.0
    clip, png = view_clip(info), None
    if image and clip:
        png = encode(await page.screenshot(clip=clip, type="png"), size)
    return {"ok": True, "t_ms": t_ms, "meta": {"bridge": m}, "view": clip, "size": size,
            "png": base64.b64encode(png).decode() if png else None, "s": round(time.monotonic() - t0, 3)}


class Stream:
    """Chrome's screencast of the page: the newest frame, with the time the compositor made it."""
    QUALITY, EVERY = 92, 6       # JPEG quality; every n-th compositor frame (about 8 a second while the view changes)

    def __init__(self, page):
        self.page, self.cdp, self.frame, self.n = page, None, None, 0     # frame = (jpeg bytes, stamp wall ms, arrival wall ms)

    async def start(self) -> "Stream":
        self.cdp = await self.page.context.new_cdp_session(self.page)
        self.cdp.on("Page.screencastFrame", self._on_frame)
        await self.cdp.send("Page.startScreencast", {"format": "jpeg", "quality": self.QUALITY, "everyNthFrame": self.EVERY})
        return self

    def _on_frame(self, ev: dict) -> None:
        now = time.time() * 1000.0
        ts = (ev.get("metadata") or {}).get("timestamp")
        self.frame = (base64.b64decode(ev["data"]), float(ts) * 1000.0 if ts else now, now); self.n += 1
        asyncio.ensure_future(self._ack(ev["sessionId"]))

    async def _ack(self, sid) -> None:
        try:
            await self.cdp.send("Page.screencastFrameAck", {"sessionId": sid})
        except Exception:  # noqa: BLE001 — the page is closing
            pass

    async def stop(self) -> None:
        try:
            if self.cdp is not None:
                await self.cdp.send("Page.stopScreencast")
        except Exception:  # noqa: BLE001
            pass


def crop_encode(jpeg: bytes, clip: dict, size: int, viewport: dict = VIEWPORT) -> bytes:
    """A screencast frame (the whole viewport) → the endoscope view's rect → size × size PNG, Lanczos (as encode())."""
    from PIL import Image
    im = Image.open(io.BytesIO(jpeg)).convert("RGB")
    fx, fy = im.size[0] / viewport["width"], im.size[1] / viewport["height"]
    im = im.crop((clip["x"] * fx, clip["y"] * fy, (clip["x"] + clip["width"]) * fx, (clip["y"] + clip["height"]) * fy)).resize((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


async def observe_stream(page, stream: Stream, size: int = 256, image: bool = True, state: dict | None = None, max_age_ms: float = 1500.0) -> dict:
    """observe() without the screenshot: metrics() from the page, the image from the stream's newest frame. `state` keeps
    the endoscope view's rect between calls (info() is asked again every 10th time: the layout does not change in a run).
    A frame older than max_age_ms while the view should be changing is not an observation of now: the screenshot is
    taken instead (and the reply says so: "stream": false)."""
    t0 = time.monotonic(); state = state if state is not None else {}
    state["n"] = state.get("n", 0) + 1
    if state.get("clip") is None or state["n"] % 10 == 1:
        info = await page.evaluate(INFO_JS)
        if info is None:
            return {"ok": False, "error": "the page has no window.__ubeRender (a dev build opened with &render?)"}
        state["clip"] = view_clip(info)
    m = await page.evaluate(METRICS_JS)
    clip, fr, now = state["clip"], stream.frame, time.time() * 1000.0
    if image and clip and (fr is None or now - fr[2] > max_age_ms):
        d = await observe(page, size, image); d["stream"] = False      # nothing new from the compositor: look the slow way
        return d
    png = await asyncio.to_thread(crop_encode, fr[0], clip, size) if (image and clip) else None
    return {"ok": True, "t_ms": fr[1] if (png and fr) else now, "meta": {"bridge": m}, "view": clip, "size": size, "stream": True,
            "frame_age_ms": round(now - fr[1], 1) if fr else None, "png": base64.b64encode(png).decode() if png else None,
            "s": round(time.monotonic() - t0, 3)}


class Server:
    """GET /obs and /health on 127.0.0.1:port, answered from `page` (a Playwright page) or from `source` (an async
    function (size, image) → observe()'s dict: the dry run's stand-in). One observation at a time. With `stream` (a
    started Stream) a request with stream=1 is answered from the screencast."""

    def __init__(self, page, port: int, source=None, stream: Stream | None = None):
        self.page, self.port, self.source, self.stream, self._st = page, port, source, stream, {}
        self.lock, self.n, self.last_s, self.srv = asyncio.Lock(), 0, None, None

    async def start(self) -> "Server":
        self.srv = await asyncio.start_server(self._client, "127.0.0.1", self.port)
        return self

    async def stop(self) -> None:
        if self.srv is not None:
            self.srv.close()
            try:
                await asyncio.wait_for(self.srv.wait_closed(), 5.0)
            except asyncio.TimeoutError:
                pass

    async def _answer(self, target: str) -> tuple[int, dict]:
        u = urllib.parse.urlsplit(target)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/obs":
            size, image = int(q.get("size", ["256"])[0]), q.get("image", ["1"])[0] != "0"
            if not 16 <= size <= 2048:
                return 400, {"ok": False, "error": f"size {size}"}
            want_stream = q.get("stream", ["0"])[0] == "1"
            async with self.lock:
                if self.source:
                    d = await self.source(size, image)
                elif want_stream and self.stream is not None:
                    d = await observe_stream(self.page, self.stream, size, image, self._st)
                else:
                    d = await observe(self.page, size, image)
            self.n += 1
            self.last_s = d.get("s")
            return (200 if d.get("ok") else 503), d
        if u.path == "/health":
            has = True if self.source else bool(await self.page.evaluate("() => !!window.__ubeRender"))
            return 200, {"ok": True, "bridge": has, "requests": self.n}
        return 404, {"ok": False, "error": f"no {u.path}"}

    async def _client(self, rd: asyncio.StreamReader, wr: asyncio.StreamWriter) -> None:
        try:
            line = (await asyncio.wait_for(rd.readline(), 10.0)).decode("latin-1").split()
            while (await asyncio.wait_for(rd.readline(), 10.0)) not in (b"\r\n", b"\n", b""):
                pass
            if len(line) < 2 or line[0] != "GET":
                code, body = 405, {"ok": False, "error": "GET only"}
            else:
                code, body = await self._answer(line[1])
        except Exception as e:  # noqa: BLE001 — the page may be closing; the caller gets the reason, the run script the log
            code, body = 500, {"ok": False, "error": f"{type(e).__name__}: {e}"[:300]}
        try:
            raw = json.dumps(body).encode()
            wr.write(f"HTTP/1.0 {code} X\r\nContent-Type: application/json\r\nContent-Length: {len(raw)}\r\n"
                     "Connection: close\r\n\r\n".encode() + raw)
            await wr.drain()
        except Exception:  # noqa: BLE001 — the client went away
            pass
        finally:
            wr.close()


def _fetch(url: str, timeout_s: float) -> dict:
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # a local port: never through a proxy
    try:
        with op.open(url, timeout=timeout_s) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:   # the server's own refusals carry a JSON body with the reason
        try:
            return json.loads(e.read())
        except ValueError:
            return {"ok": False, "error": f"HTTP {e.code}"}


def health(url: str, timeout_s: float = 5.0) -> dict | None:
    """The obs server's /health, or None when nothing answers there."""
    try:
        return _fetch(f"{url.rstrip('/')}/health", timeout_s)
    except (OSError, ValueError):
        return None


def get(url: str, size: int = 256, image: bool = True, timeout_s: float = 60.0, stream: bool = False) -> tuple[np.ndarray | None, dict]:
    """(endoscope image, RGB uint8, size × size × 3 | None, reply) — the reply's "meta" is a step session's (rollout.step).
    stream: the image from the page's screencast (no screenshot: may be asked while the arms move)."""
    from PIL import Image
    try:
        d = _fetch(f"{url.rstrip('/')}/obs?size={size}&image={int(image)}" + ("&stream=1" if stream else ""), timeout_s)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"no observation from the recorder page at {url} ({type(e).__name__}: {e})") from e
    if not d.get("ok"):
        raise RuntimeError(f"no observation from the recorder page at {url}: {d.get('error')}")
    png = d.pop("png", None)
    img = np.asarray(Image.open(io.BytesIO(base64.b64decode(png))).convert("RGB")) if png else None
    return img, d


def bridge(m: dict | None) -> dict:
    return ((m or {}).get("meta") or {}).get("bridge") or {}


def medium(m: dict | None, tol_deg: float = 1.0) -> str | None:
    """air | water from the endoscope camera's view angle, None when it is neither (no optics pass, or no camera yet)."""
    va = (bridge(m).get("endoCamera") or {}).get("viewAngle")
    if va is None:
        return None
    return next((k for k, a in VIEW_ANGLE.items() if abs(float(va) - a) <= tol_deg), None)


def frame_meta(m: dict, removed_mm3: float, cutting: bool) -> dict:
    """The page's observation as ONE frame's reply for rollout.PhaseTracker. The page is looked at once per chunk, so what
    it removed while the arms moved is known only as a total: the caller shares it among the chunk's frames (bone is removed
    only with the trigger on) by giving each frame its own removed volume and cutting flag; everything else is the page's."""
    b = dict(bridge(m))
    d = dict(b.get("drill") or {})
    d["proxy"] = {**(d.get("proxy") or {}), "cutting": bool(cutting)}
    b["drill"], b["removedMm3"] = d, float(removed_mm3)
    return {"meta": {"bridge": b}}


def chunk_cuts(trig: list[bool], rose: bool) -> list[bool]:
    """Which frames of a chunk count as cutting, given the frames' trigger and whether the removed volume rose over the
    chunk: the frames with the trigger on — or, when it rose with no trigger frame at all (the trigger-off command lands
    just after the chunk's last frame), the last one."""
    if not rose:
        return [False] * len(trig)
    return list(trig) if any(trig) else [False] * (len(trig) - 1) + [True] * min(1, len(trig))


async def _fake(a) -> None:
    """A stand-in page: a grey frame and metrics in which the endoscope is inside and each request up to --cuts removes
    --cut-mm3 more — enough for live_loop.py to enter, drill, see the cutting stop, and leave."""
    from PIL import Image
    n = {"k": 0}

    async def source(size: int, image: bool) -> dict:
        n["k"] += 1
        png = None
        if image:
            buf = io.BytesIO()
            Image.new("RGB", (size, size), (90, 60, 50)).save(buf, format="PNG")
            png = buf.getvalue()
        k = max(0, min(a.cuts, n["k"] - a.skip))
        m = {"endoPose": {"entry": a.entry, "depth": 30.0}, "endoCamera": {"viewAngle": VIEW_ANGLE[a.medium]},
             "drill": {"B": None, "proxy": {"cutting": False}, "inView": {"inFov": True, "visible": True, "x": 0.0, "y": 0.0}},
             "removedMm3": round(k * a.cut_mm3, 3)}
        return {"ok": True, "t_ms": time.time() * 1000.0, "meta": {"bridge": m}, "view": {"x": 0, "y": 0, "width": size, "height": size},
                "size": size, "png": base64.b64encode(png).decode() if png else None, "s": 0.0}

    srv = await Server(None, a.fake, source).start()
    print(f"[page_obs] fake page on 127.0.0.1:{a.fake} ({a.medium}, {a.cuts} × {a.cut_mm3} mm3 after {a.skip} requests)", flush=True)
    try:
        await asyncio.Event().wait()
    finally:
        await srv.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fake", type=int, required=True, help="serve a stand-in page on this port (dry runs)")
    ap.add_argument("--medium", choices=list(VIEW_ANGLE), default="water")
    ap.add_argument("--entry", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="the endoscope's skin entry (scene mm) — "
                    "within 20 mm of the demos' portal, or the phase rules never see the endoscope inside")
    ap.add_argument("--skip", type=int, default=3, help="requests before the first cut (the scripted entry)")
    ap.add_argument("--cuts", type=int, default=12, help="requests that remove bone")
    ap.add_argument("--cut-mm3", type=float, default=4.0)
    try:
        asyncio.run(_fake(ap.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
