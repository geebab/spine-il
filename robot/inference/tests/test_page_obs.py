"""page_obs.py without a browser: the server and client over a real local socket with a stand-in page, the clip chosen
from info().views, and a chunk's removed volume going through rollout.PhaseTracker frame by frame."""

import asyncio
import io
import socket
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[1]))

import page_obs as PG  # noqa: E402
import rollout as RO  # noqa: E402

CLIP = {"x": 752.0, "y": 63.0, "width": 752.0, "height": 752.0}
INFO = {"views": [{"id": "3d", "slot": 0, "rect": {"x": 0, "y": 48, "width": 500, "height": 470}, "canvas": None, "bridge": ["scene_raw"]},
                  {"id": "endoscope", "slot": None, "rect": {"x": 750, "y": 60, "width": 760, "height": 760},
                   "canvas": {"width": 752, "height": 752, "rect": CLIP}, "bridge": ["endo_raw"]}]}


def metrics(removed=0.0, depth=30.0, angle=67.4013):
    return {"endoPose": {"entry": [1.0, 2.0, 3.0], "depth": depth}, "endoCamera": {"viewAngle": angle},
            "drill": {"B": [0, 0, 0], "proxy": {"cutting": False, "gate": "air", "penMm": 0.0},
                      "inView": {"inFov": True, "visible": True, "x": 0.1, "y": -0.2}}, "removedMm3": removed}


class Page:
    """What page_obs needs of a Playwright page: evaluate(info | metrics | has bridge) and screenshot(clip)."""

    def __init__(self, info=INFO, bridge=True):
        self.info, self.bridge, self.removed, self.shots = info, bridge, 0.0, []

    async def evaluate(self, js, *a):
        if not self.bridge:
            return None if js in (PG.INFO_JS, PG.METRICS_JS) else False
        return self.info if js == PG.INFO_JS else metrics(self.removed) if js == PG.METRICS_JS else True

    async def screenshot(self, clip=None, type="png"):
        self.shots.append(clip)
        im = np.zeros((int(clip["height"]), int(clip["width"]), 4), np.uint8)
        im[..., 0], im[..., 3] = 200, 255   # red, opaque: the RGBA a browser screenshot has
        buf = io.BytesIO(); Image.fromarray(im, "RGBA").save(buf, format="PNG")
        return buf.getvalue()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(page, fn):
    """Run fn(base url) in a thread while the server answers on this loop — the two processes of a run."""
    async def go():
        port = free_port()
        srv = await PG.Server(page, port).start()
        try:
            return await asyncio.to_thread(fn, f"http://127.0.0.1:{port}"), srv.n
        finally:
            await srv.stop()
    return asyncio.run(go())


def test_clip_is_the_endoscope_views_main_canvas():
    assert PG.view_clip(INFO) == CLIP
    no_canvas = {"views": [{"id": "endoscope", "rect": {"x": 1, "y": 2, "width": 30, "height": 40}, "canvas": None}]}
    assert PG.view_clip(no_canvas) == {"x": 1.0, "y": 2.0, "width": 30.0, "height": 40.0}
    assert PG.view_clip({"views": [INFO["views"][0]]}) is None and PG.view_clip(None) is None


def test_observation_over_http_is_a_step_reply():
    page = Page(); page.removed = 12.5
    (img, m), n = serve(page, lambda url: PG.get(url, 64))
    assert img.shape == (64, 64, 3) and img.dtype == np.uint8 and tuple(img[10, 10]) == (200, 0, 0)
    assert page.shots == [CLIP] and n == 1 and m["view"] == CLIP and "png" not in m
    assert PG.bridge(m)["removedMm3"] == 12.5 and PG.medium(m) == "water"
    assert RO.drill_visible(m, "water")   # rollout reads the reply as it reads a step session's


def test_no_image_when_asked_not_to_or_when_the_view_is_closed():
    page = Page()
    (img, m), _ = serve(page, lambda url: PG.get(url, 64, image=False))
    assert img is None and page.shots == [] and PG.bridge(m)["endoPose"]["depth"] == 30.0
    page = Page(info={"views": [INFO["views"][0]]})
    (img, m), _ = serve(page, lambda url: PG.get(url, 64))
    assert img is None and m["view"] is None


def test_a_page_without_the_bridge_is_an_error_not_an_empty_observation():
    with pytest.raises(RuntimeError, match="__ubeRender"):
        serve(Page(bridge=False), lambda url: PG.get(url, 64))
    h, _ = serve(Page(bridge=False), PG.health)
    assert h == {"ok": True, "bridge": False, "requests": 0}
    assert PG.health(f"http://127.0.0.1:{free_port()}", timeout_s=1.0) is None   # nothing listens there
    with pytest.raises(RuntimeError, match="no observation"):
        PG.get(f"http://127.0.0.1:{free_port()}", timeout_s=1.0)


def test_medium_from_the_view_angle():
    m = lambda a: {"meta": {"bridge": metrics(angle=a)}}
    assert PG.medium(m(67.40)) == "water" and PG.medium(m(92.93)) == "air"
    assert PG.medium(m(30.0)) is None and PG.medium({}) is None   # the default pinhole camera / no camera yet


def test_chunk_cuts():
    assert PG.chunk_cuts([False, True, True, False], True) == [False, True, True, False]
    assert PG.chunk_cuts([False, True, True, False], False) == [False] * 4
    assert PG.chunk_cuts([False, False, False], True) == [False, False, True]   # cut after the last frame's trigger-off
    assert PG.chunk_cuts([], True) == []


def run_chunks(chunks, pause_s=1.0, stall_s=None, stall_mm3=30.0):
    """chunks = [(trigger per frame, removed volume the page shows after the chunk)] through the tracker as live_loop.look does."""
    tr = RO.PhaseTracker([1.0, 2.0, 3.0], 1 / 15, pause_s=pause_s, stall_s=stall_s, stall_mm3=stall_mm3)
    v = np.zeros(15); v[3] = v[10] = 1.0
    phases = []
    for trig, vol in chunks:
        m = {"meta": {"bridge": metrics(vol)}}
        for t, cut in zip(trig, PG.chunk_cuts(trig, vol > tr.vol + 1e-6)):
            v[14] = float(t)
            phases.append(tr.update(PG.frame_meta(m, vol if cut else tr.vol, cut), v.copy()))
    return tr, phases


def test_phase_rules_on_chunk_totals():
    on, off = [True] * 20, [False] * 20
    tr, ph = run_chunks([(off, 0.0), (on, 6.0), (on, 11.0)])
    assert ph[:20] == [3] * 20 and ph[20:] == [4] * 40 and tr.vol == 11.0 and tr.end is None   # inside → 3, first cut → 4
    assert tr.drill_s == pytest.approx(40 / 15)
    # the trigger released: about 1 s without a cut and the trigger off ends the drilling (pause), at the 15th such frame
    tr, ph = run_chunks([(on, 6.0), (off, 6.0)], pause_s=0.97)
    assert tr.end == "pause" and ph.index(5) == 20 + 14
    # half a chunk with the trigger on: only those frames cut, the pause counts from the release
    tr, ph = run_chunks([([True] * 10 + [False] * 10, 6.0)], pause_s=0.5)
    assert tr.end == "pause" and ph.index(5) == 10 + 7
    # the trigger held but nothing removed any more: stall once a whole window (2 s here) of drilling gained under 1 mm³ —
    # the first cut (frame 0) is the window's start at frame 30
    tr, ph = run_chunks([(on, 6.0), (on, 6.0), (on, 6.0)], pause_s=15.0, stall_s=1.97, stall_mm3=1.0)
    assert tr.end == "stall" and ph.index(5) == 30
    # scraping a voxel off every chunk is a stall too (10-07: 33 mm³ in 25 s with the trigger held) …
    tr, ph = run_chunks([(on, 6.0)] + [(on, 6.0 + 0.5 * k) for k in range(1, 30)], pause_s=15.0, stall_s=25.0, stall_mm3=30.0)
    assert tr.end == "stall" and ph.index(5) == 375   # the first frame whose window (25 s = 375 frames back) starts in the drilling
    # … and removing 3.3 mm³ a chunk (≈ 62 mm³ in 25 s, twice the threshold) is not
    tr, ph = run_chunks([(on, 6.0)] + [(on, 6.0 + 3.3 * k) for k in range(1, 40)], pause_s=15.0, stall_s=25.0, stall_mm3=30.0)
    assert tr.end is None and 5 not in ph


def test_frame_meta_keeps_the_pages_other_values():
    m = {"meta": {"bridge": metrics(40.0)}, "view": CLIP}
    f = PG.frame_meta(m, 33.0, True)
    b = f["meta"]["bridge"]
    assert b["removedMm3"] == 33.0 and b["drill"]["proxy"] == {"cutting": True, "gate": "air", "penMm": 0.0}
    assert b["endoPose"] == m["meta"]["bridge"]["endoPose"] and b["drill"]["inView"]["x"] == 0.1
    assert m["meta"]["bridge"]["removedMm3"] == 40.0 and m["meta"]["bridge"]["drill"]["proxy"]["cutting"] is False   # not changed


# ---------------------------------------------------------------- the stream (10-12: observations while the arms move)
class Cast:
    """A stand-in for page_obs.Stream: one screencast frame — the whole viewport, blue, the endoscope view's rect green."""

    def __init__(self, age_ms=50.0, scale=1.0):
        import time
        W, H = int(PG.VIEWPORT["width"] * scale), int(PG.VIEWPORT["height"] * scale)
        im = np.zeros((H, W, 3), np.uint8); im[..., 2] = 220
        x0, y0, x1, y1 = (int(CLIP[k] * scale) for k in ("x", "y", "x", "y")); x1 += int(CLIP["width"] * scale); y1 += int(CLIP["height"] * scale)
        im[y0:y1, x0:x1] = (0, 200, 0)
        buf = io.BytesIO(); Image.fromarray(im, "RGB").save(buf, format="JPEG", quality=92)
        now = time.time() * 1000.0
        self.frame, self.n = (buf.getvalue(), now - age_ms, now - age_ms), 1


def serve_stream(page, cast, fn):
    async def go():
        port = free_port()
        srv = await PG.Server(page, port, stream=cast).start()
        try:
            return await asyncio.to_thread(fn, f"http://127.0.0.1:{port}")
        finally:
            await srv.stop()
    return asyncio.run(go())


def test_a_stream_observation_is_the_views_rect_of_the_newest_frame_and_takes_no_screenshot():
    page, cast = Page(), Cast()
    page.removed = 12.5
    img, d = serve_stream(page, cast, lambda u: PG.get(u, 256, stream=True))
    assert img.shape == (256, 256, 3) and d["stream"] is True and page.shots == []           # the page was not stopped for it
    assert abs(int(img[..., 1].mean()) - 200) < 6 and img[..., 2].mean() < 12                 # the green rect, none of the blue around it
    assert d["t_ms"] == pytest.approx(cast.frame[1]) and 0 <= d["frame_age_ms"] < 2000        # the frame's own time, not the request's
    assert PG.bridge(d)["removedMm3"] == 12.5 and d["view"] == CLIP
    img2, d2 = serve_stream(page, Cast(scale=0.5), lambda u: PG.get(u, 256, stream=True))     # a scaled frame is cropped at its own scale
    assert abs(int(img2[..., 1].mean()) - 200) < 8 and page.shots == []


def test_without_the_parameter_or_without_a_stream_the_screenshot_is_taken_as_before():
    page = Page()
    img, d = serve_stream(page, Cast(), lambda u: PG.get(u, 256))                             # stream there, not asked for
    assert page.shots == [CLIP] and "stream" not in d and int(img[..., 0].mean()) == 200
    page = Page()
    img, d = serve_stream(page, None, lambda u: PG.get(u, 256, stream=True))                  # asked for, no stream running
    assert page.shots == [CLIP] and "stream" not in d


def test_a_stale_frame_is_not_an_observation_of_now():
    page = Page()
    img, d = serve_stream(page, Cast(age_ms=5000.0), lambda u: PG.get(u, 256, stream=True))
    assert d["stream"] is False and page.shots == [CLIP] and int(img[..., 0].mean()) == 200   # the screenshot instead, and the reply says so


def test_the_stream_keeps_the_newest_frame_and_acknowledges_each():
    import base64

    class CDP:
        def __init__(self):
            self.sent, self.h = [], None

        def on(self, name, h):
            self.h = h

        async def send(self, name, params=None):
            self.sent.append((name, params))

    class Ctx:
        def __init__(self, cdp):
            self.cdp = cdp

        async def new_cdp_session(self, page):
            return self.cdp

    class P:
        pass

    async def go():
        cdp = CDP(); p = P(); p.context = Ctx(cdp)
        st = await PG.Stream(p).start()
        assert cdp.sent[0] == ("Page.startScreencast", {"format": "jpeg", "quality": 92, "everyNthFrame": 6})      # full size: no maxWidth
        cdp.h({"data": base64.b64encode(b"one").decode(), "metadata": {"timestamp": 1000.5}, "sessionId": 7})
        cdp.h({"data": base64.b64encode(b"two").decode(), "metadata": {"timestamp": 1001.25}, "sessionId": 8})
        await asyncio.sleep(0.01)
        assert st.frame[0] == b"two" and st.frame[1] == 1001250.0 and st.n == 2
        assert [x for x in cdp.sent if x[0] == "Page.screencastFrameAck"] == [("Page.screencastFrameAck", {"sessionId": 7}), ("Page.screencastFrameAck", {"sessionId": 8})]
        await st.stop()
        assert cdp.sent[-1][0] == "Page.stopScreencast"
    asyncio.run(go())
