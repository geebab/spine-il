#!/usr/bin/env python3
"""One playable video per hand recording with its phase labels laid over the endoscope images (2026-10-06).

    python phase_report.py --json R.json                                  # labels + the signals behind them
    python phase_video.py --report R.json --cache STEP_CACHE --out DIR [--only IDS] [--missing]

Left: the step-rendered endoscope image of every 15 Hz frame (the dataset's own images, 2×), framed in the colour of
the frame's phase. Right: the phase (number, name), the time to the next boundary, and what the rules read at that
moment — trigger, bone cut rows, each tool inside / outside the skin with its depth past the entry and its tip speed.
Bottom: the session's phase bands, cut rows and trigger spans with the play position.
The phase of a frame is phases.PhaseResult.phase_at() of the report's boundaries — what export_lerobot.py stores.
A session is skipped while its images are not in the cache (render_step.py); --missing skips finished videos.
"""
from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import phases as PH  # noqa: E402
    import render_cache as RC  # noqa: E402
else:
    from . import phases as PH
    from . import render_cache as RC

W, H, IMG = 1040, 640, 512
BG, FG, MUTED, RULE, WARN = (18, 22, 29), (230, 233, 239), (154, 164, 181), (46, 54, 67), (240, 179, 90)
COL = {1: (111, 122, 140), 2: (79, 143, 214), 3: (53, 174, 157), 4: (224, 154, 53), 5: (156, 122, 218), 6: (176, 92, 126)}
ON, OFF = (111, 207, 151), (70, 80, 96)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-{}.ttc"
NAME = {1: "Ready", 2: "Approach", 3: "Advance", 4: "Drilling", 5: "Withdraw", 6: "Return"}   # every text in the video is English (user, 10-06)
EN = {k: v.split(" / ")[1] for k, v in PH.PHASES.items()}
TL = (84, W - 20, 556)   # timeline: x0, x1, y


@functools.lru_cache(maxsize=None)
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT.format("Bold" if bold else "Regular"), size, index=1)   # index 1 = the KR face


def frames_of(cache: Path, session: str, fps: float, size: int) -> dict[int, Path]:
    base = Path(cache) / session / RC.spec_tag(fps, ("endo",), size)
    out = {}
    for d in sorted(p for p in base.glob("c*-*") if (p / "DONE").exists()) if base.exists() else []:
        a, b = (int(x) for x in d.name[1:].split("-"))
        out.update({k: d / "frames/endo" / f"{k - a:06d}.png" for k in range(a, b + 1)})
    return out


def in_spans(sp: list, t: float) -> bool:
    return any(a <= t < b for a, b in sp)


def wrap(d: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    lines, cur = [], ""
    for ch in text:
        if d.textlength(cur + ch, font=f) > width:
            lines.append(cur); cur = ch.lstrip()
        else:
            cur += ch
    return lines + [cur] if cur else lines


def background(r: dict) -> Image.Image:
    """Everything that does not change during the session: the timeline, the row labels, the notes."""
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    x0, x1, y = TL; sx = lambda t: x0 + t / r["t_end"] * (x1 - x0)
    for k, (a, b) in r["phases"].items():
        if b > a:
            d.rectangle([sx(a), y, max(sx(b), sx(a) + 1), y + 24], fill=COL[int(k)])
            if sx(b) - sx(a) > 18:
                d.text(((sx(a) + sx(b)) / 2, y + 12), str(k), font=font(15, True), fill=(255, 255, 255), anchor="mm")
    for t in r["cut_t"]:
        d.line([sx(t), y + 28, sx(t), y + 35], fill=FG)
    for a, b in r["trigger"]:
        d.rectangle([sx(a), y + 42, max(sx(b), sx(a) + 1), y + 49], fill=MUTED)
    for yy, lab in ((y + 12, "phase"), (y + 31, "bone cut"), (y + 46, "trigger")):
        d.text((x0 - 8, yy), lab, font=font(12), fill=MUTED, anchor="rm")
    step = next(s for s in (5, 10, 20, 30, 60) if r["t_end"] / s <= 14)
    for t in np.arange(0, r["t_end"] + 1e-9, step):
        d.text((sx(t), y + 66), f"{t:g}", font=font(12), fill=MUTED, anchor="mm")
    d.text((x1, y + 66), "s", font=font(12), fill=MUTED, anchor="rm")
    px = IMG + 44
    for i, lab in enumerate(("Trigger", "Bone cut", "Endoscope", "Drill")):
        d.text((px, 218 + i * 46), lab, font=font(20), fill=MUTED, anchor="lm")
    yy, f = 402, font(14)
    for c in [c for c in r["checks"] if c["level"] == "warn"][:4]:
        for line in wrap(d, "check: " + c.get("en", c["text"]), f, W - px - 20)[:3]:
            d.text((px, yy), line, font=f, fill=WARN); yy += 19
        yy += 4
    return im


def pill(d: ImageDraw.ImageDraw, x: float, y: float, text: str, on: bool, colour=ON) -> float:
    f = font(18, True); w = d.textlength(text, font=f) + 22
    d.rounded_rectangle([x, y - 15, x + w, y + 15], radius=6, fill=colour if on else OFF)
    d.text((x + w / 2, y), text, font=f, fill=(12, 16, 22) if on else MUTED, anchor="mm")
    return x + w


STRIP_H = 108


def strip_background(r: dict, width: int) -> Image.Image:
    """The part of the strip that does not change: phase bands, bone-cut rows and trigger spans over the whole session."""
    im = Image.new("RGB", (width, STRIP_H), BG); d = ImageDraw.Draw(im)
    x0, x1, y = 84, width - 16, 50; sx = lambda t: x0 + t / r["t_end"] * (x1 - x0)
    for k, (a, b) in r["phases"].items():
        if b > a:
            d.rectangle([sx(a), y, max(sx(b), sx(a) + 1), y + 16], fill=COL[int(k)])
            if sx(b) - sx(a) > 16:
                d.text(((sx(a) + sx(b)) / 2, y + 8), str(k), font=font(12, True), fill=(255, 255, 255), anchor="mm")
    for t in r["cut_t"]:
        d.line([sx(t), y + 20, sx(t), y + 27], fill=FG)
    for a, b in r["trigger"]:
        d.rectangle([sx(a), y + 31, max(sx(b), sx(a) + 1), y + 38], fill=ON)
    for yy, lab in ((y + 8, "phase"), (y + 23, "bone cut"), (y + 35, "trigger")):
        d.text((x0 - 8, yy), lab, font=font(11), fill=MUTED, anchor="rm")
    step = next(s for s in (5, 10, 20, 30, 60, 120, 300, 600) if r["t_end"] / s <= 14)
    for t in np.arange(0, r["t_end"] + 1e-9, step):
        d.text((sx(t), y + 49), f"{t:g}", font=font(11), fill=MUTED, anchor="mm")
    d.text((x1, y + 49), "s", font=font(11), fill=MUTED, anchor="rm")
    return im


def strip(bg: Image.Image, r: dict, t: float) -> Image.Image:
    """The strip at session time t (s): phase, trigger and bone-cut state, play position — as in the phase check videos."""
    b = r["bounds"]
    p = int(PH.PhaseResult({}, b["t_move"], b["t_in"], b["t_cut0"], b["t_cut1"], b["t_out"]).phase_at(np.array([t]))[0])
    im = bg.copy(); d = ImageDraw.Draw(im); w = im.size[0]
    d.rounded_rectangle([12, 8, 196, 40], radius=8, fill=COL[p])
    d.text((104, 24), f"{p}  {NAME[p]}", font=font(20, True), fill=(255, 255, 255), anchor="mm")
    d.text((214, 24), "Trigger", font=font(17), fill=MUTED, anchor="lm")
    on = in_spans(r["trigger"], t); x = pill(d, 282, 24, "ON" if on else "OFF", on)
    d.text((x + 22, 24), "Bone cut", font=font(17), fill=MUTED, anchor="lm")
    cut = np.asarray(r["cut_t"], float); cutting = bool(len(cut) and np.abs(cut - t).min() <= 0.1)
    pill(d, x + 104, 24, "CUTTING" if cutting else "–", cutting, COL[4])
    d.text((w - 16, 24), f"{t:6.1f} s / {r['t_end']:.1f} s", font=font(17, True), fill=FG, anchor="rm")
    xp = 84 + min(max(t, 0.0), r["t_end"]) / r["t_end"] * (w - 16 - 84)
    d.line([xp, 46, xp, 92], fill=(255, 255, 255), width=2)
    return im


def video(r: dict, frames: dict[int, Path], out: Path, fps: float) -> None:
    b = r["bounds"]
    ph = PH.PhaseResult({}, b["t_move"], b["t_in"], b["t_cut0"], b["t_cut1"], b["t_out"])
    ks = list(range(0, max(frames) + 1))
    phase = ph.phase_at(np.array(ks) / fps)
    nxt = [b["t_move"], b["t_in"], b["t_cut0"], b["t_cut1"], b["t_out"], r["t_end"]]
    cut_t = np.asarray(r["cut_t"], float)
    bg = background(r); x0, x1, y = TL; px = IMG + 44
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "27", "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for k in ks:
        t, p = k / fps, int(phase[k])
        im = bg.copy(); d = ImageDraw.Draw(im)
        d.rounded_rectangle([14, 14, 26 + IMG, 26 + IMG], radius=10, outline=COL[p], width=6)
        e = Image.open(frames[k]).convert("RGB").resize((IMG, IMG), Image.BICUBIC)
        im.paste(e, (20, 20))
        d = ImageDraw.Draw(im)
        d.text((28, 24), r["id"], font=font(22, True), fill=FG)    # the image's corners are black
        d.text((28, 52), f'{r["date"][:4]}-{r["date"][4:6]}-{r["date"][6:]}', font=font(13), fill=MUTED)
        # phase banner
        d.rounded_rectangle([px, 20, W - 20, 112], radius=10, fill=COL[p])
        d.text((px + 40, 66), str(p), font=font(60, True), fill=(255, 255, 255), anchor="mm")
        d.text((px + 84, 50), NAME[p], font=font(30, True), fill=(255, 255, 255), anchor="lm")
        d.text((px + 84, 88), EN[p], font=font(15), fill=(255, 255, 255), anchor="lm")
        d.text((px, 138), f"{t:6.1f} s / {r['t_end']:.1f} s", font=font(22, True), fill=FG, anchor="lm")
        left = nxt[p - 1] - t
        d.text((px, 170), (f"next: phase {p + 1} {NAME[p + 1]} in {max(left, 0):.1f} s" if p < 6 else f"end of the recording in {max(left, 0):.1f} s"),
               font=font(16), fill=MUTED, anchor="lm")
        # what the rules read
        pill(d, px + 112, 218, "ON" if in_spans(r["trigger"], t) else "OFF", in_spans(r["trigger"], t))
        cutting = bool(len(cut_t) and np.abs(cut_t - t).min() <= 0.1)
        pill(d, px + 112, 264, "CUTTING" if cutting else "–", cutting, COL[4])
        i = min(int(round(t / r["grid_s"])), len(r["endo"]["speed"]) - 1)
        for row, name in ((310, "endo"), (356, "drill")):
            ins = in_spans(r[name]["inside"], t); dep, sp = r[name]["depth"][i], r[name]["speed"][i]
            xe = pill(d, px + 112, row, "inside skin" if ins else "outside skin", ins, COL[3])
            away = dep is not None and dep > 0 and not ins     # past the skin along the shaft, but not at the working portal (the ready pose)
            d.text((xe + 10, row), ("not at the portal" if away else "depth –" if dep is None else f"depth {dep:+.0f} mm") + f" · {sp:.0f} mm/s",
                   font=font(15), fill=FG, anchor="lm")
        xp = x0 + t / r["t_end"] * (x1 - x0)
        d.line([xp, y - 8, xp, y + 51], fill=(255, 255, 255), width=2)
        d.polygon([(xp - 6, y - 14), (xp + 6, y - 14), (xp, y - 5)], fill=(255, 255, 255))
        ff.stdin.write(im.tobytes())
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", type=Path, required=True, help="phase_report.py --json")
    ap.add_argument("--cache", type=Path, required=True, help="step render cache")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--missing", action="store_true", help="skip sessions whose video exists")
    a = ap.parse_args()
    out = a.out.expanduser(); out.mkdir(parents=True, exist_ok=True)
    only = {x for x in a.only.split(",") if x}
    for r in json.loads(a.report.expanduser().read_text()):
        if only and r["id"] not in only:
            continue
        r["phases"] = {int(k): v for k, v in r["phases"].items()}
        f = out / f"phase_{r['id']}.mp4"
        if a.missing and f.exists():
            continue
        fr = frames_of(a.cache.expanduser(), r["session"], a.fps, a.size)
        if not fr or set(range(max(fr) + 1)) - set(fr):
            print(f"{r['id']}: images not in the cache yet — skipped", flush=True)
            continue
        tmp = f.with_suffix(".tmp.mp4")
        video(r, fr, tmp, a.fps)
        tmp.rename(f)
        print(f"{r['id']}: {max(fr) + 1} frames → {f} ({f.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
