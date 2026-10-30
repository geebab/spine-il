#!/usr/bin/env python3
"""The phase labels of the hand recordings, laid out for a human check (2026-10-06).

    python phase_report.py --json OUT.json [--only IDS] [--sessions DIR]
    python phase_video.py --report OUT.json --cache STEP_CACHE --out DIR      # then: one playable video per session

Per session: the six phase spans of phases.py and the signals its rules read, on a 0.1 s grid —
    speed    tip speed of each tool (mm/s; phase 1 ends where both were last below V_STILL_MMS)
    depth    tip depth past the skin entry of each tool (mm) and the inside test (depth > 0, entry within PORTAL_MM
             of the working portal; phase 3 starts when the first tool is inside, phase 5 ends when the last is out)
    cuts     bone-cut rows merged into bursts (phase 4 = first … last burst), trigger on spans
and what a reader should look at ("checks"): the segmenter's own flags, missing phases, a first or last burst that
stands alone (phase 4 would start early / end late), pauses in the drilling, time with both tools out of the skin
inside phase 4, the trigger held after the last cut. phase_video.py lays the labels over the endoscope images.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import phases as PH  # noqa: E402
    import render_cache as RC  # noqa: E402
    import sources as S  # noqa: E402
else:
    from . import phases as PH
    from . import render_cache as RC
    from . import sources as S

from lerobot_robot_spinearm.recording import parse_rigid, read_rows

GRID_S = 0.1
ALONE_GAP_S = 5.0     # a first / last burst this far from the next one, and short, "stands alone"
ALONE_ROWS = 15
PAUSE_S = 3.0         # pauses in the drilling worth listing


def spans(mask: np.ndarray, g: np.ndarray) -> list[list[float]]:
    """True runs of a mask on grid g → [[start, end], …] (s)."""
    m = np.r_[False, np.asarray(mask, bool), False]
    e = np.flatnonzero(np.diff(m.astype(np.int8)))
    step = g[1] - g[0] if len(g) > 1 else GRID_S
    return [[round(float(g[a]), 2), round(float(g[b - 1] + step), 2)] for a, b in zip(e[::2], e[1::2])]


def speed(tr: "PH.ToolTrack", g: np.ndarray, win_s: float = 0.2) -> np.ndarray:
    """Tip speed on the grid, the way phases._still measures it (interpolated tip over a centred window)."""
    ok = np.isfinite(tr.tip).all(axis=1)
    t, p = tr.t[ok], tr.tip[ok]
    if len(t) < 2:
        return np.full(len(g), np.nan)
    h = win_s / 2
    a = np.stack([np.interp(g - h, t, p[:, c]) for c in range(3)], 1)
    b = np.stack([np.interp(g + h, t, p[:, c]) for c in range(3)], 1)
    return np.linalg.norm(b - a, axis=1) / win_s


def flag_en(f: str) -> str:
    """phases.segment's flags (Korean) in English."""
    m = re.match(r"(\d)단계 없음", f)
    if m:
        return f"phase {m.group(1)} missing (zero length)"
    return {"준비자세 정지 구간을 못 찾음(처음부터 움직임)": "no still period found before the skin entry (the tools move from the start)",
            "첫 절삭 순간 두 기구 모두 피부 밖으로 판정됨": "both tools read outside the skin at the first cut",
            "경계 순서 이상": "phase boundaries out of order"}.get(f, f)


def analyse(s: "S.Session") -> dict:
    return analyse_path(s.path)


def analyse_path(path, expect_full: bool = True) -> dict:
    """One session file (hand, or a robot-origin session with expect_full=False: its missing phases are not flagged)."""
    hdr, rows, _ = read_rows(path)
    sid = hdr.get("sessionId") or Path(path).stem
    T0 = parse_rigid(hdr["registration"]["T0"])
    t0 = RC.timeline(path)["t_exp_first"]
    e, d, cuts = PH.from_rows(rows, t0, T0)
    t_end = max(float(e.t[-1]) if len(e.t) else 0.0, float(d.t[-1]) if len(d.t) else 0.0)
    ph = PH.segment(e, d, cuts, 0.0, t_end, expect_full=expect_full)
    g = np.arange(0.0, t_end + 1e-9, GRID_S)
    out = {"id": S.short_id(sid), "session": sid, "date": sid.split("_")[1][:8], "t_end": round(t_end, 2),
           "bounds": {k: round(float(v), 2) for k, v in (("t_move", ph.t_move), ("t_in", ph.t_in), ("t_cut0", ph.t_cut0),
                                                         ("t_cut1", ph.t_cut1), ("t_out", ph.t_out))},
           "phases": ph.table(), "flags": list(ph.flags), "grid_s": GRID_S, "t_exp_first": t0}
    inside = {}
    for name, tr in (("endo", e), ("drill", d)):
        portal = (ph.info.get("portal_mm") or {}).get(name)
        ins = PH._zoh(tr.t, PH.inside_mask(tr, np.asarray(portal, float)), g, False).astype(bool) if portal is not None and len(tr.t) \
            else np.zeros(len(g), bool)
        inside[name] = ins
        dep = PH._zoh(tr.t, tr.depth, g, np.nan).astype(float) if len(tr.t) else np.full(len(g), np.nan)
        out[name] = {"speed": np.round(speed(tr, g), 1).tolist(), "depth": [None if not np.isfinite(x) else round(float(x), 1) for x in dep],
                     "inside": spans(ins, g)}
    both_out = ~(inside["endo"] | inside["drill"])
    bursts = PH.cut_bursts(cuts)
    kept = [b for b in bursts if b[2] >= PH.BLIP_ROWS or b[1] - b[0] >= 0.2] or bursts
    out["bursts"] = [[round(a, 2), round(b, 2), n] for a, b, n in kept]
    out["blips"] = [[round(a, 2), round(b, 2), n] for a, b, n in bursts if (a, b, n) not in kept]
    out["cut_t"] = np.round(np.sort(cuts), 2).tolist()
    tt, tv = [], []
    for r in rows:
        if r.get("k") == "f" and r.get("tool") == "drill" and isinstance(r.get("tExp"), (int, float)):
            tt.append((r["tExp"] - t0) / 1000.0); tv.append(r.get("btn") == 1 and not r.get("btnStale"))
    o = np.argsort(tt, kind="stable")
    trig = PH._zoh(np.asarray(tt, float)[o], np.asarray(tv, bool)[o], g, False).astype(bool) if tt else np.zeros(len(g), bool)
    out["trigger"] = spans(trig, g)
    # ---- what to look at
    c, b = [], out["bounds"]
    c += [{"level": "warn", "text": f, "en": flag_en(f)} for f in ph.flags if "블립" not in f]
    stray = [x for x in out["blips"] if not b["t_cut0"] <= x[0] <= b["t_cut1"]]   # a dropped blip outside phase 4 = a bone cut labelled 3 or 5
    if stray:
        c.append({"level": "warn", "text": "4단계 밖의 짧은 뼈 절삭(무시됨): " + ", ".join(f"{x[0]:.1f} s({x[2]}행, {'3' if x[0] < b['t_cut0'] else '5'}단계)" for x in stray),
                  "en": "short bone cut outside phase 4 (ignored): " + ", ".join(f"{x[0]:.1f} s ({x[2]} rows, phase {'3' if x[0] < b['t_cut0'] else '5'})" for x in stray)})
    if len(out["blips"]) > len(stray):
        c.append({"level": "info", "text": f"4단계 안의 짧은 절삭 묶음 {len(out['blips']) - len(stray)}개는 따로 세지 않음(라벨 영향 없음)",
                  "en": f"{len(out['blips']) - len(stray)} short cut burst(s) inside phase 4 not counted separately (no effect on the labels)"})
    gaps = [(p[1], q[0]) for p, q in zip(kept[:-1], kept[1:])]
    if len(kept) > 1:
        if gaps[0][1] - gaps[0][0] >= ALONE_GAP_S and kept[0][2] <= ALONE_ROWS:
            c.append({"level": "warn", "text": f"첫 절삭 묶음이 혼자 떨어져 있음: {kept[0][0]:.1f} s에 {kept[0][2]}행, 다음 절삭은 {gaps[0][1]:.1f} s "
                                               f"→ 4단계가 {gaps[0][1] - kept[0][0]:.1f} s 일찍 시작했을 수 있음",
                      "en": f"the first cut burst stands alone: {kept[0][2]} rows at {kept[0][0]:.1f} s, the next cut at {gaps[0][1]:.1f} s — phase 4 may start "
                            f"{gaps[0][1] - kept[0][0]:.1f} s early"})
        if gaps[-1][1] - gaps[-1][0] >= ALONE_GAP_S and kept[-1][2] <= ALONE_ROWS:
            c.append({"level": "warn", "text": f"마지막 절삭 묶음이 혼자 떨어져 있음: {kept[-1][0]:.1f} s에 {kept[-1][2]}행, 그 앞 절삭은 {gaps[-1][0]:.1f} s에 끝남 "
                                               f"→ 4단계가 {kept[-1][1] - gaps[-1][0]:.1f} s 늦게 끝났을 수 있음",
                      "en": f"the last cut burst stands alone: {kept[-1][2]} rows at {kept[-1][0]:.1f} s, the cut before it ended at {gaps[-1][0]:.1f} s — phase 4 "
                            f"may end {kept[-1][1] - gaps[-1][0]:.1f} s late"})
    w4 = (g >= b["t_cut0"]) & (g <= b["t_cut1"])
    out_in4 = [x for x in spans(both_out & w4, g) if x[1] - x[0] >= PH.MIN_RUN_S]
    if out_in4:
        c.append({"level": "warn", "text": "4단계 안에 두 도구가 모두 피부 밖인 구간: " + ", ".join(f"{a:.1f}–{z:.1f} s" for a, z in out_in4)
                                           + " → 뺐다가 다시 넣은 구간이 드릴링으로 라벨됨",
                  "en": "both tools outside the skin inside phase 4: " + ", ".join(f"{a:.1f}–{z:.1f} s" for a, z in out_in4)
                        + " — a withdrawal and re-entry labelled as drilling"})
    long = [(a, z) for a, z in gaps if z - a >= PAUSE_S]
    if long:
        c.append({"level": "info", "text": f"드릴링 중 {PAUSE_S:g} s 이상 쉼 {len(long)}번(4단계로 라벨됨): " + ", ".join(f"{a:.1f}–{z:.1f} s" for a, z in long),
                  "en": f"{len(long)} pause(s) of {PAUSE_S:g} s or more while drilling (labelled phase 4): " + ", ".join(f"{a:.1f}–{z:.1f} s" for a, z in long)})
    after = float((trig & (g > b["t_cut1"]) & (g < b["t_out"])).sum() * GRID_S)
    if after >= 1.0:
        c.append({"level": "info", "text": f"마지막 절삭 뒤에도 방아쇠가 {after:.1f} s 켜져 있었음(뼈를 깎지 않아 5단계로 라벨됨)",
                  "en": f"trigger still on for {after:.1f} s after the last cut (no bone cut, labelled phase 5)"})
    first = min((n for n in ("endo", "drill") if out[n]["inside"]), key=lambda n: min((a for a, z in out[n]["inside"] if z > b["t_in"] - 0.05), default=1e9), default=None)
    out["first_in"] = first
    dur = {int(k): round(v[1] - v[0], 2) for k, v in out["phases"].items()}
    out["dur"] = dur
    out["pause_total_s"] = round(sum(z - a for a, z in gaps), 1)
    out["trigger_on_in4"] = round(float(trig[w4].mean()), 3) if w4.any() else None
    out["checks"] = c
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=Path, default=None, help="default ~/il-robot/viewer-data/sessions")
    ap.add_argument("--only", default="", help="comma list of session short ids")
    ap.add_argument("--json", type=Path, required=True)
    a = ap.parse_args()
    used, _ = RC.usable_sessions(("hand",), a.sessions, set(a.only.split(",")) if a.only else None)
    used.sort(key=lambda s: s.session_id)
    R = []
    for s in used:
        try:
            R.append(analyse(s))
        except (ValueError, KeyError, OSError) as e:
            print(f"{s.sid}: skipped ({e})")
            continue
        r = R[-1]
        print(f"{r['id']} {r['t_end']:6.1f} s | " + " ".join(f"{k}:{r['dur'][k]:5.1f}" for k in range(1, 7))
              + f" | pause {r['pause_total_s']:5.1f} | " + ("; ".join(c["text"] for c in r["checks"] if c["level"] == "warn") or "-"), flush=True)
    a.json.expanduser().write_text(json.dumps(R, ensure_ascii=False))


if __name__ == "__main__":
    main()
