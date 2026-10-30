"""Automatic phase segmentation of one FE recorder v2 session (hand or robot-origin) into the 6 surgical phases.

  1 ready      until the motion starts: the end of the last still period (both tips < V_STILL_MMS) before phase 3
  2 approach   moving, both tools still outside the skin at the working portal
  3 advance    a tool is inside the skin, before the first bone cut
  4 drill      first → last bone cut (cut rows, target=bone; cut bursts closer than CUT_GAP_S merged, isolated blips of
               < BLIP_ROWS rows dropped when other bursts exist)
  5 withdraw   after the last cut until both tools are out of the skin
  6 return     outside the skin until the end

Skin signal (no recorder field says "inside the skin"; hdr/f rows were checked — `split`, `proxy.gate`, `real` are
not it, and the viewer's "진입점 확보" badge is a UI state = viewActive, not skin contact). What the recorder does
carry per f row is the viewer's portal geometry on the drawn pose:
  endo   drawn.entry = where the shaft line crosses the skin (castSkinRay), drawn.depth = signed tip distance past it
  drill  drawn.entry, drawn.B (burr), drawn.u (shaft, pointing to the tip) → depth = (B − entry)·u
A tool is *inside* when depth > DEPTH_IN_MM and its entry is within PORTAL_MM of the working portal (the median
entry of that tool while drilling). The portal test is needed: at the ready pose the tools rest on the dummy's flank
and the virtual skin there gives depth +10…+15 mm with the entry 45–70 mm from the portal (rev.4 set, all 8).
Reliability (rev.4): the in→out transitions are sharp (entry distance < 15 mm while working, depth ≤ 0 or the entry
jumps > 40 mm within ~1 s of leaving); a boundary is uncertain by about one tracker frame plus the dwell MIN_RUN_S.
Robot-origin sessions carry the same drawn fields (the live viewer under the anchor registration).

Pure numpy; `from_rows()` reads recorder rows, `segment()` does the work (tested on synthetic tracks).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PHASES = {
    1: "1 준비자세 / hold the ready pose",
    2: "2 피부진입점으로 이동 / move to the skin entry point",
    3: "3 피부에서 타겟으로 이동 / advance from the skin to the target",
    4: "4 L4-5 후궁 드릴링 / drill the L4-5 lamina",
    5: "5 피부 밖으로 이동 / withdraw out of the skin",
    6: "6 준비자세로 복귀 / return to the ready pose",
}
V_STILL_MMS = 20.0   # both tips slower than this (mm/s) = still (ready pose)
DEPTH_IN_MM = 0.0    # depth past the skin entry that counts as inside
PORTAL_MM = 20.0     # entry within this of the working portal
MIN_RUN_S = 0.3      # in/out runs shorter than this are ignored (flicker at the skin surface)
CUT_GAP_S = 1.0      # cut rows closer than this belong to one burst
BLIP_ROWS = 3        # a burst with fewer rows (and < 0.2 s) is a blip
GRID_S = 0.02        # evaluation grid for the in/out logic


@dataclass
class ToolTrack:
    t: np.ndarray        # s (session axis)
    tip: np.ndarray      # (N, 3) mm, tracker tip in the scene
    depth: np.ndarray    # (N,) mm past the skin entry (nan = unknown)
    entry: np.ndarray    # (N, 3) mm skin entry (nan = unknown)


@dataclass
class PhaseResult:
    bounds: dict[int, tuple[float, float]]      # phase → [start, end) s (end of 6 = t_end, inclusive)
    t_move: float
    t_in: float
    t_cut0: float
    t_cut1: float
    t_out: float
    flags: list[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    def phase_at(self, t) -> np.ndarray:
        """Phase index (1..6) at times t (s)."""
        t = np.asarray(t, float)
        edges = np.array([self.t_move, self.t_in, self.t_cut0, self.t_cut1, self.t_out])
        ph = 1 + np.searchsorted(edges, t, side="right")
        ph[(t >= self.t_cut0) & (t <= self.t_cut1)] = 4   # the last cut instant is still drilling
        return ph.astype(np.int64)

    def table(self) -> dict:
        return {k: [round(a, 2), round(b, 2)] for k, (a, b) in self.bounds.items()}


def _zoh(t_src: np.ndarray, v: np.ndarray, t: np.ndarray, default=False) -> np.ndarray:
    i = np.searchsorted(t_src, t, side="right") - 1
    out = np.where(i >= 0, v[np.clip(i, 0, len(v) - 1)], default)
    return out


def _clean_runs(x: np.ndarray, min_n: int) -> np.ndarray:
    """Flip runs (True or False) shorter than min_n samples, shortest first, never the first/last run's value at the
    very ends unless it is short too."""
    x = x.copy()
    for _ in range(2000):
        if len(x) == 0:
            return x
        edges = np.flatnonzero(np.diff(x.astype(np.int8))) + 1
        if len(edges) == 0:
            break
        starts = np.r_[0, edges]
        ends = np.r_[edges, len(x)]
        n = ends - starts
        k = int(np.argmin(n))
        if n[k] >= min_n:
            break
        x[starts[k]:ends[k]] = not x[starts[k]]
    return x


def cut_bursts(cut_t: np.ndarray, gap_s: float = CUT_GAP_S) -> list[tuple[float, float, int]]:
    """Cut row times → [(start, end, n rows)] with gaps < gap_s merged."""
    c = np.sort(np.asarray(cut_t, float))
    if len(c) == 0:
        return []
    out, a, n = [], c[0], 1
    for p, q in zip(c[:-1], c[1:]):
        if q - p >= gap_s:
            out.append((a, p, n))
            a, n = q, 0
        n += 1
    out.append((a, c[-1], n))
    return out


def _still(tr: ToolTrack, g: np.ndarray, win_s: float = 0.2, max_age_s: float = 1.0) -> np.ndarray:
    """On grid g: the tool is tracked and its tip (linearly interpolated) moved < V_STILL_MMS·win_s over the centred
    window [g − win_s/2, g + win_s/2]. The recorder thins rows of a still tool (0.25–0.6 s apart at rest, 919d), so
    samples up to max_age_s apart still count as tracked."""
    ok = np.isfinite(tr.tip).all(axis=1)
    t, p = tr.t[ok], tr.tip[ok]
    if len(t) < 2:
        return np.zeros(len(g), bool)
    h = win_s / 2
    a = np.stack([np.interp(g - h, t, p[:, c]) for c in range(3)], 1)
    b = np.stack([np.interp(g + h, t, p[:, c]) for c in range(3)], 1)
    i = np.clip(np.searchsorted(t, g, side="right") - 1, 0, len(t) - 1)
    j = np.minimum(i + 1, len(t) - 1)
    tracked = (g >= t[0] - h) & (g <= t[-1] + h) & ((t[j] - t[i]) <= max_age_s)
    return tracked & (np.linalg.norm(b - a, axis=1) / win_s < V_STILL_MMS)


def inside_mask(tr: ToolTrack, portal: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(tr.entry - portal, axis=1)
    with np.errstate(invalid="ignore"):
        return (tr.depth > DEPTH_IN_MM) & (d < PORTAL_MM)


def segment(endo: ToolTrack, drill: ToolTrack, cut_t, t_start: float, t_end: float, expect_full: bool = True
            ) -> PhaseResult:
    """Phase boundaries of one session on its own time axis (s). expect_full: a hand session should show all six
    phases in order — a missing one is flagged; a robot replay window is expected to lack 1/2 (and often 5/6)."""
    flags: list[str] = []
    bursts = cut_bursts(cut_t)
    kept = [b for b in bursts if b[2] >= BLIP_ROWS or b[1] - b[0] >= 0.2] or bursts
    if len(kept) < len(bursts):
        flags.append(f"절삭 블립 {len(bursts) - len(kept)}개 무시(<{BLIP_ROWS}행)")
    if not kept:
        raise ValueError("no bone cut rows — phase 4 undefined")
    t_cut0, t_cut1 = kept[0][0], kept[-1][1]
    gaps = [q[0] - p[1] for p, q in zip(kept[:-1], kept[1:])]

    g = np.arange(t_start, t_end + 1e-9, GRID_S)
    inside = np.zeros(len(g), bool)
    portals = {}
    for name, tr in (("endo", endo), ("drill", drill)):
        if len(tr.t) == 0:
            continue
        w = (tr.t >= t_cut0) & (tr.t <= t_cut1) & np.isfinite(tr.entry).all(axis=1)
        if not w.any():
            w = np.isfinite(tr.entry).all(axis=1)
        if not w.any():
            continue
        portal = np.median(tr.entry[w], axis=0)
        portals[name] = portal
        inside |= _zoh(tr.t, inside_mask(tr, portal), g, False).astype(bool)
    inside = _clean_runs(inside, max(1, int(round(MIN_RUN_S / GRID_S))))

    # skin entry: start of the inside run that reaches the first cut; exit: end of the run that holds the last cut
    i0 = min(np.searchsorted(g, t_cut0), len(g) - 1)
    i1 = min(np.searchsorted(g, t_cut1), len(g) - 1)
    if not inside[i0]:
        flags.append("첫 절삭 순간 두 기구 모두 피부 밖으로 판정됨")
        prev = np.flatnonzero(inside[:i0 + 1])
        i_in = prev[-1] if len(prev) else i0
    else:
        i_in = i0
    while i_in > 0 and inside[i_in - 1]:
        i_in -= 1
    t_in = float(g[i_in])
    j = i1
    while j < len(g) and inside[j]:
        j += 1
    t_out = float(g[j]) if j < len(g) else float(t_end)
    if t_out < t_cut1:
        t_out = t_cut1

    # motion start = the end of the last still period before the skin entry: both tips tracked and slower than
    # V_STILL_MMS for ≥ MIN_RUN_S (a recording may begin while the tools are still being brought to the ready pose —
    # 0338 — so "the first position" is not the rest pose)
    t_move = t_in
    if i_in > 0:
        n_run = max(1, int(round(MIN_RUN_S / GRID_S)))
        still = np.ones(len(g), bool)
        for tr in (endo, drill):
            still &= _still(tr, g)
        still = _clean_runs(still, n_run)
        hit = np.flatnonzero(still[:i_in])
        if len(hit):
            t_move = float(g[min(hit[-1] + 1, i_in)])
        else:
            t_move = float(t_start)
            flags.append("준비자세 정지 구간을 못 찾음(처음부터 움직임)")
    else:
        t_move = t_in = float(t_start)

    bounds = {1: (t_start, t_move), 2: (t_move, t_in), 3: (t_in, t_cut0), 4: (t_cut0, t_cut1),
              5: (t_cut1, t_out), 6: (t_out, t_end)}
    if expect_full:
        for k, (a, b) in bounds.items():
            if b - a <= 0:
                flags.append(f"{k}단계 없음(길이 0)")
    if not (t_start <= t_move <= t_in <= t_cut0 <= t_cut1 <= t_out <= t_end + 1e-9):
        flags.append("경계 순서 이상")
    info = {"portal_mm": {k: np.round(v, 1).tolist() for k, v in portals.items()},
            "cut_bursts": len(kept), "cut_gaps_over_1s": int(sum(1 for x in gaps if x >= CUT_GAP_S)),
            "cut_gap_total_s": round(float(sum(gaps)), 2)}
    return PhaseResult(bounds, t_move, t_in, t_cut0, t_cut1, t_out, flags, info)


# ---------------------------------------------------------------- recorder rows
def from_rows(rows: list[dict], t0_ms: float, T0) -> tuple[ToolTrack, ToolTrack, np.ndarray]:
    """Recorder rows (ordered) → (endo, drill, bone cut times) on the axis (tExp − t0_ms)/1000. T0 = (R, t) scene←cam
    for rows without scene.tip."""
    R, tt = T0
    tracks = {}
    for tool in ("endo", "drill"):
        t, tip, dep, ent = [], [], [], []
        for r in rows:
            if r.get("k") != "f" or (r.get("tool") or "endo") != tool or r.get("lost"):
                continue
            te = r.get("tExp")
            if not isinstance(te, (int, float)):
                continue
            sc = (r.get("scene") or {}).get("tip")
            if sc is None and r.get("tip") is not None:
                sc = (R @ np.asarray(r["tip"], float) + tt).tolist()
            if sc is None:
                continue
            d = r.get("drawn") or {}
            e = d.get("entry")
            if tool == "endo":
                dp = d.get("depth")
            elif e is not None and d.get("B") is not None and d.get("u") is not None:
                dp = float(np.dot(np.asarray(d["B"], float) - np.asarray(e, float), np.asarray(d["u"], float)))
            else:
                dp = None
            t.append((te - t0_ms) / 1000.0)
            tip.append(sc)
            dep.append(np.nan if dp is None or d.get("lost") else float(dp))
            ent.append([np.nan] * 3 if e is None or d.get("lost") else e)
        o = np.argsort(t, kind="stable")
        tracks[tool] = ToolTrack(np.asarray(t, float)[o], np.asarray(tip, float).reshape(-1, 3)[o],
                                 np.asarray(dep, float)[o], np.asarray(ent, float).reshape(-1, 3)[o])
    cuts = np.array([(r["tExp"] - t0_ms) / 1000.0 for r in rows
                     if r.get("k") == "cut" and r.get("target") == "bone" and isinstance(r.get("tExp"), (int, float))])
    return tracks["endo"], tracks["drill"], cuts
