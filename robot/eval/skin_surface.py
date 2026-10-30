#!/usr/bin/env python3
"""Where a tool's shaft crosses the skin of the model — from the viewer's own answers in the recordings.

    python skin_surface.py --build [--dataset DS] [--every 20]      → runs/jitter_iter/work/skin_points.npz (+ a check)
    python skin_surface.py --check                                  the prediction against recorded entries
    python skin_surface.py --build --recordings DIR [--only LIST] --out PATH
                                                                    the same from a folder of recordings (the public copies,
                                                                    <date>/<session>.jsonl) instead of a dataset's metadata;
                                                                    LIST = the session files to take: a text file, one name a
                                                                    line, or an earlier samples file (its `recordings`)
    python skin_surface.py --build --surface 4,5 --dataset DS --out DIR/skin_points.npz
                                                                    + one-surface models (skin_surface_deg<d>.json) and the
                                                                    held-out comparison with the local plane (skin_surface_check.json)

The viewer casts every tool's shaft line onto the skin of the model (castSkinRay) and the recorder writes the crossing into
every row (`drawn.entry`, both tools). The harness needs that point for a pose that is in no recording: a run's start pose
is a random draw, and the tool pivots about the skin point it entered through — the point is the run's own, found in the
model, not a point fixed beforehand (user, 2026-10-07: "강제로 피벗점을 정하게해선 안됨"; until then the pivot rule and
the ready pose used the median entry of the demonstrations). The page's live metrics carry the entry of the endoscope
only, and the case file is not ours to open — but the recorded entries are samples of the model's skin (thousands, over
the whole working area). `entry_point(p, u)` intersects a shaft line with the plane fitted to the samples nearest to it,
twice (the second fit around the first crossing): the skin there is smooth, and `--check` says how close this comes to
the viewer's own crossing on rows that were left out.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HOME = Path.home()
OUT = HOME / "il-robot/runs/jitter_iter/work/skin_points.npz"
DS = HOME / "il-robot/datasets/spine_hand24_replay_ph_trim_water"
GRID_MM, NEAR_MM, K_MIN = 1.0, 8.0, 12


def rows(path: str, every: int):
    """(tool, point on the shaft, unit direction to the tip, entry) of every `every`-th row with a skin crossing."""
    n = 0
    with open(path) as fh:
        for line in fh:
            if '"k":"f"' not in line[:60] or '"entry"' not in line:
                continue
            n += 1
            if n % every:
                continue
            r = json.loads(line); d = r.get("drawn") or {}
            e = d.get("entry")
            if not e or d.get("lost") or d.get("entryFallback"):
                continue
            p = d.get("B") if r.get("tool") == "drill" else d.get("tip")
            if p is None:
                continue
            p, e = np.asarray(p, float), np.asarray(e, float)
            u = p - e; L = float(np.linalg.norm(u))
            if L < 5.0:   # the tip at the skin: the line is not defined well enough
                continue
            yield r.get("tool"), p, u / L, e


def collect(recordings: list[str], every: int) -> dict:
    T, P, U, E = [], [], [], []
    for f in recordings:
        for tool, p, u, e in rows(f, every):
            T.append(tool == "drill"); P.append(p); U.append(u); E.append(e)
    return {"drill": np.asarray(T, bool), "p": np.asarray(P), "u": np.asarray(U), "entry": np.asarray(E)}


def downsample(E: np.ndarray, grid: float = GRID_MM) -> np.ndarray:
    """One point per grid cell (the mean of its samples): the recordings dwell at the portals."""
    key = np.floor(E / grid).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = np.asarray(inv).ravel()
    out = np.zeros((inv.max() + 1, 3)); np.add.at(out, inv, E)
    return out / np.bincount(inv)[:, None]


def _plane(Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    return c, vt[-1]


def entry_point(p, u, pts: np.ndarray, near_mm: float = NEAR_MM) -> np.ndarray | None:
    """The skin crossing of the line through p with unit direction u (pointing to the tip; the skin lies behind p), from
    the skin samples `pts`: plane through the samples nearest to the line, then again through those nearest to the first
    crossing. None when there are too few samples near the line (a pose far from every recorded one)."""
    p, u = np.asarray(p, float), np.asarray(u, float)
    u = u / np.linalg.norm(u)
    rel = pts - p
    along = rel @ u
    d_line = np.linalg.norm(rel - np.outer(along, u), axis=1)
    x = None
    for step in range(2):
        d = d_line if x is None else np.linalg.norm(pts - x, axis=1)
        idx = np.flatnonzero(d <= near_mm)
        if len(idx) < K_MIN:
            idx = np.argsort(d)[:K_MIN * 3]
            if d[idx[-1]] > 3 * near_mm:
                return None
        c, nrm = _plane(pts[idx])
        den = float(nrm @ u)
        if abs(den) < 0.05:   # the shaft runs along the skin
            return None
        x = p + u * float(nrm @ (c - p)) / den
    return x


def load(path: Path = OUT) -> np.ndarray:
    return np.load(path)["pts"]


# ---- the skin as one surface (user 10-23 via robot-mac: the public recordings may lose drawn.entry; what is published is
# a polynomial height field over the samples' own principal plane, its coefficients, axes and valid range — no samples)
HULL_K, HULL_MARGIN_MM = 16, 2.0   # the valid range: a convex 16-gon around the samples in (s, t), 2 mm outside them
# (10-23 via robot-mac: a cell list of the samples would show the skin's footprint — a few half-planes do not)


def _exps(deg: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(deg + 1) for j in range(deg + 1 - i)]


def fit_surface(pts: np.ndarray, degree: int = 5) -> dict:
    """Height h = f(s, t) over the principal plane of the skin samples (centre c, rows of R = e1, e2, e3 = normal), f a
    polynomial of total degree `degree` in s, t scaled to [-1, 1] over the samples' box (least squares). Valid range =
    the convex HULL_K-gon whose sides lie HULL_MARGIN_MM outside the outermost samples, as half-planes n·(s, t) <= d."""
    c = pts.mean(0)
    _, _, R = np.linalg.svd(pts - c, full_matrices=False)
    L = (pts - c) @ R.T
    lo, hi = L[:, :2].min(0), L[:, :2].max(0)
    mid, half = (lo + hi) / 2, (hi - lo) / 2
    S = (L[:, :2] - mid) / half
    E = _exps(degree)
    A = np.stack([S[:, 0] ** i * S[:, 1] ** j for i, j in E], 1)
    coef, *_ = np.linalg.lstsq(A, L[:, 2], rcond=None)
    res = L[:, 2] - A @ coef
    th = np.arange(HULL_K) * 2 * np.pi / HULL_K
    N = np.stack([np.cos(th), np.sin(th)], 1)
    D = (L[:, :2] @ N.T).max(0) + HULL_MARGIN_MM
    return {"format": "skin-surface-v1", "degree": degree, "centre_mm": c.tolist(), "axes_rows": R.tolist(),
            "st_mid_mm": mid.tolist(), "st_half_mm": half.tolist(), "exponents": E, "coef_mm": coef.tolist(),
            "box_lo_mm": lo.tolist(), "box_hi_mm": hi.tolist(), "valid_halfplanes": [[float(n[0]), float(n[1]), float(d)] for n, d in zip(N, D)],
            "fit_points": int(len(pts)), "fit_rms_mm": float(np.sqrt(np.mean(res ** 2))),
            "note": "scene frame (LPS, mm). local q = R (x - c); h = q[2] = sum coef_k * s'^i * t'^j, s' = (q[0] - mid[0]) / half[0], "
                    "t' = (q[1] - mid[1]) / half[1]; valid where nx*q[0] + ny*q[1] <= d for every [nx, ny, d] in valid_halfplanes"}


def _surf_eval(M: dict, s: float, t: float) -> tuple[float, float, float]:
    """h, dh/ds, dh/dt at local (s, t) in mm."""
    mid, half = M["st_mid_mm"], M["st_half_mm"]
    a, b = (s - mid[0]) / half[0], (t - mid[1]) / half[1]
    h = ds = dt = 0.0
    for (i, j), k in zip(M["exponents"], M["coef_mm"]):
        h += k * a ** i * b ** j
        if i:
            ds += k * i * a ** (i - 1) * b ** j / half[0]
        if j:
            dt += k * j * a ** i * b ** (j - 1) / half[1]
    return h, ds, dt


def _valid(M: dict, s: float, t: float) -> bool:
    return all(nx * s + ny * t <= d for nx, ny, d in M["valid_halfplanes"])


def _height(M: dict, s: float, t: float) -> float:
    """The surface height at local (s, t): one polynomial, or the patches blended with smooth weights (partition of unity)."""
    if M.get("format") == "skin-patches-v1":
        num = den = 0.0
        for P in M["patches"]:
            ws = 1.0
            for k, v in ((0, s), (1, t)):
                r = (v - P["st_mid_mm"][k]) / P["st_half_mm"][k]
                if abs(r) >= 1.0:
                    ws = 0.0; break
                ws *= (1.0 - r * r) ** 2
            if ws > 0.0:
                num += ws * _surf_eval(P, s, t)[0]; den += ws
        return num / den if den > 0.0 else float("nan")
    return _surf_eval(M, s, t)[0]


def _height_vec(M: dict, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """_height for arrays of (s, t)."""
    def poly(P, s, t):
        a, b = (s - P["st_mid_mm"][0]) / P["st_half_mm"][0], (t - P["st_mid_mm"][1]) / P["st_half_mm"][1]
        return sum(k * a ** i * b ** j for (i, j), k in zip(P["exponents"], P["coef_mm"]))
    if M.get("format") == "skin-patches-v1":
        num = np.zeros_like(s); den = np.zeros_like(s)
        for P in M["patches"]:
            ra = (s - P["st_mid_mm"][0]) / P["st_half_mm"][0]; rb = (t - P["st_mid_mm"][1]) / P["st_half_mm"][1]
            w_ = np.where((np.abs(ra) < 1) & (np.abs(rb) < 1), (1 - ra ** 2) ** 2 * (1 - rb ** 2) ** 2, 0.0)
            num += w_ * poly(P, s, t); den += w_
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
    return poly(M, s, t)


def _valid_vec(M: dict, s: np.ndarray, t: np.ndarray) -> np.ndarray:
    ok = np.ones_like(s, bool)
    for nx, ny, d in M["valid_halfplanes"]:
        ok &= nx * s + ny * t <= d
    return ok


def entry_point_surface(p, u, M: dict, reach_mm: float = 60.0, step_mm: float = 0.5) -> np.ndarray | None:
    """The crossing of the line through p (unit direction u) with the surface M: g(λ) = q[2] − h(q[0], q[1]) along the line,
    scanned every step_mm within reach_mm of its crossing with the principal plane, then bisected at the sign change nearest
    to it (both ends inside the valid range). None when the line runs along the plane or no crossing lies in the valid range."""
    p, u = np.asarray(p, float), np.asarray(u, float)
    u = u / np.linalg.norm(u)
    c, R = np.asarray(M["centre_mm"]), np.asarray(M["axes_rows"])
    q0, w = R @ (p - c), R @ u
    if abs(w[2]) < 0.05:
        return None
    lam0 = -q0[2] / w[2]

    def g(lam):
        q = q0 + lam * w
        return (q[2] - _height(M, q[0], q[1])) if _valid(M, q[0], q[1]) else float("nan")
    L = lam0 + np.arange(-reach_mm, reach_mm + step_mm / 2, step_mm)
    Q = q0[None, :] + L[:, None] * w[None, :]
    G = Q[:, 2] - _height_vec(M, Q[:, 0], Q[:, 1])
    G[~_valid_vec(M, Q[:, 0], Q[:, 1])] = np.nan
    best = None
    for k in range(len(L) - 1):
        if np.isfinite(G[k]) and np.isfinite(G[k + 1]) and G[k] * G[k + 1] <= 0.0:
            if best is None or abs(L[k] - lam0) < abs(L[best] - lam0):
                best = k
    if best is None:
        return None
    lo, hi, glo = L[best], L[best + 1], G[best]
    for _ in range(40):
        mid = (lo + hi) / 2; gm = g(mid)
        if not np.isfinite(gm):
            break
        if glo * gm <= 0.0:
            hi = mid
        else:
            lo, glo = mid, gm
    return p + (lo + hi) / 2 * u


def fit_patches(pts: np.ndarray, degree: int = 3, grid: int = 3) -> dict:
    """The skin as overlapping patches in one frame: the samples' (s, t) box cut into grid × grid tiles, each tile widened to
    twice its size (half-overlap with its neighbours) and fitted with its own polynomial of `degree`; heights blend with
    smooth weights. Same frame and valid range as fit_surface."""
    base = fit_surface(pts, 1)
    c, R = np.asarray(base["centre_mm"]), np.asarray(base["axes_rows"])
    L = (pts - c) @ R.T; lo, hi = np.asarray(base["box_lo_mm"]), np.asarray(base["box_hi_mm"])
    cell = (hi - lo) / grid; E = _exps(degree); patches = []; res = []
    for i in range(grid):
        for j in range(grid):
            mid = lo + cell * np.array([i + 0.5, j + 0.5]); half = cell.copy()      # tile ± one cell: overlaps its neighbours
            sel = np.all(np.abs(L[:, :2] - mid) < half, axis=1)
            if sel.sum() < 3 * len(E):
                continue
            S = (L[sel, :2] - mid) / half
            A = np.stack([S[:, 0] ** a * S[:, 1] ** b for a, b in E], 1)
            coef, *_ = np.linalg.lstsq(A, L[sel, 2], rcond=None)
            patches.append({"st_mid_mm": mid.tolist(), "st_half_mm": half.tolist(), "exponents": E, "coef_mm": coef.tolist()})
    M = {"format": "skin-patches-v1", "degree": degree, "grid": grid, "centre_mm": base["centre_mm"], "axes_rows": base["axes_rows"],
         "box_lo_mm": base["box_lo_mm"], "box_hi_mm": base["box_hi_mm"], "valid_halfplanes": base["valid_halfplanes"], "patches": patches,
         "fit_points": int(len(pts)),
         "note": "scene frame (LPS, mm). local q = R (x - c); h(s, t) = sum_k w_k f_k / sum_k w_k over patches, f_k the patch "
                 "polynomial in s' = (s - mid_k[0]) / half_k[0], t' likewise, w_k = prod (1 - s'^2)^2 (1 - t'^2)^2 inside the patch; "
                 "valid where nx*s + ny*t <= d for every [nx, ny, d] in valid_halfplanes"}
    r = [L[k, 2] - _height(M, L[k, 0], L[k, 1]) for k in range(len(L))]
    M["fit_rms_mm"] = float(np.sqrt(np.nanmean(np.square(r))))
    return M


def save_surface(M: dict, path: Path) -> None:
    path.write_text(json.dumps({k: v for k, v in M.items() if not k.startswith("_")}, indent=1))


def load_surface(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def recordings_of(ds: Path) -> list[str]:
    out = []
    for line in (ds / "meta/spine_il/episodes.jsonl").read_text().splitlines():
        m = json.loads(line) if line.strip() else {}
        if m.get("source") == "hand" and m.get("recording") not in out:
            out.append(m["recording"])
    return out


def recordings_in(folder: Path, only: Path | None = None) -> list[str]:
    """The recordings under `folder`, in file-name order (= a dataset's order: the names start with the session's time;
    the order decides which rows are held out). `only`: the names to take — a text file or a samples file (.npz)."""
    recs = sorted(folder.rglob("s_*.jsonl"), key=lambda p: p.name)
    if only is not None:
        names = set(np.load(only)["recordings"].tolist()) if only.suffix == ".npz" else \
            {x.strip() for x in only.read_text().splitlines() if x.strip()}
        recs = [p for p in recs if p.name in names]
        if len(recs) != len(names):
            raise SystemExit(f"{len(names) - len(recs)} of the {len(names)} recordings in {only} are not under {folder}")
    return [str(p) for p in recs]


def check(C: dict, pts: np.ndarray, n: int = 600, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(C["p"]), size=min(n, len(C["p"])), replace=False)
    err = {"endo": [], "drill": []}; miss = 0
    for i in idx:
        x = entry_point(C["p"][i], C["u"][i], pts)
        if x is None:
            miss += 1; continue
        err["drill" if C["drill"][i] else "endo"].append(float(np.linalg.norm(x - C["entry"][i])))
    st = lambda v: {"n": len(v), "median_mm": round(float(np.median(v)), 2), "p95_mm": round(float(np.percentile(v, 95)), 2), "max_mm": round(float(np.max(v)), 2)} if v else None
    return {"rows": int(len(idx)), "no_answer": miss, "endo": st(err["endo"]), "drill": st(err["drill"])}


def surfaces_of(spec: str, pts: np.ndarray, pts_tool: dict) -> dict:
    """'g<d>': one surface of degree d over both tools' samples; 't<d>': one per tool, each over that tool's own samples
    (the two portals lie apart); 'p<d>x<n>': per tool, n × n overlapping patches of degree d. → {"endo": model, "drill": model}."""
    if spec[0] == "p":
        d, n = (int(x) for x in spec[1:].split("x"))
        return {t: fit_patches(pts_tool[t], d, n) for t in ("endo", "drill")}
    d = int(spec[1:])
    if spec[0] == "g":
        M = fit_surface(pts, d); return {"endo": M, "drill": M}
    return {t: fit_surface(pts_tool[t], d) for t in ("endo", "drill")}


def compare(held: dict, pts: np.ndarray, variants: dict[str, dict], ref: dict) -> dict:
    """Every held-out row: the local-plane answer and each variant's answer against the viewer's own crossing (castSkinRay
    → drawn.entry); per tool median / p95 / max, and the error by the shaft's angle to the skin normal (of `ref`)."""
    st = lambda v: {"n": len(v), "median_mm": round(float(np.median(v)), 3), "p95_mm": round(float(np.percentile(v, 95)), 3),
                    "max_mm": round(float(np.max(v)), 3)} if len(v) else None
    c, R = np.asarray(ref["centre_mm"]), np.asarray(ref["axes_rows"])
    ang = []
    for i in range(len(held["p"])):
        q = R @ (held["entry"][i] - c); _, ds, dt = _surf_eval(ref, q[0], q[1])
        n = R.T @ np.array([-ds, -dt, 1.0]); n /= np.linalg.norm(n)
        ang.append(float(np.degrees(np.arccos(min(1.0, abs(n @ held["u"][i]))))))
    ang = np.asarray(ang); bins = [(0, 20), (20, 35), (35, 50), (50, 90)]
    meth = {"local_plane": None, **variants}
    out = {"held_rows": int(len(held["p"])), "angle_to_normal_deg": {"median": round(float(np.median(ang)), 1),
           "p95": round(float(np.percentile(ang, 95)), 1)}, "methods": {}}
    for name, V in meth.items():
        err = np.full(len(held["p"]), np.nan)
        for i in range(len(held["p"])):
            tool = "drill" if held["drill"][i] else "endo"
            x = entry_point(held["p"][i], held["u"][i], pts) if V is None else entry_point_surface(held["p"][i], held["u"][i], V[tool])
            if x is not None:
                err[i] = float(np.linalg.norm(x - held["entry"][i]))
        ok = ~np.isnan(err)
        out["methods"][name] = {"no_answer": int((~ok).sum()), "endo": st(err[ok & ~held["drill"]]), "drill": st(err[ok & held["drill"]]),
                                "by_angle_deg": {f"{a}-{b}": st(err[ok & (ang >= a) & (ang < b)]) for a, b in bins},
                                **({"fit_rms_mm": {t: round(V[t]["fit_rms_mm"], 3) for t in V}} if V else {})}
        print(f"  {name} done", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--surface", default="", help="with --build: also fit the one-surface model of these degrees (e.g. 4,5), "
                    "write skin_surface_deg<d>.json beside --out and compare it with the local plane on the held-out rows")
    ap.add_argument("--out", default=str(OUT), help="the samples file to write (default: the one the robot runs use)")
    ap.add_argument("--reuse", action="store_true", help="with --surface: take the rows from --out (written by an earlier --surface build)")
    ap.add_argument("--dataset", default=str(DS))
    ap.add_argument("--every", type=int, default=20)
    ap.add_argument("--recordings", default="", help="a folder of recordings (<date>/<session>.jsonl) instead of --dataset")
    ap.add_argument("--only", default="", help="with --recordings: the session files to take (text file or samples .npz)")
    a = ap.parse_args()
    recs = recordings_in(Path(a.recordings).expanduser(), Path(a.only).expanduser() if a.only else None) if a.recordings \
        else recordings_of(Path(a.dataset).expanduser())
    if a.build and a.surface:
        out = Path(a.out).expanduser()
        if a.reuse and out.exists():            # the rows collected before (reading 98 recordings takes minutes)
            z = np.load(out); C = {"p": z["all_p"], "u": z["all_u"], "entry": z["all_entry"], "drill": z["all_drill"]}
        else:
            C = collect(recs, a.every)
        fit = np.arange(len(C["entry"])) % 5 != 0            # every fifth sampled row is kept out of all fits
        pts = downsample(C["entry"][fit])
        pts_tool = {t: downsample(C["entry"][fit & (C["drill"] == (t == "drill"))]) for t in ("endo", "drill")}
        held = {k: v[~fit] for k, v in C.items()}
        np.savez_compressed(out, pts=pts, recordings=np.array([Path(r).name for r in recs]),
                            all_p=C["p"], all_u=C["u"], all_entry=C["entry"], all_drill=C["drill"],
                            held_p=held["p"], held_u=held["u"], held_entry=held["entry"], held_drill=held["drill"])
        variants = {}
        for spec in [x.strip() for x in a.surface.split(",") if x.strip()]:
            spec = spec if spec[0] in "gtp" else "g" + spec
            variants[spec] = surfaces_of(spec, pts, pts_tool)
            for t, M in variants[spec].items():
                if spec[0] != "g" or t == "endo":       # per-tool variants: one file per tool
                    save_surface(M, out.with_name(f"skin_surface_{spec}{'_' + t if spec[0] != 'g' else ''}.json"))
        ref = fit_surface(pts, 6)
        rep = {"recordings": len(recs), "rows": int(len(C["entry"])), "skin_points": int(len(pts)),
               "skin_points_per_tool": {t: int(len(v)) for t, v in pts_tool.items()}, "check": compare(held, pts, variants, ref)}
        out.with_name("skin_surface_check.json").write_text(json.dumps(rep, indent=1))
        for name, m in rep["check"]["methods"].items():
            f = lambda q: f"{q['median_mm']:.2f}/{q['p95_mm']:.2f}/{q['max_mm']:.2f}" if q else "-"
            print(f"{name:<12} none {m['no_answer']:3} | endo {f(m['endo'])} | drill {f(m['drill'])} | by angle "
                  + " ".join(f"{k}:{f(v)}" for k, v in m["by_angle_deg"].items()))
        print(f"→ {out.with_name('skin_surface_check.json')}")
    elif a.build:
        C = collect(recs, a.every)
        fit = np.arange(len(C["entry"])) % 5 != 0            # every fifth sampled row is kept out of the surface, for the check
        pts = downsample(C["entry"][fit])
        held = {k: v[~fit] for k, v in C.items()}
        np.savez_compressed(Path(a.out).expanduser(), pts=pts, recordings=np.array([Path(r).name for r in recs]))
        print(json.dumps({"recordings": len(recs), "rows": int(len(C["entry"])), "skin_points": int(len(pts)),
                          "extent_mm": [pts.min(0).round(1).tolist(), pts.max(0).round(1).tolist()], "held_out_check": check(held, pts)}))
        print(f"→ {a.out}")
    elif a.check:
        print(json.dumps(check(collect(recs, a.every * 7), load())))


if __name__ == "__main__":
    main()
