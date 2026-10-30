"""A policy's cut vs the 8 hand demos — whole sessions, outcome metrics (2026-09-30).

    python policy_eval.py POLICY_RECORDING [--robot ROBOT_SESSION ...] [--label NAME]
    python policy_eval.py --demos-only                     # the demo × demo baseline (8×8)

Cuts (removed-voxel sets, render API, same SDF/grid required):
  demo k      each hand recording, whole session, its own recorded cut rows (replay)      cache iou_cache_hfull_<sid>.npz
  policy      the policy recording (policy2rec output), whole, simulated at the live page's 15 Hz tick
  robot       a robot-origin session of it (optional), its own recorded cut (replay)
Per cut vs each demo: IoU, Dice, tol1_dice, assd_mm, hd95_mm (metrics_ext.py) and the volume ratio; summarised as the
best-matching demo and the mean over demos. Region metrics: consensus = voxels cut by ≥ K of the demos (default 4 of 8)
→ coverage (share of it the cut contains) and precision (share of the cut inside it); off-target = cut volume outside
the demos' union dilated by 2 voxels (1.6 mm). The demo × demo table is the yardstick: a policy scoring like a demo vs
the others is as close to "a surgeon" as the surgeons are to each other. Writes ../policy_eval_<label>.json.
"""
import argparse, glob, hashlib, json, os, time
from pathlib import Path

import numpy as np

import iou_eval as E
import metrics_ext as X

DEMOS = ["3868", "91cb", "bef5", "d340", "bad1", "0338", "919d", "6639"]
REPLAY = {"views": [], "status": False, "fps": 10.0, "removed_voxels": True, "removed_every_s": 3}


def demo_path(rid: str) -> str:
    return sorted(f for f in glob.glob(str(Path.home() / f"il-robot/viewer-data/sessions/2026-09-28/*_trk_{rid}.jsonl")))[0]


def demo_cut(rid: str) -> dict:
    p = demo_path(rid)
    return E.cached("hfull_" + Path(p).stem, lambda: E.voxels(E.render(p, dict(REPLAY))))


def view_stats(rows: list[dict], optics: str | None = None) -> dict:
    """Endoscope sees the drill? Over frames where the viewer is cutting (bridge.drill.proxy.cutting):
    on_screen_while_cutting — share with the drill tip on the endoscope's screen: in the field of view and inside the image
        circle of the medium (scope_screen.tip_on_screen), hidden behind bone or not. This is the measure the run criteria
        use (user, 10-07: "드릴이 없으면" = the tip has left the screen, nothing else).
    visible_while_cutting — the stricter share used until 10-07: in view and not hidden (inFov and visible).
    Median |x|, |y| offset from the image centre, median distance.
    Demos (render_cache, 15 fps, the stricter share): 83–100 % while cutting, |x| 0.03–0.11, 21–35 mm."""
    from scope_screen import tip_on_screen
    cut = [((r.get("bridge") or {}).get("drill") or {}, ((r.get("bridge") or {}).get("endoCamera") or {}).get("viewAngle")) for r in rows]
    cut = [(d, va) for d, va in cut if (d.get("proxy") or {}).get("cutting")]
    iv = [d.get("inView") or {} for d, _ in cut]
    ok = [bool(v.get("inFov") and v.get("visible")) for v in iv]
    scr = [tip_on_screen(v, va, optics) for v, (_, va) in zip(iv, cut)]
    xs = [abs(v["x"]) for v in iv if v.get("x") is not None]; ys = [abs(v["y"]) for v in iv if v.get("y") is not None]
    zs = [v["zMm"] for v in iv if v.get("zMm") is not None]
    return {"cutting_frames": len(cut), "visible_while_cutting": round(float(np.mean(ok)), 3) if ok else None,
            "on_screen_while_cutting": round(float(np.mean(scr)), 3) if scr else None,
            "hidden_on_screen_frames": int(sum(s and not o for s, o in zip(scr, ok))), "off_screen_frames": int(sum(not s for s in scr)),
            "optics": optics,
            "x_med": round(float(np.median(xs)), 3) if xs else None, "y_med": round(float(np.median(ys)), 3) if ys else None,
            "z_med_mm": round(float(np.median(zs)), 1) if zs else None}


def render_with_view(key: str, inp, opts: dict) -> tuple[dict, dict]:
    """Removed voxels (cached npz) + the view stats of the same render (cached json next to it)."""
    import io, zipfile
    vf = E.OUT / f"view_{key}.json"
    if (E.OUT / f"iou_cache_{key}.npz").exists() and vf.exists():   # (a view file from before 10-07 16:00 has no screen share: view_recount.py adds it)
        return E.cached(key, None), json.loads(vf.read_text())
    z = E.render(inp, {**opts, "metrics": True})
    vox = E.cached(key, lambda: E.voxels(z)) if not (E.OUT / f"iou_cache_{key}.npz").exists() else E.cached(key, None)
    rows = [json.loads(l) for l in z.read("metrics.jsonl").decode().splitlines() if l.strip()] if "metrics.jsonl" in z.namelist() else []
    st = view_stats(rows, opts.get("optics")); vf.write_text(json.dumps(st))
    return vox, st


def compare(a: np.ndarray, b: np.ndarray, hdr: dict) -> dict:
    j, d, *_ = E.iou(a, b)
    return {"iou": round(j, 3), "dice": round(d, 3), **X.ext_metrics(a, b, hdr),
            "vol_ratio": round(len(a) / len(b), 3) if len(b) else None}


def region(cut: np.ndarray, demos: dict, hdr: dict, k: int) -> dict:
    allv = np.concatenate([d["final"] for d in demos.values()]).astype(np.int64)
    u, n = np.unique(allv, return_counts=True)
    cons = u[n >= k]
    union = X.dilate(X.dilate(u, hdr["dims"]), hdr["dims"])
    c = cut.astype(np.int64); v = hdr.get("voxelMm3", 0.512)
    inside = np.isin(c, cons).sum()
    return {"consensus_k": k, "consensus_mm3": round(len(cons) * v, 1),
            "coverage": round(float(np.isin(cons, c).mean()), 3) if len(cons) else None,
            "precision": round(float(inside / len(c)), 3) if len(c) else None,
            "off_target_mm3": round(float((~np.isin(c, union)).sum() * v), 1)}


def summarize(name: str, cut: dict, demos: dict, hdr: dict, k: int) -> dict:
    per = {rid: compare(cut["final"], d["final"], hdr) for rid, d in demos.items()}
    best = max(per, key=lambda r: per[r]["iou"] if per[r]["iou"] == per[r]["iou"] else -1)
    mean = {m: round(float(np.nanmean([per[r][m] for r in per])), 3) for m in ("iou", "tol1_dice", "assd_mm", "hd95_mm", "vol_ratio")}
    return {"name": name, "mm3": round(len(cut["final"]) * hdr.get("voxelMm3", 0.512), 1), "best_demo": best,
            "best": per[best], "mean": mean, "per_demo": per, "region": region(cut["final"], demos, hdr, k)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy", nargs="?")
    ap.add_argument("--robot", nargs="*", default=[])
    ap.add_argument("--label", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--demos-only", action="store_true")
    ap.add_argument("--optics", choices=("air", "water"), default=None,
                    help="endoscope medium of the policy / robot renders (default: the recordings' flag, air). The cuts do not "
                         "depend on it; the view statistics (drill in view) do")
    a = ap.parse_args()
    ox = {"optics": a.optics} if a.optics else {}
    demos = {rid: demo_cut(rid) for rid in DEMOS}
    hdr = demos[DEMOS[0]]["hdr"]
    if any(d["hdr"].get("sdfSha256") != hdr.get("sdfSha256") for d in demos.values()):
        raise SystemExit("demos on different SDFs")
    out = {"time": time.strftime("%m-%d %H:%M"), "demo_mm3": {r: round(len(d["final"]) * 0.512, 1) for r, d in demos.items()}}
    if a.demos_only or not a.policy:   # leave-one-out: each demo vs the other 7, the same summary as a policy gets
        rows = {}
        for rid in DEMOS:
            others = {r: d for r, d in demos.items() if r != rid}
            rows[rid] = summarize(f"demo {rid}", demos[rid], others, hdr, max(1, round(a.k * 7 / 8)))
        out["demo_loo"] = rows
        label = "demos"
    else:
        views, cuts = {}, {}
        # POLICY_EVAL_ROBOT_ONLY=1: without the step re-simulation of the policy recording (the measured poses cut again in
        # the simulator — a comparison of simulator and robot, 12–14 min for a 2,000-frame run, not used by the run
        # criteria). For a batch of deferred evaluations: the robot cuts first, this row when the render API is free.
        if not os.environ.get("POLICY_EVAL_ROBOT_ONLY"):
            cuts["policy"], views["policy"] = render_with_view("psimfull15_" + Path(a.policy).stem, a.policy, {**E.OPTS, "tick_hz": 15.0, "fps": 15.0, **ox})
        for r in a.robot:
            n = f"robot {Path(r).stem[-4:]}"
            # POLICY_EVAL_PAGE_CLOCK=jump: the bridge redraws once per output frame instead of every rAF (render.py
            # page_clock) — the replay of a robot session takes a fraction of the time. The removed set, its snapshots and
            # the view statistics are the replay's own state (seek), checked identical on a session before it is set
            # (ab_page_clock.py); the cache key stays the session's.
            pc = {"page_clock": os.environ["POLICY_EVAL_PAGE_CLOCK"]} if os.environ.get("POLICY_EVAL_PAGE_CLOCK") else {}
            cuts[n], views[n] = render_with_view("rofull_" + Path(r).stem, r, {**REPLAY, "fps": 5.0, **ox, **pc})
        for c in cuts.values():
            if c["hdr"].get("sdfSha256") != hdr.get("sdfSha256"):
                raise SystemExit("policy/robot cut on a different SDF")
        out["policy_recording"] = a.policy
        out["cuts"] = {n: {**summarize(n, c, demos, hdr, a.k), "view": views.get(n)} for n, c in cuts.items()}
        label = a.label or Path(a.policy).stem[-4:]
    f = E.OUT / f"policy_eval_{label}.json"
    f.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "cuts" and k != "demo_loo"}, ensure_ascii=False))
    for n, s in (out.get("cuts") or out.get("demo_loo")).items():
        print(f"{n:14s} {s['mm3']:7.1f} mm3 | best {s['best_demo']} IoU {s['best']['iou']:.3f} tol1 {s['best']['tol1_dice']:.3f} "
              f"ASSD {s['best']['assd_mm']:.2f} | mean IoU {s['mean']['iou']:.3f} tol1 {s['mean']['tol1_dice']:.3f} "
              f"ASSD {s['mean']['assd_mm']:.2f} HD95 {s['mean']['hd95_mm']:.2f} | cover {s['region']['coverage']} "
              f"prec {s['region']['precision']} off {s['region']['off_target_mm3']} mm3"
              + (f" | drill on screen while cutting {s['view'].get('on_screen_while_cutting')} (seen {s['view']['visible_while_cutting']}; {s['view']['cutting_frames']} fr)" if s.get("view") else ""))


if __name__ == "__main__":
    main()
