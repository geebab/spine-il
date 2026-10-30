"""deep_1018 A (after 2d, before any further robot run): what the policy saw and did when its progress output jumped.
For every answer of the L4 H90g runs (R_H90g_pf_* · R_H90g_e965_*) in the drilling phase: the progress value (highest of
the rows the window executes), the page image the answer was made from (the latest saved look ≤ window frame − 16: lead
16), the tool state measured at that look and the volume removed so far. Answers are split into
  spike  ≥ 0.85 with the previous answer < 0.8      high  ≥ 0.85 after ≥ 0.8      low  < 0.8
and the groups are compared: trigger, drill tip speed and distance to its skin entry, drill on screen, removal rate over
the last 3 s, image brightness / white (bone) share / change from the previous look, removed volume. Contact sheets of
spikes (previous look · spike look · next look) and of random low looks go to OUT/sheets/.

  python spike_diag.py OUT_DIR"""
import glob, json, os, re, sys
import numpy as np
from PIL import Image, ImageDraw

LV = os.path.expanduser("~/il-robot/runs/act_smoke/live"); OUT = sys.argv[1]; os.makedirs(f"{OUT}/sheets", exist_ok=True)
FPS = 15; LEAD = 16
rng = np.random.default_rng(1019)


def img_feats(p):
    a = np.asarray(Image.open(p).convert("RGB"), float)
    g = a.mean(2)
    return a, {"bright": float(g.mean()), "white": float((g > 200).mean()), "dark": float((g < 40).mean()),
               "red": float(((a[..., 0] > a[..., 1] + 40) & (a[..., 0] > 90)).mean())}


rows = []; sheets = {"spike": [], "low": []}
runs = sorted(r for r in glob.glob(f"{LV}/R_H90g_pf_*") + glob.glob(f"{LV}/R_H90g_e965_*") if os.path.isdir(r))
for r in runs:
    name = os.path.basename(r); d = json.load(open(f"{r}/live.json")); m = d["meta"]
    a = np.asarray(d["frames"], float); vol = np.asarray(m["removed_mm3"], float); ph = np.asarray(m["phases_seen"])
    vis = m["drill_in_view"]; portal = np.asarray(m["portals_mm"]["drill"], float)
    saved = np.array(sorted(int(f[:-4]) for f in os.listdir(f"{r}/frames") if f.endswith(".png")))
    g = re.search(r"_(ncontra|nipsi|contra|ipsi)\d", name).group(1)
    prev_p, prev_img = None, None
    dl = [e for e in m.get("done_log") or [] if e[1]]
    for k, (t, vals, jd) in enumerate(dl):
        p = float(max(vals)); f = saved[saved <= t - LEAD]
        if not len(f):
            prev_p = p; continue
        f = int(f[-1]); fn = min(f, len(a) - 1)
        if fn >= len(ph) or ph[fn] != 4:
            prev_p = p; continue
        im, F = img_feats(f"{r}/frames/{f:05d}.png")
        w0 = max(0, fn - 8); sp = float(np.linalg.norm(a[fn, 7:10] - a[w0, 7:10]) / max(1, fn - w0) * FPS)
        w3 = max(0, fn - 3 * FPS); rate = float((vol[fn] - vol[w3]) / max(1e-9, (fn - w3) / FPS))
        vv = vis[fn] if fn < len(vis) else None
        row = {"run": name, "group": g, "t": t, "img": f, "p": p, "prev": prev_p, "ended": jd is not None, "vol": float(vol[fn]),
               "trig": float(a[max(0, fn - 8):fn + 1, 14].mean()), "speed_mm_s": sp, "from_entry_mm": float(np.linalg.norm(a[fn, 7:10] - portal)),
               "rate_mm3_s": rate, "on_screen": None if not vv or vv[0] is None else int(vv[0]),
               "img_change": None if prev_img is None else float(np.abs(im - prev_img).mean()), **F}
        row["kind"] = "low" if p < 0.85 else ("spike" if prev_p is not None and prev_p < 0.8 else "high")
        if p >= 0.8 and p < 0.85:
            row["kind"] = "mid"
        rows.append(row); prev_p, prev_img = p, im
json.dump(rows, open(f"{OUT}/spike_diag.json", "w"), indent=1)

keys = ["p", "vol", "trig", "speed_mm_s", "from_entry_mm", "rate_mm3_s", "on_screen", "img_change", "bright", "white", "dark", "red"]
L = [f"A — progress spikes, H90g L4 runs (pf · e965): {len(set(r['run'] for r in rows))} runs, {len(rows)} answers in the drilling phase", "",
     "medians (IQR) per kind; spike = ≥ 0.85 after < 0.8, high = ≥ 0.85 after ≥ 0.8, low < 0.8"]
for kind in ("low", "mid", "spike", "high"):
    rs = [r for r in rows if r["kind"] == kind]
    if not rs:
        continue
    cells = []
    for k_ in keys:
        v = np.array([r[k_] for r in rs if r[k_] is not None], float)
        cells.append(f"{k_} {np.median(v):.2f} ({np.percentile(v, 25):.2f}–{np.percentile(v, 75):.2f})" if len(v) else f"{k_} —")
    L.append(f"  {kind:<5} n {len(rs):4}: " + " · ".join(cells))

# within-run: does the progress output follow the removed volume?
L += ["", "per run: Spearman(progress, removed volume) over its drilling answers · share of answers ≥ 0.85 · spikes · the ending answer"]
from math import erfc, sqrt


def rank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    for v in np.unique(x):                    # ties: mean rank
        k = x == v; r[k] = r[k].mean()
    return r


def spearman(x, y):
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def mwu_p(x, y):
    """Mann-Whitney U, two-sided, normal approximation (no tie correction)"""
    n1, n2 = len(x), len(y); r = rank(list(x) + list(y)); u = r[:n1].sum() + n1 - n1 * (n1 + 1) / 2
    z = (u - n1 * n2 / 2) / sqrt(n1 * n2 * (n1 + n2 + 1) / 12); return erfc(abs(z) / sqrt(2))
for name in sorted(set(r["run"] for r in rows)):
    rs = [r for r in rows if r["run"] == name]
    rho = spearman([r["p"] for r in rs], [r["vol"] for r in rs]) if len(rs) > 3 else float("nan")
    end = next((r for r in rs if r["ended"]), None)
    L.append(f"  {name:<32} n {len(rs):3}  rho {rho:+.2f}  ≥0.85 {np.mean([r['p'] >= 0.85 for r in rs]):.2f}  spikes {sum(r['kind'] == 'spike' for r in rs):2}"
             + (f"  end: {end['kind']} p {end['p']:.2f} prev {end['prev']:.2f} trig {end['trig']:.2f} speed {end['speed_mm_s']:.1f} rate {end['rate_mm3_s']:.1f}" if end else ""))

L += ["", "spike vs low (Mann-Whitney, two-sided; within the same runs)"]
for k_ in keys[1:]:
    s = [r[k_] for r in rows if r["kind"] == "spike" and r[k_] is not None]; lo = [r[k_] for r in rows if r["kind"] == "low" and r[k_] is not None]
    if len(s) > 2 and len(lo) > 2:
        L.append(f"  {k_:<14} spike {np.median(s):8.2f}  low {np.median(lo):8.2f}  p {mwu_p(s, lo):.3g}")

# the answer right after a spike: does it come back down?
nxt = []
for i, r in enumerate(rows[:-1]):
    if r["kind"] == "spike" and not r["ended"] and rows[i + 1]["run"] == r["run"]:
        nxt.append(rows[i + 1]["p"])
L += ["", f"answer after a spike that did not end the run: n {len(nxt)}, median {np.median(nxt):.2f}, back below 0.8 in {np.mean(np.array(nxt) < 0.8):.0%}" if nxt else ""]


def sheet(sel, fn, title):
    W = 256; out = Image.new("RGB", (W * 3, (W + 18) * len(sel)), "white"); dr = ImageDraw.Draw(out)
    for j, r in enumerate(sel):
        rd = f"{LV}/{r['run']}"; saved = sorted(int(f[:-4]) for f in os.listdir(f"{rd}/frames") if f.endswith(".png"))
        i = saved.index(r["img"])
        for c, ii in enumerate((i - 1, i, i + 1)):
            if 0 <= ii < len(saved):
                out.paste(Image.open(f"{rd}/frames/{saved[ii]:05d}.png").convert("RGB"), (c * W, j * (W + 18) + 18))
        dr.text((4, j * (W + 18) + 3), f"{r['run'][7:30]} t{r['t']} p {r['prev']:.2f}->{r['p']:.2f} vol {r['vol']:.0f} trig {r['trig']:.1f} v {r['speed_mm_s']:.1f} {'END' if r['ended'] else ''}", fill="black")
    out.save(f"{OUT}/sheets/{fn}")


sp = [r for r in rows if r["kind"] == "spike"]; lo = [r for r in rows if r["kind"] == "low"]
for n0 in range(0, min(len(sp), 24), 6):
    sheet(sp[n0:n0 + 6], f"spikes_{n0 // 6}.png", "spikes")
sheet([lo[i] for i in rng.choice(len(lo), 6, replace=False)], "low_random.png", "low")
open(f"{OUT}/spike_diag.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
