"""deep_1018 stage 2 (PREREG.md v1), the part that needs neither the robot nor the VM: where a contra run's depth is lost.
2d ending   — reach as a function of the removed volume (3 s snapshots of the voxel caches): the level's lang2-contra demos at
              the run's final volume vs at their own end. Demo-like path that stops early → ending; short of the demos at the
              same volume → path (model / execution).
2b execution — commanded vs measured drill-tip reach while the trigger is on (live.json frames_cmd / frames, column 7), clamped
              frames, RCM degrees; the policy's progress output at the end (done_log).
Also which demo class a run resembles at the same volume (contra / unspecified / ipsi).

  python stage2.py OUT_DIR    (after stage1.py: reads OUT_DIR/stage1.json)"""
import json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dataset import instructions as INS          # noqa: E402

H = os.path.expanduser("~"); C = f"{H}/il-robot/runs/jitter_iter"; LV = f"{H}/il-robot/runs/act_smoke/live"
MID = INS.MIDLINE_X_MM; OUT = sys.argv[1]
S1 = json.load(open(f"{OUT}/stage1.json"))
HD = None


def curve(f, lv):
    """[(mm3, reach)] over the snapshots, then the final"""
    global HD
    z = np.load(f, allow_pickle=True); h = json.loads(str(z["hdr"])); HD = h; nx = h["dims"][0]
    sn = z["snaps"].item(); out = []
    for t in sorted(sn, key=float):
        idx = np.asarray(sn[t], np.int64)
        if len(idx) >= 20:
            x = h["origin"][0] + (idx % nx) * h["spacing"][0]
            out.append((len(idx) * h["voxelMm3"], float(MID[lv] - np.percentile(x, 5))))
    idx = z["final"].astype(np.int64); x = h["origin"][0] + (idx % nx) * h["spacing"][0]
    out.append((len(idx) * h["voxelMm3"], float(MID[lv] - np.percentile(x, 5))))
    return np.array(out)


def at(cv, v):
    """reach of a curve at removed volume v (None beyond its end)"""
    if v > cv[-1, 0]:
        return None
    return float(np.interp(v, cv[:, 0], cv[:, 1]))


def group(name):
    m = re.search(r"_(L[35])?(ncontra|nipsi|contra|ipsi|neutral)\d", name)
    return (m.group(1) or "L4", m.group(2)) if m else (None, None)


demo_c = {}
for d in S1["demos"]:
    f = f"{C}/iou_cache_hfull_{d['sid']}.npz"
    demo_c[d["sid"]] = (d["level"], d["label"], curve(f, d["level"]))

rows = []
for w in S1["runs"]:
    lv, g = group(w["run"])
    if g is None:
        continue
    cv = curve(f"{C}/iou_cache_rofull_{w['sid']}.npz", lv); V = cv[-1, 0]
    row = {"run": w["run"], "model": w["model"], "level": lv, "group": g, "end": w["end"], "reach": cv[-1, 1], "mm3": V}
    for lab in ("contralateral", "unspecified", "ipsilateral"):
        ds = [c for (l_, lb, c) in demo_c.values() if l_ == lv and lb == lab]
        vals = [at(c, V) for c in ds]; vals = [v for v in vals if v is not None]
        row[f"demo_{lab[:6]}_at_V"] = float(np.median(vals)) if len(vals) >= 3 else None
        row[f"demo_{lab[:6]}_at_V_n"] = len(vals)
        row[f"demo_{lab[:6]}_final"] = float(np.median([c[-1, 1] for c in ds])) if ds else None
    d = json.load(open(f"{LV}/{w['run']}/live.json")); m = d["meta"]
    a = np.asarray(d["frames"], float); b = np.asarray(d["frames_cmd"], float); on = a[:, 14] > 0.5
    if on.any():
        row["tip_meas"] = float(MID[lv] - a[on, 7].min()); row["tip_cmd"] = float(MID[lv] - b[on, 7].min())
        row["tip_meas_p2"] = float(MID[lv] - np.percentile(a[on, 7], 2)); row["tip_cmd_p2"] = float(MID[lv] - np.percentile(b[on, 7], 2))
    row["clamped"] = m.get("clamped_frames"); row["rcm_deg"] = m.get("rcm_deg_total"); row["drill_s"] = m.get("drill_s")
    dl = m.get("done_log") or []
    row["progress_end"] = float(max(dl[-1][1])) if dl and dl[-1][1] else None
    row["progress_max"] = float(max(max(e[1]) for e in dl if e[1])) if dl else None
    cmf = (m.get("continuous") or {}).get("measured_from_command_mm") or {}
    row["meas_from_cmd_median"] = cmf.get("median")
    rows.append(row)

json.dump(rows, open(f"{OUT}/stage2.json", "w"), indent=1)

# ---- table per model × level × group (contra-type groups first) ----
L = ["2d / 2b per run: reach (voxel) · volume · contra demos at the same volume · their end · unspecified / ipsi demos at that volume"
     " · tip reach commanded / measured (max) · clamped · progress at the end", ""]
key = lambda r: (r["model"], r["level"], r["group"], r["run"])
for r in sorted(rows, key=key):
    f = lambda v: "  —  " if v is None else f"{v:+5.1f}"
    L.append(f"  {r['run']:<32} {r['end'] or '':<7} reach {r['reach']:+5.1f} @ {r['mm3']:5.0f} mm3 | contra demos @V {f(r['demo_contra_at_V'])} (n {r['demo_contra_at_V_n']:2}) end {f(r['demo_contra_final'])}"
             f" | unspec @V {f(r['demo_unspec_at_V'])} ipsi @V {f(r['demo_ipsila_at_V'])} | tip cmd {f(r.get('tip_cmd'))} meas {f(r.get('tip_meas'))} | clamp {r['clamped']} | prog end {r['progress_end']}")

L += ["", "summary per model × level × group (means): run reach | contra demos at the run's volume | contra demos' end | "
      "→ ending loss = demo end − demo @V, path loss = demo @V − run | commanded − measured tip"]
summ = {}
for r in rows:
    summ.setdefault((r["model"], r["level"], r["group"]), []).append(r)
for k in sorted(summ):
    rs = summ[k]
    ok = [r for r in rs if r["demo_contra_at_V"] is not None]
    rr = np.mean([r["reach"] for r in rs]); vv = np.mean([r["mm3"] for r in rs])
    if ok:
        dav = np.mean([r["demo_contra_at_V"] for r in ok]); dend = np.mean([r["demo_contra_final"] for r in ok]); rok = np.mean([r["reach"] for r in ok])
        ce = np.mean([r["tip_cmd"] - r["tip_meas"] for r in rs if "tip_cmd" in r])
        L.append(f"  {k[0]} {k[1]} {k[2]:<8} n {len(rs)}  reach {rr:+5.1f} @ {vv:5.0f} mm3 | contra demos @V {dav:+5.1f} (runs with a match {len(ok)}) end {dend:+5.1f}"
                 f" → ending loss {dend - dav:+4.1f} · path loss {dav - rok:+4.1f} mm | cmd − meas tip {ce:+4.1f} mm")
    else:
        L.append(f"  {k[0]} {k[1]} {k[2]:<8} n {len(rs)}  reach {rr:+5.1f} @ {vv:5.0f} mm3 | no contra demo reaches this volume")

# demo curves: volume at which contra demos first pass +5 mm
L += ["", "contra demos: removed volume when the reach first passes 0 / +5 mm, and at the end (median, per level)"]
for lv in MID:
    cs = [c for (l_, lb, c) in demo_c.values() if l_ == lv and lb == "contralateral"]
    def first(c, thr):
        k = np.nonzero(c[:, 1] > thr)[0]; return c[k[0], 0] if len(k) else np.nan
    v0 = [first(c, 0) for c in cs]; v5 = [first(c, 5) for c in cs]; ve = [c[-1, 0] for c in cs]
    L.append(f"  {lv}: n {len(cs)}  past 0 mm at {np.nanmedian(v0):5.0f} mm3 · past +5 mm at {np.nanmedian(v5):5.0f} mm3 (IQR {np.nanpercentile(v5, 25):.0f}–{np.nanpercentile(v5, 75):.0f}) · end {np.median(ve):5.0f} mm3"
             f"  (share of the end volume at +5 mm: {np.nanmedian(np.array(v5) / np.array(ve)):.2f})")
open(f"{OUT}/stage2.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
