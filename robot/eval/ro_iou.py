"""Robot-origin session vs hand: real bone IoU of the two RECORDED cuts (both rendered in replay mode, their own cut rows).

    python ro_iou.py ROBOT_SESSION.jsonl HAND_TAG [--tag TAG]    (HAND_TAG = a p?_ tag of that recording: window + caches)

Also: robot session vs the hand SIMULATION of the window (hsim cache) and the robot session re-simulated (drill_rule r4).
Each comparison: IoU, Dice + metrics_ext (tol1_dice, assd_mm, hd95_mm). Appends ../ro_iou.csv, writes ../ro_<sid>_iou.json,
keeps the robot's recorded voxel set in ../ro_cache_<sid>.npz.
"""
import argparse, csv, hashlib, json, time
from pathlib import Path

import numpy as np

import iou_eval as E
import metrics_ext as X

ap = argparse.ArgumentParser()
ap.add_argument("session"); ap.add_argument("hand_tag"); ap.add_argument("--tag", default="")
ap.add_argument("--no-resim", action="store_true", help="skip re-simulating the robot session")
ap.add_argument("--resim-tick-hz", type=float, default=None,
                help="re-simulate at this engine tick (15 × replay speed = the live page's ticks per RECORDING second)")
a = ap.parse_args()
sess, tag = a.session, a.hand_tag
hand = json.loads((E.OUT / f"{tag}_hand.json").read_text()); src = hand["_source"]
size = E.V.drill_size(src["recording"]); opts = {**E.OPTS, "drill_size": size}
win = [src["recording"], src["from_s"], src["len_s"]]
key = hashlib.sha1(json.dumps([win, opts, sorted({v.get("source", "") for v in (src.get("tools") or {}).values()})]).encode()).hexdigest()[:10]
H = E.cached("hsim_" + key, lambda: E.voxels(E.render(hand, opts)))
rec_key = hashlib.sha1(json.dumps([win, size]).encode()).hexdigest()[:10]
with open(src["recording"]) as fh:
    src_hdr = next(d for d in map(json.loads, fh) if d.get("k") == "hdr")
policy = (src_hdr.get("origin") or {}).get("source") == "policy"
if policy:   # policy2rec output has no cut rows: its "recorded" cut = the viewer's live cut, i.e. simulate at the live 15 Hz
    Hr = E.cached("psim15_" + rec_key, lambda: E.voxels(E.render(src["recording"], {
        **{k: v for k, v in E.OPTS.items()}, "tick_hz": 15.0, "t_start_s": src["from_s"], "t_end_s": src["from_s"] + src["len_s"]})))
else:
    Hr = E.cached("hrec_" + rec_key, lambda: E.voxels(E.render(src["recording"], {
        "views": [], "status": False, "fps": 10.0, "removed_voxels": True, "removed_every_s": 3,
        "t_start_s": src["from_s"], "t_end_s": src["from_s"] + src["len_s"]})))
demo = None   # rehearsal: the demo hand recording the policy output came from, same window, its own recorded cut
demo_sid = ((src_hdr.get("origin") or {}).get("policy") or {}).get("demo_session")
if policy and demo_sid:
    import glob as _g0
    dp = [f for f in _g0.glob(str(Path.home() / f"il-robot/viewer-data/sessions/*/{demo_sid}.jsonl")) if "/robot/" not in f]
    if dp:
        dkey = hashlib.sha1(json.dumps([[dp[0], src["from_s"], src["len_s"]], size]).encode()).hexdigest()[:10]
        demo = E.cached("hrec_" + dkey, lambda: E.voxels(E.render(dp[0], {
            "views": [], "status": False, "fps": 10.0, "removed_voxels": True, "removed_every_s": 3,
            "t_start_s": src["from_s"], "t_end_s": src["from_s"] + src["len_s"]})))
sid = Path(sess).stem
Rr = E.cached("ro_" + sid, lambda: E.voxels(E.render(sess, {"views": [], "status": False, "fps": 10.0,
                                                             "removed_voxels": True, "removed_every_s": 3})))
ropts = {k: v for k, v in opts.items() if k != "drill_size"}
if a.resim_tick_hz:
    ropts["tick_hz"] = a.resim_tick_hz
Rs = None if a.no_resim else E.cached(f"rosim{a.resim_tick_hz or 60:g}_" + sid, lambda: E.voxels(E.render(sess, ropts)))
out_tick = a.resim_tick_hz or 60.0
sets = [x for x in (H, Rr, Rs) if x is not None]
if not all(x["hdr"].get("sdfSha256") == Hr["hdr"].get("sdfSha256") and x["hdr"].get("gridSha256") == Hr["hdr"].get("gridSha256") for x in sets):
    raise SystemExit("different SDF/grid — not comparable")
v = Hr["hdr"].get("voxelMm3", 0.512)
out = {"time": time.strftime("%m-%d %H:%M"), "tag": a.tag, "session": Path(sess).name, "hand_tag": tag,
       "recording": Path(src["recording"]).name, "window": [src["from_s"], src["len_s"]], "drill": size,
       "hand_rec_mm3": round(len(Hr["final"]) * v, 1), "hand_sim_mm3": round(len(H["final"]) * v, 1),
       "robot_rec_mm3": round(len(Rr["final"]) * v, 1), "robot_sim_mm3": round(len(Rs["final"]) * v, 1) if Rs else None,
       "resim_tick_hz": None if a.no_resim else out_tick}
pairs = [("rr_hr", Rr, Hr), ("rr_hs", Rr, H), ("hs_hr", H, Hr)] + ([("rs_hr", Rs, Hr), ("rs_hs", Rs, H), ("rr_rs", Rr, Rs)] if Rs else [])
if demo is not None:   # rehearsal round trip: robot live cut vs the demo's recorded cut, and the policy's live cut vs the demo
    pairs += [("rr_demo", Rr, demo), ("hr_demo", Hr, demo)]
    out["demo_rec_mm3"] = round(len(demo["final"]) * v, 1)
out["policy_source"] = policy
for name, p, q in pairs:
    j, d, *_ = E.iou(p["final"], q["final"])
    out[name] = {"iou": round(j, 3), "dice": round(d, 3), **X.ext_metrics(p["final"], q["final"], Hr["hdr"])}
lost = {"endo": [0, 0], "drill": [0, 0]}   # session QC: marker-lost share per tool (a starved tracker shows here)
with open(sess) as fh:
    for line in fh:
        if '"k":"f"' in line[:12]:
            r = json.loads(line); t = r.get("tool") or "endo"
            if t in lost:
                lost[t][0] += 1; lost[t][1] += bool(r.get("lost"))
out["lost_frac"] = {t: round(b / a, 3) if a else None for t, (a, b) in lost.items()}
try:   # drill tip motion while the trigger is on, in recording time (tip_motion.py): robot vs the hand window
    import glob as _g, tip_motion as TM
    st_ = _g.glob(str(Path.home() / f"il-robot/runs/robot_origin/*_replay_{a.tag}/settings.json"))
    speed = float(json.loads(Path(st_[-1]).read_text())["speed"]) if st_ else 0.33
    out["tip_robot"] = TM.stats(*TM.load(sess, None, speed))
    out["tip_hand"] = TM.stats(*TM.load(src["recording"], (src["from_s"], src["from_s"] + src["len_s"])))
except Exception as e:   # the metrics above stand on their own
    out["tip_error"] = str(e)[:200]
print(json.dumps(out, ensure_ascii=False))
(E.OUT / f"ro_{sid}_iou.json").write_text(json.dumps(out, ensure_ascii=False))
row = {k: out[k] for k in ("time", "tag", "session", "recording", "hand_rec_mm3", "hand_sim_mm3", "robot_rec_mm3", "robot_sim_mm3")}
row["lost_endo"], row["lost_drill"] = out["lost_frac"]["endo"], out["lost_frac"]["drill"]
for who in ("robot", "hand"):
    for m in ("path_raw_mm", "path_smooth_mm", "jitter_rms_mm"):
        row[f"tip_{who}_{m}"] = (out.get(f"tip_{who}") or {}).get(m)
row["resim_tick_hz"] = out.get("resim_tick_hz")
for name in ("rr_hr", "rr_hs", "hs_hr", "rs_hr", "rs_hs", "rr_rs", "rr_demo", "hr_demo"):
    for m in ("iou", "tol1_dice", "assd_mm", "hd95_mm"):
        row[f"{name}_{m}"] = (out.get(name) or {}).get(m)
f = E.OUT / "ro_iou.csv"; new = not f.exists()
with open(f, "a", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
