"""Per-recording hand windows: start vector (drill tip - endo tip, camera), start orientations, common time axis."""
import glob, json, pickle, numpy as np
from common import *
REC = {"3868": 17, "91cb": 15, "bef5": 17, "d340": 27, "bad1": 12, "0338": 18, "919d": 10, "6639": 14}
_WIN = P / "windows.json"   # new recordings (new_recording.py): {rid: {"file": path, "from_s": s}}
if _WIN.exists():
    REC.update({k: v["from_s"] for k, v in json.loads(_WIN.read_text()).items()})
LEN = {k: float(v.get("len_s", 30.0)) for k, v in (json.loads(_WIN.read_text()).items() if _WIN.exists() else [])}
import hand_calib   # robot/ is on sys.path (common.py)
CAL = H / "il-robot/calib/tracker_models/hand_20260928"
CALF = {"endo": CAL / "tool_calib.json", "drill": CAL / "tool_calib_drill.json"}   # the 09-28 demos' (a recording of unknown calibration)

def calib_of(path):
    """The hand tool calibration the recording names in its header — the 10-06 demos' drill tip is 5.1 mm from the 09-28 one."""
    try:
        return hand_calib.for_recording(path)
    except LookupError as e:
        print(f"[recs] {Path(path).name}: {e} — using the 09-28 calibration", flush=True)
        return CALF

def tip_at(rp, t):
    R, p, _ = rec.sample_pose(rp, t, "linear", 0.25)
    return R, R @ rp["tip_local_m"] + p

def load(rid):
    hits = [f for pat in (f"*/*_trk_{rid}.jsonl", f"policy/*/*_trk_{rid}.jsonl")
            for f in glob.glob(str(H / "il-robot/viewer-data/sessions" / pat)) if "/robot/" not in f]
    path = sorted(hits)[0]
    t0 = REC[rid]
    calf = calib_of(path)
    # a recording of another rig, registered to this one (fit_registration.py → registration_<rid>.json): the loader's absolute
    # mode moves every pose into this rig's camera through the scene — the search then asks the arms for the hand's own pose
    T_now = None
    if (P / f"registration_{rid}.json").exists():
        g = json.loads((P / f"registration_{rid}.json").read_text()); T_now = {"R": g["R"], "t": g["t"]}
        print(f"[recs] {rid}: registration fitted to the arms of {g.get('arms_of')} ({(g.get('fit') or {}).get('turn_deg')}°)", flush=True)
    rp = {tool: rec.load_recording(path, tool, calib_path=str(calf[tool]), t_start_s=t0, t_end_s=t0 + LEN.get(rid, 30.0), T_now=T_now) for tool in ("endo", "drill")}
    # common axis: drill's own t (0 = drill's first sample); endo sampled at drill time + (drill offset - endo offset)
    sh = {"drill": 0.0, "endo": rp["drill"]["t_offset_s"] - rp["endo"]["t_offset_s"]}
    ts = rp["drill"]["t"]
    R0, P0 = {}, {}
    for tool in rp:
        R0[tool], P0[tool] = tip_at(rp[tool], 0.0 + sh[tool])
    return dict(path=path, rp=rp, sh=sh, T=float(ts[-1]), R0=R0, P0=P0, vec=P0["drill"] - P0["endo"])

if __name__ == "__main__":
    out = {}
    for rid in REC:
        r = load(rid)
        disp = {}
        for tool in ("endo", "drill"):
            tt = np.arange(0, r["T"], 0.1)
            d = np.array([np.linalg.norm(tip_at(r["rp"][tool], t + r["sh"][tool])[1] - r["P0"][tool]) for t in tt]) * 1000
            disp[tool] = (round(d.max(), 1), round(np.percentile(d, 90), 1))
        sepv = []
        for t in np.arange(0, r["T"], 0.1):
            sepv.append(np.linalg.norm(tip_at(r["rp"]["drill"], t)[1] - tip_at(r["rp"]["endo"], t + r["sh"]["endo"])[1]) * 1000)
        print(rid, "vec", (r["vec"] * 1000).round(1), "|v|", round(np.linalg.norm(r["vec"]) * 1000, 1), "T", round(r["T"], 1),
              "sh_endo", round(r["sh"]["endo"], 3), "disp max/p90", disp, "sep min/med/max", np.round([min(sepv), np.median(sepv), max(sepv)], 1),
              "n", {k: len(v["t"]) for k, v in r["rp"].items()}, flush=True)
        out[rid] = dict(vec=r["vec"], R0=r["R0"], P0=r["P0"], sh=r["sh"], T=r["T"])
    pickle.dump(out, open(P / "recs.pkl", "wb"))
