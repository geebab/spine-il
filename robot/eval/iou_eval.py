"""Real bone IoU / Dice between the hand's and the robot's cut, rev.4 (2026-09-28).

    python iou_eval.py TAG      # ~/il-robot/runs/jitter_iter/TAG_hand.json + TAG_robot.json (iter_eval, filtered pose)

Both trajectories are rendered by the same render API build with drill_rule r4 (simulate), removed_voxels and 3 s
snapshots, drill size from the recording's cut rows. IoU = |H ∩ R| / |H ∪ R| on the removed-bone voxel sets (compared
only when the two headers' sdfSha256 and gridSha256 agree). The hand's render is cached per window. Once per window
the recording itself is also rendered in replay mode (its own rev.4 cut rows) → hand-sim vs recorded IoU = how well
the simulation reproduces the real cut (the ceiling for robot vs hand). Appends ~/il-robot/runs/jitter_iter/iou.csv.
"""
import csv, hashlib, io, json, sys, time, zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import vol_eval as V   # call(), drill_size(), token header

OUT = Path.home() / "il-robot/runs/jitter_iter"
OPTS = {"fps": 10.0, "views": [], "status": False, "ae": "on", "flags_override": {"rawabs": False},
        "drill_rule": "r4", "removed_voxels": True, "removed_every_s": 3}


def render(inp, opts: dict) -> zipfile.ZipFile:
    j = json.loads(V.call("/v1/jobs", {"input": V.job_input(inp), "options": opts}, 600))
    while True:
        s = json.loads(V.call(f"/v1/jobs/{j['id']}"))
        if s["status"] in ("done", "failed", "error"):
            break
        time.sleep(3)
    if s["status"] != "done":
        raise RuntimeError(f"render job {j['id']}: {s['status']} {s.get('error')}")
    return zipfile.ZipFile(io.BytesIO(V.call(f"/v1/jobs/{j['id']}/result.zip", t=600)))


def voxels(z: zipfile.ZipFile) -> dict:
    h = json.loads(z.read("removed/header.json"))
    snaps = {}
    if "removed/index.json" in z.namelist():
        for e in json.loads(z.read("removed/index.json")).get("rows", []):
            f = e.get("file")
            if f and f in z.namelist():
                snaps[round(float(e.get("t_ms", 0)) / 1000.0, 1)] = np.frombuffer(z.read(f), "<u4")
    return {"hdr": h, "final": np.frombuffer(z.read("removed/final.bin"), "<u4"), "snaps": snaps}


def iou(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int, int, int]:
    i = len(np.intersect1d(a, b, assume_unique=True)); u = len(a) + len(b) - i
    return (i / u if u else float("nan")), (2 * i / (len(a) + len(b)) if (len(a) + len(b)) else float("nan")), i, len(a), len(b)


def cached(key: str, make) -> dict:
    f = OUT / f"iou_cache_{key}.npz"
    if f.exists():
        d = np.load(f, allow_pickle=True)
        return {"hdr": json.loads(str(d["hdr"])), "final": d["final"], "snaps": d["snaps"].item()}
    v = make()
    np.savez_compressed(f, hdr=json.dumps(v["hdr"]), final=v["final"], snaps=np.array(v["snaps"], dtype=object))
    return v


def main():
    tag = sys.argv[1]
    hand = json.loads((OUT / f"{tag}_hand.json").read_text()); robot = json.loads((OUT / f"{tag}_robot.json").read_text())
    src = hand["_source"]; size = V.drill_size(src["recording"])
    opts = {**OPTS, "drill_size": size}
    win = [src["recording"], src["from_s"], src["len_s"]]
    key = hashlib.sha1(json.dumps([win, opts, sorted({v.get("source", "") for v in (src.get("tools") or {}).values()})])
                       .encode()).hexdigest()[:10]
    H = cached("hsim_" + key, lambda: voxels(render(hand, opts)))
    Rv = voxels(render(robot, opts))
    same = H["hdr"].get("sdfSha256") == Rv["hdr"].get("sdfSha256") and H["hdr"].get("gridSha256") == Rv["hdr"].get("gridSha256")
    if not same:
        raise SystemExit(f"[iou] {tag}: different SDF/grid — not comparable")
    rec_key = hashlib.sha1(json.dumps([win, size]).encode()).hexdigest()[:10]
    Hr = cached("hrec_" + rec_key, lambda: voxels(render(src["recording"], {
        "views": [], "status": False, "fps": 10.0, "removed_voxels": True, "removed_every_s": 3,
        "t_start_s": src["from_s"], "t_end_s": src["from_s"] + src["len_s"]})))
    j_rh, d_rh, i_rh, n_r, n_h = iou(Rv["final"], H["final"])
    rec_start = Hr["hdr"].get("start", {}).get("count", 0)
    j_hh, d_hh, *_ = iou(H["final"], Hr["final"])
    vox = H["hdr"].get("voxelMm3", 0.512)
    per = {}
    for t in sorted(set(H["snaps"]) & set(Rv["snaps"])):
        per[t] = round(iou(Rv["snaps"][t], H["snaps"][t])[0], 3)
    row = {"time": time.strftime("%m-%d %H:%M"), "tag": tag, "recording": Path(src["recording"]).name,
           "from_s": src["from_s"], "len_s": src["len_s"], "drill": size, "rule": Rv["hdr"].get("drillRule", {}).get("rule"),
           "hand_mm3": round(n_h * vox, 1), "robot_mm3": round(n_r * vox, 1), "ratio": round(n_r / n_h, 3) if n_h else None,
           "iou": round(j_rh, 3), "dice": round(d_rh, 3), "inter_mm3": round(i_rh * vox, 1),
           "handsim_vs_rec_iou": round(j_hh, 3), "rec_mm3": round(len(Hr["final"]) * vox, 1), "rec_start_vox": rec_start,
           "viewer": Rv["hdr"].get("viewer", {}).get("gitSha")}
    (OUT / f"{tag}_iou.json").write_text(json.dumps({**row, "iou_by_t": per}))
    f = OUT / "iou.csv"; new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[iou] {tag}: IoU {row['iou']} Dice {row['dice']} | hand {row['hand_mm3']} robot {row['robot_mm3']} mm³ "
          f"(ratio {row['ratio']}) | hand-sim vs recorded IoU {row['handsim_vs_rec_iou']}", flush=True)


if __name__ == "__main__":
    main()
