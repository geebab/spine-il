"""Bone volume the FE viewer removes with the arms' motion vs with the hand's, same render settings.

    python vol_eval.py TAG            # uses ~/il-robot/runs/jitter_iter/TAG_hand.json and TAG_robot.json

Render API simulate, 30 fps, no image capture (views []), rig look (ae on, rawabs off). The authored-trajectory
render feeds the poses in one batch per captured frame, so the volume depends on the frame rate: 10 fps gave
252 mm³, 30 fps 286, 60 fps 294 for the same hand (1449 0–20 s) — 30 fps is within 3 % of 60 and runs ~3x faster
than real time. The hand's volume is cached per window (it does not change between iterations).
Appends to ~/il-robot/runs/jitter_iter/volume.csv.
"""
import csv, hashlib, io, json, os, sys, time, urllib.request, zipfile
from pathlib import Path

API = os.environ.get("RENDER_API_URL", "http://127.0.0.1:18780")   # another service: ube-render-b :18781, a Mac node's tunnel
OUT = Path.home() / "il-robot/runs/jitter_iter"
OPTS = {"fps": 10.0, "views": [], "status": False, "ae": "on", "flags_override": {"rawabs": False}}  # fps-independent with bridge 130cff3 (10 = 30 fps: 128 mm³ both) — 10 is the fastest
RENDERER = "bridge 130cff3"   # per-message clock (fps-independent) + drill_size option


RADIUS_TO_SIZE = {1.5: "small", 2.0: "medium", 4.0: "large"}


def drill_size(recording: str) -> str:
    """The drill the recording actually cut with: the radius of its cut rows (a size changed mid-session — 5646:
    header large, every cut row r 2 mm — is only in the rows), else hdr.drill.sizeKey."""
    hdr_key, radii = None, {}
    for line in open(recording):
        if hdr_key is None and ('"k":"hdr"' in line[:40] or '"k": "hdr"' in line[:40]):
            hdr_key = (json.loads(line).get("drill") or {}).get("sizeKey")
        elif '"k":"cut"' in line[:30] or '"k": "cut"' in line[:30]:
            r = json.loads(line).get("rMm")
            if r is not None:
                radii[float(r)] = radii.get(float(r), 0) + 1
    if radii:
        return RADIUS_TO_SIZE.get(max(radii, key=radii.get), hdr_key or "large")
    return hdr_key or "large"


def _auth() -> dict:
    """Bearer token of the hardened render API (bridge e789b2e): RENDER_API_TOKEN, else the server's token file."""
    tok = os.environ.get("RENDER_API_TOKEN") or ""
    f = Path.home() / ".config/spinearm-render-api/token"
    if not tok and f.exists():
        tok = f.read_text().strip()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def job_input(src):
    """A job's input for the service in API: the path as it is (the service on this machine reads the file itself), or —
    RENDER_API_INLINE=1, a service on another machine such as a Mac render node behind an SSH tunnel — the file's text."""
    if os.environ.get("RENDER_API_INLINE") == "1" and isinstance(src, (str, Path)) and "\n" not in str(src) and Path(src).is_file():
        return Path(src).read_text()
    return src


def call(p, b=None, t=120):
    r = urllib.request.Request(API + p, data=None if b is None else json.dumps(b).encode(),
                               headers={"Content-Type": "application/json", **_auth()})
    return urllib.request.urlopen(r, timeout=t).read()


def volume(traj: dict, size: str = "large") -> tuple[float, list]:
    j = json.loads(call("/v1/jobs", {"input": traj, "options": {**OPTS, "drill_size": size}}))
    while True:
        s = json.loads(call(f"/v1/jobs/{j['id']}"))
        if s["status"] in ("done", "failed", "error"):
            break
        time.sleep(3)
    if s["status"] != "done":
        raise RuntimeError(f"render job {j['id']}: {s['status']} {s.get('error')}")
    z = zipfile.ZipFile(io.BytesIO(call(f"/v1/jobs/{j['id']}/result.zip", t=600)))
    m = [json.loads(x) for x in z.read("metrics.jsonl").decode().splitlines() if x.strip()]
    curve = [(round(f["t_ms"] / 1000, 2), f["bridge"].get("removedMm3")) for f in m[::30]]
    return float(m[-1]["bridge"].get("removedMm3") or 0.0), curve


def main():
    tag = sys.argv[1]
    hand = json.loads((OUT / f"{tag}_hand.json").read_text())
    robot = json.loads((OUT / f"{tag}_robot.json").read_text())
    src = hand["_source"]
    size = drill_size(src["recording"])
    pose = sorted({v.get("source", "") for v in (src.get("tools") or {}).values()})   # raw vs filtered hand
    key = hashlib.sha1(json.dumps([src["recording"], src["from_s"], src["len_s"], OPTS, size, RENDERER, pose]).encode()).hexdigest()[:10]
    cache = OUT / f"handvol_{key}.json"
    if cache.exists():
        hv = json.loads(cache.read_text())
    else:
        v, c = volume(hand, size)
        hv = {"volume": v, "curve": c, "window": [src["recording"], src["from_s"], src["len_s"]], "opts": OPTS}
        cache.write_text(json.dumps(hv))
    rv, rc = volume(robot, size)
    row = {"time": time.strftime("%m-%d %H:%M"), "tag": tag, "recording": Path(src["recording"]).name,
           "from_s": src["from_s"], "len_s": src["len_s"], "hand_mm3": round(hv["volume"], 1),
           "robot_mm3": round(rv, 1), "ratio": round(rv / hv["volume"], 3) if hv["volume"] else None,
           "drill": size, "fps": OPTS["fps"], "renderer": RENDERER}
    (OUT / f"{tag}_volume.json").write_text(json.dumps({**row, "hand_curve": hv["curve"], "robot_curve": rc}))
    f = OUT / "volume.csv"
    new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"[vol] {tag}: hand {row['hand_mm3']} mm³, robot {row['robot_mm3']} mm³, ratio {row['ratio']}", flush=True)


if __name__ == "__main__":
    main()
