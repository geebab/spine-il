#!/usr/bin/env python3
"""IoU / Dice between two render jobs' removed-bone voxel sets (options removed_voxels / removed_every_s).

    python3 removed_iou.py A.zip B.zip                  # the final sets (removed/final.bin)
    python3 removed_iou.py A.zip B.zip --at 6000         # the snapshots of the 6 s multiple (removed_every_s)
    python3 removed_iou.py A.zip B.zip --at 6000 --since 3000   # the voxels removed in (3 s, 6 s] only
    python3 removed_iou.py A.zip B.zip --at baseline     # the first frames

A and B are result.zip files or their extracted folders. numpy only. Sets on different grids or from different
original SDFs are refused (removed/header.json: sdfSha256, gridSha256, dims, origin, spacing, …). Printed as warnings —
comparisons you may want, but they must be the ones you meant: a drill rule that is unknown or differs, a different cut
source (recorded / simulated), snapshots whose frames are not at the same t_ms (a snapshot is the first frame at/after
its multiple: fps / window / tExp phase decide which), and t_ms axes with different origins (a recording's grid origin
vs an authored input's first sample).
"""

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

SAME = ("sdfSha256", "gridSha256", "dims", "origin", "spacing", "criterion", "indexOrder", "coord", "encoding")


def _read(src: Path, name: str) -> bytes:
    if src.is_dir():
        return (src / name).read_bytes()
    with zipfile.ZipFile(src) as z:
        return z.read(name)


def load(src, at="final") -> tuple[dict, np.ndarray, dict]:
    """(header, uint32 indices, the entry used) — at = "final" or a snapshot's threshold in ms (index.json)."""
    src = Path(src)
    header = json.loads(_read(src, "removed/header.json"))
    if at == "final":
        entry = header["final"]
    else:
        rows = json.loads(_read(src, "removed/index.json"))["rows"]
        if at == "baseline":
            entry = next((r for r in rows if r.get("baseline")), None)
        else:
            entry = next((r for r in rows if r["threshold_ms"] is not None and abs(r["threshold_ms"] - float(at)) < 1e-3),
                         None)
        if entry is None:
            raise SystemExit(f"{src}: no snapshot at {at} (thresholds: {[r['threshold_ms'] for r in rows]})")
    return header, np.frombuffer(_read(src, entry["file"]), dtype="<u4"), entry


def check_same_grid(ha: dict, hb: dict) -> list[str]:
    """Refuse (SystemExit) unless voxel-by-voxel comparable; return warnings for rule / cut differences."""
    bad = [k for k in SAME if ha.get(k) != hb.get(k)]
    if bad:
        raise SystemExit(f"not comparable — removed/header.json differs in {bad}")
    warn = []
    ra, rb = (h.get("drillRule") or {} for h in (ha, hb))
    if "unknown" in (ra.get("rule"), rb.get("rule")) or (ra.get("key"), ra.get("requested")) != (rb.get("key"),
                                                                                                  rb.get("requested")):
        warn.append(f"drill rule A {ra.get('rule')!r} vs B {rb.get('rule')!r} (requested {ra.get('requested')!r} / "
                    f"{rb.get('requested')!r})")
    if ha.get("cuts") != hb.get("cuts"):
        warn.append(f"cuts A {ha.get('cuts')!r} vs B {hb.get('cuts')!r}")
    if ha.get("input") != hb.get("input"):
        warn.append(f"t_ms origins differ: A is {ha.get('input')!r}, B is {hb.get('input')!r} (recording grid origin "
                    "vs authored first sample)")
    return warn


def check_same_time(ea: dict, eb: dict) -> list[str]:
    if abs(ea["t_ms"] - eb["t_ms"]) > 1e-3:
        return [f"the sets are from frames at different times: A t_ms {ea['t_ms']} vs B t_ms {eb['t_ms']}"]
    return []


def iou_dice(a: np.ndarray, b: np.ndarray) -> dict:
    inter = np.intersect1d(a, b, assume_unique=True).size
    union = np.union1d(a, b).size
    return {"a": int(a.size), "b": int(b.size), "intersection": int(inter), "union": int(union),
            "iou": inter / union if union else None, "dice": 2 * inter / (a.size + b.size) if a.size + b.size else None}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--at", default="final", help="final (default) or a snapshot threshold in ms")
    ap.add_argument("--since", help="subtract the snapshot at this threshold (ms): the interval (since, at]")
    x = ap.parse_args(argv)
    ha, a, ea = load(x.a, x.at)
    hb, b, eb = load(x.b, x.at)
    warn = check_same_grid(ha, hb) + check_same_time(ea, eb)
    if x.since is not None:
        _, a0, sa = load(x.a, x.since)
        _, b0, sb = load(x.b, x.since)
        warn += check_same_time(sa, sb)
        a = np.setdiff1d(a, a0, assume_unique=True)
        b = np.setdiff1d(b, b0, assume_unique=True)
    r = iou_dice(a, b)
    v = ha["voxelMm3"]
    r.update(aMm3=round(r["a"] * v, 3), bMm3=round(r["b"] * v, 3), at=x.at, since=x.since,
             aT_ms=ea["t_ms"], bT_ms=eb["t_ms"], warnings=warn)
    print(json.dumps(r, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
