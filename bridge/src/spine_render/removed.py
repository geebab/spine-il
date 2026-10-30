"""Removed bone voxels — the exact voxel set behind metrics.jsonl `removedMm3`, for IoU / Dice between jobs (a hand
recording vs robot runs of the same case).

The viewer computes it (`window.__ubeRender.removedVoxels()`, FE `lib/removedVoxels.ts`, viewer branch
`render-removed-voxels`) from the same two arrays as the volume metric: the drill SDF now and the case's original copy.
  criterion  original > 0 && current <= 0          (SDF sign: bone > 0, air < 0)
  grid       the drill SDF: the case bone SDF cropped to the bone box + padding; scene frame (viewer LPS, mm);
             voxel [i, j, k] sits at origin + [i, j, k] * spacing
  indices    linear i + nx*(j + ny*k) — i = x fastest, then j = y, then k = z (the SDF's own array order);
             uint32 little-endian, ascending, no duplicates
  identity   sdfSha256  = SHA-256 of the original SDF's float32 little-endian bytes
             gridSha256 = SHA-256 of struct.pack('<3I3d3d', *dims, *origin, *spacing)   (60 bytes)
Two jobs can be compared voxel by voxel iff both hashes are equal (same grid, same original SDF). The renderer re-derives
gridSha256 from the dims / origin / spacing the viewer reports and refuses a mismatch, so the header's numbers are the
ones that were hashed.

Files in result.zip (job options `removed_voxels`, `removed_every_s`):
  removed/header.json   grid, identity, criterion, case, cuts (recorded | simulated), effective drill rule, final entry
  removed/final.bin     the set at the last rendered frame (the metrics.jsonl row of the same i / t_ms)
  removed/index.json    removed_every_s only: rows {i, t_ms, threshold_ms, baseline, count, mm3, file} — the set at
                        that frame (everything removed since the job started)
  removed/t<ms>.bin     removed_every_s only: the first frame (baseline), then the first frame at/after every multiple
                        k·removed_every_s of the metrics t_ms axis. threshold_ms = the multiple, t_ms = the frame whose
                        state it is — equal only when the frame grid hits the multiple (fps·removed_every_s an integer,
                        grid from 0); a baseline off the multiples has threshold_ms null.
"""

from __future__ import annotations

import base64
import hashlib
import math
import re
import struct
from dataclasses import dataclass

import numpy as np

FE_FORMAT = "ube-removed-voxels"  # what the viewer's removedVoxels() returns
FE_VERSION = 1
FORMAT = "spine-render-removed-voxels"  # removed/header.json
VERSION = 1
DIR = "removed"
HEADER = f"{DIR}/header.json"
FINAL = f"{DIR}/final.bin"
INDEX = f"{DIR}/index.json"
CRITERION = "original > 0 && current <= 0"
INDEX_ORDER = "i + nx*(j + ny*k)"
GRID_STRUCT = "<3I3d3d"
TIME_AXIS = ("t_ms = the metrics.jsonl t_ms of frame i — ms from the script's first sample (authored inputs) or from "
             "the recording's grid origin (tExp / tMono)")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_EPS_MS = 1e-3  # grid times carry float noise far below this; two frames are never this close
MIN_EVERY_S = 0.1  # removed_every_s floor: finer than the tracker / carve granularity, and result.zip grows per set


def grid_sha256(dims, origin, spacing) -> str:
    """SHA-256 of the canonical grid bytes — struct.pack('<3I3d3d', *dims, *origin, *spacing)."""
    return hashlib.sha256(struct.pack(GRID_STRUCT, *[int(d) for d in dims], *[float(v) for v in origin],
                                      *[float(v) for v in spacing])).hexdigest()


def _vec3(v, what: str, positive: bool = False) -> list[float]:
    ok = isinstance(v, list) and len(v) == 3 and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and (x > 0 or not positive)
        for x in v)
    if not ok:
        raise RuntimeError(f"removedVoxels(): {what} {v!r} is not 3 finite numbers{' > 0' if positive else ''}")
    return [float(x) for x in v]


def decode(rep, indices: bool = True) -> tuple[dict, bytes]:
    """(the report without its indices, the indices as uint32 LE bytes) from window.__ubeRender.removedVoxels() —
    checked end to end (format, criterion, grid, count, order, range, grid hash); any doubt fails the job.
    indices=False reads a count-and-hash report (removedVoxels({indices: false})): no index checks, bytes b""."""
    if not isinstance(rep, dict):
        raise RuntimeError(f"removedVoxels() returned {type(rep).__name__}, not a report object")
    if rep.get("format") != FE_FORMAT or rep.get("version") != FE_VERSION:
        raise RuntimeError(f"removedVoxels() report {rep.get('format')!r} v{rep.get('version')!r} — this renderer reads "
                           f"{FE_FORMAT!r} v{FE_VERSION}")
    for k, want in (("criterion", CRITERION), ("coord", "scene"), ("indexOrder", INDEX_ORDER), ("encoding", "uint32le")):
        if rep.get(k) != want:
            raise RuntimeError(f"removedVoxels(): {k} {rep.get(k)!r}, expected {want!r}")
    dims = rep.get("dims")
    if not (isinstance(dims, list) and len(dims) == 3 and all(isinstance(d, int) and not isinstance(d, bool) and d > 0
                                                               for d in dims)):
        raise RuntimeError(f"removedVoxels(): dims {dims!r} are not 3 positive integers")
    origin, spacing = _vec3(rep.get("origin"), "origin"), _vec3(rep.get("spacing"), "spacing", positive=True)
    n = dims[0] * dims[1] * dims[2]
    if rep.get("nVoxels") != n or rep.get("sdfBytes") != 4 * n:
        raise RuntimeError(f"removedVoxels(): nVoxels {rep.get('nVoxels')!r} / sdfBytes {rep.get('sdfBytes')!r} do not "
                           f"match the grid {dims} ({n} voxels)")
    count = rep.get("count")
    if not (isinstance(count, int) and not isinstance(count, bool) and 0 <= count <= n):
        raise RuntimeError(f"removedVoxels(): count {count!r} is not an integer in [0, {n}]")
    data = b""
    if indices:
        b64 = rep.get("indicesB64")
        if not isinstance(b64, str):
            raise RuntimeError("removedVoxels(): no indicesB64 (called with indices: false?)")
        data = base64.b64decode(b64, validate=True)
        if len(data) % 4:
            raise RuntimeError(f"removedVoxels(): {len(data)} index bytes is not a whole number of uint32")
        idx = np.frombuffer(data, dtype="<u4")
        if idx.size != count:
            raise RuntimeError(f"removedVoxels(): {idx.size} indices but count {count!r}")
        if idx.size > 1 and not bool(np.all(idx[1:] > idx[:-1])):
            raise RuntimeError("removedVoxels(): indices are not strictly ascending (unsorted or duplicated)")
        if idx.size and int(idx[-1]) >= n:
            raise RuntimeError(f"removedVoxels(): index {int(idx[-1])} is outside the {n}-voxel grid")
    mm3 = rep.get("mm3")
    want = count * spacing[0] * spacing[1] * spacing[2]
    if not isinstance(mm3, (int, float)) or abs(mm3 - want) > 1e-9 * max(1.0, want):
        raise RuntimeError(f"removedVoxels(): mm3 {mm3!r} ≠ count × voxel volume {want!r}")
    g = grid_sha256(dims, origin, spacing)
    if rep.get("gridSha256") != g:
        raise RuntimeError(f"removedVoxels(): gridSha256 {rep.get('gridSha256')!r} ≠ {g} re-derived from dims / origin / "
                           f"spacing ({GRID_STRUCT}) — the header numbers would not be the hashed ones")
    if not _HEX64.fullmatch(str(rep.get("sdfSha256"))):
        raise RuntimeError(f"removedVoxels(): sdfSha256 {rep.get('sdfSha256')!r} is not a SHA-256 hex digest")
    return {k: v for k, v in rep.items() if k != "indicesB64"}, data


def identity(meta: dict) -> dict:
    """What must be equal between two sets for voxel-by-voxel comparison (and within one job)."""
    return {k: meta.get(k) for k in ("dims", "origin", "spacing", "sdfSha256", "gridSha256")}


@dataclass
class RemovedSet:
    """One removedVoxels() answer, at frame i / t_ms of the job's metrics timeline."""
    i: int
    t_ms: float
    meta: dict
    data: bytes  # uint32 LE, ascending
    threshold_ms: float | None = None  # removed_every_s: the multiple this snapshot stands for (None: off-grid baseline)
    wall_ms: float | None = None  # the page call, wall clock
    baseline: bool = False  # removed_every_s: the job's first frame

    @property
    def count(self) -> int:
        return len(self.data) // 4 if self.data else int(self.meta.get("count") or 0)

    def row(self, file: str | None) -> dict:
        return {"i": self.i, "t_ms": round(self.t_ms, 3), "threshold_ms": None if self.threshold_ms is None else
                round(self.threshold_ms, 3), "baseline": self.baseline, "count": self.count,
                "mm3": self.meta.get("mm3"), "file": file}


class SnapshotPlan:
    """removed_every_s: the first frame (baseline), then the first frame at/after each multiple k·every_s of the
    metrics t_ms axis — one snapshot per multiple however many frames fall between (a coarse fps skips multiples).
    The multiple a snapshot stands for is its threshold; a baseline that is not on a multiple has none."""

    def __init__(self, every_s: float):
        self.every_ms = 1000.0 * float(every_s)
        self.next_ms: float | None = None

    def due(self, t_ms: float) -> tuple[bool, float | None, bool]:
        """(snapshot this frame?, the multiple in ms it stands for or None, is it the baseline)."""
        first = self.next_ms is None
        if not first and t_ms + _EPS_MS < self.next_ms:
            return False, None, False
        k = math.floor((t_ms + _EPS_MS) / self.every_ms)
        self.next_ms = (k + 1) * self.every_ms
        thr = k * self.every_ms
        if first and abs(t_ms - thr) > _EPS_MS:
            return True, None, True  # a window starting between multiples: the baseline stands for no multiple
        return True, thr, first


def snapshot_file(t_ms: float, used: set[str]) -> str:
    """removed/t<ms>.bin (integer ms, 8 digits) — a second frame on the same ms (fps > 1000 only) gets a suffix."""
    name = f"{DIR}/t{int(math.floor(t_ms + 0.5)):08d}.bin"
    if name in used:
        name = f"{DIR}/t{int(math.floor(t_ms + 0.5)):08d}_{len(used):04d}.bin"
    used.add(name)
    return name


def build_header(final: RemovedSet, facts: dict, every_s: float | None = None, n_snapshots: int = 0) -> dict:
    """removed/header.json — the viewer's grid / identity facts, the final set's entry, then the job's facts (job,
    input, case, replayMode, cuts, drillRule, source, recording, replay, viewer, renderer). The caller adds `start`
    (the set before the first frame — 0 voxels unless the case's .acp carries a carve)."""
    m = final.meta
    return {
        "format": FORMAT, "version": VERSION,
        "criterion": m["criterion"], "sdfSign": m.get("sdfSign"),
        "coord": "scene", "coordNote": "viewer scene frame (LPS, mm); voxel [i, j, k] sits at origin + [i, j, k] * spacing",
        "dims": m["dims"], "origin": m["origin"], "spacing": m["spacing"], "voxelMm3": m.get("voxelMm3"),
        "nVoxels": m["nVoxels"],
        "encoding": "uint32le", "indexOrder": INDEX_ORDER, "axes": m.get("axes"), "sorted": "ascending-unique",
        "sdfSha256": m["sdfSha256"], "gridSha256": m["gridSha256"], "sdfBytes": m["sdfBytes"],
        "gridBytes": f"struct.pack('{GRID_STRUCT}', *dims, *origin, *spacing)",
        "comparable": "two jobs' sets compare voxel by voxel iff sdfSha256 and gridSha256 are both equal",
        "final": final.row(FINAL),
        "start": None,
        "snapshots": {"everyS": every_s, "index": INDEX, "n": n_snapshots} if every_s else None,
        "timeAxis": TIME_AXIS,
        "viewerCase": {"caseKey": m.get("caseKey"), "acpUrl": m.get("acpUrl")},
        "drillRadiusMm": m.get("drillRadiusMm"),
        **facts,
    }


def index_doc(snaps: list[tuple[RemovedSet, str]], every_s: float) -> dict:
    """removed/index.json — the cumulative snapshots in time order."""
    return {"format": f"{FORMAT}-index", "version": VERSION, "everyS": every_s, "timeAxis": TIME_AXIS,
            "rows": [s.row(f) for s, f in snaps]}
