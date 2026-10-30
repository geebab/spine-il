"""Supplementary cut-agreement metrics next to the exact voxel IoU (2026-09-29, user-approved as additions):

  tol1_dice  — overlap with a 1-voxel (0.8 mm) tolerance: (|A ∩ dil(B)| + |B ∩ dil(A)|) / (|A| + |B|), dil = 26-neighbour
               dilation. 1 when every voxel of each set lies within one voxel of the other; no longer punishes the
               sub-voxel shifts the noise floor showed (0.3 mm offset → IoU 0.89–0.94).
  assd_mm    — average symmetric surface distance between the two cut regions' boundary voxels (mm)
  hd95_mm    — 95th percentile of those symmetric boundary distances (mm): where the cuts disagree most, robustly
Sets are removed-voxel index arrays (uint32, index = i + nx*(j + ny*k)) on the same grid (hdr dims/spacing).
"""
import numpy as np

N26 = np.array([(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)])
N6 = np.array([(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)])


def ijk(idx: np.ndarray, dims) -> np.ndarray:
    nx, ny, _ = dims
    idx = idx.astype(np.int64)
    return np.stack([idx % nx, (idx // nx) % ny, idx // (nx * ny)], 1)


def enc(p: np.ndarray, dims) -> np.ndarray:
    nx, ny, nz = dims
    ok = (p >= 0).all(1) & (p[:, 0] < nx) & (p[:, 1] < ny) & (p[:, 2] < nz)
    q = np.where(ok[:, None], p, 0)
    return np.where(ok, q[:, 0] + nx * (q[:, 1] + ny * q[:, 2]), -1)


def dilate(idx: np.ndarray, dims) -> np.ndarray:
    p = ijk(idx, dims)
    nb = (p[:, None, :] + N26[None]).reshape(-1, 3)
    e = enc(nb, dims)
    return np.union1d(idx.astype(np.int64), e[e >= 0])


def boundary(idx: np.ndarray, dims) -> np.ndarray:
    """Voxels of the set with a 6-neighbour outside it (ijk)."""
    p = ijk(idx, dims)
    nb = enc((p[:, None, :] + N6[None]).reshape(-1, 3), dims).reshape(len(p), 6)
    inside = np.isin(nb, idx.astype(np.int64))
    return p[~inside.all(1)]


def _nearest(a: np.ndarray, b: np.ndarray, chunk: int = 2048) -> np.ndarray:
    out = np.empty(len(a))
    bf = b.astype(np.float32)
    for s in range(0, len(a), chunk):
        d = ((a[s:s + chunk, None, :].astype(np.float32) - bf[None]) ** 2).sum(-1)
        out[s:s + chunk] = np.sqrt(d.min(1))
    return out


def ext_metrics(a: np.ndarray, b: np.ndarray, hdr: dict) -> dict:
    dims, sp = hdr["dims"], float(hdr["spacing"][0])
    if len(a) == 0 or len(b) == 0:
        return {"tol1_dice": float("nan"), "assd_mm": float("nan"), "hd95_mm": float("nan")}
    ai, bi = a.astype(np.int64), b.astype(np.int64)
    tol = (np.isin(ai, dilate(bi, dims)).sum() + np.isin(bi, dilate(ai, dims)).sum()) / (len(a) + len(b))
    ba, bb = boundary(ai, dims), boundary(bi, dims)
    d = np.concatenate([_nearest(ba, bb), _nearest(bb, ba)]) * sp
    return {"tol1_dice": round(float(tol), 3), "assd_mm": round(float(d.mean()), 2), "hd95_mm": round(float(np.percentile(d, 95)), 2)}
