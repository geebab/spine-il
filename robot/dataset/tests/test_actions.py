"""actions.py: delta round trip and the chunk-relative (cumulative) ↔ per-step conversion; train/lerobot-train-rel.py's torch
conversion against the numpy reference."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import actions as AC  # noqa: E402


def _state(rng):
    return np.r_[rng.normal(size=3) * 50, AC.R2q(AC.exp_R(rng.normal(size=3))), rng.normal(size=3) * 50,
                 AC.R2q(AC.exp_R(rng.normal(size=3))), float(rng.integers(2))]


def _walk(rng, n=20):
    st = [_state(rng)]
    for k in range(n):
        d = np.r_[rng.normal(size=3) * 0.3, rng.normal(size=3) * 0.02, rng.normal(size=3) * 0.3, rng.normal(size=3) * 0.02, k % 2]
        st.append(AC.from_delta(st[-1], d))
    return st


def test_delta_roundtrip():
    rng = np.random.default_rng(0)
    st = _walk(rng)
    for a, b in zip(st[:-1], st[1:]):
        c = AC.from_delta(a, AC.to_delta(a, b))
        assert np.abs(c[:3] - b[:3]).max() < 1e-5 and np.abs(c[7:10] - b[7:10]).max() < 1e-5
        assert abs(abs(c[3:7] @ b[3:7]) - 1) < 1e-6 and c[14] == b[14]


def test_cum_to_steps_reintegrates():
    rng = np.random.default_rng(1)
    st = _walk(rng)
    cum = np.array([AC.to_delta(st[0], s) for s in st[1:]])     # every later pose relative to the first
    v = st[0].copy()
    for j, d in enumerate(AC.cum_to_steps(cum)):
        v = AC.from_delta(v, d)
        assert np.abs(v[:3] - st[j + 1][:3]).max() < 1e-4 and np.abs(v[7:10] - st[j + 1][7:10]).max() < 1e-4
        assert abs(abs(v[10:14] @ st[j + 1][10:14]) - 1) < 1e-6 and v[14] == st[j + 1][14]


def test_torch_chunk_relative_matches_numpy():
    import torch
    spec = importlib.util.spec_from_file_location("rel", HERE.parents[1] / "train" / "lerobot-train-rel.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    rng = np.random.default_rng(2)
    S = np.array([_state(rng) for _ in range(4)])
    A = np.array([[AC.from_delta(s, np.r_[rng.normal(size=3), rng.normal(size=3) * 0.2, rng.normal(size=3),
                                          rng.normal(size=3) * 0.2, 1.0]) for _ in range(6)] for s in S])
    t = m.to_chunk_relative(torch.tensor(S), torch.tensor(A)).numpy()
    ref = np.array([[AC.to_delta(S[i], A[i, j]) for j in range(6)] for i in range(4)])
    assert np.abs(t - ref).max() < 1e-5
