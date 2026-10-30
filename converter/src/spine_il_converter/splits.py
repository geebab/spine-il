"""Deterministic episode-level splits (PLAN-0 §6.5, evidence E3).

Each session is ranked by sha256(seed:sessionId), so a session keeps its split when new sessions are
added (only the cut points move). `test` doubles as the held-out basal-vector set for virtual
closed-loop evaluation.
"""

from __future__ import annotations

import hashlib

SPLIT_NAMES = ("train", "val", "test")


def assign_splits(session_ids: list[str], seed: int = 0, fractions=(0.8, 0.1, 0.1)) -> dict[str, str]:
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1")
    ranked = sorted(session_ids, key=lambda sid: hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest())
    n = len(ranked)
    n_test = round(n * fractions[2])
    n_val = round(n * fractions[1])
    n_train = n - n_val - n_test
    out = {}
    for i, sid in enumerate(ranked):
        out[sid] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return out
