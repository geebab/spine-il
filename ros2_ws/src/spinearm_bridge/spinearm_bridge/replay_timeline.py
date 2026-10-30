"""Playback side of recording_player, ROS-free (numpy only): the playback window, the wall-clock schedule and the
cumulative cut markers. recording.py parses a recording into events; the node publishes what these hand it."""

from __future__ import annotations

import numpy as np

from .recording import Event, Recording

CUT_CHUNK = 2000   # stamps per cuts marker: a growing sphere list is re-sent one chunk at a time
_ID_STRIDE = 10000   # marker id = group × _ID_STRIDE + chunk index


def window(rec: Recording, start_s: float = 0.0, end_s: float = 0.0) -> tuple[list[Event], list[Event], float, float]:
    """(cuts before the window, events inside it, window start ms, window end ms) for [first + start_s, first + end_s]
    (end_s ≤ 0 → to the end). The earlier cuts are applied at once when playback starts, so a start in the middle shows
    the bone as it was; a dynamic registration's last transform before the window is re-stamped at its start."""
    if not rec.events:
        raise ValueError("recording has no events")
    a = rec.t_first + max(0.0, float(start_s)) * 1000.0
    b = rec.t_last if end_s <= 0 else min(rec.t_last, rec.t_first + float(end_s) * 1000.0)
    if b < a:
        raise ValueError(f"empty window: start_s={start_s} end_s={end_s} (recording {rec.duration_s:.2f} s)")
    pre = [e for e in rec.events if e.t < a and e.kind == "cut"]
    inside = [e for e in rec.events if a <= e.t <= b]
    tf_before = [e for e in rec.events if e.t < a and e.kind == "tf"]
    if tf_before and not any(e.kind == "tf" and e.t == a for e in inside):
        inside.insert(0, Event(a, tf_before[-1].r, "tf", tf_before[-1].data))
    return pre, inside, a, b


class Schedule:
    """Wall-clock pacing of a sorted timeline: an event at t (ms) is due (t − t_start) / rate after start(). Nothing is
    dropped — a late tick gets every due event in order (catch-up), at most `limit` per call."""

    def __init__(self, t_ms, t_start_ms: float, t_end_ms: float, rate: float):
        if not (np.isfinite(rate) and rate > 0):
            raise ValueError(f"rate must be > 0, got {rate}")
        self.t = np.asarray(t_ms, float)
        if self.t.size > 1 and np.any(np.diff(self.t) < 0):
            raise ValueError("event times must be sorted")
        if t_end_ms < t_start_ms:
            raise ValueError("window end before start")
        self.t_start, self.t_end, self.rate = float(t_start_ms), float(t_end_ms), float(rate)
        self.cursor, self.mono0, self.wall0_ms = 0, 0.0, 0.0

    def start(self, mono_s: float, wall_ms: float) -> None:
        """(Re)start at monotonic mono_s; wall_ms = the epoch ms that t_start maps to in wall-clock stamping."""
        self.cursor, self.mono0, self.wall0_ms = 0, float(mono_s), float(wall_ms)

    def position_ms(self, mono_s: float) -> float:
        """Recording time reached at mono_s (held at the window end)."""
        return min(self.t_end, self.t_start + (mono_s - self.mono0) * 1000.0 * self.rate)

    def due(self, mono_s: float, limit: int | None = None) -> range:
        pos = self.t_start + (mono_s - self.mono0) * 1000.0 * self.rate
        end = int(np.searchsorted(self.t, pos, side="right"))
        if limit is not None:
            end = min(end, self.cursor + limit)
        out = range(self.cursor, max(self.cursor, end))
        self.cursor = out.stop
        return out

    def due_mono_s(self, t_ms: float) -> float:
        return self.mono0 + (t_ms - self.t_start) / (1000.0 * self.rate)

    def wall_ms(self, t_ms: float) -> float:
        """Wall-clock stamp of recording time t_ms: start wall time + (t − t_start) / rate (live-like)."""
        return self.wall0_ms + (t_ms - self.t_start) / self.rate

    def finished(self, mono_s: float) -> bool:
        return self.cursor >= self.t.size and self.position_ms(mono_s) >= self.t_end

    def progress(self, mono_s: float) -> float:
        span = self.t_end - self.t_start
        return 1.0 if span <= 0 else (self.position_ms(mono_s) - self.t_start) / span


class CutAccumulator:
    """Cumulative drill stamps for the cuts markers: one sphere list per (target, label, radius), cut into chunks of
    `chunk` stamps with stable marker ids (group × 10000 + chunk index).

    take_new() hands out only the stamps added since its last call, chunk by chunk, so the player turns each stamp into
    a message once and re-sends only the chunks that grew (a long session never rebuilds or re-sends the whole list per
    cut). take_removed() gives the marker ids the player has been handed that must go (soft reset, clear)."""

    def __init__(self, chunk: int = CUT_CHUNK):
        if not 0 < chunk < _ID_STRIDE:
            raise ValueError(f"chunk must be in 1..{_ID_STRIDE - 1}")
        self.chunk = chunk
        self.gid: dict[tuple, int] = {}                # group key → number (stable, never reused)
        self.parts: dict[tuple, list[np.ndarray]] = {}   # every stamp of a live group
        self.fresh: dict[tuple, list[np.ndarray]] = {}   # stamps not handed out yet
        self.sent: dict[tuple, int] = {}               # stamps handed out per group
        self.removed: set[int] = set()
        self.n_rows = self.n_points = self.mod = 0

    def marker_id(self, key: tuple, index: int) -> int:
        return self.gid[key] * _ID_STRIDE + index

    def apply(self, d: dict) -> None:
        self.n_rows += 1
        if d.get("op") == "reset":           # soft-tissue journal restored: its stamps are gone
            for key in [k for k in self.parts if k[0] == d["target"]]:
                self._drop(key)
            return
        pts = np.asarray(d.get("points", ()), float).reshape(-1, 3)
        if d.get("radius_m") is None or not len(pts):
            return
        key = (d["target"], d.get("label") or "", round(float(d["radius_m"]), 6))
        self.gid.setdefault(key, len(self.gid))
        self.parts.setdefault(key, []).append(pts)
        self.fresh.setdefault(key, []).append(pts)
        self.n_points += len(pts)
        self.mod += int(d.get("mod") or 0)

    def _drop(self, key: tuple) -> None:
        handed = self.sent.pop(key, 0)
        self.removed.update(self.marker_id(key, i) for i in range(-(-handed // self.chunk)))
        self.parts.pop(key, None)
        self.fresh.pop(key, None)

    def clear(self) -> None:
        for key in list(self.parts):
            self._drop(key)
        self.n_rows = self.n_points = self.mod = 0

    def take_new(self) -> list[tuple[int, tuple, np.ndarray]]:
        """[(marker id, group key, stamps (n, 3) m added to that chunk since the last call)]"""
        out = []
        for key, parts in self.fresh.items():
            new = np.concatenate(parts)
            s = self.sent.get(key, 0)
            i = 0
            while i < len(new):
                c = (s + i) // self.chunk
                j = i + (c + 1) * self.chunk - (s + i)
                out.append((self.marker_id(key, c), key, new[i:j]))
                i = j
            self.sent[key] = s + len(new)
        self.fresh.clear()
        return out

    def take_removed(self) -> list[int]:
        out = sorted(self.removed)
        self.removed.clear()
        return out

    def groups(self) -> list[tuple[tuple, np.ndarray]]:
        """[(group key, every stamp (n, 3) m)] — the whole current state."""
        return [(k, np.concatenate(v)) for k, v in self.parts.items()]
