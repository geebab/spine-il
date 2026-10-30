"""A recorder v2 session handed to the viewer as is.

The render keeps the file (a job saves it next to its results and serves it to the page by URL — `replay.load({url})`)
and reads only the header and a few timestamps: to pick the frame times, name the case and check the counts. It never
turns rows into tracker messages — the replay rules live in the viewer, in one place (fid-common v2).

Modes — the same words in the viewer, the player and here:
- `replay` (default for recordings): the recording as it happened — cuts = the engine's own cut rows re-applied
  (byte-identical SDF), screen = the recorded drawn poses applied directly. Needs timingSpec v2-t0 + hdr.replay.cutRows
  (rec-2.3.0+).
- `simulate`: the engine cuts by itself. For a recording that is the recorded hand re-run through the current engine
  (virtual clock, `tickHz`) — a diagnostic, allowed only with diagnostic=true. Authored trajectories, deltas and WS steps
  are always `simulate` (Python messages → the in-page tracker socket).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .gate import recording_gate
from .geom import Registration
from .normalize.recording import is_path
from .script import CASE_RE, rig_default  # CASE_RE re-exported (moved to script to avoid an import cycle)

MODES = ("replay", "simulate")
GRIDS = ("tExp", "tMono")
DEFAULT_CASE = "altician_lumbar1"
DEFAULT_GRID_TOOL = "endo"  # contract: seek({tExpShown, tool}) — the reference tool, default endo, else the first tool


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


class RecordingRef:
    """The file (path, or bytes until a job saves it), its sha256, the header, and the timeline the render reads:
    first camera frame (tExp), last row tExp, last frame each tool's screen showed (drawn.tExp), tools in order of
    appearance, the first row's tMono (the viewer's tMs origin), the last tMono, and the cut rows' tMono (one per `r` —
    a re-sent upload chunk repeats rows)."""

    def __init__(self, hdr: dict, raw: bytes | None, path: Path | None, **timeline):
        self.hdr, self.path = hdr, path
        self._raw = raw if path is None else None
        self.sha256 = timeline.pop("sha256")
        self.nbytes = timeline.pop("nbytes")
        self.warnings: list[str] = timeline.pop("warnings")
        self.t_exp_first: float | None = timeline.pop("t_exp_first")
        self.t_exp_last: float | None = timeline.pop("t_exp_last")
        self.shown_last: dict[str, float] = timeline.pop("shown_last")
        self.tools: list[str] = timeline.pop("tools")
        self.t0_mono: float | None = timeline.pop("t0_mono")
        self.t_mono_last: float | None = timeline.pop("t_mono_last")
        self.cut_mono: list[float] = sorted(timeline.pop("cut_mono"))

    @classmethod
    def from_src(cls, src) -> "RecordingRef | None":
        """RecordingRef for a recorder v2 session (path, JSONL text/bytes or parsed rows); None for anything else.
        A recording with a line that is not a JSON object raises ValueError (a torn last line only warns — the viewer
        decides)."""
        path = None
        if is_path(src):
            path = Path(src)
            raw = path.read_bytes()
        elif isinstance(src, bytes):
            raw = src
        elif isinstance(src, str):
            raw = src.encode("utf-8")
        elif isinstance(src, list) and all(isinstance(r, dict) for r in src):
            raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in src).encode("utf-8")
        else:
            return None
        try:
            lines = raw.decode("utf-8").splitlines()
        except UnicodeDecodeError:
            return None
        rows, bad = [], []
        for n, line in enumerate(lines, 1):
            if line.strip():
                try:
                    rows.append((n, json.loads(line)))
                except json.JSONDecodeError:
                    bad.append(n)
        hdr = next((o for _, o in rows if isinstance(o, dict) and o.get("k") == "hdr" and "schemaVersion" in o), None)
        if hdr is None:
            return None  # not a recording: normalize() judges JSON documents and scripts
        not_obj = [n for n, o in rows if not isinstance(o, dict)]
        if not_obj:
            raise ValueError(f"recording line {not_obj[0]} is not a JSON object")
        last_line = max((n for n, line in enumerate(lines, 1) if line.strip()), default=0)
        if [n for n in bad if n != last_line]:
            raise ValueError(f"recording line {bad[0]} is not JSON")
        warnings = [f"recording line {last_line} (the last) is not JSON — a torn tail? skipped here, the viewer decides"
                    ] if bad else []
        tl = dict(t_exp_first=None, t_exp_last=None, shown_last={}, tools=[], t0_mono=None, t_mono_last=None)
        first, cuts = None, {}
        for j, r in rows:
            k, mono, n = r.get("k"), _num(r.get("tMono")), _num(r.get("r"))
            if mono is not None and k != "hdr":
                tl["t_mono_last"] = mono if tl["t_mono_last"] is None else max(tl["t_mono_last"], mono)
                if n is not None and (first is None or n < first[0]):
                    first = (n, mono)
            if k == "f":
                tool = r.get("tool")
                if tool not in tl["tools"]:
                    tl["tools"].append(tool)
                te = _num(r.get("tExp"))
                if te is not None:
                    tl["t_exp_first"] = te if tl["t_exp_first"] is None else min(tl["t_exp_first"], te)
                    tl["t_exp_last"] = te if tl["t_exp_last"] is None else max(tl["t_exp_last"], te)
                d = r.get("drawn")
                dte = _num(d.get("tExp")) if isinstance(d, dict) else None
                if dte is not None:
                    tl["shown_last"][tool] = max(tl["shown_last"].get(tool, -math.inf), dte)
            elif k == "cut" and mono is not None:
                cuts.setdefault(n if n is not None else ("line", j), mono)
        tl["t0_mono"] = first[1] if first else None
        return cls(hdr, raw, path, sha256=hashlib.sha256(raw).hexdigest(), nbytes=len(raw), warnings=warnings,
                   cut_mono=list(cuts.values()), **tl)

    # ── the file ──

    def save(self, dest: Path) -> Path:
        """Write the file to `dest` (a job's recording.jsonl) and keep only the path — nothing large stays in memory."""
        dest = Path(dest)
        if self.path is None or self.path.resolve() != dest.resolve():
            dest.write_bytes(self.data())
        self.path, self._raw = dest, None
        return dest

    def data(self) -> bytes:
        return self._raw if self._raw is not None else self.path.read_bytes()

    # ── what the render needs ──

    @property
    def case(self) -> str:
        """The viewer case (mock registry key) — the recorder writes it as meshId. Checked: it goes into the page URL."""
        mesh = self.hdr.get("meshId") or DEFAULT_CASE
        if not isinstance(mesh, str) or not CASE_RE.fullmatch(mesh):
            raise ValueError(f"hdr.meshId {mesh!r} is not a viewer case key ([A-Za-z0-9_.-], ≤ 64)")
        return mesh

    def registration(self) -> Registration:
        """T0 the page is seeded with (the viewer's replay applies the recording's own T; this keeps the page
        registered like the live session)."""
        t0 = (self.hdr.get("registration") or {}).get("T0")
        return Registration.from_dict(t0 or rig_default()["registration"])

    def gate(self) -> dict:
        return recording_gate(self.hdr)

    @property
    def screen_duration_ms(self) -> float | None:
        if self.t0_mono is None or self.t_mono_last is None:
            return None
        return self.t_mono_last - self.t0_mono

    def replay_capable(self) -> tuple[bool, str]:
        h = self.hdr
        if h.get("timingSpec") != "v2-t0":
            return False, f"timingSpec is {h.get('timingSpec')!r}, replay needs 'v2-t0' (drawn poses, tExp)"
        rp = h.get("replay") or {}
        if not rp.get("cutRows"):
            return False, "no hdr.replay.cutRows: the recorder predates cut rows (rec-2.3.0)"
        if rp.get("drawn") is False:
            return False, "hdr.replay.drawn is false: no drawn poses to show"
        return True, ""

    def grid_tool(self, tool: str | None = None) -> str | None:
        """The tExpShown reference tool: `tool` if asked, else endo, else the first tool with drawn frames."""
        if tool is not None:
            if tool not in self.shown_last:
                raise ValueError(f"grid_tool {tool!r} has no drawn frames in this recording (tools {self.tools})")
            return tool
        if DEFAULT_GRID_TOOL in self.shown_last:
            return DEFAULT_GRID_TOOL
        return next((t for t in self.tools if t in self.shown_last), None)

    def texp_span(self, tool: str | None = None) -> tuple[float, float, str] | None:
        """(first camera frame tExp, last frame the reference tool's screen showed, that tool) — the tExp grid span."""
        ref = self.grid_tool(tool)
        if self.t_exp_first is None or ref is None:
            return None
        return self.t_exp_first, self.shown_last[ref], ref

    def seek_grid(self, axis: str, fps: float, duration_ms: float | None = None, t_start_s: float | None = None,
                  t_end_s: float | None = None, tool: str | None = None) -> list[tuple[float, dict]]:
        """[(t_ms from the grid origin, seek target)] every 1/fps.

        tExp  — dataset-aligned: {tExpShown: first tExp + k/fps, tool}, up to the last frame the reference tool's screen
                showed.
        tMono — screen time: {tMs: k/fps} from the first row's tMono, up to `duration_ms` (replay.load's durationMs;
                the file's own span before the page opens)."""
        if axis not in GRIDS:
            raise ValueError(f"grid must be one of {GRIDS}")
        step = 1000.0 / fps
        lo = 1000.0 * (t_start_s or 0.0)
        ref = None
        if axis == "tExp":
            span = self.texp_span(tool)
            if span is None:
                raise ValueError("no tExp / drawn.tExp in this recording: use grid='tMono'")
            origin, end_abs, ref = span
            hi = end_abs - origin
        else:
            if duration_ms is None:
                raise ValueError("tMono grid needs the replay duration")
            origin, hi = 0.0, float(duration_ms)
        if t_end_s is not None:
            hi = min(hi, 1000.0 * t_end_s)
        n = int(math.floor((hi + 0.01) / step)) + 1 if hi >= 0 else 0  # 10 µs slack: epoch-ms floats round at 2e-4
        out = []
        for k in range(n):
            t = k * step
            if t < lo - 1e-9:
                continue
            out.append((t, {"tExpShown": origin + t, "tool": ref} if axis == "tExp" else {"tMs": t}))
        return out

    def expected_cut_rows(self, t_ms: float | None, t0_mono: float | None = None) -> int | None:
        """Recorded cut rows the viewer must have applied by screen time t_ms: tMono − t0Mono ≤ tMs (t0Mono = the
        viewer's origin from load, default the first row's tMono)."""
        t0 = self.t0_mono if t0_mono is None else t0_mono
        if t_ms is None or t0 is None:
            return None
        return sum(1 for m in self.cut_mono if m - t0 <= t_ms + 1e-6)

    def summary(self) -> dict:
        """What /v1/normalize and `spine-render normalize` report for a recording (nothing is converted); also the
        manifest's `source`."""
        ok, why = self.replay_capable()
        span = self.texp_span()
        h = self.hdr
        return {"kind": "recording", "sessionId": h.get("sessionId"), "case": self.case, "sha256": self.sha256,
                "bytes": self.nbytes, "recorderVersion": h.get("recorderVersion"), "timingSpec": h.get("timingSpec"),
                "modes": ["replay", "simulate (diagnostic)"] if ok else ["simulate (diagnostic)"],
                "replayUnavailable": why or None, "tools": self.tools, "nCutRows": len(self.cut_mono),
                "screenDurationMs": round(self.screen_duration_ms, 3) if self.screen_duration_ms is not None else None,
                "tExpFirst": self.t_exp_first, "tExpLast": self.t_exp_last,
                "tExpLastShown": span[1] if span else None, "gridTool": span[2] if span else None,
                "warnings": self.warnings, "gate": self.gate()}


def resolve_mode(mode: str | None, rec: "RecordingRef | None", diagnostic: bool | None = None,
                 drill_rule: str | None = None) -> str:
    """'replay' | 'simulate' for a request. Recordings: replay by default; simulate only with diagnostic=true — or with
    a `drill_rule`, which re-cuts the recorded hand under that rule (simulate; the diagnostic is implied, an explicit
    diagnostic=false or mode 'replay' is refused: replay re-applies the recorded cut rows, which ignore any rule).
    Authored inputs: simulate only."""
    if mode not in (None, "auto", *MODES):
        raise ValueError(f"mode must be one of {MODES}")
    if rec is None:
        if mode == "replay":
            raise ValueError("mode 'replay' needs a recorder session (the viewer replays the file); authored "
                             "trajectories and deltas are simulated")
        return "simulate"
    if drill_rule is not None:
        if mode == "replay":
            raise ValueError(f"drill_rule {drill_rule!r} re-cuts the recording under that rule — mode 'simulate' (a "
                             "diagnostic); mode 'replay' re-applies the recorded cut rows, which ignore any drill rule")
        if diagnostic is False:
            raise ValueError(f"drill_rule {drill_rule!r} on a recording runs mode 'simulate' — a diagnostic (the "
                             "recorded hand re-cut by the current engine); diagnostic=false contradicts it")
        return "simulate"
    if mode == "simulate":
        if diagnostic is not True:
            raise ValueError("mode 'simulate' on a recording re-runs the recorded hand through the current engine — a "
                             "diagnostic: pass diagnostic=true (cuts are recomputed, the recording's own are the "
                             "reference)")
        return "simulate"
    ok, why = rec.replay_capable()
    if not ok:
        raise ValueError(f"this recording cannot be replayed: {why} — mode 'simulate' with diagnostic=true re-runs it "
                         "through the current engine instead (it has no tExp grid: grid='tMono', chosen by default)")
    return "replay"


def resolve_grid(grid: str | None, rec: RecordingRef, tool: str | None = None) -> str:
    """The frame grid for a recording, decided before any page opens: tExp when the recording has camera-frame stamps
    and drawn frames (the reference tool's), else tMono. An explicit tExp the recording cannot give is an error."""
    if grid not in (None, "auto", *GRIDS):
        raise ValueError(f"grid must be one of {GRIDS}")
    has_texp = rec.texp_span(tool) is not None
    if grid in (None, "auto"):
        return "tExp" if has_texp else "tMono"
    if grid == "tExp" and not has_texp:
        raise ValueError("grid 'tExp' needs tExp + drawn.tExp (timingSpec v2-t0) — this recording has none; use "
                         "grid='tMono' or leave grid unset")
    return grid


def check_window(rec: RecordingRef, grid: str, fps: float, t_start_s: float | None, t_end_s: float | None,
                 tool: str | None = None, duration_ms: float | None = None) -> int:
    """Frames the window gives on the file's own timeline (before the page opens); a window outside it is an error."""
    frames = rec.seek_grid(grid, fps, duration_ms if duration_ms is not None else rec.screen_duration_ms, t_start_s,
                           t_end_s, tool)
    if not frames:
        if grid == "tExp":
            span = rec.texp_span(tool)
            extent = f"tExp span 0–{(span[1] - span[0]) / 1000:.3f} s ({span[2]})" if span else "no tExp span"
        else:
            d = duration_ms if duration_ms is not None else rec.screen_duration_ms
            extent = f"screen time 0–{d / 1000:.3f} s" if d is not None else "no screen time"
        raise ValueError(f"window t_start_s={t_start_s} t_end_s={t_end_s} is outside the recording ({extent}) — no frames")
    return len(frames)
