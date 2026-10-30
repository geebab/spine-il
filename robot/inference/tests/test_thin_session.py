"""thin_session.py on a made-up session: frame rows thinned to the asked rate, every row near a cut or a trigger edge kept,
all other rows and the order untouched, a small file left alone, never written into the sessions tree."""

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import thin_session as T  # noqa: E402


def session(path: Path, seconds: float = 20.0, hz: float = 50.0, cut_at=(8.0,), trig=(7.5, 9.0)) -> list[dict]:
    rows = [{"k": "hdr", "r": 0, "sessionId": "s_x"}]
    n = int(seconds * hz); r = 1
    for i in range(n):
        t = 1000.0 + i * 1000.0 / hz
        for tool in ("endo", "drill"):
            d = {"k": "f", "r": r, "tool": tool, "tMono": t, "pad": "x" * 200}
            if tool == "drill":
                d["btn"] = int(trig[0] * 1000 + 1000 <= t < trig[1] * 1000 + 1000)
            rows.append(d); r += 1
        for c in cut_at:
            if abs(t - (1000.0 + c * 1000.0)) < 1e-6:
                rows.append({"k": "cut", "r": r, "tool": "drill", "tMono": t + 1.0, "mod": 3}); r += 1
        if i == n // 2:
            rows.append({"k": "ev", "r": r, "tMono": t, "ev": "zr_down"}); r += 1
    rows.append({"k": "ftr", "r": r, "endReason": "completed"})
    path.write_text("".join(json.dumps(d, separators=(",", ":")) + "\n" for d in rows))
    return rows


def test_thins_frames_keeps_the_rest(tmp_path):
    src = tmp_path / "in" / "s_x.jsonl"; src.parent.mkdir()
    rows = session(src)
    res = T.thin(src, tmp_path / "out", hz=10.0, keep_s=0.5, min_mb=0.0)
    out = [json.loads(l) for l in Path(res["path"]).read_text().splitlines()]
    assert res["thinned"] and Path(res["path"]).name == src.name and Path(res["path"]).parent == tmp_path / "out"
    assert [d for d in out if d["k"] != "f"] == [d for d in rows if d["k"] != "f"]       # header, event, cut, footer: all, as they were
    assert [d["r"] for d in out] == sorted(d["r"] for d in out) and out[-1]["k"] == "ftr"   # order kept
    for tool in ("endo", "drill"):
        t = [d["tMono"] for d in out if d["k"] == "f" and d["tool"] == tool]
        full = [d["tMono"] for d in rows if d["k"] == "f" and d["tool"] == tool]
        assert t[0] == full[0] and t[-1] == full[-1]                                     # first and last frame of the tool
        near = lambda x: any(abs(x - (1000.0 + m * 1000.0)) <= 500.0 + 1.0 for m in (8.0, 7.5, 9.0))
        assert all(x in t for x in full if near(x))                                       # full detail around the cut and the trigger edges
        far = [b - a for a, b in zip(t, t[1:]) if not near(a) and not near(b)][:-1]      # the tool's last frame is kept whatever its distance
        assert min(far) >= 100.0 - 1e-6 and max(far) <= 120.0 + 1e-6                      # 10 Hz elsewhere (50 Hz rows: every 5th)
        assert res["rows"][tool] == [len(full), len(t)] and len(t) < 0.5 * len(full)


def test_small_session_is_left_alone_and_sessions_tree_refused(tmp_path):
    src = tmp_path / "s_y.jsonl"; session(src, seconds=2.0)
    res = T.thin(src, tmp_path / "out", min_mb=80.0)
    assert res == {"path": str(src), "thinned": False, "mb": res["mb"], "rows": {}} and not (tmp_path / "out").exists()
    with pytest.raises(SystemExit):
        T.thin(src, tmp_path / "viewer-data" / "sessions" / "robot", min_mb=0.0)
