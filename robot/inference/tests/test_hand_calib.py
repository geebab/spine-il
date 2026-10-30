"""hand_calib.py: a recording header → the hand tool calibration files it names (two days' calibrations in a made-up home)."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import hand_calib as HC  # noqa: E402

TIP = {"a": {"endo": [-153.2, 22.6, -59.1], "drill": [-126.4, 57.6, 110.7]},
       "b": {"endo": [-153.2, 22.6, -59.1], "drill": [-127.0, 58.8, 115.6]}}   # the same endoscope tip, another drill tip


def put(d: Path, tips: dict, note: str) -> dict:
    """Write the two calibration files of one day → {tool: sha256's first 12 hex}."""
    d.mkdir(parents=True)
    out = {}
    for tool, name in HC.FILES.items():
        raw = json.dumps({"tip_local": tips[tool], "meta": {"note": note}}).encode()
        (d / name).write_bytes(raw)
        out[tool] = hashlib.sha256(raw).hexdigest()[:12]
    return out


@pytest.fixture
def home(tmp_path):
    sha = {"a": put(tmp_path / "il-robot/calib/tracker_models/hand_20260928", TIP["a"], "a"),
           "b": put(tmp_path / "il-robot/runs/hand_incoming_20261006", TIP["b"], "b")}
    return tmp_path, sha


def hdr(sha: dict | None, tips: dict) -> dict:
    return {"k": "hdr", "toolCalib": {t: {**({"sha256_12": sha[t]} if sha else {}), "tipLocal": tips[t]} for t in HC.FILES}}


def test_each_days_recording_gets_its_own_files(home):
    root, sha = home
    a = HC.for_header(hdr(sha["a"], TIP["a"]), root)
    b = HC.for_header(hdr(sha["b"], TIP["b"]), root)
    assert a["drill"].parent.name == "hand_20260928" and b["drill"].parent.name == "hand_incoming_20261006"
    assert a["endo"].name == "tool_calib.json" and b["drill"].name == "tool_calib_drill.json"
    assert a["endo"] != b["endo"]   # the endoscope tips are equal, the files (hashes) are not: the hash decides


def test_the_archive_is_preferred_over_the_staging_folder(home):
    root, sha = home
    put(root / "il-robot/calib/tracker_models/hand_20261006", TIP["b"], "b")   # the same bytes, archived
    assert HC.for_header(hdr(sha["b"], TIP["b"]), root)["drill"].parent.name == "hand_20261006"


def test_a_header_without_hashes_is_matched_by_its_tip(home):
    root, _ = home
    got = HC.for_header(hdr(None, TIP["b"]), root)
    assert got["drill"].parent.name == "hand_incoming_20261006"
    assert got["endo"].parent.name == "hand_20260928"   # both days have this endoscope tip: the first known file


def test_an_unknown_calibration_is_an_error_not_a_guess(home):
    root, sha = home
    with pytest.raises(LookupError, match="tool_calib_drill.json"):
        HC.for_header(hdr({"endo": sha["a"]["endo"], "drill": "0" * 12}, TIP["a"]), root)   # a hash no file has: the tip is not tried
    with pytest.raises(LookupError):
        HC.for_header({"k": "hdr"}, root)   # a robot session's header names the arms' blocks, or nothing


def test_for_recording_reads_the_header_row(home, tmp_path):
    root, sha = home
    f = tmp_path / "s.jsonl"
    f.write_text(json.dumps({"k": "f", "tool": "endo"}) + "\n" + json.dumps(hdr(sha["b"], TIP["b"])) + "\n")   # a late header
    assert HC.for_recording(f, root)["drill"].parent.name == "hand_incoming_20261006"
    g = tmp_path / "n.jsonl"; g.write_text(json.dumps({"k": "f"}) + "\n")
    with pytest.raises(LookupError, match="no header"):
        HC.for_recording(g, root)
