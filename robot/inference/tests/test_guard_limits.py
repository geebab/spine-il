"""eval/guard_pairs.py run_limits: a run's own travel limits and its tip-separation stop (guard.json)."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "eval"))

import guard_pairs as G  # noqa: E402


def limits(tmp_path, monkeypatch, content, sep_default=None):
    monkeypatch.setattr(G, "RO", tmp_path)
    d = tmp_path / "RUN"; d.mkdir(exist_ok=True)
    if content is not None:
        (d / "guard.json").write_text(content if isinstance(content, str) else json.dumps(content))
    return G.run_limits("RUN", 30.0) if sep_default is None else G.run_limits("RUN", 30.0, sep_default)


def test_the_tip_stop_is_5_mm_for_every_run():   # user, 10-07: "5mm로 완전히 변경"
    assert G.MIN_SEP_MM == 5.0


def test_a_run_without_a_file_keeps_the_defaults(tmp_path, monkeypatch):
    assert limits(tmp_path, monkeypatch, None) == {"endo": 30.0, "drill": 30.0, "sep": 5.0}
    assert limits(tmp_path, monkeypatch, "not json") == {"endo": 30.0, "drill": 30.0, "sep": 5.0}
    assert limits(tmp_path, monkeypatch, "[1, 2]") == {"endo": 30.0, "drill": 30.0, "sep": 5.0}


def test_travel_limits_only_widen_and_are_capped(tmp_path, monkeypatch):
    assert limits(tmp_path, monkeypatch, {"endo_mm": 95, "drill_mm": 115}) == {"endo": 95.0, "drill": 115.0, "sep": 5.0}
    assert limits(tmp_path, monkeypatch, {"endo_mm": 5, "drill_mm": 500})["endo"] == 30.0
    assert limits(tmp_path, monkeypatch, {"endo_mm": 5, "drill_mm": 500})["drill"] == G.MAX_RUN_LIMIT_MM


def test_the_tip_stop_can_only_be_lowered_and_not_below_the_floor(tmp_path, monkeypatch):
    assert limits(tmp_path, monkeypatch, {"endo_mm": 95, "drill_mm": 115, "min_sep_mm": 4})["sep"] == 4.0
    assert limits(tmp_path, monkeypatch, {"min_sep_mm": 4}) == {"endo": 30.0, "drill": 30.0, "sep": 4.0}   # without travel limits
    assert limits(tmp_path, monkeypatch, {"min_sep_mm": 25})["sep"] == 5.0           # never above the guard's own
    assert limits(tmp_path, monkeypatch, {"min_sep_mm": 0.5})["sep"] == G.MIN_SEP_FLOOR_MM
    assert limits(tmp_path, monkeypatch, {"min_sep_mm": "x", "endo_mm": 95, "drill_mm": 115}) == {"endo": 95.0, "drill": 115.0, "sep": 5.0}
    # a guard started with another value of its own (--min-sep-mm 10): a run's 5 still lowers it
    assert limits(tmp_path, monkeypatch, {"min_sep_mm": 5}, sep_default=10.0)["sep"] == 5.0
    assert limits(tmp_path, monkeypatch, {"min_sep_mm": 25}, sep_default=10.0)["sep"] == 10.0
