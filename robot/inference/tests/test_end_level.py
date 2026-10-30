"""end_level.py — the level at which a progress prediction ends the drilling, on made-up demonstrations: the loop's rule
walked along a demonstration, the two losses the level trades, the stand-in read-out, the choice under several predictors."""
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import end_level as EL      # noqa: E402
import rollout as RO        # noqa: E402


def demo(n=600, first=60, last=480, seed=0):
    """Progress as a demonstration has it: 0 until the first cut, rising (unevenly) to 1 at the last cut, 1 to the end."""
    rng = np.random.default_rng(seed)
    inc = rng.gamma(0.6, 1.0, last - first); inc[-1] += 0.2
    p = np.zeros(n); p[first + 1:last + 1] = np.cumsum(inc) / inc.sum(); p[last:] = 1.0
    return {"progress": p, "state": np.zeros((n, 15)), "phase": np.where(np.arange(n) <= last, 4, 5), "first_cut": first + 1, "last_cut": last,
            "session": f"s_2026{'0928' if seed % 2 else '1006'}T0000_x", "day": "20260928" if seed % 2 else "20261006"}


EPS = {i: demo(seed=i, last=420 + 15 * i) for i in range(6)}


def test_targets_are_the_progress_of_the_frames_ahead():
    p = np.linspace(0, 1, 11)
    t = EL.targets(p, 4)
    assert t.shape == (11, 4) and np.allclose(t[0], p[1:5]) and np.allclose(t[8], [p[9], p[10], p[10], p[10]]) and np.allclose(t[10], 1.0)


def test_a_perfect_predictor_ends_where_the_progress_passes_the_level():
    d = EPS[0]
    ch = EL.oracle({0: d}, 16)[0]
    for L in (0.6, 0.9, 0.95):
        k = EL.walk(ch, d["progress"], 0, 16, L)
        cross = int(np.nonzero(d["progress"] > L)[0][0])
        # the chunk's value j is the progress of frame t + j + 1: the steps before it are executed, so the drilling ends on
        # the frame before the first one above the level — or up to one grid step later (the rule is looked at every 16)
        assert cross - 1 <= k <= cross + 16
        assert d["progress"][min(k + 1, len(d["progress"]) - 1)] > L
    assert EL.walk(ch, d["progress"], 0, 16, 0.999) is not None      # the label reaches 1: every level below fires


def test_an_end_before_the_first_cut_counts_and_is_worth_nothing():
    """The loop takes the policy's end from the first frame of its part (dry run of 10-09: 0 mm³ removed) — so does the walk."""
    d = EPS[1]
    high = {t: np.ones(16) for t in range(0, len(d["progress"]), 4)}                # a predictor that says "done" from the start
    k = EL.walk(high, d["progress"], 0, 16, 0.95)
    assert k == 0 and d["progress"][k] == 0.0
    s = EL.sweep({1: d}, {1: high}, 16)
    assert s["best"]["U"] == 0.0 and s["best"]["missed"] == 0.0                     # ended every time, with nothing removed
    assert EL.walk(high, d["progress"], d["first_cut"], 16, 0.95) >= d["first_cut"]  # a later start of the count is still possible


def test_a_prediction_that_falls_short_misses_above_its_ceiling():
    """The loss on the high side: a predictor compressed to 0.9 of the truth never passes 0.92 — those runs count 0."""
    pred = {e: {t: 0.9 * v for t, v in EL.oracle({e: d}, 16)[e].items()} for e, d in EPS.items()}
    s = EL.sweep(EPS, pred, 16)
    row = {r["level"]: r for r in s["rows"]}
    assert row[0.92]["missed"] == 1.0 and row[0.92]["U"] == 0.0
    assert row[0.85]["missed"] == 0.0 and row[0.85]["removed_when_ended"] > 0.85 / 0.9 - 0.01     # ends where the TRUE progress is ~0.94
    assert 0.85 <= s["best"]["level"] < 0.9 and s["best"]["U"] > 0.94                              # the best level sits just under the ceiling


def test_the_perfect_predictor_wants_the_highest_level_and_loses_one_minus_it():
    s = EL.sweep(EPS, EL.oracle(EPS, 16), 16)
    assert s["best"]["level"] >= 0.98 and s["best"]["missed"] == 0.0 and s["best"]["U"] > 0.98
    row = {r["level"]: r for r in s["rows"]}
    assert 0.75 < row[0.8]["U"] < 0.9 and row[0.8]["end_vs_last_cut_s"]["median"] < 0          # early: before the last cut
    assert all(a <= b + 1e-9 for a, b in zip(s["U"], s["U"][1:]))                              # no risk: higher is never worse


def test_the_perfect_predictor_does_not_hold_an_array_per_frame():
    """10-09: every row kept its own (n, horizon) array alive — 4 GB on the real dataset, and the machine ran out of memory."""
    d = demo(n=3000, last=2800)
    ch = EL.oracle({0: d}, 16)[0]
    bases = {id(v.base) for v in ch.values()}
    assert len(ch) == 3000 and len(bases) == 1 and next(iter(ch.values())).base.nbytes < 3000 * 16 * 8 * 1.1


def test_minimax_takes_the_level_no_predictor_loses_much_at():
    perfect = EL.sweep(EPS, EL.oracle(EPS, 16), 16)
    short = EL.sweep(EPS, {e: {t: 0.9 * v for t, v in EL.oracle({e: d}, 16)[e].items()} for e, d in EPS.items()}, 16)
    m = EL.minimax({"perfect": perfect, "short": short})
    assert m["level"] < 0.9                                   # above 0.9 the short predictor loses everything
    assert m["loss"]["short"] < 0.02 and 0.05 < m["loss"]["perfect"] < 0.2 and m["worst_loss"] == max(m["loss"].values())
    assert m["own_best"]["perfect"] > m["level"] >= m["own_best"]["short"] - 0.05


def test_the_stand_in_reads_progress_out_of_features_it_has_not_seen():
    """Features that carry the progress (plus noise): the held-out read-out follows it; features that carry nothing: it
    predicts the mean, never reaches a high level, and the best level drops."""
    rng = np.random.default_rng(1)
    X, E, K = [], [], []
    for e, d in EPS.items():
        k = np.arange(0, len(d["progress"]), 3)
        f = np.c_[d["progress"][k] + rng.normal(0, 0.02, len(k)), rng.normal(0, 1, (len(k), 7))]
        X.append(f); E += [e] * len(k); K += list(k)
    feat = {"X": np.concatenate(X).astype(np.float32), "ep": np.asarray(E), "k": np.asarray(K)}
    pred, info = EL.probe(EPS, feat, 16)
    assert info["folds"] == 6 and info["mae"] < 0.05 and set(pred) == set(EPS)
    s = EL.sweep(EPS, pred, 16, offsets=[0, 3, 6, 9])
    assert s["best"]["level"] >= 0.9 and s["best"]["missed"] <= 0.1
    blind = {"X": rng.normal(0, 1, feat["X"].shape).astype(np.float32), "ep": feat["ep"], "k": feat["k"]}
    pred0, info0 = EL.probe(EPS, blind, 16)
    assert info0["mae"] > 0.25 and info0["highest_prediction_per_episode"]["max"] < 0.9
    s0 = EL.sweep(EPS, pred0, 16, offsets=[0, 3, 6, 9])
    assert s0["best"]["level"] < s["best"]["level"] and s0["best"]["U"] < s["best"]["U"]
    by_day, info_d = EL.probe(EPS, feat, 16, group=lambda e: EPS[e]["day"])
    assert info_d["folds"] == 2 and set(by_day) == set(EPS)


def test_the_real_dataset_reads(tmp_path):
    ds = Path.home() / "il-robot/datasets/spine_hand24_replay_trim_water_progress"
    if not ds.exists():
        pytest.skip("dataset not on this machine")
    eps = EL.load(ds)
    assert len(eps) == 25 and sum(len(d["progress"]) for d in eps.values()) == 47233
    d = eps[18]
    assert d["state"].shape == (2034, 15) and 0 <= d["progress"].min() and d["progress"].max() == pytest.approx(1.0, abs=1e-6)
    assert d["first_cut"] < d["last_cut"] and d["day"] == "20261006" and eps[2]["day"] == "20260928"
    vm = EL.video_map(ds)
    assert vm[0][1] == 0 and vm[0][2] == 524 and vm[1][1] == 524 and vm[18][2] == 2034
    got = [(e, k, img.shape) for e, k, img in EL.frames(ds, {1}, stride=300)]
    assert got == [(1, 0, (256, 256, 3)), (1, 300, (256, 256, 3)), (1, 600, (256, 256, 3))]



def test_open_loop_scores_of_answers_that_are_the_demonstration_and_of_answers_that_stand_still():
    """The scores openloop_eval.py gives a local model, from a server's answers: a model that answers with the
    demonstration's own continuation has gain 1 and direction 1, one that answers "stay where you are" has gain 0."""
    rng = np.random.default_rng(3)
    d = demo(n=400, first=40, last=330)
    st = np.zeros((400, 15)); st[:, 7:10] = np.cumsum(rng.normal(0.2, 0.3, (400, 3)), axis=0); st[:, 0:3] = np.cumsum(rng.normal(0, 0.1, (400, 3)), axis=0)
    st[41:331, 14] = 1.0; d["state"] = st
    eps = {0: d}

    def answer(t, still=False, shift=0.0):
        idx = np.minimum(np.arange(t + 1, t + 41), 399)
        rows = np.repeat(st[t][None], 40, 0) if still else st[idx].copy()
        return np.c_[rows, np.clip(d["progress"][idx] + shift, 0, 1)]
    same = EL.open_loop(eps, {0: {t: answer(t) for t in range(0, 360, 16)}})
    assert same["drilling"]["drill_H16"]["gain_slope"] == pytest.approx(1.0, abs=1e-6) and same["drilling"]["drill_H16"]["cos_median"] == pytest.approx(1.0, abs=1e-6)
    assert same["drilling"]["trigger_agree_16"] == 1.0 and same["drilling"]["progress_error"]["mean_abs"] == pytest.approx(0.0, abs=1e-9)
    assert "approach" in same and same["approach"]["drill_H40"]["gain_slope"] == pytest.approx(1.0, abs=1e-6)
    still = EL.open_loop(eps, {0: {t: answer(t, still=True, shift=-0.2) for t in range(0, 360, 16)}})
    assert still["drilling"]["drill_H16"]["gain_slope"] == pytest.approx(0.0, abs=1e-9) and still["drilling"]["drill_H16"]["pred_mm_median"] == 0.0
    assert still["drilling"]["progress_error"]["mean"] < -0.1
    assert still["progress_error_where_true_progress_is_0.9_or_more"]["mean"] < -0.05      # short near the end: what moves the best level down
