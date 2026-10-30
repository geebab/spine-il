import numpy as np

from spine_il_converter.geometry import quat_to_matrix
from spine_il_converter.resample import interpolate_track, match_nearest, phase_locked_grid
from spine_il_converter.session import PoseTrack


def rz(a):
    c, s = np.cos(a), np.sin(a)
    out = np.zeros(np.shape(a) + (3, 3))
    out[..., 0, 0], out[..., 0, 1], out[..., 1, 0], out[..., 1, 1], out[..., 2, 2] = c, -s, s, c, 1
    return out


def track(t):
    return PoseTrack(
        t_src=t,
        tip=np.stack([t * 0.1, -t * 0.05, np.full_like(t, 3.0)], axis=1),
        rot=rz(t * 1e-3),
        on=(t > 500).astype(float),
        n_rows=len(t),
        n_lost=0,
        n_up_missing=0,
    )


def test_linear_and_slerp_are_exact_for_constant_rates():
    t = np.sort(np.random.default_rng(1).uniform(0, 1000, 90))
    t[0], t[-1] = 0.0, 1000.0
    grid = np.linspace(0, 1000, 31)
    rs = interpolate_track(track(t), grid, max_gap_ms=250)
    assert rs.valid.all()
    np.testing.assert_allclose(rs.tip[:, 0], grid * 0.1, atol=1e-9)
    np.testing.assert_allclose(rs.rot, rz(grid * 1e-3), atol=1e-12)


def test_gap_longer_than_limit_is_invalid_and_short_gap_is_bridged():
    t = np.concatenate([np.arange(0, 300, 11.1), np.arange(700, 1000, 11.1)])  # 400 ms hole
    rs = interpolate_track(track(t), np.array([150.0, 500.0, 800.0]), max_gap_ms=250)
    assert rs.valid.tolist() == [True, False, True]
    t2 = np.concatenate([np.arange(0, 300, 11.1), np.arange(450, 1000, 11.1)])  # ~150 ms hole
    rs2 = interpolate_track(track(t2), np.array([100.0, 380.0]), max_gap_ms=250)
    assert rs2.valid.all() and rs2.bridged.tolist() == [False, True]


def test_out_of_range_and_exact_endpoints():
    t = np.array([0.0, 10.0, 20.0])
    rs = interpolate_track(track(t), np.array([-1.0, 0.0, 20.0, 21.0]), max_gap_ms=250)
    assert rs.valid.tolist() == [False, True, True, False]


def test_drill_on_is_zero_order_hold():
    t = np.array([0.0, 400.0, 600.0])
    rs = interpolate_track(track(t), np.array([450.0, 599.0, 600.0]), max_gap_ms=250)
    assert rs.on.tolist() == [0.0, 0.0, 1.0]


def test_match_nearest_tolerance():
    ref = np.array([100.0, 0.0, 50.0])
    idx = match_nearest(np.array([1.0, 26.0, 49.0, 130.0]), ref, tol_ms=20)  # 26 is 24 ms from 50
    assert idx.tolist() == [1, -1, 2, -1]


def test_grid_is_phase_locked_to_first_capture():
    anchors = np.array([3.0, 36.4, 69.9, 103.1])
    grid = phase_locked_grid(anchors, t_min=0.0, t_max=110.0, fps=30)
    np.testing.assert_allclose(grid, 3.0 + np.arange(4) * 1000 / 30)
    assert quat_to_matrix(np.array([1.0, 0, 0, 0])).shape == (3, 3)
