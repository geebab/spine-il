"""Latency analysis helpers — synthetic trajectories with a known delay."""

import numpy as np

from lerobot_robot_spinearm import latency as L


def traj(t_ms):
    t = t_ms / 1000.0
    return np.column_stack([20 * np.sin(2 * np.pi * 0.7 * t) + 5 * np.sin(2 * np.pi * 1.9 * t),
                            10 * np.cos(2 * np.pi * 0.4 * t), 3 * np.sin(2 * np.pi * 1.3 * t)])


def test_delay_recovered_to_a_millisecond():
    rng = np.random.default_rng(0)
    t_ref = np.arange(0, 20000, 11.1) + rng.uniform(0, 1, 1802)
    t_obs = np.arange(3, 20000, 11.1)
    x_obs = traj(t_obs - 37.0) + np.array([4.0, -2.0, 1.0]) + rng.normal(scale=0.2, size=(t_obs.size, 3))
    r = L.delay_fit(t_ref, traj(t_ref), t_obs, x_obs, max_ms=300)
    assert abs(r["tau_ms"] - 37.0) < 1.0 and r["rms_at_tau"] < 0.5 < r["rms_at_0"]


def test_interp_gaps_and_errors():
    t = np.array([0.0, 10.0, 20.0, 200.0])
    x = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0.0]])
    y = L.interp3([5.0, 15.0, 100.0, 300.0], t, x)
    assert y[0, 0] == 5 and y[1, 0] == 15 and np.isnan(y[2]).all() and np.isnan(y[3]).all()
    e = L.err_stats(np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [np.nan, 0, 0]]))
    assert e["n"] == 2 and e["max"] == 5.0 and e["bias_xyz"] == [1.5, 2.0, 0.0]


def test_step_metrics_first_order_with_dead_time():
    t = np.arange(0, 3000, 11.1)
    tc = 1000.0
    x = np.zeros((t.size, 3))
    on = t >= tc + 60
    x[on, 0] = 15 * (1 - np.exp(-(t[on] - tc - 60) / 100.0))
    t_cmd = np.array([0.0, tc, 2999.0])
    x_cmd = np.array([[0, 0, 0], [15, 0, 0], [15, 0, 0.0]])
    m = L.step_metrics(t_cmd, x_cmd, t, x, 0, tc, 2990.0, 15.0)
    assert 60 <= m["dead_ms"] <= 80 and 200 <= m["rise_ms"] <= 240 and abs(m["ss_err_mm"]) < 0.1


def test_still_windows_and_sine_fit():
    t = np.arange(0, 4000, 10.0)
    x = np.zeros((t.size, 3))
    x[t > 2000, 0] = (t[t > 2000] - 2000) * 0.05  # 50 mm/s after 2 s
    w = L.still_windows(t, x)
    assert len(w) == 1 and w[0][0] == 0 and 1900 <= w[0][1] <= 2010
    amp, ph = L.sine_fit(t, 7 * np.sin(2 * np.pi * 1.0 * t / 1000 - np.radians(30)), 1.0)
    assert abs(amp - 7) < 1e-6 and abs(ph + 30) < 1e-6


def test_pick_robot_relative_identity():
    from lerobot_robot_spinearm.tap import pick_robot

    ref = np.array([0.0, 0.0, 0.5])
    P = lambda x: {"tip": ref + np.array([x, 0, 0]), "tool": "t"}  # noqa: E731
    r, h, s = pick_robot({"endo": P(0.047), "endo2": P(0.185)}, ref)  # re-acquired 47 mm off, hand far: keep going
    assert s == "ok" and r["tip"][0] == 0.047
    assert pick_robot({"endo": P(0.03), "endo2": P(0.06)}, ref)[2] == "ambiguous"  # both near
    assert pick_robot({"endo": P(0.02), "endo2": None}, ref)[2] == "hand_lost"
    assert pick_robot({"endo": P(0.17), "endo2": None}, ref)[2] == "robot_lost"
    assert pick_robot({"endo": P(0.08), "endo2": None}, ref)[2] == "ambiguous"
