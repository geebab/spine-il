"""Hand-eye (eye-to-hand, constellation on the arm) — synthetic poses, no hardware."""

import numpy as np
import pytest

from lerobot_robot_spinearm import handeye as he

PTS = np.array([[0, 0, 0], [145.8, 0, 0], [32.7, 31.1, 0], [89.0, -21.0, 1.0]]) / 1000.0
TIP = np.array([-154.9, 11.3, -59.2]) / 1000.0


def rand_T(rng, rot_deg=180.0, trans_m=0.3):
    v = rng.normal(size=3)
    v *= np.radians(rng.uniform(0, rot_deg)) / np.linalg.norm(v)
    return he.compose(he.rotvec_to_R(v), rng.uniform(-trans_m, trans_m, 3))


def scene(rng, n=25, noise_mm=0.0, noise_deg=0.0):
    X = rand_T(rng, 90, 0.06)
    Z = rand_T(rng, 180, 0.5)
    fk = [rand_T(rng, 40, 0.1) @ he.compose(np.eye(3), [0.25, 0, 0.15]) for _ in range(n)]
    obs = []
    for F in fk:
        C = he.inv(Z) @ F @ X
        if noise_mm or noise_deg:
            dv = rng.normal(size=3)
            C = C @ he.compose(he.rotvec_to_R(dv / np.linalg.norm(dv) * np.radians(noise_deg)),
                               rng.normal(scale=noise_mm / 1000, size=3))
        obs.append(C)
    return X, Z, fk, obs


def test_exact_recovery():
    X, Z, fk, obs = scene(np.random.default_rng(1))
    s = he.solve(fk, obs, PTS, TIP)
    assert np.abs(s["X"] - X).max() < 1e-6 and np.abs(s["Z"] - Z).max() < 1e-6
    summ = he.summarize(he.residuals(s["X"], s["Z"], fk, obs, PTS, TIP))
    assert summ["tip_mm"]["max"] < 1e-3


def test_noisy_recovery_and_outlier_rejection():
    rng = np.random.default_rng(2)
    X, Z, fk, obs = scene(rng, n=30, noise_mm=0.3, noise_deg=0.1)
    obs[4] = obs[4] @ he.compose(np.eye(3), [0.02, 0, 0])  # one 20 mm mislabel
    s = he.solve(fk, obs, PTS, TIP)
    assert 4 not in s["keep"]
    assert np.linalg.norm(s["Z"][:3, 3] - Z[:3, 3]) * 1000 < 1.0
    assert he.R_angle_deg(s["Z"][:3, :3].T @ Z[:3, :3]) < 0.2
    assert np.linalg.norm(s["X"][:3, 3] - X[:3, 3]) * 1000 < 1.0


def test_urdf_tip_frame_matches_composition(tmp_path):
    pytest.importorskip("placo")
    from lerobot_robot_spinearm.ee import URDF, MarkerKinematics

    X = rand_T(np.random.default_rng(3), 90, 0.06)
    path = he.urdf_with_tip(URDF, X, TIP, tmp_path / "tip.urdf")
    link = MarkerKinematics()
    tip = MarkerKinematics(urdf=path, frame=he.TIP_FRAME)
    q = [10, -30, 40, 20, 15, -10]
    want = link.fk(q) @ X @ he.compose(np.eye(3), TIP)
    assert np.abs(tip.fk(q) - want).max() < 1e-6


def test_joint_offsets_recovered_with_real_kinematics():
    pytest.importorskip("placo")
    from lerobot_robot_spinearm.ee import MarkerKinematics

    kin = MarkerKinematics(bounds=None)
    rng = np.random.default_rng(4)
    X = rand_T(rng, 60, 0.05)
    Z = rand_T(rng, 180, 0.5)
    true = np.array([0, 2.5, -8.0, 1.5, 4.8, 0])
    q0 = np.array([0, -55, 20, 90, 0, -59.0])
    q = [q0 + rng.uniform(-1, 1, 6) * [10, 8, 8, 15, 20, 15] for _ in range(30)]
    obs = [he.inv(Z) @ kin.fk(qi + true) @ X for qi in q]
    s = he.solve([kin.fk(qi) for qi in q], obs, PTS, TIP, q_deg=q, fk_fn=kin.fk)
    assert np.abs(s["dq"][1:5] - true[1:5]).max() < 0.05
    summ = he.summarize(he.residuals(s["X"], s["Z"], [kin.fk(qi + s["dq"]) for qi in q], obs, PTS, TIP))
    assert summ["tip_mm"]["max"] < 0.05
