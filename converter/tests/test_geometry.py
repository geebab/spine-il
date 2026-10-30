import numpy as np
import pytest

from spine_il_converter.geometry import (
    GeometryError,
    basal_frame,
    matrix_to_quat,
    points_from_frame,
    points_to_frame,
    quat_to_matrix,
    rot6d,
    rot6d_to_matrix,
    slerp,
    tool_rotation,
)

rng = np.random.default_rng(0)


def random_rotations(n):
    return quat_to_matrix(rng.normal(size=(n, 4)))


def test_basal_frame_definition():
    entry, target = np.array([45.0, 95.0, 160.0]), np.array([38.0, 60.0, 165.0])
    origin, R = basal_frame(entry, target)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(R[:, 2], (target - entry) / np.linalg.norm(target - entry))
    assert R[2, 0] > 0  # x axis points cranial (+S)
    assert abs(R[:, 0] @ R[:, 2]) < 1e-12
    np.testing.assert_allclose(points_to_frame(target, origin, R), [0, 0, np.linalg.norm(target - entry)], atol=1e-12)
    pts = rng.normal(size=(20, 3)) * 50
    np.testing.assert_allclose(points_from_frame(points_to_frame(pts, origin, R), origin, R), pts, atol=1e-10)


def test_basal_frame_rejects_cranial_vector():
    with pytest.raises(GeometryError):
        basal_frame([0, 0, 0], [0, 0, 10])


def test_rot6d_roundtrip():
    R = random_rotations(500)
    np.testing.assert_allclose(rot6d_to_matrix(rot6d(R)), R, atol=1e-12)
    assert rot6d(R).shape == (500, 6)
    np.testing.assert_allclose(rot6d(R)[:, :3], R[:, :, 0])  # column-major layout


def test_quaternion_roundtrip_including_half_turns():
    R = np.concatenate([random_rotations(500), [np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1]), np.diag([-1.0, -1, 1])]])
    q = matrix_to_quat(R)
    assert (q[:, 0] >= 0).all()
    np.testing.assert_allclose(quat_to_matrix(q), R, atol=1e-12)


def test_slerp_constant_rate_and_hemisphere():
    def rz(a):
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

    q0, q1 = matrix_to_quat(rz(0.2)), matrix_to_quat(rz(1.4))
    for t in (0.0, 0.25, 0.5, 1.0):
        np.testing.assert_allclose(quat_to_matrix(slerp(q0, q1, t)), rz(0.2 + 1.2 * t), atol=1e-12)
    np.testing.assert_allclose(quat_to_matrix(slerp(q0, -q1, 0.5)), rz(0.8), atol=1e-12)  # q ≡ −q


def test_tool_rotation_axes_and_fallback():
    d = np.array([0.3, -0.9, 0.2])
    up = np.array([0.0, 0.0, 1.0])
    R = tool_rotation(d, up)
    np.testing.assert_allclose(R[:, 2], d / np.linalg.norm(d))
    assert abs(R[:, 0] @ R[:, 2]) < 1e-12 and R[2, 0] > 0
    Rs = tool_rotation(np.array([[0.0, 0, 1], [0, 1, 0]]), np.array([[0.0, 0, 1], [0, 1, 0]]))  # up ∥ dir
    for Ri in Rs:
        np.testing.assert_allclose(Ri.T @ Ri, np.eye(3), atol=1e-12)
