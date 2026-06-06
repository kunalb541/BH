"""Principal angle pipeline unit tests. Pre-reg §10."""

import numpy as np
import pytest

from src.principal_angles import (
    max_principal_angle,
    observer_matrix_zscore,
    principal_angles,
    top_r_subspace,
)


def random_orthonormal(n: int, r: int, rng: np.random.Generator) -> np.ndarray:
    G = rng.standard_normal(size=(n, r))
    Q, _ = np.linalg.qr(G)
    return Q


def test_identical_subspace_zero_angle():
    rng = np.random.default_rng(0)
    Q = random_orthonormal(8, 2, rng)
    angles = principal_angles(Q, Q)
    # arccos near 1 amplifies machine epsilon: tolerance ~ sqrt(2*eps) ~ 1e-7
    assert np.all(angles < 1e-6), f"identical subspace gave angles {angles}"


def test_orthogonal_subspaces_pi_over_two():
    """Two disjoint orthogonal subspaces in R^8 should give all angles = pi/2."""
    rng = np.random.default_rng(1)
    G = rng.standard_normal(size=(8, 8))
    Q, _ = np.linalg.qr(G)
    A = Q[:, :2]
    B = Q[:, 2:4]
    angles = principal_angles(A, B)
    assert np.allclose(angles, np.pi / 2, atol=1e-8)


def test_one_shared_direction_one_orthogonal():
    """If A and B share one direction and the other is orthogonal,
    angles = (0, pi/2).
    """
    rng = np.random.default_rng(2)
    G = rng.standard_normal(size=(6, 6))
    Q, _ = np.linalg.qr(G)
    shared = Q[:, 0:1]
    A = np.concatenate([shared, Q[:, 1:2]], axis=1)
    B = np.concatenate([shared, Q[:, 2:3]], axis=1)
    angles = principal_angles(A, B)
    # arccos amplifies near 1; loosen near-zero tolerance to 1e-6
    assert abs(angles[0]) < 1e-6
    assert abs(angles[1] - np.pi / 2) < 1e-8


def test_angles_invariant_under_basis_rotation():
    rng = np.random.default_rng(3)
    A = random_orthonormal(7, 2, rng)
    B = random_orthonormal(7, 2, rng)
    angles_AB = principal_angles(A, B)
    R = random_orthonormal(2, 2, rng)
    A2 = A @ R
    angles_A2B = principal_angles(A2, B)
    assert np.allclose(angles_AB, angles_A2B, atol=1e-10)


def test_top_r_subspace_orthonormal_columns():
    rng = np.random.default_rng(4)
    M = rng.standard_normal(size=(20, 8))
    V = top_r_subspace(M, 2)
    assert V.shape == (8, 2)
    G = V.T @ V
    assert np.allclose(G, np.eye(2), atol=1e-10)


def test_top_r_subspace_recovers_planted_signal():
    """If M = (random scores) * v1^T + (smaller random scores) * v2^T + tiny noise,
    top_r_subspace should recover the span of {v1, v2}.
    """
    rng = np.random.default_rng(5)
    n = 9
    G = rng.standard_normal(size=(n, n))
    Q, _ = np.linalg.qr(G)
    v1 = Q[:, 0]
    v2 = Q[:, 1]
    m = 30
    a = rng.standard_normal(m) * 5
    b = rng.standard_normal(m) * 1
    noise = rng.standard_normal(size=(m, n)) * 0.01
    M = np.outer(a, v1) + np.outer(b, v2) + noise
    V = top_r_subspace(M, 2)
    # angle between span(V) and span([v1, v2]) should be tiny
    truth = np.stack([v1, v2], axis=1)
    angles = principal_angles(V, truth)
    assert np.all(angles < 0.1), f"recovered subspace angles too large: {angles}"


def test_observer_matrix_zscore_per_row():
    rng = np.random.default_rng(6)
    cells = [rng.standard_normal(7) * 5 + 3 for _ in range(10)]
    Z = observer_matrix_zscore(cells)
    assert Z.shape == (10, 7)
    # Each row should have ~zero mean and unit std
    for r in range(10):
        assert abs(float(Z[r].mean())) < 1e-10
        assert abs(float(Z[r].std()) - 1.0) < 1e-10


def test_observer_matrix_zscore_handles_constant_row():
    cells = [np.full(5, 2.0), np.array([1.0, 2.0, 3.0, 4.0, 5.0])]
    Z = observer_matrix_zscore(cells)
    # constant row z-score is zero (we use sd_safe=1 to avoid divide-by-zero)
    assert np.all(Z[0] == 0.0)
