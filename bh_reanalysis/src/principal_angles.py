"""Principal angles between subspaces.

Standard reference: Golub & Van Loan, Matrix Computations, §6.4.3.

For two matrices A (n x r1), B (n x r2) with orthonormal columns,
principal angles theta_1 <= ... <= theta_min(r1, r2) satisfy
  cos(theta_k) = sigma_k(A^T B)
where sigma_k are singular values in descending order.

We expose:
  - top_r_subspace(M, r): orthonormal basis (n x r) for the top-r right singular
    subspace of M (cell-axis x site-axis matrix). Effectively the right-singular
    vectors of M, top r of them. Lives in site-space (R^L).
  - principal_angles(A, B): vector of principal angles, ascending.
  - max_principal_angle(A, B): the worst-case angle (largest principal angle).
"""

from __future__ import annotations

import numpy as np


def top_r_subspace(M: np.ndarray, r: int) -> np.ndarray:
    """Return n x r orthonormal basis for the top-r right-singular subspace of M.

    M is m x n (cells x sites). The right-singular vectors live in R^n (site
    space). Returns the top-r right-singular vectors as columns of an n x r
    matrix. If rank(M) < r, the trailing columns come from any orthogonal
    completion via QR; this is well-defined and the returned matrix is always
    n x r with orthonormal columns.
    """
    if r < 1:
        raise ValueError(f"r must be >= 1, got {r}")
    n = M.shape[1]
    if r > n:
        raise ValueError(f"r={r} exceeds site-space dimension n={n}")
    U, s, Vt = np.linalg.svd(M, full_matrices=True)
    # Vt rows are right-singular vectors; take top-r
    return Vt[:r, :].T  # n x r


def principal_angles(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Principal angles between two subspaces, ascending.

    A is n x r1, B is n x r2, both with orthonormal columns. Returns
    a length-min(r1, r2) vector of angles in [0, pi/2].
    """
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must live in the same ambient space")
    M = A.T @ B
    # cos(theta_k) = sigma_k(A^T B)
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    return np.sort(angles)


def max_principal_angle(A: np.ndarray, B: np.ndarray) -> float:
    return float(principal_angles(A, B).max())


def observer_matrix_zscore(values_per_cell: list[np.ndarray]) -> np.ndarray:
    """Stack per-cell observer values (each shape (L,)) into a (n_cells, L)
    matrix and z-score each ROW (each cell) independently.

    Per-cell normalization is locked in pre-reg §4.2: 'observers are
    computed per-cell separately, then per-cell-normalized by subtracting
    cell-wise mean and dividing by cell-wise std (z-score per cell).'
    """
    M = np.stack([np.asarray(v, dtype=float) for v in values_per_cell], axis=0)
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, keepdims=True)
    sd_safe = np.where(sd > 0, sd, 1.0)
    Z = (M - mu) / sd_safe
    return Z
