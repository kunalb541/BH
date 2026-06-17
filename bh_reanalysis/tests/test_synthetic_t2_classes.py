"""T2 synthetic gate. Pre-reg §4.6.

Build a synthetic observer family with KNOWN class structure:
  - Class A: 5 observers, all sharing a planted top-2 subspace in R^L (with small noise).
  - Class B: 5 observers, sharing a different planted top-2 subspace.
  - Pipeline: same as the real-data T2 — z-score per cell, top-r=2 right-subspace, principal angles.

Pass criteria (locked in pre-reg §4.6):
  - Synthetic same-subspace observers: median within-class angle < 0.05 rad.
  - Disjoint synthetic observers: median across-class angle > 1.4 rad.
  - KS distinguishability vs Haar null at p < 0.001.
"""

import numpy as np
from scipy.stats import ks_2samp

from src.principal_angles import (
    observer_matrix_zscore,
    principal_angles,
    top_r_subspace,
)
from src.nulls import haar_max_angle_samples


L_AMBIENT = 8
N_CELLS = 16
N_PER_CLASS = 5
NOISE = 0.005
R = 2


def build_synthetic_class(planted_subspace: np.ndarray,
                          n_observers: int,
                          n_cells: int,
                          noise: float,
                          rng: np.random.Generator) -> list[list[np.ndarray]]:
    """Each observer is a list of per-cell L-vectors. Each per-cell vector is
    a random combination of the columns of planted_subspace, plus small noise.
    """
    L = planted_subspace.shape[0]
    r = planted_subspace.shape[1]
    out = []
    for _ in range(n_observers):
        cells = []
        for _ in range(n_cells):
            coeff = rng.standard_normal(r)
            v = planted_subspace @ coeff + rng.standard_normal(L) * noise
            cells.append(v)
        out.append(cells)
    return out


def compute_pairwise_max_angles(observer_matrices_zscored: list[np.ndarray]) -> np.ndarray:
    """Compute the max principal angle between every pair of observers' top-r
    right-singular subspaces.
    """
    subspaces = [top_r_subspace(M, R) for M in observer_matrices_zscored]
    n = len(subspaces)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            ang = principal_angles(subspaces[i], subspaces[j]).max()
            out.append(float(ang))
    return np.asarray(out)


def test_t2_synthetic_gate():
    rng = np.random.default_rng(20260506)
    # Pick two disjoint planted subspaces in R^L
    G = rng.standard_normal((L_AMBIENT, L_AMBIENT))
    Q, _ = np.linalg.qr(G)
    planted_A = Q[:, 0:R]
    planted_B = Q[:, R:R + R]   # disjoint

    obs_A = build_synthetic_class(planted_A, N_PER_CLASS, N_CELLS, NOISE, rng)
    obs_B = build_synthetic_class(planted_B, N_PER_CLASS, N_CELLS, NOISE, rng)

    # Build observer matrices, z-score per cell (per row of cell x site matrix)
    def to_matrix(obs: list[list[np.ndarray]]) -> list[np.ndarray]:
        return [observer_matrix_zscore(o) for o in obs]

    M_A = to_matrix(obs_A)
    M_B = to_matrix(obs_B)

    # Within-class angles
    within_A = compute_pairwise_max_angles(M_A)
    within_B = compute_pairwise_max_angles(M_B)
    within = np.concatenate([within_A, within_B])

    # Across-class angles
    sub_A = [top_r_subspace(M, R) for M in M_A]
    sub_B = [top_r_subspace(M, R) for M in M_B]
    across = []
    for a in sub_A:
        for b in sub_B:
            across.append(float(principal_angles(a, b).max()))
    across = np.asarray(across)

    # Pre-reg locked thresholds
    median_within = float(np.median(within))
    median_across = float(np.median(across))
    assert median_within < 0.05, (
        f"synthetic same-subspace median angle = {median_within:.4f}, "
        f"expected < 0.05 rad"
    )
    assert median_across > 1.4, (
        f"synthetic disjoint-subspace median angle = {median_across:.4f}, "
        f"expected > 1.4 rad"
    )

    # KS test against Haar null
    null_samples = haar_max_angle_samples(L=L_AMBIENT, r=R, n_samples=2000, seed=12345)
    # Use the within-class distribution (should be FAR LEFT of null)
    ks_within = ks_2samp(within, null_samples)
    assert ks_within.pvalue < 1e-3, (
        f"synthetic within-class KS p = {ks_within.pvalue:.4g}, expected < 1e-3"
    )
