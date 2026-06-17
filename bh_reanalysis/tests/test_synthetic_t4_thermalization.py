"""T4 synthetic gate. Pre-reg §6.4.

Build a synthetic observer family at three 'thermalization stages' with
imposed angle shrinkage. Verify the pipeline detects shrinkage with median-angle
separation > 0.2 rad between stage 1 and stage 3.
"""

import numpy as np

from src.principal_angles import (
    observer_matrix_zscore,
    principal_angles,
    top_r_subspace,
)


L_AMBIENT = 8
N_CELLS = 12
N_OBSERVERS = 6
R = 2


def build_observers_at_stage(planted_subspaces: list[np.ndarray],
                             alpha: float,
                             n_cells: int,
                             rng: np.random.Generator) -> list[list[np.ndarray]]:
    """Each observer i has its own per-observer planted subspace planted_i.
    At thermalization parameter `alpha` in [0, 1], the per-cell vectors blend
    toward a SHARED common subspace as alpha -> 1.

    alpha=0: each observer keeps its private subspace (high disagreement)
    alpha=1: all observers collapse to the same subspace (low disagreement)
    """
    L = planted_subspaces[0].shape[0]
    r = planted_subspaces[0].shape[1]
    # Common subspace (the late-time attractor)
    common = planted_subspaces[0]   # arbitrary choice
    out = []
    for i, private in enumerate(planted_subspaces):
        cells = []
        for _ in range(n_cells):
            coeff = rng.standard_normal(r)
            v_private = private @ coeff
            v_common = common @ coeff
            v = (1 - alpha) * v_private + alpha * v_common
            v += rng.standard_normal(L) * 0.005
            cells.append(v)
        out.append(cells)
    return out


def median_pairwise_angle(observers_per_cell: list[list[np.ndarray]]) -> float:
    matrices = [observer_matrix_zscore(o) for o in observers_per_cell]
    subspaces = [top_r_subspace(M, R) for M in matrices]
    angles = []
    for i in range(len(subspaces)):
        for j in range(i + 1, len(subspaces)):
            angles.append(float(principal_angles(subspaces[i], subspaces[j]).max()))
    return float(np.median(angles))


def test_t4_synthetic_gate():
    rng = np.random.default_rng(20260506)
    # Build N_OBSERVERS distinct planted subspaces in R^L
    G = rng.standard_normal((L_AMBIENT, L_AMBIENT))
    Q, _ = np.linalg.qr(G)
    # Each observer gets a distinct random 2-d subspace
    planted = []
    for i in range(N_OBSERVERS):
        idx = rng.choice(L_AMBIENT, size=R, replace=False)
        planted.append(Q[:, idx])

    # Stage 1: alpha=0 (max disagreement); Stage 2: alpha=0.5; Stage 3: alpha=0.95
    obs_stage1 = build_observers_at_stage(planted, alpha=0.0,  n_cells=N_CELLS, rng=rng)
    obs_stage2 = build_observers_at_stage(planted, alpha=0.5,  n_cells=N_CELLS, rng=rng)
    obs_stage3 = build_observers_at_stage(planted, alpha=0.95, n_cells=N_CELLS, rng=rng)

    m1 = median_pairwise_angle(obs_stage1)
    m2 = median_pairwise_angle(obs_stage2)
    m3 = median_pairwise_angle(obs_stage3)

    # Pre-reg: median(stage 1) - median(stage 3) > 0.2 rad
    assert m1 - m3 > 0.2, (
        f"synthetic shrinkage too small: m1={m1:.4f}, m3={m3:.4f}, diff={m1-m3:.4f}"
    )
    # Sanity: monotonic
    assert m1 >= m2 - 0.05, f"non-monotonic: m1={m1:.4f}, m2={m2:.4f}"
    assert m2 >= m3 - 0.05, f"non-monotonic: m2={m2:.4f}, m3={m3:.4f}"
