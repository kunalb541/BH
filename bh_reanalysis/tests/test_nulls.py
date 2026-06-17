"""Haar null sampling tests. Pre-reg §10."""

import numpy as np

from src.nulls import (
    SEED,
    haar_all_angle_samples,
    haar_max_angle_samples,
    haar_orthonormal_frame,
)


def test_haar_orthonormal_frame_columns_unit_norm():
    rng = np.random.default_rng(0)
    Q = haar_orthonormal_frame(8, 2, rng)
    assert Q.shape == (8, 2)
    G = Q.T @ Q
    assert np.allclose(G, np.eye(2), atol=1e-10)


def test_haar_max_angles_in_valid_range():
    angles = haar_max_angle_samples(L=6, r=2, n_samples=200, seed=0)
    assert angles.shape == (200,)
    assert np.all(angles >= 0)
    assert np.all(angles <= np.pi / 2 + 1e-8)


def test_haar_max_angles_reproducible():
    a1 = haar_max_angle_samples(L=6, r=2, n_samples=50, seed=SEED)
    a2 = haar_max_angle_samples(L=6, r=2, n_samples=50, seed=SEED)
    assert np.allclose(a1, a2)


def test_haar_all_angles_sorted_ascending():
    samples = haar_all_angle_samples(L=8, r=3, n_samples=20, seed=1)
    assert samples.shape == (20, 3)
    for row in samples:
        assert np.all(row[:-1] <= row[1:] + 1e-10)


def test_haar_max_angle_distribution_concentrates_high_for_small_r_in_large_L():
    """For r=2 in L=6, the max principal angle distribution between two random
    Haar 2-frames concentrates around values not far below pi/2 (most random
    pairs are nearly orthogonal). Loose check: median > pi/4.
    """
    angles = haar_max_angle_samples(L=6, r=2, n_samples=500, seed=2)
    median = float(np.median(angles))
    assert median > np.pi / 4, f"expected median > pi/4, got {median}"
