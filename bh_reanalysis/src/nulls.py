"""Haar-uniform null distributions for principal angles.

For the T2 test we need a null distribution of principal angles between
random r-dim subspaces in R^L. Sample by drawing two Gaussian L x r matrices,
QR-decomposing them, and computing principal angles between the resulting
orthonormal frames.

Pre-registered seed: 20260506.
"""

from __future__ import annotations

import numpy as np

from .principal_angles import principal_angles


SEED = 20260506


def haar_orthonormal_frame(L: int, r: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a uniformly random L x r orthonormal frame (Haar measure on
    the Stiefel manifold V_r(R^L)).
    """
    G = rng.standard_normal(size=(L, r))
    Q, R = np.linalg.qr(G)
    # Sign-correct so Q is uniform on the Stiefel manifold.
    sign = np.sign(np.diag(R))
    sign[sign == 0] = 1
    Q = Q * sign[np.newaxis, :]
    return Q


def haar_max_angle_samples(L: int, r: int, n_samples: int, seed: int = SEED) -> np.ndarray:
    """Return n_samples max-principal-angles between independent Haar frames."""
    rng = np.random.default_rng(seed)
    out = np.empty(n_samples)
    for i in range(n_samples):
        A = haar_orthonormal_frame(L, r, rng)
        B = haar_orthonormal_frame(L, r, rng)
        ang = principal_angles(A, B)
        out[i] = float(ang.max())
    return out


def haar_all_angle_samples(L: int, r: int, n_samples: int, seed: int = SEED) -> np.ndarray:
    """Return n_samples x r array of all r principal angles between Haar frames.
    Each row sorted ascending.
    """
    rng = np.random.default_rng(seed)
    out = np.empty((n_samples, r))
    for i in range(n_samples):
        A = haar_orthonormal_frame(L, r, rng)
        B = haar_orthonormal_frame(L, r, rng)
        out[i, :] = principal_angles(A, B)
    return out
