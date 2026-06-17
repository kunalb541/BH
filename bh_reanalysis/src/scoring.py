"""Three scoring rules for T5 (intervention fork).

Definitions are LOCKED in pre-reg §7.2 (revised, data-bound).

For each site i in cell c, restrict to random-arm trials t in which i was
hit (i.e., i in random_sites[t]). For that subset of trials S_i:

  R_trace(i)  = mean over t in S_i of sum_j |delta_rnd[t, j]|
                (total occupation movement when i is part of the hit set)
  R_opnorm(i) = mean over t in S_i of max_j |delta_rnd[t, j]|
                (largest single-site movement)
  R_gap(i)    = mean over t in S_i of (
                  |delta_rnd[t, i]| - mean_{j not in random_sites[t]} |delta_rnd[t, j]|
                )
                (per-site advantage of having i in the hit set vs sites left out)

Each rule produces an L-vector of scores. Optimal site = argmax.
Site i with empty S_i (never hit in any random trial) is given score -inf.
"""

from __future__ import annotations

import numpy as np

from .load_bh_data import Cell


def _trial_subset_per_site(cell: Cell) -> list[np.ndarray]:
    """For each site i, return the array of trial indices in which i was hit
    by the random arm.
    """
    L = cell.L
    out: list[np.ndarray] = []
    for i in range(L):
        mask = np.any(cell.random_sites == i, axis=1)
        out.append(np.where(mask)[0])
    return out


def trace_score(cell: Cell) -> np.ndarray:
    abs_d = np.abs(cell.delta_rnd)
    per_trial = abs_d.sum(axis=1)  # shape (n_trials,)
    subsets = _trial_subset_per_site(cell)
    out = np.full(cell.L, -np.inf, dtype=float)
    for i, S in enumerate(subsets):
        if len(S) == 0:
            continue
        out[i] = float(per_trial[S].mean())
    return out


def opnorm_score(cell: Cell) -> np.ndarray:
    abs_d = np.abs(cell.delta_rnd)
    per_trial = abs_d.max(axis=1)
    subsets = _trial_subset_per_site(cell)
    out = np.full(cell.L, -np.inf, dtype=float)
    for i, S in enumerate(subsets):
        if len(S) == 0:
            continue
        out[i] = float(per_trial[S].mean())
    return out


def gap_score(cell: Cell) -> np.ndarray:
    abs_d = np.abs(cell.delta_rnd)        # (n_trials, L)
    L = cell.L
    n_trials = cell.n_trials
    # For each trial t, build mask of sites NOT in random_sites[t].
    not_hit_mask = np.ones((n_trials, L), dtype=bool)
    rows = np.repeat(np.arange(n_trials), cell.k)
    cols = cell.random_sites.reshape(-1)
    not_hit_mask[rows, cols] = False
    # For each trial t, mean of |delta_rnd[t, j]| for j NOT in random_sites[t].
    sums = (abs_d * not_hit_mask).sum(axis=1)
    counts = not_hit_mask.sum(axis=1)  # = L - k
    other_means = sums / counts
    subsets = _trial_subset_per_site(cell)
    out = np.full(L, -np.inf, dtype=float)
    for i, S in enumerate(subsets):
        if len(S) == 0:
            continue
        per_trial = abs_d[S, i] - other_means[S]
        out[i] = float(per_trial.mean())
    return out


def optimal_sites(cell: Cell) -> dict[str, int]:
    """Return optimal-site argmax for each rule."""
    return {
        "trace":  int(np.argmax(trace_score(cell))),
        "opnorm": int(np.argmax(opnorm_score(cell))),
        "gap":    int(np.argmax(gap_score(cell))),
    }
