"""T2 — pairwise principal angle structure between observers.

Pre-reg §4.

For each L:
  - Build cell-x-site z-scored observer matrices (one per observer).
  - Compute top-r=2 right-singular subspace in site-space (dim L).
  - Compute pairwise max principal angles between subspace pairs.
  - Classify pairs as within-class / across-class.
  - Test:
      P2.1 within < across by L
      P2.2 within < 0.7 rad and across > 1.0 rad
      P2.3 KS test vs Haar null at p < 0.05 with within in left tail
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp

from .load_bh_data import Cell, cells_by_L
from .nulls import haar_max_angle_samples
from .observers import OBSERVERS, Observer
from .principal_angles import (
    max_principal_angle,
    observer_matrix_zscore,
    top_r_subspace,
)


R = 2


def build_subspaces_at_L(cells_at_L: list[Cell],
                         observers: list[Observer]) -> dict[str, np.ndarray]:
    """Returns {observer_name: site-space subspace (L x r)} at this L."""
    out = {}
    L = cells_at_L[0].L
    for obs in observers:
        per_cell_vals = [obs(c) for c in cells_at_L]
        Z = observer_matrix_zscore(per_cell_vals)
        # Z is (n_cells x L). Top-r right-singular vectors live in R^L.
        out[obs.name] = top_r_subspace(Z, R)
    return out


def compute_pairs(subspaces: dict[str, np.ndarray],
                  observers: list[Observer]) -> tuple[list[tuple[str, str, float]],
                                                       list[tuple[str, str, float]]]:
    name_to_obs = {o.name: o for o in observers}
    within = []
    across = []
    names = [o.name for o in observers]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ni, nj = names[i], names[j]
            ang = max_principal_angle(subspaces[ni], subspaces[nj])
            entry = (ni, nj, ang)
            if name_to_obs[ni].cls == name_to_obs[nj].cls:
                within.append(entry)
            else:
                across.append(entry)
    return within, across


def run_t2(cells: list[Cell]) -> dict:
    by_L = cells_by_L(cells)
    out: dict = {"per_L": {}}
    p21_pass_per_L = {}
    p22_pass_per_L = {}
    p23_pass_per_L = {}
    for L, cells_at_L in by_L.items():
        if len(cells_at_L) < 2:
            continue
        subs = build_subspaces_at_L(cells_at_L, OBSERVERS)
        within, across = compute_pairs(subs, OBSERVERS)
        within_angles = np.array([a for _, _, a in within])
        across_angles = np.array([a for _, _, a in across])
        median_within = float(np.median(within_angles))
        median_across = float(np.median(across_angles))
        # P2.1
        p21 = bool(median_within < median_across)
        p21_pass_per_L[L] = p21
        # P2.2
        p22 = bool(median_within < 0.7 and median_across > 1.0)
        p22_pass_per_L[L] = p22
        # P2.3 KS test on combined empirical pairwise angles vs Haar null
        n_null = 5000
        null_samples = haar_max_angle_samples(L=L, r=R, n_samples=n_null, seed=20260506)
        all_emp = np.concatenate([within_angles, across_angles])
        ks_stat, ks_p = ks_2samp(all_emp, null_samples)
        # Pre-reg: empirical distribution should differ AND show longer left tail.
        # Operationalize "longer left tail" as: empirical 25th percentile is below
        # null 25th percentile.
        emp_q25 = float(np.quantile(all_emp, 0.25))
        null_q25 = float(np.quantile(null_samples, 0.25))
        left_tail_longer = bool(emp_q25 < null_q25)
        p23 = bool(ks_p < 0.05 and left_tail_longer)
        p23_pass_per_L[L] = p23

        out["per_L"][L] = {
            "n_cells": len(cells_at_L),
            "n_within_pairs": len(within),
            "n_across_pairs": len(across),
            "median_within_angle": median_within,
            "median_across_angle": median_across,
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_p),
            "empirical_q25": emp_q25,
            "null_q25": null_q25,
            "left_tail_longer": left_tail_longer,
            "P2.1_pass": p21,
            "P2.2_pass": p22,
            "P2.3_pass": p23,
            "within_pairs": [
                {"a": a, "b": b, "angle": float(ang)} for a, b, ang in within
            ],
            "across_pairs": [
                {"a": a, "b": b, "angle": float(ang)} for a, b, ang in across
            ],
        }

    out["P2.1_overall_pass"] = all(p21_pass_per_L.values())
    out["P2.2_overall_pass"] = all(p22_pass_per_L.values())
    # Pre-reg P2.3: "For each L, ... at p < 0.05 (KS test), with the empirical
    # distribution showing a longer left tail". "For each L" => all L must pass.
    out["P2.3_overall_pass"] = all(p23_pass_per_L.values())
    # Falsifications
    out["F2.1_falsified"] = not all(p21_pass_per_L.values())
    # F2.2: KS p>=0.05 for ALL three L values
    out["F2.2_falsified"] = all(
        out["per_L"][L]["ks_pvalue"] >= 0.05 for L in out["per_L"]
    )
    # F2.3: within median > across median at any L
    out["F2.3_falsified"] = any(
        out["per_L"][L]["median_within_angle"] > out["per_L"][L]["median_across_angle"]
        for L in out["per_L"]
    )
    return out
