"""T4 — temporal evolution of misalignment band.

Pre-reg §6.

For each L, at each available tau:
  - Build observer matrices (cells at this L, this tau, x sites).
  - Compute top-r=2 subspaces and 66 pairwise principal angles.
  - Take median.

Pre-reg P4.1 (L=6 three points): median(tau=1) >= median(tau=2) >= median(tau=3),
strict in at least one step.

Pre-reg P4.2 (L=8, L=9 two points): median(tau=2) >= median(tau=3) at both L.

Auxiliary (NOT pre-registered, Amendment A1): three-point check at L=9 restricted
to J/U in {0.12, 0.20, 0.30, 0.40} where tau=1 is stored.
"""

from __future__ import annotations

import numpy as np

from .load_bh_data import Cell
from .observers import OBSERVERS
from .principal_angles import (
    max_principal_angle,
    observer_matrix_zscore,
    top_r_subspace,
)


R = 2


def median_pairwise_angle_at(cells_subset: list[Cell]) -> float | None:
    if len(cells_subset) < 2:
        return None
    L = cells_subset[0].L
    subspaces = []
    for obs in OBSERVERS:
        per_cell_vals = [obs(c) for c in cells_subset]
        Z = observer_matrix_zscore(per_cell_vals)
        subspaces.append(top_r_subspace(Z, R))
    angles = []
    for i in range(len(subspaces)):
        for j in range(i + 1, len(subspaces)):
            angles.append(max_principal_angle(subspaces[i], subspaces[j]))
    return float(np.median(angles))


def run_t4(cells: list[Cell]) -> dict:
    out: dict = {"per_L": {}}

    # Group cells by L and tau
    by_L_tau: dict[tuple[int, float], list[Cell]] = {}
    for c in cells:
        by_L_tau.setdefault((c.L, c.tau), []).append(c)

    # P4.1: L=6, three taus
    L6_medians: dict[float, float] = {}
    for tau in [1.0, 2.0, 3.0]:
        cs = by_L_tau.get((6, tau), [])
        m = median_pairwise_angle_at(cs)
        if m is not None:
            L6_medians[tau] = m
    p41_pass = (
        all(t in L6_medians for t in [1.0, 2.0, 3.0])
        and L6_medians[1.0] >= L6_medians[2.0]
        and L6_medians[2.0] >= L6_medians[3.0]
        and not (L6_medians[1.0] == L6_medians[2.0] == L6_medians[3.0])
    )
    out["per_L"][6] = {"medians_by_tau": L6_medians, "P4.1_pass": p41_pass}

    # P4.2: L=8 and L=9, tau=2 vs tau=3
    p42_per_L = {}
    for L in [8, 9]:
        ms = {}
        for tau in [2.0, 3.0]:
            cs = by_L_tau.get((L, tau), [])
            m = median_pairwise_angle_at(cs)
            if m is not None:
                ms[tau] = m
        ok = (
            2.0 in ms and 3.0 in ms and ms[2.0] >= ms[3.0]
        )
        p42_per_L[L] = ok
        out["per_L"][L] = {"medians_by_tau": ms, "P4.2_pass": ok}
    p42_pass = all(p42_per_L.values())

    # Auxiliary: L=9 restricted to J/U in {0.12, 0.20, 0.30, 0.40} (tau=1 stored)
    aux_jus = [0.12, 0.20, 0.30, 0.40]
    aux_medians: dict[float, float] = {}
    for tau in [1.0, 2.0, 3.0]:
        cs = [c for c in cells if c.L == 9 and c.tau == tau and c.J_over_U in aux_jus]
        m = median_pairwise_angle_at(cs)
        if m is not None:
            aux_medians[tau] = m
    aux_monotonic = (
        all(t in aux_medians for t in [1.0, 2.0, 3.0])
        and aux_medians[1.0] >= aux_medians[2.0]
        and aux_medians[2.0] >= aux_medians[3.0]
    )
    out["aux_L9_3point"] = {
        "medians_by_tau": aux_medians,
        "monotonic_decreasing": aux_monotonic,
    }

    out["P4.1_pass"] = p41_pass
    out["P4.2_pass"] = p42_pass
    out["F4.1_falsified"] = (
        all(t in L6_medians for t in [1.0, 2.0, 3.0])
        and L6_medians[1.0] < L6_medians[2.0]
    )
    out["F4.2_falsified"] = bool(not p41_pass and not p42_pass)
    return out
