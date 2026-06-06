"""T3 — principal-direction shift across the J/U threshold.

Pre-reg §5.

For L in {8, 9} and tau in {2, 3}:
  - Build J/U-by-site response matrix R[j, i] = gap_i (or sensitivity variants).
  - At each J/U value j, compute leading right-singular unit vector v_j of the
    matrix at THAT J/U (rank-1 SVD of the row M_j); since each cell only has L
    sites, we use the per-cell observer vector itself, normalized.

Actually, pre-reg §5.2 says "leading right-singular vector v_j of the matrix at
J/U value j (via SVD; v_j is a unit L-vector over sites)." A matrix at a single
J/U value has only one row (one cell), so SVD reduces to direction = row /
||row||. We follow that interpretation.

  - Angle between consecutive J/U: theta_jj+1 = arccos(|<v_j, v_j+1>|).
  - theta_threshold = max angle over J/U pairs straddling J/U=0.30
    (pairs where one J/U <= 0.30 and the next J/U > 0.30, or symmetric;
    operationally, the pair where both endpoints are within [0.24, 0.32]).
  - theta_offthreshold = median angle over consecutive J/U pairs whose both
    endpoints are outside [0.24, 0.32].

Pre-reg P3.1: theta_threshold > theta_offthreshold for both (L=8, tau=2) and
(L=9, tau=2); ratio > 2 for at least one (L, tau).
Pre-reg P3.2: argmax over consecutive J/U pairs of theta lies in [0.24, 0.32]
for at least one (L, tau).
"""

from __future__ import annotations

import numpy as np

from .load_bh_data import Cell, cells_by_L_tau
from .observers import name_to_observer


THRESHOLD_LO = 0.24
THRESHOLD_HI = 0.32
THRESHOLD_CENTER = 0.30


def _per_cell_unit_direction(cell: Cell, observer_name: str) -> np.ndarray:
    obs = name_to_observer(observer_name)
    v = np.asarray(obs(cell), dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0:
        return v
    return v / n


def _angle_between_unit(u: np.ndarray, v: np.ndarray) -> float:
    cos = float(abs(np.dot(u, v)))
    cos = min(1.0, max(0.0, cos))
    return float(np.arccos(cos))


def run_t3(cells: list[Cell], observer_name: str = "gap") -> dict:
    by_Lt = cells_by_L_tau(cells)
    target_keys = [(8, 2.0), (8, 3.0), (9, 2.0), (9, 3.0)]
    out: dict = {"observer": observer_name, "per_Ltau": {}}
    p31_pairs_pass = []
    p31_ratios = []
    p32_argmax_in_band = []

    for key in target_keys:
        if key not in by_Lt:
            continue
        cs = by_Lt[key]  # sorted by J/U
        if len(cs) < 3:
            continue
        jus = np.array([c.J_over_U for c in cs])
        directions = np.stack([_per_cell_unit_direction(c, observer_name) for c in cs])
        n = len(cs)
        thetas = np.array([
            _angle_between_unit(directions[i], directions[i + 1])
            for i in range(n - 1)
        ])
        midpoints = (jus[:-1] + jus[1:]) / 2.0

        # Threshold pairs: midpoint in [THRESHOLD_LO, THRESHOLD_HI]
        threshold_mask = (midpoints >= THRESHOLD_LO) & (midpoints <= THRESHOLD_HI)
        # Off-threshold: consecutive pairs whose midpoint is outside the band
        off_mask = ~threshold_mask
        if not threshold_mask.any() or not off_mask.any():
            continue
        theta_threshold = float(thetas[threshold_mask].max())
        theta_off = float(np.median(thetas[off_mask]))
        ratio = theta_threshold / theta_off if theta_off > 0 else float("inf")
        argmax_idx = int(np.argmax(thetas))
        argmax_J_pair = (float(jus[argmax_idx]), float(jus[argmax_idx + 1]))
        argmax_in_band = bool(
            jus[argmax_idx] >= THRESHOLD_LO - 1e-9
            and jus[argmax_idx + 1] <= THRESHOLD_HI + 1e-9
        ) or bool(
            (jus[argmax_idx] <= 0.30 and jus[argmax_idx + 1] >= 0.30)
        )

        out["per_Ltau"][f"L{key[0]}_tau{key[1]}"] = {
            "L": key[0],
            "tau": key[1],
            "n_J_values": n,
            "J_over_U": jus.tolist(),
            "consecutive_angles": thetas.tolist(),
            "theta_threshold": theta_threshold,
            "theta_offthreshold": theta_off,
            "ratio": ratio,
            "argmax_consecutive_J_pair": argmax_J_pair,
            "argmax_consecutive_idx": argmax_idx,
            "argmax_in_threshold_band": argmax_in_band,
        }

        # Pre-reg P3.1 contribution: only L8tau2 and L9tau2 are required
        if key in [(8, 2.0), (9, 2.0)]:
            p31_pairs_pass.append(theta_threshold > theta_off)
            p31_ratios.append(ratio)
        # Pre-reg P3.2 (locked): argmax pair within [0.24, 0.32] (the stricter
        # prediction band, distinct from F3.2's wider [0.20, 0.36] falsification
        # band reported separately below).
        peak_in_prediction_band = bool(
            jus[argmax_idx] >= 0.24 - 1e-9
            and jus[argmax_idx + 1] <= 0.32 + 1e-9
        )
        peak_in_falsification_band = bool(
            jus[argmax_idx] >= 0.20 - 1e-9
            and jus[argmax_idx + 1] <= 0.36 + 1e-9
        )
        p32_argmax_in_band.append(peak_in_prediction_band)
        # Append falsification-band check separately for F3.2 below
        out["per_Ltau"][f"L{key[0]}_tau{key[1]}"]["argmax_in_prediction_band_0p24_0p32"] = peak_in_prediction_band
        out["per_Ltau"][f"L{key[0]}_tau{key[1]}"]["argmax_in_falsification_band_0p20_0p36"] = peak_in_falsification_band

    out["P3.1_pass"] = bool(
        len(p31_pairs_pass) == 2 and all(p31_pairs_pass)
        and any(r > 2.0 for r in p31_ratios)
    )
    out["P3.1_pairs_pass"] = p31_pairs_pass
    out["P3.1_ratios"] = p31_ratios
    out["P3.2_pass"] = bool(any(p32_argmax_in_band))
    out["F3.1_falsified"] = bool(
        len(p31_pairs_pass) == 2 and not any(p31_pairs_pass)
    )
    # F3.2 uses the wider [0.20, 0.36] falsification band per pre-reg §5.3
    f32_per_Ltau = [
        v.get("argmax_in_falsification_band_0p20_0p36", False)
        for v in out["per_Ltau"].values()
    ]
    out["F3.2_falsified"] = bool(
        len(f32_per_Ltau) >= 2 and not any(f32_per_Ltau)
    )
    return out


def run_t3_sensitivity(cells: list[Cell]) -> dict:
    """Pre-reg §5.4: re-run with redist_gap and dn_tgt observers."""
    return {
        "primary_gap": run_t3(cells, observer_name="gap"),
        "redist_gap": run_t3(cells, observer_name="redist_gap"),
        "dn_tgt": run_t3(cells, observer_name="dn_tgt"),
    }
