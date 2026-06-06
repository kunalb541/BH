"""T1 — predictive vs causal observer dissociation.

Pre-reg §3.

For each cell:
  - Predictive winner observer X*: argmax over observer family of
    |Pearson r(X(cell), dn_rnd_mean(cell))| across sites.
  - Predicted-best-site = argmax_i X*(cell)[i].
  - Causal-best-site = argmax_i gap(cell)[i].
  - Dissociation = (predicted-best-site != causal-best-site).

P1.1: dissociation rate >= 50% across all cells.
F1.1: dissociation rate <= 20%.

Sensitivity (declared in pre-reg §3.6): swap dn_rnd_mean -> dn_tgt as the
predictive target and report whether the locked conclusion flips.
"""

from __future__ import annotations

import numpy as np

from .load_bh_data import Cell
from .observers import OBSERVERS, Observer


def predictive_score(observer: Observer, cell: Cell, target_name: str) -> float:
    """|Pearson r| between observer values and a target response across sites."""
    obs_vals = observer(cell)
    if target_name == "dn_rnd_mean":
        target = cell.delta_rnd.mean(axis=0)
    elif target_name == "dn_tgt":
        target = cell.delta_tgt
    else:
        raise ValueError(target_name)
    if np.std(obs_vals) == 0 or np.std(target) == 0:
        return 0.0
    return float(abs(np.corrcoef(obs_vals, target)[0, 1]))


def predictive_winner(cell: Cell, target_name: str) -> Observer:
    scores = [(o, predictive_score(o, cell, target_name)) for o in OBSERVERS]
    scores.sort(key=lambda t: -t[1])
    return scores[0][0]


def predicted_best_site(cell: Cell, target_name: str) -> int:
    obs = predictive_winner(cell, target_name)
    return int(np.argmax(obs(cell)))


def causal_gap_per_site(cell: Cell) -> np.ndarray:
    return np.abs(cell.delta_tgt) - np.abs(cell.delta_rnd).mean(axis=0)


def causal_best_site(cell: Cell) -> int:
    return int(np.argmax(causal_gap_per_site(cell)))


def cell_dissociated(cell: Cell, target_name: str = "dn_rnd_mean") -> bool:
    return predicted_best_site(cell, target_name) != causal_best_site(cell)


def run_t1(cells: list[Cell]) -> dict:
    primary = []
    sensitivity = []
    per_cell = []
    for c in cells:
        p_site_primary = predicted_best_site(c, "dn_rnd_mean")
        p_site_sens = predicted_best_site(c, "dn_tgt")
        c_site = causal_best_site(c)
        diss_primary = (p_site_primary != c_site)
        diss_sens = (p_site_sens != c_site)
        primary.append(diss_primary)
        sensitivity.append(diss_sens)
        winner = predictive_winner(c, "dn_rnd_mean")
        winner_score = predictive_score(winner, c, "dn_rnd_mean")
        causal_gap_at_winner = float(causal_gap_per_site(c)[c_site])
        per_cell.append({
            "L": c.L, "J_over_U": c.J_over_U, "tau": c.tau,
            "predictive_winner_observer": winner.name,
            "predictive_winner_obs_class": winner.cls,
            "predictive_winner_abs_r": winner_score,
            "predicted_best_site_primary": p_site_primary,
            "predicted_best_site_sensitivity": p_site_sens,
            "causal_best_site": c_site,
            "causal_gap_at_winner_site": causal_gap_at_winner,
            "dissociated_primary": diss_primary,
            "dissociated_sensitivity": diss_sens,
        })
    n = len(cells)
    rate_primary = float(np.mean(primary))
    rate_sens = float(np.mean(sensitivity))
    return {
        "n_cells": n,
        "dissociation_rate_primary": rate_primary,
        "dissociation_rate_sensitivity": rate_sens,
        "P1.1_threshold": 0.50,
        "P1.1_pass": rate_primary >= 0.50,
        "F1.1_threshold": 0.20,
        "F1.1_falsified": rate_primary <= 0.20,
        "median_predictive_abs_r": float(np.median(
            [p["predictive_winner_abs_r"] for p in per_cell]
        )),
        "median_causal_gap": float(np.median(
            [p["causal_gap_at_winner_site"] for p in per_cell]
        )),
        "per_cell": per_cell,
    }
