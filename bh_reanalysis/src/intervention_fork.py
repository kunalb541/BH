"""T5 — scoring-rule fork.

Pre-reg §7.

For each cell, compute optimal site under three scoring rules:
  trace, opnorm, gap (definitions in src/scoring.py).

P5.1: across all cells, the three rules' optimal sites disagree (not all
identical) in >= 30% of cells.
P5.2: at least one pair of rules disagrees in >= 20% of cells.
F5.1: all three rules agree in >= 90% of cells.
"""

from __future__ import annotations

import numpy as np

from .load_bh_data import Cell
from .scoring import optimal_sites


def run_t5(cells: list[Cell]) -> dict:
    per_cell = []
    n = len(cells)
    n_all_agree = 0
    n_pair_disagree = {"trace_vs_opnorm": 0, "trace_vs_gap": 0, "opnorm_vs_gap": 0}
    for c in cells:
        s = optimal_sites(c)
        agree = (s["trace"] == s["opnorm"] == s["gap"])
        if agree:
            n_all_agree += 1
        if s["trace"] != s["opnorm"]:
            n_pair_disagree["trace_vs_opnorm"] += 1
        if s["trace"] != s["gap"]:
            n_pair_disagree["trace_vs_gap"] += 1
        if s["opnorm"] != s["gap"]:
            n_pair_disagree["opnorm_vs_gap"] += 1
        per_cell.append({
            "L": c.L, "J_over_U": c.J_over_U, "tau": c.tau,
            "optimal_trace": s["trace"],
            "optimal_opnorm": s["opnorm"],
            "optimal_gap": s["gap"],
            "all_agree": agree,
        })
    n_disagree = n - n_all_agree
    rate_disagree_3way = n_disagree / n
    pair_rates = {k: v / n for k, v in n_pair_disagree.items()}
    p51 = bool(rate_disagree_3way >= 0.30)
    p52 = bool(any(r >= 0.20 for r in pair_rates.values()))
    f51 = bool((n_all_agree / n) >= 0.90)
    return {
        "n_cells": n,
        "n_all_three_agree": n_all_agree,
        "rate_3way_disagreement": rate_disagree_3way,
        "pair_disagreement_rates": pair_rates,
        "P5.1_threshold": 0.30,
        "P5.1_pass": p51,
        "P5.2_threshold": 0.20,
        "P5.2_pass": p52,
        "F5.1_falsified": f51,
        "per_cell": per_cell,
    }
