"""Scoring rule unit tests. Pre-reg §10."""

import numpy as np
import pytest

from src.load_bh_data import Cell
from src.scoring import gap_score, opnorm_score, optimal_sites, trace_score


def make_cell(L=6, k=2, delta_rnd=None, random_sites=None) -> Cell:
    if delta_rnd is None:
        delta_rnd = np.zeros((4, L))
    if random_sites is None:
        random_sites = np.array([[0, 1], [2, 3], [4, 5], [0, 5]])
    return Cell(
        L=L, N=L // 2, J_over_U=0.3, tau=2.0, k=k,
        Fi=np.linspace(0.2, 0.4, L),
        selected=np.array([0, 1]),
        delta_tgt=np.zeros(L),
        delta_rnd=delta_rnd,
        random_sites=random_sites,
        source="(fake)",
    )


def test_trace_score_hand_computed():
    """Trial 0 hits sites 0, 1; |delta_rnd[0]| sums to 0.4.
       Trial 1 hits sites 2, 3; sum = 0.4.
       Trial 2 hits sites 4, 5; sum = 0.4.
       Trial 3 hits sites 0, 5; sum = 0.4.
    For site 0: hit in trials [0, 3]; mean total = 0.4.
    For site 1: hit in trial [0]; mean total = 0.4.
    """
    L = 6
    delta_rnd = np.zeros((4, L))
    # All trials produce |sum| = 0.4 for simplicity
    delta_rnd[0] = [0.1, 0.1, 0.1, 0.05, 0.05, 0.0]   # sum |.| = 0.4
    delta_rnd[1] = [0.0, 0.0, 0.2, 0.1, 0.05, 0.05]   # sum |.| = 0.4
    delta_rnd[2] = [0.05, 0.05, 0.0, 0.0, 0.2, 0.1]   # sum |.| = 0.4
    delta_rnd[3] = [0.15, 0.0, 0.0, 0.0, 0.05, 0.2]   # sum |.| = 0.4
    cell = make_cell(L=L, delta_rnd=delta_rnd,
                     random_sites=np.array([[0, 1], [2, 3], [4, 5], [0, 5]]))
    s = trace_score(cell)
    assert s.shape == (L,)
    # Site 0 hit in trials 0 and 3 -> mean = (0.4 + 0.4) / 2 = 0.4
    assert abs(s[0] - 0.4) < 1e-12
    # Site 1 hit in trial 0 -> mean = 0.4
    assert abs(s[1] - 0.4) < 1e-12
    # Site 2 hit in trial 1 -> 0.4
    assert abs(s[2] - 0.4) < 1e-12


def test_unhit_site_returns_neg_inf():
    L = 6
    delta_rnd = np.full((2, L), 0.1)
    # Sites 0, 1 are hit; sites 2..5 never hit
    random_sites = np.array([[0, 1], [0, 1]])
    cell = make_cell(L=L, delta_rnd=delta_rnd, random_sites=random_sites)
    s_trace = trace_score(cell)
    s_op = opnorm_score(cell)
    s_gap = gap_score(cell)
    for i in [2, 3, 4, 5]:
        assert s_trace[i] == -np.inf
        assert s_op[i] == -np.inf
        assert s_gap[i] == -np.inf
    assert s_trace[0] != -np.inf


def test_opnorm_uses_max_per_trial():
    L = 4
    delta_rnd = np.zeros((2, L))
    delta_rnd[0] = [0.1, 0.0, 0.0, 0.0]   # max = 0.1
    delta_rnd[1] = [0.0, 0.5, 0.0, 0.0]   # max = 0.5
    random_sites = np.array([[0, 1], [0, 1]])
    cell = make_cell(L=L, delta_rnd=delta_rnd, random_sites=random_sites)
    s = opnorm_score(cell)
    # Site 0 hit in both trials, mean of max = (0.1 + 0.5)/2 = 0.3
    assert abs(s[0] - 0.3) < 1e-12
    assert abs(s[1] - 0.3) < 1e-12


def test_gap_score_matches_definition():
    """For a hand-picked example: site 0 hit, site 1 hit. L = 4, k = 2.
    delta_rnd[0] = [0.5, 0.1, 0.1, 0.1].
    Mean of |delta_rnd[0, j]| for j NOT in random_sites[0] = (0.1 + 0.1) / 2 = 0.1.
    So gap for site 0 in trial 0 = 0.5 - 0.1 = 0.4.
    """
    L = 4
    delta_rnd = np.zeros((1, L))
    delta_rnd[0] = [0.5, 0.1, 0.1, 0.1]
    random_sites = np.array([[0, 1]])
    cell = make_cell(L=L, delta_rnd=delta_rnd, random_sites=random_sites)
    s = gap_score(cell)
    assert abs(s[0] - 0.4) < 1e-12


def test_optimal_sites_returns_three_keys():
    cell = make_cell()
    s = optimal_sites(cell)
    assert set(s.keys()) == {"trace", "opnorm", "gap"}
    for v in s.values():
        assert isinstance(v, int)
        assert 0 <= v < cell.L
