"""Observer unit tests. Pre-reg §10."""

import numpy as np
import pytest

from src.load_bh_data import Cell
from src.observers import OBSERVERS, class_observers, name_to_observer


def make_fake_cell(L: int = 6, k: int = 2, n_trials: int = 5, seed: int = 0) -> Cell:
    rng = np.random.default_rng(seed)
    Fi = rng.uniform(0.1, 0.5, size=L)
    selected = np.argsort(-Fi)[:k]
    delta_tgt = rng.normal(scale=0.01, size=L)
    delta_rnd = rng.normal(scale=0.01, size=(n_trials, L))
    random_sites = np.stack([
        rng.choice(L, size=k, replace=False) for _ in range(n_trials)
    ])
    return Cell(
        L=L, N=L // 2, J_over_U=0.3, tau=2.0, k=k,
        Fi=Fi, selected=selected,
        delta_tgt=delta_tgt, delta_rnd=delta_rnd, random_sites=random_sites,
        source="(fake)",
    )


def test_observer_count_per_class():
    """Pre-reg §2.4 locks: 4 F-class, 5 dn-class, 3 gap-class."""
    assert len(class_observers("F")) == 4
    assert len(class_observers("dn")) == 5
    assert len(class_observers("gap")) == 3
    assert len(OBSERVERS) == 12


def test_observer_names_unique():
    names = [o.name for o in OBSERVERS]
    assert len(set(names)) == len(names)


def test_observer_returns_correct_shape():
    cell = make_fake_cell(L=8)
    for obs in OBSERVERS:
        v = obs(cell)
        assert v.shape == (cell.L,), f"{obs.name} returned {v.shape} for L={cell.L}"
        assert np.all(np.isfinite(v)), f"{obs.name} produced non-finite values"


def test_observer_deterministic():
    cell = make_fake_cell(L=6, seed=42)
    for obs in OBSERVERS:
        v1 = obs(cell)
        v2 = obs(cell)
        assert np.allclose(v1, v2)


def test_F_rank_is_permutation():
    cell = make_fake_cell(L=8)
    rank = name_to_observer("F_rank")(cell)
    assert sorted(rank.tolist()) == list(range(1, 9))


def test_F_centered_zero_mean():
    cell = make_fake_cell()
    v = name_to_observer("F_centered")(cell)
    assert abs(float(v.mean())) < 1e-12


def test_F_zscore_unit_std_when_nondegenerate():
    cell = make_fake_cell()
    v = name_to_observer("F_zscore")(cell)
    assert abs(float(v.mean())) < 1e-12
    assert abs(float(v.std()) - 1.0) < 1e-10


def test_F_zscore_handles_constant_F():
    L = 6
    cell = Cell(
        L=L, N=L // 2, J_over_U=0.3, tau=2.0, k=2,
        Fi=np.full(L, 0.25),
        selected=np.array([0, 1]),
        delta_tgt=np.zeros(L),
        delta_rnd=np.zeros((3, L)),
        random_sites=np.array([[0, 1], [2, 3], [4, 5]]),
        source="(fake)",
    )
    v = name_to_observer("F_zscore")(cell)
    assert np.all(v == 0.0)


def test_redist_tgt_nonnegative():
    cell = make_fake_cell()
    v = name_to_observer("redist_tgt")(cell)
    assert np.all(v >= 0)


def test_dn_rnd_mean_matches_explicit_mean():
    cell = make_fake_cell()
    v = name_to_observer("dn_rnd_mean")(cell)
    expected = cell.delta_rnd.mean(axis=0)
    assert np.allclose(v, expected)


def test_gap_signed_equals_dn_tgt_minus_dn_rnd_mean():
    cell = make_fake_cell()
    g = name_to_observer("gap_signed")(cell)
    expected = cell.delta_tgt - cell.delta_rnd.mean(axis=0)
    assert np.allclose(g, expected)
