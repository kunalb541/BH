"""Observer family for the BH reanalysis battery.

See bh_reanalysis_prereg.md §2 for class definitions. Locked here:

  F-class (4 vector observers per cell):
    F, F_rank, F_centered, F_zscore
  dn-class (5 vector observers per cell):
    dn_tgt, dn_rnd_mean, abs_dn_tgt, abs_dn_rnd_mean, redist_tgt
  gap-class (3 vector observers per cell):
    gap, gap_signed, redist_gap

Each observer is a function (Cell) -> np.ndarray of shape (L,).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .load_bh_data import Cell


ObserverFn = Callable[[Cell], np.ndarray]


@dataclass(frozen=True)
class Observer:
    name: str
    cls: str            # one of {"F", "dn", "gap"}
    fn: ObserverFn

    def __call__(self, cell: Cell) -> np.ndarray:
        return self.fn(cell)


# F-class
def _F(cell: Cell) -> np.ndarray:
    return np.asarray(cell.Fi, dtype=float)


def _F_rank(cell: Cell) -> np.ndarray:
    # rank 1 = largest F_i; ties broken by site index (stable sort)
    order = np.argsort(-cell.Fi, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(order) + 1)
    return rank.astype(float)


def _F_centered(cell: Cell) -> np.ndarray:
    return np.asarray(cell.Fi, dtype=float) - float(np.mean(cell.Fi))


def _F_zscore(cell: Cell) -> np.ndarray:
    F = np.asarray(cell.Fi, dtype=float)
    s = float(np.std(F))
    return (F - float(np.mean(F))) / (s if s > 0 else 1.0)


# dn-class
def _dn_tgt(cell: Cell) -> np.ndarray:
    return np.asarray(cell.delta_tgt, dtype=float)


def _dn_rnd_mean(cell: Cell) -> np.ndarray:
    return np.mean(cell.delta_rnd, axis=0)


def _abs_dn_tgt(cell: Cell) -> np.ndarray:
    return np.abs(cell.delta_tgt).astype(float)


def _abs_dn_rnd_mean(cell: Cell) -> np.ndarray:
    return np.mean(np.abs(cell.delta_rnd), axis=0)


def _redist_tgt(cell: Cell) -> np.ndarray:
    return np.maximum(0.0, -np.asarray(cell.delta_tgt, dtype=float))


# gap-class
def _gap(cell: Cell) -> np.ndarray:
    return _abs_dn_tgt(cell) - _abs_dn_rnd_mean(cell)


def _gap_signed(cell: Cell) -> np.ndarray:
    return _dn_tgt(cell) - _dn_rnd_mean(cell)


def _redist_gap(cell: Cell) -> np.ndarray:
    rnd = np.maximum(0.0, -cell.delta_rnd)
    return _redist_tgt(cell) - np.mean(rnd, axis=0)


OBSERVERS: list[Observer] = [
    Observer("F",            "F",   _F),
    Observer("F_rank",       "F",   _F_rank),
    Observer("F_centered",   "F",   _F_centered),
    Observer("F_zscore",     "F",   _F_zscore),
    Observer("dn_tgt",       "dn",  _dn_tgt),
    Observer("dn_rnd_mean",  "dn",  _dn_rnd_mean),
    Observer("abs_dn_tgt",   "dn",  _abs_dn_tgt),
    Observer("abs_dn_rnd_mean", "dn", _abs_dn_rnd_mean),
    Observer("redist_tgt",   "dn",  _redist_tgt),
    Observer("gap",          "gap", _gap),
    Observer("gap_signed",   "gap", _gap_signed),
    Observer("redist_gap",   "gap", _redist_gap),
]


# Aggregate scalar observers (used in T1 only; not in the T2 vector-pair test)
def F_sum(cell: Cell) -> float:
    return float(np.sum(cell.Fi))


def F_max(cell: Cell) -> float:
    return float(np.max(cell.Fi))


def F_argmax(cell: Cell) -> int:
    return int(np.argmax(cell.Fi))


def class_observers(cls: str) -> list[Observer]:
    return [o for o in OBSERVERS if o.cls == cls]


def name_to_observer(name: str) -> Observer:
    for o in OBSERVERS:
        if o.name == name:
            return o
    raise KeyError(name)
