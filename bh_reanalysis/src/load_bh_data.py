"""Loader for existing BH simulation outputs.

Reads:
  - outputs/data/results_L6.csv (L=6, mechanism column as Python-literal string)
  - outputs/checkpoints/L{8,9}_N4_JU*.json (L=8, L=9, JSON)

Returns a list of Cell objects.

Each Cell holds, for one (L, J/U, tau) condition:
  - Fi: pre-intervention site variance, shape (L,)
  - selected: top-k F_i sites, length k
  - delta_tgt: per-site Delta<n_i>, targeted arm, shape (L,)
  - delta_rnd: per-trial per-site Delta<n_i>, random arm, shape (n_trials, L)
  - random_sites: per-trial selected random sites, shape (n_trials, k)

Verification rule: the existing summary `mean_diff` recomputed from delta_tgt and
delta_rnd on the cell's matched scoring rule must match the stored value to
numerical precision (verified in tests/test_loaders.py).
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = REPO_ROOT / "outputs"


@dataclass(frozen=True)
class Cell:
    L: int
    N: int
    J_over_U: float
    tau: float
    k: int
    Fi: np.ndarray             # shape (L,)
    selected: np.ndarray       # shape (k,), int site indices
    delta_tgt: np.ndarray      # shape (L,)
    delta_rnd: np.ndarray      # shape (n_trials, L)
    random_sites: np.ndarray   # shape (n_trials, k)
    source: str                # path to source file

    @property
    def n_trials(self) -> int:
        return self.delta_rnd.shape[0]

    def cell_key(self) -> tuple[int, float, float]:
        return (self.L, float(self.J_over_U), float(self.tau))


def _parse_mechanism_blob(blob: str) -> list[dict]:
    """results_L6.csv stores `mechanism` as a Python repr string. Parse it."""
    return ast.literal_eval(blob)


def _parse_array_blob(blob: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(blob), dtype=float)


def _parse_int_array_blob(blob: str) -> np.ndarray:
    return np.asarray(ast.literal_eval(blob), dtype=int)


def _cell_from_l6_row(row: pd.Series) -> Cell:
    L = int(row["L"])
    Fi = _parse_array_blob(row["Fi"])
    selected = _parse_int_array_blob(row["selected"])
    mech = _parse_mechanism_blob(row["mechanism"])
    n_trials = len(mech)
    delta_tgt = np.asarray(mech[0]["delta_tgt"], dtype=float)
    # consistency: every trial in the L=6 mechanism stores the same delta_tgt.
    for m in mech:
        assert np.allclose(np.asarray(m["delta_tgt"]), delta_tgt), \
            "delta_tgt drifted across trials in a single L=6 cell — not expected"
    delta_rnd = np.stack([np.asarray(m["delta_rnd"], dtype=float) for m in mech])
    random_sites = np.stack([np.asarray(m["random"], dtype=int) for m in mech])
    return Cell(
        L=L,
        N=int(L // 2),  # results_L6.csv does not store N; the paper used N = L/2 for L=6
        J_over_U=float(row["J_over_U"]),
        tau=float(row["tau"]),
        k=int(row["selected_k"]),
        Fi=Fi,
        selected=selected,
        delta_tgt=delta_tgt,
        delta_rnd=delta_rnd,
        random_sites=random_sites,
        source=str(OUTPUTS / "data" / "results_L6.csv"),
    )


def _cells_from_checkpoint(path: Path) -> list[Cell]:
    with path.open() as f:
        d = json.load(f)
    L = int(d["L"])
    N = int(d["N"])
    J_over_U = float(d["J_over_U"])
    k = int(d["k"])
    Fi = np.asarray(d["Fi"], dtype=float)
    selected = np.asarray(d["selected"], dtype=int)
    cells: list[Cell] = []
    for r in d["results"]:
        mech = r["mechanism"]
        delta_tgt = np.asarray(mech[0]["delta_tgt"], dtype=float)
        for m in mech:
            assert np.allclose(np.asarray(m["delta_tgt"]), delta_tgt), \
                f"delta_tgt drifted across trials in {path.name} tau={r['tau']}"
        delta_rnd = np.stack(
            [np.asarray(m["delta_rnd"], dtype=float) for m in mech]
        )
        random_sites = np.stack(
            [np.asarray(m["random"], dtype=int) for m in mech]
        )
        cells.append(Cell(
            L=L,
            N=N,
            J_over_U=J_over_U,
            tau=float(r["tau"]),
            k=k,
            Fi=Fi.copy(),
            selected=selected.copy(),
            delta_tgt=delta_tgt,
            delta_rnd=delta_rnd,
            random_sites=random_sites,
            source=str(path),
        ))
    return cells


def load_all_cells() -> list[Cell]:
    cells: list[Cell] = []

    # L=6
    l6_path = OUTPUTS / "data" / "results_L6.csv"
    if l6_path.exists():
        df = pd.read_csv(l6_path)
        for _, row in df.iterrows():
            cells.append(_cell_from_l6_row(row))

    # L=8 and L=9 from checkpoints
    ckpt_dir = OUTPUTS / "checkpoints"
    if ckpt_dir.exists():
        for path in sorted(ckpt_dir.glob("L*_N*_JU*.json")):
            cells.extend(_cells_from_checkpoint(path))

    return cells


def cells_by_L(cells: Iterable[Cell]) -> dict[int, list[Cell]]:
    out: dict[int, list[Cell]] = {}
    for c in cells:
        out.setdefault(c.L, []).append(c)
    for L in out:
        out[L].sort(key=lambda c: (c.J_over_U, c.tau))
    return out


def cells_by_L_tau(cells: Iterable[Cell]) -> dict[tuple[int, float], list[Cell]]:
    out: dict[tuple[int, float], list[Cell]] = {}
    for c in cells:
        out.setdefault((c.L, c.tau), []).append(c)
    for k in out:
        out[k].sort(key=lambda c: c.J_over_U)
    return out


def recompute_mean_diff(cell: Cell, mode: str = "redist_clip") -> float:
    """Recompute the gap between targeted and random arms at the cell's
    selected-site set. Used for cross-check against `mean_diff` column.

    The original paper used the redistribution-clip variant by default:
      gap_t = sum_{i in selected} max(0, -delta_tgt[i])
            - sum_{i in random_t} max(0, -delta_rnd_t[i])
    averaged over trials.
    """
    sel = cell.selected
    if mode == "redist_clip":
        tgt_sum = np.sum(np.maximum(0.0, -cell.delta_tgt[sel]))
        rnd_sums = np.array([
            np.sum(np.maximum(0.0, -cell.delta_rnd[t, cell.random_sites[t]]))
            for t in range(cell.n_trials)
        ])
        return float(tgt_sum - rnd_sums.mean())
    if mode == "signed_clip":
        tgt_sum = float(np.sum(cell.delta_tgt[sel]))
        rnd_sums = np.array([
            float(np.sum(cell.delta_rnd[t, cell.random_sites[t]]))
            for t in range(cell.n_trials)
        ])
        return float(tgt_sum - rnd_sums.mean())
    raise ValueError(f"unknown mode {mode}")


if __name__ == "__main__":
    cells = load_all_cells()
    by_L = cells_by_L(cells)
    print(f"Loaded {len(cells)} cells total")
    for L, lst in by_L.items():
        jus = sorted({c.J_over_U for c in lst})
        taus = sorted({c.tau for c in lst})
        print(f"  L={L}: {len(lst)} cells, J/U in {jus}, tau in {taus}")
