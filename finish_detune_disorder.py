#!/usr/bin/env python3
"""Finish the symmetry-broken detuning disorder sweep, RESUMABLE per realization.

The environment kills long processes (~30 min cap), and each mu_max=2.0 realization
takes ~5 min, so we checkpoint to the CSV after EVERY realization and skip realizations
already present. Re-running this script repeatedly completes the sweep across kill windows.
When all (mu_max, realization) cells are present, it prints the full Step-3C summary.
"""
import os
import numpy as np
import pandas as pd

import bh
import bh_hardening as bhh
from symbreak_diag import build_condition_mu, L, k, JUS
from symbreak_detune import detune_superops, current_op, diagnose, summarize, TAUS

bh._SPARSE_D_THRESHOLD = 60
OUT = "outputs/mechanism_pilot"
CSV = os.path.join(OUT, "symbreak_detune_disorder.csv")
TARGET = {0.5: 10, 1.0: 10, 2.0: 10}      # mu_max -> n_realizations

def done_cells(df):
    if df is None or len(df) == 0 or "realization" not in df:
        return set()
    return {(round(s, 4), int(r)) for s, r in zip(df.strength, df.realization)}

df = pd.read_csv(CSV) if os.path.exists(CSV) else None
have = done_cells(df)
print(f"existing rows: {0 if df is None else len(df)}  cells present: {len(have)}", flush=True)

for mu_max, nreal in TARGET.items():
    for r in range(nreal):
        if (round(mu_max, 4), r) in have:
            continue
        rng = np.random.default_rng(20260606 + 1000 * int(mu_max * 10) + r)
        mu = rng.uniform(-mu_max, mu_max, size=L)
        rows = []
        for ju in JUS:
            cond = build_condition_mu(L, ju, mu)
            det = detune_superops(cond["n_ops"], cond["D"])
            Jops = [current_op(b, ju, cond["basis"], cond["idx_map"], bhh.NMAX) for b in range(L - 1)]
            for tau in TAUS:
                d = diagnose(cond, ju, tau, det, Jops)
                d.update(setting="disorder", pattern="rand", strength=mu_max, realization=r)
                rows.append(d)
        # append-checkpoint immediately
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True) if df is not None else pd.DataFrame(rows)
        df.to_csv(CSV, index=False)
        print(f"  checkpointed mu_max={mu_max} r={r}  (total rows {len(df)})", flush=True)

have = done_cells(df)
complete = all((round(m, 4), r) in have for m, n in TARGET.items() for r in range(n))
print(f"\ncomplete: {complete}  total rows: {len(df)}", flush=True)
if complete:
    summarize(df, "disorder")
    print("[done full]")
else:
    missing = [(m, r) for m, n in TARGET.items() for r in range(n) if (round(m, 4), r) not in have]
    print(f"[partial] still missing {len(missing)} cells: {missing[:8]}{'...' if len(missing)>8 else ''}")
    print("[rerun this script to continue]")
