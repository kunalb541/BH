#!/usr/bin/env python3
"""
Response-kernel mechanism pilot for the open Bose-Hubbard redistribution handle.

Goal: explain WHY local number variance F_i is a useful intervention handle in the
positive pocket (J/U >= 0.30) but not at low coupling (J/U = 0.12).

For each site i we apply a small extra dephasing dose dg only at i and measure the
FULL signed spatial response of the occupations at horizon tau:

    R[j, i] = ( <n_j>_baseline(tau) - <n_j>_{extra dg at i}(tau) ) / dg     (loss at j per unit dose at i)

Because the dephasing jump operator L_i = n_i commutes with total N, each response
COLUMN sums to exactly zero: sum_j R[j,i] = 0. The response is therefore pure
*redistribution* -- the discriminator is not how much moves but WHERE it goes.

From R we build, per site i:
    local_resp[i]  = |R[i,i]|                 on-site (diagonal) response
    offsite[i]     = sum_{j!=i} |R[j,i]|       off-site redistribution susceptibility
    total[i]       = sum_j |R[j,i]|            total response norm (== chi_redist)
    spread[i]      = sum_j |R[j,i]| * |j-i| / total[i]   mean redistribution distance

We then ask which of these F_i actually correlates with, whether that correlation
turns on only in the positive pocket, and whether the redistribution magnitude is
gated by coherent transport (bond kinetic coherence on the burn-in state).

The handle itself is measured exactly (gamma_extra = 0.5) by the exhaustive-subset
percentile of the top-F_i subset, so the pilot reveals which mechanism scalar
predicts the handle rather than asserting it.

Reuses the validated machinery in bh_hardening.py / bh.py. Exact Lindblad only.
"""

import os
import sys
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import bh
import bh_hardening as bhh

# Force the sparse Liouvillian path for L>=7 (D>=84). The dense path reallocates
# an ~800 MB D^2 matrix per evolve and is ~30x slower at L=7; sparse is exact and
# identical numerically (same operator, different storage), as used for L=8 in the
# paper. L=6 (D=56) stays dense so the committed L=6 pilot reproduces bit-for-bit.
bh._SPARSE_D_THRESHOLD = 60

OUTDIR = os.path.join("outputs", "mechanism_pilot")
os.makedirs(OUTDIR, exist_ok=True)

L      = int(sys.argv[1]) if len(sys.argv) > 1 else 6   # usage: python3 mechanism_pilot.py [L]
k      = bhh.k_sites(L)                        # ceil(L/3)
JUS    = [0.12, 0.16, 0.20, 0.24, 0.30, 0.40] # 4 requested + 2 crossover brackets
TAUS   = [1, 2, 3]
DG     = 0.1                                  # linear-response dose
GEXTRA = bhh.GAMMA_EXTRA                       # = 0.5, the paper's intervention strength


def bond_kinetic_op(b, basis, idx_map, nmax):
    """K_b = a_b^dag a_{b+1} + a_{b+1}^dag a_b  in the fixed-N Fock basis.
    <K_b> on the (real, symmetric) burn-in state measures the magnitude of
    coherent hopping on bond b -- the local transport activity. (Net current
    <-i(a^dag a - h.c.)> is exactly zero in the reflection-symmetric state.)"""
    D = len(basis)
    K = np.zeros((D, D))
    for s, state in enumerate(basis):
        ni, nj = state[b], state[b + 1]
        if nj > 0 and ni < nmax:                          # a_b^dag a_{b+1}
            new = list(state); new[b] = ni + 1; new[b + 1] = nj - 1
            j = idx_map.get(tuple(new))
            if j is not None:
                K[j, s] += np.sqrt((ni + 1) * nj)
        if ni > 0 and nj < nmax:                          # a_{b+1}^dag a_b
            new = list(state); new[b] = ni - 1; new[b + 1] = nj + 1
            j = idx_map.get(tuple(new))
            if j is not None:
                K[j, s] += np.sqrt(ni * (nj + 1))
    return K


def safe_spear(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return np.nan
    return float(spearmanr(x, y).correlation)


def topk_set(vals, kk):
    return tuple(sorted(int(s) for s in np.argsort(vals)[-kk:]))


def selector_sets(Fi, occ_burn, offsite, L, k):
    """Return the k-site subset chosen by each selector."""
    center = (L - 1) / 2.0
    geo = tuple(sorted(sorted(range(L), key=lambda i: abs(i - center))[:k]))
    return {
        "fi":     topk_set(Fi, k),                  # top-k local number variance
        "kernel": topk_set(offsite, k),             # top-k off-site redistribution susceptibility
        "geo":    geo,                              # geometric centre
        "maxn":   topk_set(occ_burn, k),            # top-k mean occupation
        "anti":   tuple(sorted(int(s) for s in np.argsort(Fi)[:k])),  # bottom-k F_i (contrast)
    }


rows = []
for ju in JUS:
    cond     = bhh.build_condition(L, ju)
    Fi       = cond["Fi"]
    occ_burn = cond["occ_burn"]
    rho_burn = cond["rho_burn"]
    basis    = cond["basis"]
    idx_map  = cond["idx_map"]

    # --- transport diagnostic on the burn-in state (tau-independent) ---
    Cb = np.array([np.real(np.trace(bond_kinetic_op(b, basis, idx_map, bhh.NMAX) @ rho_burn))
                   for b in range(L - 1)])
    mean_bond_coh = float(np.mean(np.abs(Cb)))
    A = np.zeros(L)                                  # adjacent bond coherence per site
    for i in range(L):
        if i - 1 >= 0:   A[i] += abs(Cb[i - 1])
        if i < L - 1:    A[i] += abs(Cb[i])
    sp_Fi_A = safe_spear(Fi, A)

    for tau in TAUS:
        # baseline tau-evolved occupations (no extra dephasing)
        occ_base_t = bhh.occ_from_rho(bh.evolve_rho(rho_burn, cond["liouv_base"], tau), cond)

        # --- exhaustive handle metric (gamma_extra = 0.5), all C(L,k) subsets ---
        losses = {}
        for sub in itertools.combinations(range(L), k):
            occ_after = bhh.occ_from_rho(
                bhh.evolve_with_extra(cond, list(sub), tau, GEXTRA), cond)
            losses[sub] = float(sum(max(0.0, occ_burn[i] - occ_after[i]) for i in sub))
        allv = np.array(list(losses.values()))

        # --- linear response kernel R[j,i] (single-site dose dg) ---
        R = np.zeros((L, L))
        for i in range(L):
            occ_pert = bhh.occ_from_rho(bhh.evolve_with_extra(cond, [i], tau, DG), cond)
            R[:, i] = (occ_base_t - occ_pert) / DG          # loss at j per unit dose at i
        colsum_sanity = float(np.max(np.abs(R.sum(axis=0))))  # should be ~0 (N conservation)

        total   = np.abs(R).sum(axis=0)                      # total response norm  (chi_redist)
        local   = np.abs(np.diag(R))                         # on-site response |R_ii|
        offsite = total - local                              # off-site redistribution susceptibility
        pos = np.arange(L)
        spread = np.array([
            (np.abs(R[:, i]) * np.abs(pos - i)).sum() / total[i] if total[i] > 1e-12 else 0.0
            for i in range(L)])

        # --- selector comparison on the exact handle metric ---
        sels = selector_sets(Fi, occ_burn, offsite, L, k)
        sel_pct = {}
        sel_loss = {}
        for name, sub in sels.items():
            sub = tuple(sorted(sub))
            lv = losses.get(sub)
            if lv is None:   # selector picked <k distinct sites (shouldn't at k=2); skip
                sel_pct[name] = np.nan; sel_loss[name] = np.nan; continue
            sel_pct[name]  = float(100.0 * np.mean(allv <= lv))
            sel_loss[name] = lv

        # --- targeted redistribution decomposition for the F_i set (gamma_extra=0.5) ---
        S = tuple(sorted(sels["fi"])); Sset = set(S)
        occ_after_S = bhh.occ_from_rho(bhh.evolve_with_extra(cond, list(S), tau, GEXTRA), cond)
        dn = occ_after_S - occ_burn                          # signed change at intervention
        tot_move = float(np.abs(dn).sum())
        off_sel_frac = float(sum(abs(dn[j]) for j in range(L) if j not in Sset) / tot_move) \
            if tot_move > 1e-12 else np.nan
        on_sel_signed = float(sum(dn[i] for i in S))         # net change at selected (neg = drained)

        rows.append(dict(
            L=L, J_over_U=ju, tau=tau, k=k, S=str(S),
            # handle (measured)
            handle_pct_fi=sel_pct["fi"], handle_gap=float(losses[S] - allv.mean()),
            pct_kernel=sel_pct["kernel"], pct_geo=sel_pct["geo"],
            pct_maxn=sel_pct["maxn"], pct_anti=sel_pct["anti"],
            kernel_eq_fi=int(tuple(sorted(sels["kernel"])) == S),
            fi_eq_geo=int(tuple(sorted(sels["geo"])) == S),
            # correlations F_i vs response-kernel quantities
            sp_Fi_offsite=safe_spear(Fi, offsite),
            sp_Fi_total=safe_spear(Fi, total),
            sp_Fi_local=safe_spear(Fi, local),
            sp_Fi_spread=safe_spear(Fi, spread),
            # transport
            mean_bond_coh=mean_bond_coh, sp_Fi_A=sp_Fi_A, sp_total_A=safe_spear(total, A),
            total_redist_mean=float(total.mean()), kernel_spread_mean=float(spread.mean()),
            # directional decomposition under real intervention
            off_sel_frac=off_sel_frac, on_sel_signed=on_sel_signed,
            colsum_sanity=colsum_sanity,
        ))

df = pd.DataFrame(rows)
csv_path = os.path.join(OUTDIR, f"pilot_results_L{L}.csv")
df.to_csv(csv_path, index=False)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

print("\n" + "=" * 100)
print("RESPONSE-KERNEL MECHANISM PILOT  (L=6, exact Lindblad)")
print("=" * 100)
print(f"max column-sum residual (N-conservation sanity, want ~0): {df['colsum_sanity'].max():.2e}")

print("\n--- HANDLE (exhaustive percentile of top-F_i subset, gamma_extra=0.5) ---")
print(df.pivot(index="J_over_U", columns="tau", values="handle_pct_fi").round(1))

print("\n--- PARADOX CHECK: does F_i correlate with TOTAL response norm in ALL regimes? "
      "(Spearman(F_i, total)) ---")
print(df.pivot(index="J_over_U", columns="tau", values="sp_Fi_total").round(3))

print("\n--- DISCRIMINATOR A: transport scale  mean|<K_bond>|  on burn-in state (tau-independent) ---")
tdf = df[df.tau == 1][["J_over_U", "mean_bond_coh", "sp_Fi_A"]].set_index("J_over_U")
print(tdf.round(4))

print("\n--- DISCRIMINATOR B: off-selected redistribution fraction under real intervention "
      "(gamma_extra=0.5, F_i set) ---")
print(df.pivot(index="J_over_U", columns="tau", values="off_sel_frac").round(3))
print("\n    net signed change at selected sites (negative = selected sites drained):")
print(df.pivot(index="J_over_U", columns="tau", values="on_sel_signed").round(4))

print("\n--- SELECTOR COMPARISON: exhaustive percentile by selector (tau=3) ---")
s3 = df[df.tau == 3][["J_over_U", "handle_pct_fi", "pct_kernel", "pct_geo", "pct_maxn", "pct_anti",
                      "kernel_eq_fi", "fi_eq_geo"]].set_index("J_over_U")
print(s3.round(1))

print("\n--- F_i vs response-kernel decomposition (Spearman), tau=3 ---")
c3 = df[df.tau == 3][["J_over_U", "sp_Fi_offsite", "sp_Fi_total", "sp_Fi_local",
                      "sp_Fi_spread"]].set_index("J_over_U")
print(c3.round(3))

# ---------------- automated kill-test verdicts ----------------
print("\n" + "=" * 100)
print("KILL-TEST VERDICTS")
print("=" * 100)

def regime_mean(col, jus):
    return df[df.J_over_U.isin(jus)][col].mean()

low = [0.12]; pocket = [0.30, 0.40]
verdicts = {}

# KT1: F_i must correlate with off-site redistribution norm in the positive pocket
off_pocket = regime_mean("sp_Fi_offsite", pocket)
verdicts["KT1 F_i~offsite in pocket (want >0)"] = (off_pocket, off_pocket > 0)

# KT2: total-response correlation must NOT discriminate (paper's stated mechanism is non-discriminating)
tot_low = regime_mean("sp_Fi_total", low); tot_pocket = regime_mean("sp_Fi_total", pocket)
verdicts["KT2 Spearman(F_i,total): low vs pocket"] = ((tot_low, tot_pocket),
                                                      tot_low >= tot_pocket - 0.05)

# KT3: transport scale must turn on across the crossover (gate hypothesis)
bc_low = regime_mean("mean_bond_coh", low); bc_pocket = regime_mean("mean_bond_coh", pocket)
verdicts["KT3 transport gate: bond-coh low vs pocket"] = ((bc_low, bc_pocket),
                                                          bc_pocket > 3 * max(bc_low, 1e-9))

# KT4: kernel-optimal selector should NOT strongly beat F_i (else F_i is mere heuristic)
beat = (df["pct_kernel"] - df["handle_pct_fi"])
verdicts["KT4 kernel minus F_i percentile (max over conds)"] = (float(beat.max()),
                                                                float(beat.max()) <= 5.0)

for kkey, (val, ok) in verdicts.items():
    print(f"  [{'PASS' if ok else 'FLAG'}] {kkey}: {val}")

print(f"\n[saved] {csv_path}")
print("[done]")
