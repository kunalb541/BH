#!/usr/bin/env python3
"""
Local-loss science pilot (clean L=6), using the Stage-0-validated multi-N machinery.

Intervention: extra amplitude damping L_i=a_i at the selected set S after burn-in (N changes).
Primary target: total system particle loss  dN_total = N0 - <N(tau)>.

Critical control for an N-changing channel: baseline occupation. For loss the competing
explanation is "sites lose more simply because <n_i> is larger there", so we compare F_i
against maxn (top-<n_i>), geo, edge, and exhaustive subsets, and correlate dN_total with
sum<n_i>, sumF_i, sum(burn-in current), and geometry.
"""
import os, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import scipy.sparse as sp

import bh
import bh_hardening as bhh
from bh import (basis_index, number_op, build_hamiltonian, build_liouvillian,
                _make_site_dissipator, evolve_rho, site_expectations)
bh._SPARSE_D_THRESHOLD = 60

L, NMAX, N0, U, GAMMA = 6, 3, 3, 1.0, 0.1
JUS = [0.12, 0.20, 0.30, 0.40]
TAUS = [1, 2, 3]
GLOSS = [0.02, 0.05, 0.1]
k = bhh.k_sites(L)
OUT = "outputs/mechanism_pilot"


def annihilation_op(site, basis, idx):
    D = len(basis); A = np.zeros((D, D))
    for s, st in enumerate(basis):
        ni = st[site]
        if ni > 0:
            t = list(st); t[site] = ni - 1
            j = idx.get(tuple(t))
            if j is not None:
                A[j, s] += np.sqrt(ni)
    return A


def current_op(b, J, basis, idx, nmax):
    D = len(basis); A = np.zeros((D, D))
    for s, st in enumerate(basis):
        ni, nj = st[b], st[b + 1]
        if nj > 0 and ni < nmax:
            new = list(st); new[b] = ni + 1; new[b + 1] = nj - 1
            j = idx.get(tuple(new))
            if j is not None:
                A[j, s] += np.sqrt((ni + 1) * nj)
    return -1j * J * (A - A.T)


def sp_(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    return float("nan") if (np.std(x) < 1e-15 or np.std(y) < 1e-15) else float(spearmanr(y, x).correlation)


# multi-N basis + operators (J/U-independent except H)
mb = sorted(s for s in itertools.product(range(NMAX + 1), repeat=L) if sum(s) <= N0)
midx = basis_index(mb); D = len(mb)
n_ops = [number_op(i, D, mb) for i in range(L)]
a_ops = [annihilation_op(i, mb, midx) for i in range(L)]
Nop = sum(n_ops)
loss_diss = [_make_site_dissipator(a_ops[i], D) for i in range(L)]
subsets = list(itertools.combinations(range(L), k))

rows = []
for ju in JUS:
    J = ju * U
    H = build_hamiltonian(L, J, U, NMAX, mb, midx)
    liouv_deph = build_liouvillian(H, n_ops, [GAMMA] * L)        # multi-N, dephasing only
    cond = bhh.build_condition(L, ju)                            # fixed-N burn-in machinery
    Fi, occ_burn = cond["Fi"], cond["occ_burn"]
    emap = [midx[s] for s in cond["basis"]]
    rho0 = np.zeros((D, D), dtype=complex)
    rho0[np.ix_(emap, emap)] = cond["rho_burn"]                  # embed burn-in into multi-N
    Jops = [current_op(b, J, cond["basis"], cond["idx_map"], NMAX) for b in range(L - 1)]
    Jval = np.array([np.real(np.trace(Jo @ cond["rho_burn"])) for Jo in Jops])
    out_i = np.array([(Jval[i] if i < L - 1 else 0.0) - (Jval[i - 1] if i > 0 else 0.0) for i in range(L)])
    center = (L - 1) / 2.0
    S_fi = tuple(sorted(int(s) for s in np.argsort(Fi)[-k:]))
    S_geo = tuple(sorted(sorted(range(L), key=lambda i: abs(i - center))[:k]))
    S_maxn = tuple(sorted(int(s) for s in np.argsort(occ_burn)[-k:]))
    S_edge = tuple(sorted([0, L - 1]))
    sum_n  = np.array([sum(occ_burn[i] for i in s) for s in subsets])
    sum_fi = np.array([sum(Fi[i] for i in s) for s in subsets])
    sum_cur = np.array([sum(out_i[i] for i in s) for s in subsets])
    sum_geo = np.array([-sum(abs(i - center) for i in s) for s in subsets])

    for gl in GLOSS:
        for tau in TAUS:
            dN = {}
            for sub in subsets:
                liouv = liouv_deph + gl * sum(loss_diss[i] for i in sub)
                if sp.issparse(liouv): liouv = liouv.tocsr()
                dN[sub] = N0 - float(np.real(np.trace(Nop @ evolve_rho(rho0, liouv, tau))))
            allv = np.array([dN[s] for s in subsets])
            pct = lambda St: 100.0 * float(np.mean(allv <= dN[tuple(sorted(St))]))
            # on/off decomposition for the F_i set
            lf = liouv_deph + gl * sum(loss_diss[i] for i in S_fi)
            occ_after = site_expectations(evolve_rho(rho0, lf.tocsr() if sp.issparse(lf) else lf, tau), n_ops)
            on_loss = float(sum(occ_burn[i] - occ_after[i] for i in S_fi))
            off_chg = float(sum(occ_after[j] - occ_burn[j] for j in range(L) if j not in set(S_fi)))
            rows.append(dict(
                J_over_U=ju, tau=tau, gamma_loss=gl,
                pct_fi=pct(S_fi), pct_geo=pct(S_geo), pct_maxn=pct(S_maxn), pct_edge=pct(S_edge),
                dN_fi=dN[tuple(sorted(S_fi))], dN_edge=dN[tuple(sorted(S_edge))],
                dN_spread=float(allv.max() - allv.min()),
                sp_dN_n=sp_(allv, sum_n), sp_dN_fi=sp_(allv, sum_fi),
                sp_dN_cur=sp_(allv, sum_cur), sp_dN_geo=sp_(allv, sum_geo),
                fi_eq_maxn=int(tuple(sorted(S_fi)) == tuple(sorted(S_maxn))),
                fi_eq_geo=int(tuple(sorted(S_fi)) == tuple(sorted(S_geo))),
                on_loss=on_loss, off_chg=off_chg))
    print(f"  J/U={ju} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "loss_pilot_L6.csv"), index=False)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

print("\n" + "=" * 90)
print("LOCAL-LOSS PILOT (clean L=6) — primary target dN_total")
print("=" * 90)
print("\n-- handle percentile of selectors (mean over gamma_loss, tau), by J/U --")
g = df.groupby("J_over_U").agg(pct_fi=("pct_fi", "mean"), pct_geo=("pct_geo", "mean"),
                               pct_maxn=("pct_maxn", "mean"), pct_edge=("pct_edge", "mean"),
                               fi_eq_maxn=("fi_eq_maxn", "mean"), fi_eq_geo=("fi_eq_geo", "mean"))
print(g.round(2).to_string())

print("\n-- which observable explains dN_total? Spearman(dN_total, .) across subsets (mean) --")
print(f"   sum<n_i> (occupation)  = {df.sp_dN_n.mean():+.3f}   <- the confound to beat")
print(f"   sum F_i  (variance)    = {df.sp_dN_fi.mean():+.3f}")
print(f"   sum current (C_burn)   = {df.sp_dN_cur.mean():+.3f}")
print(f"   geometry               = {df.sp_dN_geo.mean():+.3f}")

print("\n-- nontriviality & edge test --")
print(f"   dN_total spread across subsets (mean) = {df.dN_spread.mean():.4f}  (0 => trivial)")
print(f"   dN_fi vs dN_edge (mean): {df.dN_fi.mean():.4f} vs {df.dN_edge.mean():.4f}")
print(f"   linear-in-gamma check (J/U=0.30, tau=3): " +
      "  ".join(f"g{gl}:{df[(df.J_over_U==0.30)&(df.tau==3)&(df.gamma_loss==gl)].dN_fi.values[0]:.4f}" for gl in GLOSS))

print("\n" + "=" * 90)
print("KILL TESTS / VERDICT")
print("=" * 90)
nontrivial = df.dN_spread.mean() > 1e-3
fi_beats_edge = df.pct_fi.mean() > df.pct_edge.mean()
occ_driven = df.sp_dN_n.mean() > 0.9 and df.sp_dN_n.mean() >= df.sp_dN_fi.mean() and df.sp_dN_n.mean() >= df.sp_dN_cur.mean()
cur_predicts = df.sp_dN_cur.mean() > 0.5
print(f"  [{'PASS' if nontrivial else 'FAIL'}] dN_total nontrivial (varies across subsets)")
print(f"  [{'PASS' if fi_beats_edge else 'FAIL'}] F_i/center beats edge (current-replenishment direction)")
print(f"  [{'PASS' if cur_predicts else 'WEAK'}] burn-in current predicts dN_total (rho>0.5)")
print(f"  [{'OCCUPATION-DRIVEN' if occ_driven else 'NOT purely occupation-driven'}] "
      f"baseline <n_i> vs F_i/current as the operative selector")
print("\n  interpretation: " + (
    "loss is dominated by baseline occupation (n_i is the selector; F_i not the operative variable for loss) -> N-conserving-channel-scoped"
    if occ_driven else
    "F_i/current carry predictive power beyond raw occupation -> selector may extend to the loss channel"))
print(f"\n[saved] {os.path.join(OUT, 'loss_pilot_L6.csv')}")
print("[done]")
