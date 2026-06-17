#!/usr/bin/env python3
"""
Step 3E (mini) — symmetry-broken local loss: boundary confirmation.

Clean chain separates F_i and <n_i> only at J/U=0.12. Here we break the symmetry (tilt +
disorder) so F_i, <n_i>, and geometry pick DIFFERENT sites, and ask the one decisive
question for the loss boundary:

    does sum<n_i> beat sum F_i as the predictor of dN_total (total system particle loss)?

If yes -> the boundary is confirmed: F_i selects N-conserving transport-control sites;
occupation selects particle-removal (loss) sites. Mini-scope (confirmation, not a battery).
"""
import os, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import scipy.sparse as sp

import bh
import bh_hardening as bhh
from bh import (basis_index, number_op, build_hamiltonian, build_liouvillian,
                _make_site_dissipator, evolve_rho)
from bh import _build_inhomogeneous_mu
from symbreak_diag import build_condition_mu
bh._SPARSE_D_THRESHOLD = 60

L, NMAX, N0, U, GAMMA = 6, 3, 3, 1.0, 0.1
JUS = [0.12, 0.30, 0.40]
TAU = 3
GLOSS = 0.05
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


# multi-N machinery (mu-independent)
mb = sorted(s for s in itertools.product(range(NMAX + 1), repeat=L) if sum(s) <= N0)
midx = basis_index(mb); D = len(mb)
n_ops = [number_op(i, D, mb) for i in range(L)]
a_ops = [annihilation_op(i, mb, midx) for i in range(L)]
Nop = sum(n_ops)
loss_diss = [_make_site_dissipator(a_ops[i], D) for i in range(L)]
subsets = list(itertools.combinations(range(L), k))
center = (L - 1) / 2.0


def one_condition(mu_vec, ju, label, extra):
    J = ju * U
    H = build_hamiltonian(L, J, U, NMAX, mb, midx, mu=mu_vec)
    liouv_deph = build_liouvillian(H, n_ops, [GAMMA] * L)
    cond = build_condition_mu(L, ju, mu_vec)
    Fi, occ_burn = cond["Fi"], cond["occ_burn"]
    emap = [midx[s] for s in cond["basis"]]
    rho0 = np.zeros((D, D), dtype=complex); rho0[np.ix_(emap, emap)] = cond["rho_burn"]
    Jops = [current_op(b, J, cond["basis"], cond["idx_map"], NMAX) for b in range(L - 1)]
    Jval = np.array([np.real(np.trace(Jo @ cond["rho_burn"])) for Jo in Jops])
    out_i = np.array([(Jval[i] if i < L - 1 else 0.0) - (Jval[i - 1] if i > 0 else 0.0) for i in range(L)])
    S_fi = tuple(sorted(int(s) for s in np.argsort(Fi)[-k:]))
    S_maxn = tuple(sorted(int(s) for s in np.argsort(occ_burn)[-k:]))
    S_geo = tuple(sorted(sorted(range(L), key=lambda i: abs(i - center))[:k]))
    dN = {}
    for sub in subsets:
        liouv = liouv_deph + GLOSS * sum(loss_diss[i] for i in sub)
        if sp.issparse(liouv): liouv = liouv.tocsr()
        dN[sub] = N0 - float(np.real(np.trace(Nop @ evolve_rho(rho0, liouv, TAU))))
    allv = np.array([dN[s] for s in subsets])
    pct = lambda St: 100.0 * float(np.mean(allv <= dN[tuple(sorted(St))]))
    sum_n = np.array([sum(occ_burn[i] for i in s) for s in subsets])
    sum_fi = np.array([sum(Fi[i] for i in s) for s in subsets])
    sum_cur = np.array([sum(out_i[i] for i in s) for s in subsets])
    sum_geo = np.array([-sum(abs(i - center) for i in s) for s in subsets])
    return dict(setting=label, **extra, J_over_U=ju,
                overlap_fi_maxn=len(set(S_fi) & set(S_maxn)) / k,
                overlap_fi_geo=len(set(S_fi) & set(S_geo)) / k,
                pct_fi=pct(S_fi), pct_maxn=pct(S_maxn), pct_geo=pct(S_geo),
                sp_dN_n=sp_(allv, sum_n), sp_dN_fi=sp_(allv, sum_fi),
                sp_dN_cur=sp_(allv, sum_cur), sp_dN_geo=sp_(allv, sum_geo))


rows = []
for mt in (1.0, 2.0):
    mu = _build_inhomogeneous_mu(L, mt, "tilt")
    for ju in JUS:
        rows.append(one_condition(mu, ju, "tilt", {"strength": mt, "realization": -1}))
print("  tilt done", flush=True)
for mu_max in (1.0, 2.0):
    for r in range(5):
        rng = np.random.default_rng(77000 + 1000 * int(mu_max * 10) + r)
        mu = rng.uniform(-mu_max, mu_max, size=L)
        for ju in JUS:
            rows.append(one_condition(mu, ju, "disorder", {"strength": mu_max, "realization": r}))
    print(f"  disorder mu_max={mu_max} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "loss_symbreak_mini.csv"), index=False)

# decisive: restrict to where F_i and maxn actually separate
sep = df[df.overlap_fi_maxn < 1.0]
print("\n" + "=" * 88)
print("STEP 3E — SYMMETRY-BROKEN LOSS (mini boundary confirmation)")
print("=" * 88)
print(f"conditions: {len(df)} total; F_i != maxn in {len(sep)} (mean overlap_fi_maxn={df.overlap_fi_maxn.mean():.2f})")
print("\n-- predictor of dN_total (Spearman, mean over ALL conditions) --")
print(f"   sum<n_i> (occupation) = {df.sp_dN_n.mean():+.3f}")
print(f"   sum F_i  (variance)   = {df.sp_dN_fi.mean():+.3f}")
print(f"   sum current           = {df.sp_dN_cur.mean():+.3f}")
print(f"   geometry              = {df.sp_dN_geo.mean():+.3f}")
print("\n-- decisive: ONLY where F_i and maxn pick different sites --")
if len(sep):
    print(f"   n  conditions = {len(sep)}")
    print(f"   sum<n_i> = {sep.sp_dN_n.mean():+.3f}   vs   sum F_i = {sep.sp_dN_fi.mean():+.3f}")
    print(f"   handle pct: maxn = {sep.pct_maxn.mean():.1f}   vs   F_i = {sep.pct_fi.mean():.1f}   "
          f"(maxn>=fi in {100*np.mean(sep.pct_maxn>=sep.pct_fi):.0f}%)")
print("\n" + "=" * 88)
occ_wins = (sep.sp_dN_n.mean() > sep.sp_dN_fi.mean()) and (sep.pct_maxn.mean() >= sep.pct_fi.mean()) if len(sep) else False
print(f"BOUNDARY {'CONFIRMED' if occ_wins else 'NOT confirmed'}: "
      + ("occupation beats variance for loss even when they separate -> "
         "F_i is N-conserving-channel-scoped; <n_i> is the loss selector."
         if occ_wins else "occupation does NOT cleanly beat variance — revisit."))
print("[done]")
