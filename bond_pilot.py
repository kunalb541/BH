#!/usr/bin/env python3
"""
Step 4 — bond / hopping-modulation pilot (Stage 0 sanity + clean L=6 science).

Intervention: J_b -> J_b(1+delta) on a selected bond set B (coherent, N-conserving).
Endpoint set E(B) = sites touching a selected bond.

Targets (per endpoint, |E(B)| varies 3 vs 4 for adjacent vs disjoint bond pairs):
  total    : D_E_per      = [-sum_{i in E} Delta n_i] / |E|
  INDUCED  : dD_E_per     = D_E_per(delta) - D_E_per(delta=0)   <- PRIMARY (isolates bond control;
             raw total is baseline-dominated at small delta, i.e. "which endpoints drain anyway").

k_b = 2 selected bonds; exhaustive over C(5,2)=10 pairs. Selectors: endpoint-F sum
(site-F-derived), bond coherence, |bond current|, endpoint occupation (confound), exhaustive.
"""
import os, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import bh
import bh_hardening as bhh
from bh import build_hamiltonian, build_liouvillian, evolve_rho, site_expectations

L, NMAX, N0, U, GAMMA = 6, 3, 3, 1.0, 0.1
JUS = [0.12, 0.20, 0.30, 0.40]
TAUS = [1, 2, 3]
DELTAS = [0.1, 0.2, 0.5, -0.2]
KB = 2
OUT = "outputs/mechanism_pilot"


def build_H_jbonds(L, Jb, U, nmax, basis, idx):
    D = len(basis); H = np.zeros((D, D))
    for s, st in enumerate(basis):
        for site in range(L):
            ni = st[site]; H[s, s] += 0.5 * U * ni * (ni - 1)
    for b in range(L - 1):
        for s, st in enumerate(basis):
            ni, nj = st[b], st[b + 1]
            if nj > 0 and ni < nmax:
                t = list(st); t[b] = ni + 1; t[b + 1] = nj - 1; j = idx.get(tuple(t))
                if j is not None: H[j, s] += -Jb[b] * np.sqrt((ni + 1) * nj)
            if ni > 0 and nj < nmax:
                t = list(st); t[b] = ni - 1; t[b + 1] = nj + 1; j = idx.get(tuple(t))
                if j is not None: H[j, s] += -Jb[b] * np.sqrt(ni * (nj + 1))
    return H


def bond_coh_op(b, basis, idx, nmax):
    D = len(basis); A = np.zeros((D, D))
    for s, st in enumerate(basis):
        ni, nj = st[b], st[b + 1]
        if nj > 0 and ni < nmax:
            t = list(st); t[b] = ni + 1; t[b + 1] = nj - 1; j = idx.get(tuple(t))
            if j is not None: A[j, s] += np.sqrt((ni + 1) * nj)
    return A + A.T


def bond_cur_op(b, J, basis, idx, nmax):
    D = len(basis); A = np.zeros((D, D))
    for s, st in enumerate(basis):
        ni, nj = st[b], st[b + 1]
        if nj > 0 and ni < nmax:
            t = list(st); t[b] = ni + 1; t[b + 1] = nj - 1; j = idx.get(tuple(t))
            if j is not None: A[j, s] += np.sqrt((ni + 1) * nj)
    return -1j * J * (A - A.T)


def endpoints(B):
    return sorted(set([b for b in B] + [b + 1 for b in B]))


def evolve_mod(cnd, B, ju, delta, tau):
    Jb = [ju if b not in B else ju * (1 + delta) for b in range(L - 1)]
    H = build_H_jbonds(L, Jb, U, NMAX, cnd["basis"], cnd["idx_map"])
    liouv = build_liouvillian(H, cnd["n_ops"], [GAMMA] * L)
    return site_expectations(evolve_rho(cnd["rho_burn"], liouv, tau), cnd["n_ops"])


def D_E_per_val(occ_after, occ_burn, E):
    return (-sum(occ_after[i] - occ_burn[i] for i in E)) / len(E)


def sp_(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    return float("nan") if (np.std(x) < 1e-15 or np.std(y) < 1e-15) else float(spearmanr(y, x).correlation)


# ============ STAGE 0 ============
print("=" * 84); print("STAGE 0 — per-bond-J Hamiltonian sanity"); print("=" * 84)
ok = True
cond = bhh.build_condition(L, 0.30); J = 0.30
basis, idx, n_ops, occ_burn = cond["basis"], cond["idx_map"], cond["n_ops"], cond["occ_burn"]
H_scalar = build_hamiltonian(L, J, U, NMAX, basis, idx)
H_uniform = build_H_jbonds(L, [J] * (L - 1), U, NMAX, basis, idx)
g1 = np.allclose(H_scalar, H_uniform, atol=1e-12); ok &= g1
print(f"  [{'PASS' if g1 else 'FAIL'}] per-bond-J (all equal) == scalar-J H  (max|Δ|={np.max(np.abs(H_scalar-H_uniform)):.1e})")
occ_b = evolve_mod(cond, (), 0.30, 0.0, 2.0)
occ_d0 = evolve_mod(cond, (2,), 0.30, 0.0, 2.0)
g2 = np.allclose(occ_b, occ_d0, atol=1e-12); ok &= g2
print(f"  [{'PASS' if g2 else 'FAIL'}] delta=0 reproduces baseline evolution")
Kb = bond_coh_op(2, basis, idx, NMAX)
elem = lambda d: np.max(np.abs(build_H_jbonds(L, [J if b != 2 else J*(1+d) for b in range(L-1)], U, NMAX, basis, idx)) * (np.abs(Kb) > 0))
g3 = elem(0.5) > elem(0.0) > elem(-0.5); ok &= g3
print(f"  [{'PASS' if g3 else 'FAIL'}] +delta raises / -delta lowers bond-2 hopping magnitude")
g4 = np.allclose(H_uniform, H_uniform.T, atol=1e-12); ok &= g4
print(f"  [{'PASS' if g4 else 'FAIL'}] Hamiltonian Hermitian (real symmetric)")
# INDUCED small-delta linear response
E2 = endpoints((2,))
base2 = D_E_per_val(evolve_mod(cond, (2,), 0.30, 0.0, 3.0), occ_burn, E2)
ind = [D_E_per_val(evolve_mod(cond, (2,), 0.30, d, 3.0), occ_burn, E2) - base2 for d in (0.01, 0.02, 0.04)]
ratios = [ind[1] / ind[0], ind[2] / ind[1]] if abs(ind[0]) > 1e-12 else [np.nan, np.nan]
g5 = all(1.6 < r < 2.4 for r in ratios); ok &= g5
print(f"  [{'PASS' if g5 else 'FAIL'}] INDUCED drain linear at small delta (ratios ~2: {ratios[0]:.2f}, {ratios[1]:.2f})")
print(f"\nSTAGE 0: {'ALL PASS' if ok else 'FAILED'}")
if not ok:
    print("Aborting pilot."); raise SystemExit(1)

# ============ PILOT ============
bond_subsets = list(itertools.combinations(range(L - 1), KB))
rows = []
for ju in JUS:
    cond = bhh.build_condition(L, ju)
    basis, idx, n_ops = cond["basis"], cond["idx_map"], cond["n_ops"]
    Fi, occ_burn = cond["Fi"], cond["occ_burn"]
    coh = np.array([np.real(np.trace(bond_coh_op(b, basis, idx, NMAX) @ cond["rho_burn"])) for b in range(L - 1)])
    cur = np.array([abs(np.real(np.trace(bond_cur_op(b, ju, basis, idx, NMAX) @ cond["rho_burn"]))) for b in range(L - 1)])
    Fsum = np.array([Fi[b] + Fi[b + 1] for b in range(L - 1)])
    osum = np.array([occ_burn[b] + occ_burn[b + 1] for b in range(L - 1)])
    score = {"endpointF": Fsum, "coh": coh, "cur": cur, "occ": osum}
    topB = {nm: tuple(sorted(int(b) for b in np.argsort(v)[-KB:])) for nm, v in score.items()}
    for tau in TAUS:
        occ_base_after = evolve_mod(cond, (), ju, 0.0, tau)        # baseline once per (ju,tau)
        base_per = {B: D_E_per_val(occ_base_after, occ_burn, endpoints(B)) for B in bond_subsets}
        for delta in DELTAS:
            tot, ind_ = {}, {}
            for B in bond_subsets:
                E = endpoints(B)
                tp = D_E_per_val(evolve_mod(cond, B, ju, delta, tau), occ_burn, E)
                tot[B] = tp; ind_[B] = tp - base_per[B]
            allt = np.array([tot[B] for B in bond_subsets])
            alli = np.array([ind_[B] for B in bond_subsets])
            pct_ind = lambda B: 100.0 * float(np.mean(alli <= ind_[tuple(sorted(B))]))
            pct_tot = lambda B: 100.0 * float(np.mean(allt <= tot[tuple(sorted(B))]))
            sums = {nm: np.array([sum(score[nm][b] for b in B) for B in bond_subsets]) for nm in score}
            row = dict(J_over_U=ju, delta=delta, tau=tau, spread_ind=float(alli.max() - alli.min()))
            for nm in score:
                row[f"pct_{nm}"] = pct_ind(topB[nm])          # PRIMARY: induced
                row[f"pcttot_{nm}"] = pct_tot(topB[nm])        # secondary: total
                row[f"sp_{nm}"] = sp_(alli, sums[nm])          # Spearman(induced, score)
                row[f"sptot_{nm}"] = sp_(allt, sums[nm])
            rows.append(row)
    print(f"  J/U={ju} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "bond_pilot_L6.csv"), index=False)
pos = df[df.delta > 0]

print("\n" + "=" * 84); print("BOND / HOPPING-MODULATION PILOT (clean L=6) — PRIMARY = induced per-endpoint drain"); print("=" * 84)
print("\n-- which BOND observable predicts INDUCED drain? Spearman across subsets (mean, delta>0) --")
for nm, lab in [("endpointF", "endpoint-F sum (site-F-derived)"), ("coh", "bond coherence"),
                ("cur", "|bond current|"), ("occ", "endpoint occupation (confound)")]:
    print(f"   {lab:34s} = {pos[f'sp_{nm}'].mean():+.3f}   (raw-total: {pos[f'sptot_{nm}'].mean():+.3f})")
print("\n-- selector handle percentile by INDUCED drain (mean, delta>0) --")
for nm in ["endpointF", "coh", "cur", "occ"]:
    print(f"   {nm:10s}: induced={pos[f'pct_{nm}'].mean():.1f}   total={pos[f'pcttot_{nm}'].mean():.1f}")
print(f"\n-- nontriviality: induced spread across subsets (mean) = {pos.spread_ind.mean():.4f}")
print(f"-- sign check: induced endpointF pct  +0.2 vs -0.2 = "
      f"{pos[pos.delta==0.2].pct_endpointF.mean():.1f} vs {df[df.delta==-0.2].pct_endpointF.mean():.1f}")

print("\n" + "=" * 84); print("KILL TESTS / VERDICT"); print("=" * 84)
# induced (clean bond-control) target
sp_coh, sp_eF, sp_occ, sp_cur = pos.sp_coh.mean(), pos.sp_endpointF.mean(), pos.sp_occ.mean(), pos.sp_cur.mean()
# total (baseline-confounded) target
t_coh, t_eF, t_occ, t_cur = pos.sptot_coh.mean(), pos.sptot_endpointF.mean(), pos.sptot_occ.mean(), pos.sptot_cur.mean()
nontrivial = pos.spread_ind.mean() > 1e-4
best_ind = max(abs(sp_coh), abs(sp_eF), abs(sp_occ), abs(sp_cur))
induced_predicted = best_ind > 0.4                      # a real correlation, not noise
bond_beats_site_tot = max(t_coh, t_cur) > t_eF + 0.1    # bond-intrinsic beats site-derived on total
print(f"  [{'PASS' if nontrivial else 'FAIL'}] induced endpoint drain is structured/linear (spread {pos.spread_ind.mean():.4f})")
print(f"  induced target predicted by ANY observable (|rho|>0.4)? {'yes' if induced_predicted else 'NO'}  (best |rho|={best_ind:.2f})")
print(f"  total target Spearman: endpointF={t_eF:+.2f}  occ={t_occ:+.2f}  coh={t_coh:+.2f}  cur={t_cur:+.2f}")
print(f"  bond-intrinsic beats site-derived endpoint-F on total? {'yes' if bond_beats_site_tot else 'NO'}")
if not induced_predicted and not bond_beats_site_tot:
    verdict = ("SITE-LEVEL KEPT — no bond-intrinsic observable (coherence, |current|) beats the "
               "site-derived endpoint-F; the clean induced bond-control effect is not predicted by "
               "any tested observable. Hopping modulation does NOT reveal a bond-level handle.")
elif bond_beats_site_tot and not induced_predicted:
    verdict = "WEAK/AMBIGUOUS — bond observable leads only on the baseline-confounded total target."
elif induced_predicted and max(sp_coh, sp_cur) > sp_eF + 0.1:
    verdict = "BOND-LEVEL PROMOTED — a bond-intrinsic observable genuinely predicts the induced effect."
else:
    verdict = "SITE-LEVEL KEPT — site-derived selector is as good or better."
print("\n  VERDICT: " + verdict)
print(f"\n[saved] {os.path.join(OUT, 'bond_pilot_L6.csv')}\n[done]")
