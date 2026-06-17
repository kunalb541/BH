#!/usr/bin/env python3
"""
Step 3B — local-detuning probe: is directed self-drain dephasing-specific?

Same burn-in (clean H + uniform dephasing gamma=0.1) and same F_i as before, but the
INTERVENTION is now COHERENT local detuning instead of extra dephasing:

    H -> H + mu_extra * sum_{i in S} n_i        (applied during the post-burn evolution)

The detuning's Liouvillian contribution is the commutator superoperator
    detune_i = -i ( n_i (x) I  -  I (x) n_i^T )
(precomputed per site, exactly like the dephasing dissipators were), so the whole
machinery is reused. mu_extra > 0 raises on-site energy -> expected to DRAIN.

Convention:  D_S = -sum_{i in S} Delta n_i  (>0 = selected sites drain/lose).

Question: does the earned mechanism  F_i -> C_S_burn -> D_S  still predict the response
when the local control is coherent detuning rather than dephasing? Kill tests in summary.
Exact Lindblad, fixed-N sector, L=6.
"""
import os, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import bh
import bh_hardening as bhh
bh._SPARSE_D_THRESHOLD = 60

OUTDIR = os.path.join("outputs", "mechanism_pilot")
os.makedirs(OUTDIR, exist_ok=True)

L    = 6
k    = bhh.k_sites(L)
JUS  = [0.12, 0.20, 0.30, 0.40]
TAUS = [1, 2, 3]
MU0  = 0.5
SCAN = [0.1, 0.5, 1.0]
DG   = 0.1


def detune_superops(n_ops, D):
    I = np.eye(D)
    return [-1j * (np.kron(n, I) - np.kron(I, n.T)) for n in n_ops]


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


def evolve_detune(cond, detune, sites, tau, mu_extra):
    liouv = cond["liouv_base"] + mu_extra * sum(detune[i] for i in sites)
    return bhh.occ_from_rho(bh.evolve_rho(cond["rho_burn"], liouv, tau), cond)


def safe_sp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(spearmanr(x, y).correlation)


rows = []
for ju in JUS:
    cond = bhh.build_condition(L, ju)
    Fi = cond["Fi"]; occ = cond["occ_burn"]; rho = cond["rho_burn"]
    basis = cond["basis"]; idx = cond["idx_map"]
    S_fi = tuple(sorted(int(s) for s in np.argsort(Fi)[-k:])); Sset = set(S_fi)
    center = (L - 1) / 2.0
    S_geo = tuple(sorted(sorted(range(L), key=lambda i: abs(i - center))[:k]))
    detune = detune_superops(cond["n_ops"], cond["D"])

    # burn-in outward current of the F_i set (intervention-independent)
    Jops = [current_op(b, ju, basis, idx, bhh.NMAX) for b in range(L - 1)]
    Jval = np.array([np.real(np.trace(Jo @ rho)) for Jo in Jops])
    out = np.array([(Jval[i] if i < L - 1 else 0.0) - (Jval[i - 1] if i > 0 else 0.0) for i in range(L)])
    C_S_burn = float(sum(out[i] for i in S_fi))

    for mu in SCAN:
        taus = TAUS if abs(mu - MU0) < 1e-9 else [3]      # full grid only at mu0=0.5
        for sign in (+1, -1):
            mu_e = sign * mu
            for tau in taus:
                losses = {}
                for sub in itertools.combinations(range(L), k):
                    oa = evolve_detune(cond, detune, list(sub), tau, mu_e)
                    losses[sub] = float(sum(max(0.0, occ[i] - oa[i]) for i in sub))
                allv = np.array(list(losses.values()))
                pct = lambda St: 100.0 * float(np.mean(allv <= losses[tuple(sorted(St))]))

                oa_fi = evolve_detune(cond, detune, list(S_fi), tau, mu_e)
                dn = oa_fi - occ
                D_S = float(-sum(dn[i] for i in S_fi))
                tot = float(np.abs(dn).sum())
                off = float(sum(abs(dn[j]) for j in range(L) if j not in Sset) / tot) if tot > 1e-12 else np.nan
                oa_geo = evolve_detune(cond, detune, list(S_geo), tau, mu_e)
                D_S_geo = float(-sum((oa_geo - occ)[i] for i in S_geo))

                # detune-optimal selector via single-site drain (only at mu0=0.5)
                detune_opt_eq_fi = np.nan; pct_dopt = np.nan; sp_Fi_drain = np.nan
                if abs(mu - MU0) < 1e-9:
                    drain_i = np.array([-(evolve_detune(cond, detune, [i], tau, mu_e) - occ)[i] for i in range(L)])
                    S_dopt = tuple(sorted(int(s) for s in np.argsort(drain_i)[-k:]))
                    detune_opt_eq_fi = int(S_dopt == S_fi)
                    pct_dopt = pct(S_dopt)
                    sp_Fi_drain = safe_sp(Fi, drain_i)

                rows.append(dict(
                    J_over_U=ju, tau=tau, mu0=mu, sign=sign, mu_extra=mu_e,
                    C_S_burn=C_S_burn, D_S=D_S, D_S_geo=D_S_geo,
                    pct_fi=pct(S_fi), pct_geo=pct(S_geo), off_sel_frac=off,
                    detune_opt_eq_fi=detune_opt_eq_fi, pct_dopt=pct_dopt, sp_Fi_drain=sp_Fi_drain,
                    S_fi=str(S_fi), S_geo=str(S_geo)))
    print(f"  J/U={ju} done", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTDIR, "detune_probe_L6.csv"), index=False)

prim = df[abs(df.mu0 - MU0) < 1e-9]
pos = prim[prim.sign > 0]; neg = prim[prim.sign < 0]

print("\n" + "=" * 90)
print(f"STEP 3B — LOCAL DETUNING PROBE (clean L=6, mu0={MU0})")
print("=" * 90)

print("\n[7] SIGN TEST — D_S(F_i set) by J/U for +mu0 (raise energy) vs -mu0 (lower):")
piv = prim[prim.tau == 3].pivot_table(index="J_over_U", columns="sign", values="D_S")
piv.columns = [f"sign{int(c):+d}" for c in piv.columns]
print(piv.round(4).to_string())
print(f"   +mu0 mean D_S (tau=3) = {pos[pos.tau==3].D_S.mean():+.4f}   "
      f"-mu0 mean D_S (tau=3) = {neg[neg.tau==3].D_S.mean():+.4f}")

print("\n[1-4] DRAINING SIGN (+mu0): handle percentile & D_S, F_i vs geo (tau=3):")
p3 = pos[pos.tau == 3][["J_over_U", "C_S_burn", "D_S", "D_S_geo", "pct_fi", "pct_geo"]].set_index("J_over_U")
print(p3.round(4).to_string())

print("\n[5] does burn-in current divergence predict detuning D_S?  (draining sign +mu0)")
print(f"   rho(C_S_burn, D_S)        = {safe_sp(pos.C_S_burn, pos.D_S):+.3f}")
print(f"   rho(C_S_burn, handle pct) = {safe_sp(pos.C_S_burn, pos.pct_fi):+.3f}")
print(f"   (compare: dephasing gave rho(C_S_burn,D_S)=+0.91, handle=+0.94)")

print("\n[2-3] F_i vs geo (draining sign, pocket J/U>=0.30): "
      f"pct_fi={pos[pos.J_over_U>=0.3].pct_fi.mean():.1f}  pct_geo={pos[pos.J_over_U>=0.3].pct_geo.mean():.1f}")

print("\n[8] does top-F_i stay the best detuning selector?  (draining sign, mu0=0.5)")
print(f"   detune-optimal-set == F_i-set in {100*pos.detune_opt_eq_fi.mean():.0f}% of conds; "
      f"mean pct_fi={pos.pct_fi.mean():.1f} vs pct_detune_opt={pos.pct_dopt.mean():.1f}; "
      f"mean rho(F_i, single-site drain)={pos.sp_Fi_drain.mean():+.2f}")

print("\n[mu0 scan] +mu0 mean D_S (tau=3) by mu0:")
print(df[(df.sign > 0) & (df.tau == 3)].groupby("mu0").D_S.mean().round(4).to_string())

print("\n" + "=" * 90)
print("KILL TESTS")
struct = (abs(pos.D_S).mean() > 1e-3) or (abs(neg.D_S).mean() > 1e-3)
sign_rev = (pos[pos.tau==3].D_S.mean() > 0) and (neg[neg.tau==3].D_S.mean() < 0)
generalizes = safe_sp(pos.C_S_burn, pos.D_S) > 0.6
fi_useful = pos[pos.J_over_U >= 0.3].pct_fi.mean() >= 80
print(f"  [{'PASS' if struct else 'FAIL'}] detuning produces a structured response")
print(f"  [{'YES' if sign_rev else 'NO'}] sign reversal: +mu0 drains, -mu0 fills")
print(f"  [{'PASS' if generalizes else 'FAIL'}] current mechanism generalizes to detuning (rho(C_S_burn,D_S)>0.6)")
print(f"  [{'PASS' if fi_useful else 'FAIL'}] top-F_i remains useful under detuning (pocket pct>=80)")
print("[done]")
