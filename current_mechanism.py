#!/usr/bin/env python3
"""
Step 3A — current / continuity mechanism for directed self-drain.

Continuity for the Bose-Hubbard chain (the dephasing dissipator L_i=n_i conserves
<n_i> exactly, so it contributes nothing to d<n_i>/dt):

    d<n_i>/dt = <J_{i-1,i}> - <J_{i,i+1}>            (current in from left - out to right)
    outward_i = -d<n_i>/dt = <J_{i,i+1}> - <J_{i-1,i}>

Hermitian bond-current operator (current from site i to i+1):
    J_{i,i+1} = -i J ( a_i^dag a_{i+1} - a_{i+1}^dag a_i ) = 2 J Im<a_i^dag a_{i+1}>.

Directed self-drain  D_S(tau) = -sum_{i in S} [n_i(t+tau)-n_i(t)]  (>0 = S drains)
equals, by continuity, the time-integrated net OUTWARD current across S's boundary.

Question: is the SIGN/size of D_S predicted by a current quantity computable from the
burn-in state -- i.e. is "high-F_i sites become useful when they sit on an outward-
current channel" the right physical explanation? Kill tests in the summary.

Cheap clean L=6 pilot first.
"""
import os, sys, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import bh
import bh_hardening as bhh
bh._SPARSE_D_THRESHOLD = 60

OUTDIR = os.path.join("outputs", "mechanism_pilot")
os.makedirs(OUTDIR, exist_ok=True)

L    = int(sys.argv[1]) if len(sys.argv) > 1 else 6
k    = bhh.k_sites(L)
JUS  = [0.12, 0.16, 0.20, 0.24, 0.30, 0.40]
TAUS = [1, 2, 3]
DG   = 0.1


def current_op(b, J, basis, idx, nmax):
    """J_{b,b+1} = -i J (a_b^dag a_{b+1} - a_{b+1}^dag a_b).  Hermitian (complex)."""
    D = len(basis); A = np.zeros((D, D))            # A = a_b^dag a_{b+1} (real)
    for s, st in enumerate(basis):
        ni, nj = st[b], st[b + 1]
        if nj > 0 and ni < nmax:
            new = list(st); new[b] = ni + 1; new[b + 1] = nj - 1
            j = idx.get(tuple(new))
            if j is not None:
                A[j, s] += np.sqrt((ni + 1) * nj)
    return -1j * J * (A - A.T)


def bond_currents(rho, Jops):
    return np.array([np.real(np.trace(Jop @ rho)) for Jop in Jops])   # <J_{b,b+1}>, len L-1


def outward_per_site(Jval, L):
    """outward_i = <J_{i,i+1}> - <J_{i-1,i}>  (open BC: edge bonds = 0)."""
    out = np.zeros(L)
    for i in range(L):
        right = Jval[i]   if i < L - 1 else 0.0
        left  = Jval[i-1] if i > 0     else 0.0
        out[i] = right - left
    return out


def safe_sp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(spearmanr(x, y).correlation)


rows = []
cont_err_max = 0.0
integ_demo = None
for ju in JUS:
    cond = bhh.build_condition(L, ju)
    J = ju * 1.0
    Fi = cond["Fi"]; occ_burn = cond["occ_burn"]; rho_burn = cond["rho_burn"]
    basis = cond["basis"]; idx = cond["idx_map"]
    S = tuple(sorted(int(s) for s in np.argsort(Fi)[-k:])); Sset = set(S)
    Jops = [current_op(b, J, basis, idx, bhh.NMAX) for b in range(L - 1)]

    # --- continuity validation: finite-diff d<n_i>/dt vs operator divergence, at burn-in ---
    dt = 0.02
    occ_dt = bhh.occ_from_rho(bh.evolve_rho(rho_burn, cond["liouv_base"], dt), cond)
    dndt_fd = (occ_dt - occ_burn) / dt
    out_burn = outward_per_site(bond_currents(rho_burn, Jops), L)   # = -d<n>/dt (operator)
    cont_err = float(np.max(np.abs(dndt_fd - (-out_burn))))
    cont_err_max = max(cont_err_max, cont_err)

    C_S_burn = float(sum(out_burn[i] for i in S))        # burn-in net outward current of S

    for tau in TAUS:
        # response kernel diagonal R_ii (self-drain susceptibility), single-site dose
        occ_base_t = bhh.occ_from_rho(bh.evolve_rho(rho_burn, cond["liouv_base"], tau), cond)
        Rdiag = np.zeros(L)
        for i in range(L):
            op = bhh.occ_from_rho(bhh.evolve_with_extra(cond, [i], tau, DG), cond)
            Rdiag[i] = (occ_base_t[i] - op[i]) / DG

        # targeted intervention -> D_S and handle percentile
        occ_after = bhh.occ_from_rho(bhh.evolve_with_extra(cond, list(S), tau, bhh.GAMMA_EXTRA), cond)
        D_S = float(-sum(occ_after[i] - occ_burn[i] for i in S))
        losses = {}
        for sub in itertools.combinations(range(L), k):
            oa = bhh.occ_from_rho(bhh.evolve_with_extra(cond, list(sub), tau, bhh.GAMMA_EXTRA), cond)
            losses[sub] = float(sum(max(0.0, occ_burn[i] - oa[i]) for i in sub))
        allv = np.array(list(losses.values()))
        handle_pct = 100.0 * float(np.mean(allv <= losses[S]))

        rows.append(dict(
            L=L, J_over_U=ju, tau=tau, S=str(S),
            C_S_burn=C_S_burn, D_S=D_S, handle_pct=handle_pct,
            sp_out_Rii=safe_sp(out_burn, Rdiag),     # per-site: burn-in outward current vs self-drain susceptibility
            sp_Fi_out=safe_sp(Fi, out_burn),         # does F_i select outward-current sites?
            cont_err=cont_err,
        ))

    # --- one continuity-integral demo: D_S(tau=3) == integral of outward current under intervention ---
    if abs(ju - 0.30) < 1e-9:
        tau = 3; nsteps = 24; ts = np.linspace(0, tau, nsteps + 1)
        liouv_int = cond["liouv_base"] + bhh.GAMMA_EXTRA * sum(cond["site_diss"][s] for s in S)
        Jout = []
        for t in ts:
            rt = bh.evolve_rho(rho_burn, liouv_int, t) if t > 0 else rho_burn
            ov = outward_per_site(bond_currents(rt, Jops), L)
            Jout.append(sum(ov[i] for i in S))
        Jout = np.array(Jout)
        integ = float(np.sum((Jout[1:] + Jout[:-1]) / 2.0 * np.diff(ts)))
        occ_after = bhh.occ_from_rho(bh.evolve_rho(rho_burn, liouv_int, tau), cond)
        D_S_direct = float(-sum(occ_after[i] - occ_burn[i] for i in S))
        integ_demo = (integ, D_S_direct)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUTDIR, f"current_mech_L{L}.csv"), index=False)

print("=" * 88)
print(f"STEP 3A — CURRENT / CONTINUITY MECHANISM  (clean L={L})")
print("=" * 88)
print(f"[continuity sanity] max | d<n_i>/dt(finite-diff) - (in-out current) | = {cont_err_max:.2e}  (want ~0)")
if integ_demo:
    print(f"[continuity demo ] J/U=0.30,tau=3:  integral of outward current = {integ_demo[0]:+.5f}   "
          f"D_S direct = {integ_demo[1]:+.5f}   diff = {abs(integ_demo[0]-integ_demo[1]):.2e}")

print("\n-- burn-in outward current of F_i-set  C_S_burn,  D_S, handle  by J/U (tau=3) --")
t3 = df[df.tau == 3][["J_over_U", "C_S_burn", "D_S", "handle_pct"]].set_index("J_over_U")
print(t3.round(4).to_string())

print("\n-- DOES current predict directed self-drain? (Spearman over all rows) --")
print(f"   rho(C_S_burn, D_S)        = {safe_sp(df.C_S_burn, df.D_S):+.3f}")
print(f"   rho(C_S_burn, handle_pct) = {safe_sp(df.C_S_burn, df.handle_pct):+.3f}")
print(f"   per-site rho(out_burn, R_ii):  mean = {df.sp_out_Rii.mean():+.3f}  (does burn-in outward current track self-drain susceptibility?)")
print(f"   per-site rho(F_i, out_burn):   mean = {df.sp_Fi_out.mean():+.3f}  (does F_i select outward-current sites?)")

print("\n-- KILL TESTS --")
r1 = abs(safe_sp(df.C_S_burn, df.D_S))
r2 = df.sp_out_Rii.mean()
print(f"   [{'PASS' if r1 > 0.6 else 'FAIL'}] burn-in current divergence predicts D_S (|rho|>0.6): |rho|={r1:.2f}")
print(f"   [{'PASS' if abs(r2) > 0.4 else 'WEAK'}] burn-in outward current tracks per-site self-drain susceptibility: rho={r2:+.2f}")
print("\n[saved] " + os.path.join(OUTDIR, f"current_mech_L{L}.csv"))
print("[done]")
