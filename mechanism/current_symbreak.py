#!/usr/bin/env python3
"""
Step 3A (cont.) — current/continuity mechanism under symmetry breaking.

Computes the burn-in net-outward-current of the F_i-selected set, C_S_burn, for the
SAME tilt and disorder conditions already run in symbreak_diag.py (matched seeds),
then tests whether C_S_burn predicts the directed self-drain D_S = -sum_{i in S} Δn_i
and the handle percentile. Cheap: only burn-in currents (no exhaustive, no kernel) --
D_S and handle are read from the existing symbreak CSVs.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import bh
import bh_hardening as bhh
from bh import _build_inhomogeneous_mu
from symbreak_diag import build_condition_mu, L, k, JUS   # reuse exact builder + grid

bh._SPARSE_D_THRESHOLD = 60
OUTDIR = os.path.join("outputs", "mechanism_pilot")


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


def C_S_burn_of(cond, ju):
    Fi = cond["Fi"]; basis = cond["basis"]; idx = cond["idx_map"]; Ln = cond["L"]
    S = sorted(int(s) for s in np.argsort(Fi)[-k:])
    Jops = [current_op(b, ju, basis, idx, bhh.NMAX) for b in range(Ln - 1)]
    Jval = np.array([np.real(np.trace(Jop @ cond["rho_burn"])) for Jop in Jops])
    out = np.zeros(Ln)
    for i in range(Ln):
        right = Jval[i] if i < Ln - 1 else 0.0
        left = Jval[i - 1] if i > 0 else 0.0
        out[i] = right - left
    return float(sum(out[i] for i in S))


def safe_sp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(spearmanr(x, y).correlation)


# ---- tilt ----
tilt_rows = []
for pat in ["tilt", "step"]:
    for mt in [0.5, 1.0, 2.0]:
        mu = _build_inhomogeneous_mu(L, mt, pat)
        for ju in JUS:
            cond = build_condition_mu(L, ju, mu)
            tilt_rows.append(dict(pattern=pat, strength=mt, J_over_U=ju,
                                  C_S_burn=C_S_burn_of(cond, ju)))
tilt_c = pd.DataFrame(tilt_rows)

# ---- disorder (seeds MUST match symbreak_diag.run_disorder) ----
dis_rows = []
for mu_max in [0.5, 1.0, 2.0]:
    for r in range(10):
        rng = np.random.default_rng(20260606 + 1000 * int(mu_max * 10) + r)
        mu = rng.uniform(-mu_max, mu_max, size=L)
        for ju in JUS:
            cond = build_condition_mu(L, ju, mu)
            dis_rows.append(dict(strength=mu_max, realization=r, J_over_U=ju,
                                 C_S_burn=C_S_burn_of(cond, ju)))
dis_c = pd.DataFrame(dis_rows)

# ---- merge with existing D_S / handle and correlate ----
def report(name, cdf, keys):
    sb = pd.read_csv(os.path.join(OUTDIR, f"symbreak_{name}.csv"))
    sb["D_S"] = -sb["on_signed"]
    m = sb.merge(cdf, on=keys, how="left")
    print("=" * 84)
    print(f"{name.upper()}  (n={len(m)})")
    print(f"  rho(C_S_burn, D_S)        = {safe_sp(m.C_S_burn, m.D_S):+.3f}")
    print(f"  rho(C_S_burn, handle pct) = {safe_sp(m.C_S_burn, m.pct_fi):+.3f}")
    pkt = m[m.J_over_U >= 0.30]
    print(f"  pocket only: rho(C_S_burn, D_S) = {safe_sp(pkt.C_S_burn, pkt.D_S):+.3f}")
    print(f"  C_S_burn by J/U (mean): " +
          "  ".join(f"{ju}:{m[m.J_over_U==ju].C_S_burn.mean():+.4f}" for ju in sorted(m.J_over_U.unique())))
    m.to_csv(os.path.join(OUTDIR, f"current_symbreak_{name}.csv"), index=False)
    return safe_sp(m.C_S_burn, m.D_S)

print("STEP 3A under symmetry breaking — does burn-in current divergence predict D_S?\n")
r_tilt = report("tilt", tilt_c, ["pattern", "strength", "J_over_U"])
r_dis = report("disorder", dis_c, ["strength", "realization", "J_over_U"])
print("\n" + "=" * 84)
print("VERDICT")
print(f"  [{'PASS' if abs(r_tilt) > 0.5 else 'FLAG'}] tilt:     rho(C_S_burn, D_S) = {r_tilt:+.3f}")
print(f"  [{'PASS' if abs(r_dis) > 0.5 else 'FLAG'}] disorder: rho(C_S_burn, D_S) = {r_dis:+.3f}")
print("[done]")
