#!/usr/bin/env python3
"""
Step 2 — symmetry-broken response-kernel diagnostic.

Tests whether the directed self-draining mechanism survives when F_i is NOT
identical to geometry. In the clean chain F_i = geo by reflection symmetry, so a
referee can say "maybe F_i just selects central sites." Here we break the symmetry
(deterministic tilt/step, and random disorder) and ask:

  - Does top-F_i still select the self-draining response sites (R_ii > 0)?
  - Does top-F_i beat / stay competitive with geometry in the positive pocket?
  - Does the net-self-drain sign crossover persist near J/U ~ 0.20-0.30?
  - Is the old F_i<->redistribution-susceptibility still non-discriminating?

Selectors (k = ceil(L/3)):
  fi   : top-k by F_i
  geo  : k sites nearest chain centre
  sd   : top-k by diagonal self-drain susceptibility R_ii (response-kernel-optimal
         self-drain selector); R_ii > 0 means dephasing site i drains site i.

Exact Lindblad only. Reuses validated bh / bh_hardening machinery.

Usage:
  python3 symbreak_diag.py tilt
  python3 symbreak_diag.py disorder [n_real]
  python3 symbreak_diag.py tilt --smoke
"""
import os, sys, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import bh
import bh_hardening as bhh
from bh import (build_basis, basis_index, number_op, build_hamiltonian,
                build_liouvillian, _make_site_dissipator, evolve_rho,
                _fast_variances, _fast_expectations, _build_inhomogeneous_mu)

OUTDIR = os.path.join("outputs", "mechanism_pilot")
os.makedirs(OUTDIR, exist_ok=True)

L      = 6
k      = bhh.k_sites(L)
JUS    = [0.12, 0.20, 0.30, 0.40]
TAUS   = [1, 2, 3]
DG     = 0.1
GEX    = bhh.GAMMA_EXTRA


def build_condition_mu(L, ju, mu):
    """bh_hardening.build_condition but with an on-site potential mu (length L)."""
    U = 1.0; J = ju * U; N = bhh.filling(L); nmax = bhh.NMAX
    basis = build_basis(L, N, nmax); idx = basis_index(basis); D = len(basis)
    H = build_hamiltonian(L, J, U, nmax, basis, idx, mu=mu)
    n_ops = [number_op(i, D, basis) for i in range(L)]
    n_diags = np.array([np.diag(n) for n in n_ops], dtype=np.float64); n2 = n_diags ** 2
    liouv = build_liouvillian(H, n_ops, [bhh.GAMMA_BASE] * L)
    site_diss = [_make_site_dissipator(n_ops[i], D) for i in range(L)]
    w, v = np.linalg.eigh(H); psi = v[:, 0]; rho0 = np.outer(psi, psi.conj())
    rho_burn = evolve_rho(rho0, liouv, bhh.BURN_IN)
    rd = np.real(np.diag(rho_burn))
    Fi = _fast_variances(rd, n_diags, n2); occ = _fast_expectations(rd, n_diags)
    return dict(L=L, N=N, D=D, J_over_U=ju, basis=basis, idx_map=idx, H=H, n_ops=n_ops,
                n_diags=n_diags, n2_diags=n2, liouv_base=liouv, site_diss=site_diss,
                rho_burn=rho_burn, rho_diag=rd, Fi=Fi, occ_burn=occ)


def _safe_sp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(spearmanr(x, y).correlation)


def diagnose(cond, tau):
    L = cond["L"]; Fi = cond["Fi"]; occ = cond["occ_burn"]
    S_fi = tuple(sorted(int(s) for s in np.argsort(Fi)[-k:]))
    center = (L - 1) / 2.0
    S_geo = tuple(sorted(sorted(range(L), key=lambda i: abs(i - center))[:k]))

    occ_base = bhh.occ_from_rho(evolve_rho(cond["rho_burn"], cond["liouv_base"], tau), cond)
    R = np.zeros((L, L))
    for i in range(L):
        op = bhh.occ_from_rho(bhh.evolve_with_extra(cond, [i], tau, DG), cond)
        R[:, i] = (occ_base - op) / DG
    diag = np.diag(R).copy()                 # R_ii: >0 = dephasing i drains i
    total = np.abs(R).sum(axis=0)
    S_sd = tuple(sorted(int(s) for s in np.argsort(diag)[-k:]))

    losses = {}
    for sub in itertools.combinations(range(L), k):
        oa = bhh.occ_from_rho(bhh.evolve_with_extra(cond, list(sub), tau, GEX), cond)
        losses[sub] = float(sum(max(0.0, occ[i] - oa[i]) for i in sub))
    allv = np.array(list(losses.values()))
    pct = lambda St: 100.0 * float(np.mean(allv <= losses[tuple(sorted(St))]))
    sd_of = lambda St: float(sum(diag[i] for i in St))

    oa_fi = bhh.occ_from_rho(bhh.evolve_with_extra(cond, list(S_fi), tau, GEX), cond)
    dn = oa_fi - occ; tot = float(np.abs(dn).sum())
    on_signed = float(sum(dn[i] for i in S_fi))
    off_frac = float(sum(abs(dn[j]) for j in range(L) if j not in set(S_fi)) / tot) if tot > 1e-12 else np.nan

    return dict(
        tau=tau, S_fi=str(S_fi), S_geo=str(S_geo), S_sd=str(S_sd),
        overlap_fi_geo=len(set(S_fi) & set(S_geo)) / k,
        overlap_fi_sd=len(set(S_fi) & set(S_sd)) / k,
        pct_fi=pct(S_fi), pct_geo=pct(S_geo), pct_sd=pct(S_sd),
        sd_fi=sd_of(S_fi), sd_geo=sd_of(S_geo), sd_opt=sd_of(S_sd),
        on_signed=on_signed, off_frac=off_frac,
        sp_Fi_total=_safe_sp(Fi, total), sp_Fi_diag=_safe_sp(Fi, diag),
    )


def run_tilt(smoke=False):
    tilts = [1.0] if smoke else [0.5, 1.0, 2.0]
    patterns = ["tilt"] if smoke else ["tilt", "step"]
    jus = [0.30] if smoke else JUS
    taus = [3] if smoke else TAUS
    rows = []
    for pat in patterns:
        for mt in tilts:
            mu = _build_inhomogeneous_mu(L, mt, pat)
            for ju in jus:
                cond = build_condition_mu(L, ju, mu)
                for tau in taus:
                    d = diagnose(cond, tau)
                    d.update(setting="tilt", pattern=pat, strength=mt, J_over_U=ju)
                    rows.append(d)
            print(f"  [tilt] pattern={pat} strength={mt} done", flush=True)
    return pd.DataFrame(rows)


def run_disorder(n_real=10, smoke=False):
    mus = [1.0] if smoke else [0.5, 1.0, 2.0]
    jus = [0.30] if smoke else JUS
    taus = [3] if smoke else TAUS
    nr = 1 if smoke else n_real
    rows = []
    for mu_max in mus:
        for r in range(nr):
            rng = np.random.default_rng(20260606 + 1000 * int(mu_max * 10) + r)
            mu = rng.uniform(-mu_max, mu_max, size=L)
            for ju in jus:
                cond = build_condition_mu(L, ju, mu)
                for tau in taus:
                    d = diagnose(cond, tau)
                    d.update(setting="disorder", pattern="rand", strength=mu_max,
                             realization=r, J_over_U=ju)
                    rows.append(d)
        print(f"  [disorder] mu_max={mu_max} ({nr} real) done", flush=True)
    return pd.DataFrame(rows)


def summarize(df, label):
    print("\n" + "=" * 90)
    print(f"SUMMARY — {label}   (n rows = {len(df)})")
    print("=" * 90)
    pkt = df[df.J_over_U >= 0.30]
    broke = df[df.overlap_fi_geo < 1.0]          # rows where F_i != geometry
    print(f"symmetry actually broken (F_i != geo): {len(broke)}/{len(df)} rows "
          f"(mean overlap fi-geo = {df.overlap_fi_geo.mean():.2f})")

    print("\n-- handle percentile (mean), pocket J/U>=0.30 --")
    print(f"   fi={pkt.pct_fi.mean():.1f}  geo={pkt.pct_geo.mean():.1f}  "
          f"sd_opt={pkt.pct_sd.mean():.1f}   fi>=geo in {100*np.mean(pkt.pct_fi>=pkt.pct_geo):.0f}% of pocket rows")

    print("\n-- self-drain susceptibility of selected set (mean sum R_ii), pocket --")
    print(f"   fi={pkt.sd_fi.mean():+.4f}  geo={pkt.sd_geo.mean():+.4f}  "
          f"opt={pkt.sd_opt.mean():+.4f}   fi>geo in {100*np.mean(pkt.sd_fi>pkt.sd_geo):.0f}% of pocket rows")

    if len(broke):
        bp = broke[broke.J_over_U >= 0.30]
        if len(bp):
            print("\n-- WHERE F_i != geo, pocket only (the decisive geometry-separation test) --")
            print(f"   n={len(bp)}  self-drain fi={bp.sd_fi.mean():+.4f} vs geo={bp.sd_geo.mean():+.4f}  "
                  f"(fi>geo in {100*np.mean(bp.sd_fi>bp.sd_geo):.0f}%)")
            print(f"   handle pct fi={bp.pct_fi.mean():.1f} vs geo={bp.pct_geo.mean():.1f}  "
                  f"(fi>=geo in {100*np.mean(bp.pct_fi>=bp.pct_geo):.0f}%)")

    print("\n-- net self-drain sign at F_i set by J/U (mean on_signed; <0 = drained) --")
    print(df.groupby("J_over_U").on_signed.mean().round(4).to_string())

    print("\n-- predictors of the handle (Spearman over all rows) --")
    print(f"   rho(pct_fi, off_frac)         = {_safe_sp(df.pct_fi, df.off_frac):+.3f}")
    print(f"   rho(pct_fi, on_signed)        = {_safe_sp(df.pct_fi, df.on_signed):+.3f}")
    print(f"   rho(pct_fi, OLD Fi-redist)    = {_safe_sp(df.pct_fi, df.sp_Fi_total):+.3f}")
    print(f"   F_i tracks diagonal self-drain: mean sp_Fi_diag (pocket) = {pkt.sp_Fi_diag.mean():+.3f}")

    # decision flags
    fi_competitive = (pkt.pct_fi.mean() >= pkt.pct_geo.mean() - 3.0)
    fi_self_drains = (np.mean(pkt.sd_fi > pkt.sd_geo) >= 0.5) if len(pkt) else False
    pred = (abs(_safe_sp(df.pct_fi, df.on_signed)) > 0.7 or abs(_safe_sp(df.pct_fi, df.off_frac)) > 0.7)
    old_nondiscrim = abs(_safe_sp(df.pct_fi, df.sp_Fi_total)) < 0.5
    print("\n-- DECISION FLAGS --")
    print(f"   [{'PASS' if fi_competitive else 'FAIL'}] F_i competitive-with/beats geo in pocket")
    print(f"   [{'PASS' if fi_self_drains else 'FLAG'}] F_i-set more self-draining than geo-set in pocket")
    print(f"   [{'PASS' if pred else 'FAIL'}] self-drain/off-frac predicts handle (|rho|>0.7)")
    print(f"   [{'PASS' if old_nondiscrim else 'FLAG'}] old F_i-redistribution-susceptibility non-discriminating")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tilt"
    smoke = "--smoke" in sys.argv
    if mode == "tilt":
        df = run_tilt(smoke=smoke)
        path = os.path.join(OUTDIR, "symbreak_tilt.csv")
    else:
        nreal = next((int(a) for a in sys.argv[2:] if a.isdigit()), 10)
        df = run_disorder(n_real=nreal, smoke=smoke)
        path = os.path.join(OUTDIR, "symbreak_disorder.csv")
    if not smoke:
        df.to_csv(path, index=False)
    summarize(df, f"{mode}{' [SMOKE]' if smoke else ''}")
    if not smoke:
        print(f"\n[saved] {path}")
    print("[done]")
