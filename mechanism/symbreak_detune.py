#!/usr/bin/env python3
"""
Step 3C — symmetry-broken local detuning: does F_i beat geometry under COHERENT control?

Clean-chain detuning (Step 3B) showed the current/self-drain LAW generalizes to coherent
detuning, but F_i=geo there so it could not separate F_i from geometry. Here we break the
symmetry (tilt + disorder) and apply detuning instead of dephasing:

    H -> H + mu_extra * sum_{i in S} n_i        (mu_extra = +0.5 drains; -0.5 check)

and ask whether top-F_i still beats geometry, and whether burn-in current divergence
C_S_burn still predicts the response. Exact Lindblad, L=6.

Usage: python3 symbreak_detune.py tilt   |   python3 symbreak_detune.py disorder [n_real]
"""
import os, sys, itertools
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import bh
import bh_hardening as bhh
from bh import _build_inhomogeneous_mu
from symbreak_diag import build_condition_mu, L, k, JUS    # L=6, k=2, JUS=[.12,.20,.30,.40]

bh._SPARSE_D_THRESHOLD = 60
OUTDIR = os.path.join("outputs", "mechanism_pilot")
TAUS = [1, 2, 3]
MU = 0.5


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


def evolve_detune(cond, detune, sites, tau, mu):
    liouv = cond["liouv_base"] + mu * sum(detune[i] for i in sites)
    return bhh.occ_from_rho(bh.evolve_rho(cond["rho_burn"], liouv, tau), cond)


def safe_sp(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float("nan") if (np.std(x) < 1e-15 or np.std(y) < 1e-15) else float(spearmanr(x, y).correlation)


def diagnose(cond, ju, tau, detune, Jops):
    Ln = cond["L"]; Fi = cond["Fi"]; occ = cond["occ_burn"]
    S_fi = tuple(sorted(int(s) for s in np.argsort(Fi)[-k:])); Sset = set(S_fi)
    center = (Ln - 1) / 2.0
    S_geo = tuple(sorted(sorted(range(Ln), key=lambda i: abs(i - center))[:k]))

    Jval = np.array([np.real(np.trace(Jo @ cond["rho_burn"])) for Jo in Jops])
    out = np.array([(Jval[i] if i < Ln - 1 else 0.0) - (Jval[i - 1] if i > 0 else 0.0) for i in range(Ln)])
    C_S_burn = float(sum(out[i] for i in S_fi))

    losses = {}
    for sub in itertools.combinations(range(Ln), k):
        oa = evolve_detune(cond, detune, list(sub), tau, MU)
        losses[sub] = float(sum(max(0.0, occ[i] - oa[i]) for i in sub))
    allv = np.array(list(losses.values()))
    pct = lambda St: 100.0 * float(np.mean(allv <= losses[tuple(sorted(St))]))

    oa_fi = evolve_detune(cond, detune, list(S_fi), tau, MU)
    dn = oa_fi - occ
    D_S = float(-sum(dn[i] for i in S_fi))
    tot = float(np.abs(dn).sum())
    off = float(sum(abs(dn[j]) for j in range(Ln) if j not in Sset) / tot) if tot > 1e-12 else np.nan
    D_S_geo = float(-sum((evolve_detune(cond, detune, list(S_geo), tau, MU) - occ)[i] for i in S_geo))

    drain_i = np.array([-(evolve_detune(cond, detune, [i], tau, MU) - occ)[i] for i in range(Ln)])
    S_dopt = tuple(sorted(int(s) for s in np.argsort(drain_i)[-k:]))

    D_S_neg = np.nan
    if tau == 3:                                  # sign-reversal check
        D_S_neg = float(-sum((evolve_detune(cond, detune, list(S_fi), tau, -MU) - occ)[i] for i in S_fi))

    return dict(
        tau=tau, J_over_U=ju, overlap_fi_geo=len(Sset & set(S_geo)) / k,
        C_S_burn=C_S_burn, D_S=D_S, D_S_geo=D_S_geo, off_sel_frac=off,
        pct_fi=pct(S_fi), pct_geo=pct(S_geo), pct_dopt=pct(S_dopt),
        dopt_eq_fi=int(S_dopt == S_fi), sp_Fi_drain=safe_sp(Fi, drain_i), D_S_neg=D_S_neg)


def run(mode, n_real=10):
    rows = []
    if mode == "tilt":
        combos = [("tilt", mt) for mt in (0.5, 1.0, 2.0)] + [("step", mt) for mt in (0.5, 1.0, 2.0)]
        for pat, mt in combos:
            mu = _build_inhomogeneous_mu(L, mt, pat)
            for ju in JUS:
                cond = build_condition_mu(L, ju, mu)
                detune = detune_superops(cond["n_ops"], cond["D"])
                Jops = [current_op(b, ju, cond["basis"], cond["idx_map"], bhh.NMAX) for b in range(L - 1)]
                for tau in TAUS:
                    d = diagnose(cond, ju, tau, detune, Jops)
                    d.update(setting="tilt", pattern=pat, strength=mt)
                    rows.append(d)
            print(f"  [tilt] {pat} {mt} done", flush=True)
    else:
        for mu_max in (0.5, 1.0, 2.0):
            for r in range(n_real):
                rng = np.random.default_rng(20260606 + 1000 * int(mu_max * 10) + r)
                mu = rng.uniform(-mu_max, mu_max, size=L)
                for ju in JUS:
                    cond = build_condition_mu(L, ju, mu)
                    detune = detune_superops(cond["n_ops"], cond["D"])
                    Jops = [current_op(b, ju, cond["basis"], cond["idx_map"], bhh.NMAX) for b in range(L - 1)]
                    for tau in TAUS:
                        d = diagnose(cond, ju, tau, detune, Jops)
                        d.update(setting="disorder", pattern="rand", strength=mu_max, realization=r)
                        rows.append(d)
            pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "symbreak_detune_disorder.csv"), index=False)
            print(f"  [disorder] mu_max={mu_max} ({n_real} real) done [checkpointed]", flush=True)
    return pd.DataFrame(rows)


def summarize(df, mode):
    pkt = df[df.J_over_U >= 0.30]
    bp = df[(df.J_over_U >= 0.30) & (df.overlap_fi_geo < 0.5)]
    t3 = df[df.tau == 3]
    print("\n" + "=" * 86)
    print(f"STEP 3C — SYMMETRY-BROKEN DETUNING ({mode}, n={len(df)})")
    print("=" * 86)
    print(f"[1] mean overlap(F_i, geo) = {df.overlap_fi_geo.mean():.2f}  "
          f"(rows with F_i!=geo: {int((df.overlap_fi_geo<1).sum())}/{len(df)})")
    print(f"[2] handle pct (pocket): fi={pkt.pct_fi.mean():.1f}  geo={pkt.pct_geo.mean():.1f}  "
          f"fi>=geo in {100*np.mean(pkt.pct_fi>=pkt.pct_geo):.0f}%")
    print(f"[3] D_S (pocket): fi={pkt.D_S.mean():+.3f}  geo={pkt.D_S_geo.mean():+.3f}")
    print(f"[4] rho(C_S_burn, D_S)={safe_sp(df.C_S_burn, df.D_S):+.3f}  "
          f"rho(C_S_burn, handle)={safe_sp(df.C_S_burn, df.pct_fi):+.3f}")
    print(f"[5] sign reversal (tau=3): +mu mean D_S={t3.D_S.mean():+.3f}  -mu mean D_S={t3.D_S_neg.mean():+.3f}")
    print(f"[6] where overlap<0.5 (pocket): n={len(bp)}  "
          + (f"fi={bp.pct_fi.mean():.1f} vs geo={bp.pct_geo.mean():.1f} (fi>=geo {100*np.mean(bp.pct_fi>=bp.pct_geo):.0f}%)  "
             f"D_S fi={bp.D_S.mean():+.3f} vs geo={bp.D_S_geo.mean():+.3f}" if len(bp) else "(none)"))
    print(f"[7] detune-optimal vs F_i (pocket): pct_fi={pkt.pct_fi.mean():.1f} vs pct_dopt={pkt.pct_dopt.mean():.1f}  "
          f"dopt==fi in {100*pkt.dopt_eq_fi.mean():.0f}%  rho(F_i,drain)={pkt.sp_Fi_drain.mean():+.2f}")
    # decisions
    beats = (len(bp) > 0 and np.mean(bp.pct_fi >= bp.pct_geo) >= 0.6)
    pred = abs(safe_sp(df.C_S_burn, df.D_S)) > 0.5 or abs(safe_sp(df.C_S_burn, df.pct_fi)) > 0.5
    not_dethroned = np.mean(pkt.pct_fi >= pkt.pct_dopt) >= 0.5
    print("\n[8] DECISION FLAGS")
    print(f"   [{'PASS' if beats else 'FAIL'}] F_i beats geo when overlap<0.5 (F_i beyond geometry, coherent control)")
    print(f"   [{'PASS' if pred else 'FAIL'}] C_S_burn predicts detuning response")
    print(f"   [{'PASS' if not_dethroned else 'FLAG'}] F_i not dethroned by detune-optimal selector")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "tilt"
    nr = next((int(a) for a in sys.argv[2:] if a.isdigit()), 10)
    df = run(mode, nr)
    df.to_csv(os.path.join(OUTDIR, f"symbreak_detune_{mode}.csv"), index=False)
    summarize(df, mode)
    print(f"\n[saved] symbreak_detune_{mode}.csv\n[done]")
