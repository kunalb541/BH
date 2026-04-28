"""
bh_hardening.py
===============
Three referee-killing hardening tests for the BH redistribution-handle paper.

TEST 1  — Susceptibility benchmark
    Does F_i predict per-site response to a small extra local dephasing?
    Spearman/Pearson correlations + top-k overlap.

TEST 2  — Exhaustive subset ranking
    Among ALL C(L,k) subsets, where does the top-F_i subset rank?
    Percentile rank, best/worst/median, exact random mean.

TEST 3  — Target robustness
    Is the positive-pocket result an artefact of clipping?
    Gap for clipped / signed / absolute / global-redistribution metrics.

Usage
-----
    python bh_hardening.py --quick          # L=6 only
    python bh_hardening.py --full           # L=6,7,8 (paper conditions)
    python bh_hardening.py --full --delta-gamma 0.05 0.1 0.2

Outputs → outputs/bh_hardening/
    susceptibility_results.csv
    subset_ranking_results.csv
    target_robustness_results.csv
    summary_hardening.json
    fig_susceptibility_scatter.pdf/png
    fig_subset_rank_heatmap.pdf/png
    fig_target_robustness.pdf/png
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from math import comb
from pathlib import Path

# Pin BLAS before numpy import
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Import core physics from bh.py (same directory)
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from bh import (
    build_basis, basis_index, number_op,
    build_hamiltonian, build_liouvillian, _make_site_dissipator,
    evolve_rho, _fast_expectations, _fast_variances,
)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
ROOT    = Path(__file__).parent
OUTDIR  = ROOT / "outputs" / "bh_hardening"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# PAPER-STANDARD PARAMS
# ---------------------------------------------------------------------------
NMAX        = 3
GAMMA_BASE  = 0.1
GAMMA_EXTRA = 0.5
BURN_IN     = 5.0
SEED        = 20260325

def filling(L): return L // 2

def k_sites(L): return max(1, int(np.ceil(L / 3)))

# ---------------------------------------------------------------------------
# SHARED SETUP  (reused across all three tests)
# ---------------------------------------------------------------------------

def build_condition(L, J_over_U):
    """Build all static objects for one (L, J/U) condition."""
    U   = 1.0
    J   = J_over_U * U
    N   = filling(L)
    basis   = build_basis(L, N, NMAX)
    idx_map = basis_index(basis)
    D       = len(basis)

    H       = build_hamiltonian(L, J, U, NMAX, basis, idx_map)
    n_ops   = [number_op(i, D, basis) for i in range(L)]
    n_diags  = np.array([np.diag(nop) for nop in n_ops], dtype=np.float64)  # (L,D)
    n2_diags = n_diags ** 2

    gammas_base  = [GAMMA_BASE] * L
    liouv_base   = build_liouvillian(H, n_ops, gammas_base)
    site_diss    = [_make_site_dissipator(n_ops[i], D) for i in range(L)]

    # Ground state + burn-in
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0     = eigvecs[:, 0]
    rho0     = np.outer(psi0, psi0.conj())
    rho_burn = evolve_rho(rho0, liouv_base, BURN_IN)

    rho_diag  = np.real(np.diag(rho_burn))
    Fi        = _fast_variances(rho_diag, n_diags, n2_diags)
    occ_burn  = _fast_expectations(rho_diag, n_diags)

    return dict(L=L, N=N, D=D, J_over_U=J_over_U,
                basis=basis, idx_map=idx_map,
                H=H, n_ops=n_ops, n_diags=n_diags, n2_diags=n2_diags,
                liouv_base=liouv_base, site_diss=site_diss,
                rho_burn=rho_burn, rho_diag=rho_diag,
                Fi=Fi, occ_burn=occ_burn)


def evolve_with_extra(cond, sites, tau, gamma_extra=GAMMA_EXTRA):
    """Evolve rho_burn under liouv_base + gamma_extra * sum_{i in sites} D_i."""
    liouv = cond["liouv_base"] + gamma_extra * sum(
        cond["site_diss"][s] for s in sites)
    if sp.issparse(liouv):
        liouv = liouv.tocsr()
    return evolve_rho(cond["rho_burn"], liouv, tau)


def occ_from_rho(rho, cond):
    return _fast_expectations(np.real(np.diag(rho)), cond["n_diags"])


# ---------------------------------------------------------------------------
# SANITY CHECKS
# ---------------------------------------------------------------------------

def sanity_check(cond, tau=1.0, delta_gamma=0.1):
    """Run 5 sanity checks; raise AssertionError on failure."""
    L, D = cond["L"], cond["D"]
    N    = cond["N"]
    rho  = cond["rho_burn"]

    # 1. Trace ~1
    tr = np.real(np.trace(rho))
    assert abs(tr - 1.0) < 1e-8, f"trace={tr}"

    # 2. Occupations sum to N
    occ = occ_from_rho(rho, cond)
    assert abs(occ.sum() - N) < 1e-6, f"sum(occ)={occ.sum()} != N={N}"

    # 3. Response is zero when delta_gamma=0
    rho_t0 = evolve_rho(rho, cond["liouv_base"], tau)
    occ_t0 = occ_from_rho(rho_t0, cond)
    rho_t_extra = evolve_with_extra(cond, [0], tau, gamma_extra=0.0)
    occ_extra   = occ_from_rho(rho_t_extra, cond)
    assert np.allclose(occ_t0, occ_extra, atol=1e-10), \
        "delta_gamma=0 gives nonzero response"

    # 4. Subset count
    k   = k_sites(L)
    cnt = comb(L, k)
    actual = sum(1 for _ in itertools.combinations(range(L), k))
    assert actual == cnt, f"comb mismatch: {actual} != {cnt}"

    # 5. Top-F_i subset is deterministic
    Fi  = cond["Fi"]
    sel1 = frozenset(np.argsort(Fi)[-k:])
    sel2 = frozenset(np.argsort(Fi)[-k:])
    assert sel1 == sel2, "top-F_i subset is not deterministic"

    print(f"  [sanity OK] L={L} J/U={cond['J_over_U']:.2f}  "
          f"trace={tr:.10f}  sum(occ)={occ.sum():.8f}  subsets=C({L},{k})={cnt}")


# ===========================================================================
# TEST 1: SUSCEPTIBILITY BENCHMARK
# ===========================================================================

def test_susceptibility(cond, tau_list, delta_gammas):
    """
    For each site i, apply delta_gamma only at i; measure response chi_i.
    Correlate F_i vs chi_i with Spearman + Pearson.
    """
    L    = cond["L"]
    k    = k_sites(L)
    Fi   = cond["Fi"]

    # Top-F_i site indices (deterministic)
    top_fi = set(np.argsort(Fi)[-k:])

    rows = []

    for dg in delta_gammas:
        # Baseline (no extra dephasing) at each tau
        for tau in tau_list:
            rho_base_t = evolve_rho(cond["rho_burn"], cond["liouv_base"], tau)
            occ_base_t = occ_from_rho(rho_base_t, cond)

            chi_clip   = np.zeros(L)
            chi_signed = np.zeros(L)
            chi_redist = np.zeros(L)

            for site in range(L):
                rho_pert  = evolve_with_extra(cond, [site], tau, gamma_extra=dg)
                occ_pert  = occ_from_rho(rho_pert, cond)
                diff      = occ_base_t - occ_pert  # positive = loss at site

                chi_clip[site]   = max(0.0, diff[site]) / dg
                chi_signed[site] = diff[site] / dg
                chi_redist[site] = np.abs(occ_pert - occ_base_t).sum() / dg

            # Correlations (L>=3 for Spearman to be meaningful)
            def corr(x, y):
                if len(x) < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
                    return np.nan, np.nan, np.nan, np.nan
                sp_r, sp_p = spearmanr(x, y)
                pe_r, pe_p = pearsonr(x, y)
                return float(sp_r), float(sp_p), float(pe_r), float(pe_p)

            sp_clip,  _, pe_clip,  _ = corr(Fi, chi_clip)
            sp_sign,  _, pe_sign,  _ = corr(Fi, chi_signed)
            sp_red,   _, pe_red,   _ = corr(Fi, chi_redist)

            # Top-k overlap: sites where chi is highest vs top-F_i sites
            top_chi_clip  = set(np.argsort(chi_clip)[-k:])
            top_chi_sign  = set(np.argsort(chi_signed)[-k:])
            top_chi_red   = set(np.argsort(chi_redist)[-k:])
            overlap_clip  = len(top_fi & top_chi_clip)  / k
            overlap_sign  = len(top_fi & top_chi_sign)  / k
            overlap_red   = len(top_fi & top_chi_red)   / k

            # Top-1 match
            top1_fi   = int(np.argmax(Fi))
            top1_clip = int(np.argmax(chi_clip))
            top1_sign = int(np.argmax(chi_signed))

            rows.append(dict(
                L=L, J_over_U=cond["J_over_U"], tau=tau, k=k,
                delta_gamma=dg,
                spearman_clip=sp_clip, spearman_signed=sp_sign, spearman_redist=sp_red,
                pearson_clip=pe_clip,  pearson_signed=pe_sign,  pearson_redist=pe_red,
                topk_overlap_clip=overlap_clip,
                topk_overlap_signed=overlap_sign,
                topk_overlap_redist=overlap_red,
                top1_match_clip=int(top1_fi == top1_clip),
                top1_match_signed=int(top1_fi == top1_sign),
                Fi=Fi.tolist(),
                chi_clip=chi_clip.tolist(),
                chi_signed=chi_signed.tolist(),
                chi_redist=chi_redist.tolist(),
            ))

    return rows


# ===========================================================================
# TEST 2: EXHAUSTIVE SUBSET RANKING
# ===========================================================================

def test_exhaustive_subsets(cond, tau_list):
    """
    Evaluate ALL C(L,k) subsets.  Rank top-F_i subset by percentile.
    """
    L  = cond["L"]
    k  = k_sites(L)
    Fi = cond["Fi"]
    occ_before = cond["occ_burn"]

    all_subsets = list(itertools.combinations(range(L), k))
    n_subsets   = len(all_subsets)
    assert n_subsets == comb(L, k)

    # Top-F_i subset
    top_fi_subset = tuple(sorted(np.argsort(Fi)[-k:]))

    rows = []

    for tau in tau_list:
        # Pre-evolve all subsets
        vals_clip   = np.zeros(n_subsets)
        vals_signed = np.zeros(n_subsets)
        vals_abs    = np.zeros(n_subsets)
        vals_redist = np.zeros(n_subsets)

        rho_base_t  = evolve_rho(cond["rho_burn"], cond["liouv_base"], tau)
        occ_base_t  = occ_from_rho(rho_base_t, cond)

        desc = f"  subsets L={L} J/U={cond['J_over_U']:.2f} tau={tau:.0f}"
        for sidx, subset in enumerate(tqdm(all_subsets, desc=desc, leave=False, ncols=80)):
            rho_s   = evolve_with_extra(cond, list(subset), tau)
            occ_s   = occ_from_rho(rho_s, cond)
            diff_s  = occ_before - occ_s          # positive = lost occupation
            diff_t  = occ_before - occ_base_t     # baseline loss (no extra dephasing)

            vals_clip[sidx]   = sum(max(0.0, diff_s[i]) for i in subset)
            vals_signed[sidx] = sum(diff_s[i] for i in subset)
            vals_abs[sidx]    = sum(abs(occ_s[i] - occ_before[i]) for i in subset)
            vals_redist[sidx] = float(np.abs(occ_s - occ_before).sum())

        fi_idx = all_subsets.index(top_fi_subset)

        def rank_percentile(vals, fi_val, ascending=True):
            """Percentile: fraction of subsets with value <= fi_val (higher=better for loss)."""
            return 100.0 * np.mean(vals <= fi_val)

        def percentile_rank_desc(vals, fi_val):
            """Descending: fi_val beats this fraction of subsets."""
            return 100.0 * np.mean(vals <= fi_val)

        row = dict(
            L=L, J_over_U=cond["J_over_U"], tau=tau, k=k,
            n_subsets=n_subsets,
            fi_subset=list(top_fi_subset),
            # Clipped
            fi_val_clip=float(vals_clip[fi_idx]),
            best_clip=float(vals_clip.max()),
            worst_clip=float(vals_clip.min()),
            median_clip=float(np.median(vals_clip)),
            mean_clip=float(vals_clip.mean()),
            std_clip=float(vals_clip.std()),
            percentile_clip=float(percentile_rank_desc(vals_clip, vals_clip[fi_idx])),
            rank_clip=int(np.sum(vals_clip >= vals_clip[fi_idx])),  # subsets fi beats or ties
            # Signed
            fi_val_signed=float(vals_signed[fi_idx]),
            best_signed=float(vals_signed.max()),
            worst_signed=float(vals_signed.min()),
            median_signed=float(np.median(vals_signed)),
            mean_signed=float(vals_signed.mean()),
            std_signed=float(vals_signed.std()),
            percentile_signed=float(percentile_rank_desc(vals_signed, vals_signed[fi_idx])),
            rank_signed=int(np.sum(vals_signed >= vals_signed[fi_idx])),
            # Absolute
            fi_val_abs=float(vals_abs[fi_idx]),
            best_abs=float(vals_abs.max()),
            worst_abs=float(vals_abs.min()),
            median_abs=float(np.median(vals_abs)),
            mean_abs=float(vals_abs.mean()),
            std_abs=float(vals_abs.std()),
            percentile_abs=float(percentile_rank_desc(vals_abs, vals_abs[fi_idx])),
            rank_abs=int(np.sum(vals_abs >= vals_abs[fi_idx])),
            # Redistribution
            fi_val_redist=float(vals_redist[fi_idx]),
            best_redist=float(vals_redist.max()),
            worst_redist=float(vals_redist.min()),
            median_redist=float(np.median(vals_redist)),
            mean_redist=float(vals_redist.mean()),
            std_redist=float(vals_redist.std()),
            percentile_redist=float(percentile_rank_desc(vals_redist, vals_redist[fi_idx])),
            rank_redist=int(np.sum(vals_redist >= vals_redist[fi_idx])),
            # All subset values for figures
            all_vals_clip=vals_clip.tolist(),
            all_vals_signed=vals_signed.tolist(),
            all_vals_abs=vals_abs.tolist(),
            all_vals_redist=vals_redist.tolist(),
        )
        rows.append(row)

    return rows


# ===========================================================================
# TEST 3: TARGET ROBUSTNESS
# ===========================================================================

def test_target_robustness(cond, tau_list, n_boot=1000):
    """
    Compare top-F_i vs exact subset mean for clipped/signed/abs/redist.
    Also bootstrap CI over all-subset distribution.
    """
    L  = cond["L"]
    k  = k_sites(L)
    Fi = cond["Fi"]
    occ_before = cond["occ_burn"]

    all_subsets    = list(itertools.combinations(range(L), k))
    n_subsets      = len(all_subsets)
    top_fi_subset  = tuple(sorted(np.argsort(Fi)[-k:]))

    rng = np.random.default_rng(SEED + hash((L, cond["J_over_U"])) % (2**31))

    rows = []

    for tau in tau_list:
        rho_base_t = evolve_rho(cond["rho_burn"], cond["liouv_base"], tau)
        occ_base_t = occ_from_rho(rho_base_t, cond)

        # Evaluate all subsets (same as Test 2 — recompute here for standalone correctness)
        vals_clip   = np.zeros(n_subsets)
        vals_signed = np.zeros(n_subsets)
        vals_abs    = np.zeros(n_subsets)
        vals_redist = np.zeros(n_subsets)

        desc = f"  robustness L={L} J/U={cond['J_over_U']:.2f} tau={tau:.0f}"
        for sidx, subset in enumerate(tqdm(all_subsets, desc=desc, leave=False, ncols=80)):
            rho_s  = evolve_with_extra(cond, list(subset), tau)
            occ_s  = occ_from_rho(rho_s, cond)
            diff_s = occ_before - occ_s

            vals_clip[sidx]   = sum(max(0.0, diff_s[i]) for i in subset)
            vals_signed[sidx] = sum(diff_s[i] for i in subset)
            vals_abs[sidx]    = sum(abs(occ_s[i] - occ_before[i]) for i in subset)
            vals_redist[sidx] = float(np.abs(occ_s - occ_before).sum())

        fi_idx = all_subsets.index(top_fi_subset)

        def gap_and_ci(vals, fi_idx, rng, n_boot):
            fi_val  = vals[fi_idx]
            mean_v  = vals.mean()
            gap     = fi_val - mean_v
            # Bootstrap CI on gap: resample subsets
            boot_gaps = np.array([
                fi_val - vals[rng.integers(0, len(vals), size=len(vals))].mean()
                for _ in range(n_boot)
            ])
            ci_lo = float(np.percentile(boot_gaps, 2.5))
            ci_hi = float(np.percentile(boot_gaps, 97.5))
            return float(gap), ci_lo, ci_hi, float(fi_val), float(mean_v)

        gap_clip,   ci_lo_clip,   ci_hi_clip,   fi_clip,   mean_clip   = gap_and_ci(vals_clip,   fi_idx, rng, n_boot)
        gap_signed, ci_lo_signed, ci_hi_signed, fi_signed, mean_signed = gap_and_ci(vals_signed, fi_idx, rng, n_boot)
        gap_abs,    ci_lo_abs,    ci_hi_abs,    fi_abs,    mean_abs    = gap_and_ci(vals_abs,    fi_idx, rng, n_boot)
        gap_redist, ci_lo_redist, ci_hi_redist, fi_redist, mean_redist = gap_and_ci(vals_redist, fi_idx, rng, n_boot)

        same_sign_clipped_signed = int(np.sign(gap_clip) == np.sign(gap_signed))
        same_sign_clipped_abs    = int(np.sign(gap_clip) == np.sign(gap_abs))

        rows.append(dict(
            L=L, J_over_U=cond["J_over_U"], tau=tau, k=k,
            n_subsets=n_subsets,
            # Gaps
            gap_clip=gap_clip,     ci_lo_clip=ci_lo_clip,     ci_hi_clip=ci_hi_clip,
            gap_signed=gap_signed, ci_lo_signed=ci_lo_signed, ci_hi_signed=ci_hi_signed,
            gap_abs=gap_abs,       ci_lo_abs=ci_lo_abs,       ci_hi_abs=ci_hi_abs,
            gap_redist=gap_redist, ci_lo_redist=ci_lo_redist, ci_hi_redist=ci_hi_redist,
            # Values
            fi_clip=fi_clip, mean_clip=mean_clip,
            fi_signed=fi_signed, mean_signed=mean_signed,
            fi_abs=fi_abs, mean_abs=mean_abs,
            fi_redist=fi_redist, mean_redist=mean_redist,
            # Sign consistency
            same_sign_clip_signed=same_sign_clipped_signed,
            same_sign_clip_abs=same_sign_clipped_abs,
        ))

    return rows


# ===========================================================================
# FIGURES
# ===========================================================================

def fig_susceptibility_scatter(susc_rows, outdir):
    """F_i vs chi_i for representative conditions."""
    targets = [(0.12, 8, 3), (0.40, 8, 3), (0.12, 6, 3), (0.40, 6, 3)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()

    idx = 0
    for ju, L, tau in targets:
        match = [r for r in susc_rows
                 if abs(r["J_over_U"] - ju) < 1e-6
                 and r["L"] == L and r["tau"] == tau]
        if not match:
            continue
        r = match[0]
        ax = axes[idx]; idx += 1
        fi  = np.array(r["Fi"])
        chi = np.array(r["chi_signed"])
        ax.scatter(fi, chi, s=60, color="steelblue", zorder=3)
        for s, (x, y) in enumerate(zip(fi, chi)):
            ax.annotate(str(s), (x, y), fontsize=8, ha="left", va="bottom")
        sp_r = r["spearman_signed"]
        ax.set_title(f"L={L}, J/U={ju:.2f}, τ={tau}  ρ={sp_r:.2f}", fontsize=9)
        ax.set_xlabel("$F_i$", fontsize=9)
        ax.set_ylabel(r"$\chi_{\rm signed}$", fontsize=9)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    for a in axes[idx:]:
        a.set_visible(False)

    fig.suptitle("Susceptibility: $F_i$ vs per-site signed response", fontsize=10)
    fig.tight_layout()
    stem = str(outdir / "fig_susceptibility_scatter")
    for ext in [".pdf", ".png"]:
        fig.savefig(stem + ext, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {stem}.pdf/png")


def fig_subset_rank_heatmap(subset_rows, outdir):
    """Heatmap of top-F_i percentile rank over (J/U, L, tau)."""
    df = pd.DataFrame(subset_rows)[["L", "J_over_U", "tau", "percentile_clip"]]
    ju_vals  = sorted(df["J_over_U"].unique())
    L_vals   = sorted(df["L"].unique())
    tau_vals = sorted(df["tau"].unique())

    fig, axes = plt.subplots(1, len(tau_vals), figsize=(4 * len(tau_vals), 4), sharey=True)
    if len(tau_vals) == 1:
        axes = [axes]

    for ax, tau in zip(axes, tau_vals):
        pivot = df[df["tau"] == tau].pivot(index="L", columns="J_over_U", values="percentile_clip")
        im = ax.imshow(pivot.values, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(ju_vals)))
        ax.set_xticklabels([f"{j:.2f}" for j in ju_vals], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"L={l}" for l in pivot.index], fontsize=8)
        ax.set_title(f"τ={tau}", fontsize=9)
        ax.set_xlabel("J/U", fontsize=8)
        for i in range(len(pivot.index)):
            for j in range(len(ju_vals)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=8, color="black" if 20 < val < 80 else "white")

    plt.colorbar(im, ax=axes[-1], label="Percentile rank (clipped)", shrink=0.8)
    fig.suptitle("Top-$F_i$ subset percentile rank among all $C(L,k)$ subsets", fontsize=10)
    fig.tight_layout()
    stem = str(outdir / "fig_subset_rank_heatmap")
    for ext in [".pdf", ".png"]:
        fig.savefig(stem + ext, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {stem}.pdf/png")


def fig_target_robustness(robust_rows, outdir):
    """Gap (fi vs all-subset mean) for clipped/signed/abs/redist."""
    df = pd.DataFrame(robust_rows)
    metrics = ["gap_clip", "gap_signed", "gap_abs", "gap_redist"]
    labels  = ["Clipped", "Signed", "Absolute", "Redistribution"]
    colors  = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    ju_vals = sorted(df["J_over_U"].unique())
    tau_ref = df["tau"].max()
    L_ref   = df["L"].max()

    sub = df[(df["tau"] == tau_ref) & (df["L"] == L_ref)]

    fig, axes = plt.subplots(1, 4, figsize=(13, 4), sharey=False)
    for ax, metric, label, color in zip(axes, metrics, labels, colors):
        vals = [float(sub[sub["J_over_U"] == ju][metric].iloc[0]) if len(sub[sub["J_over_U"] == ju]) > 0 else np.nan
                for ju in ju_vals]
        lo_col = metric.replace("gap_", "ci_lo_")
        hi_col = metric.replace("gap_", "ci_hi_")
        los  = [float(sub[sub["J_over_U"] == ju][lo_col].iloc[0]) if len(sub[sub["J_over_U"] == ju]) > 0 else np.nan for ju in ju_vals]
        his  = [float(sub[sub["J_over_U"] == ju][hi_col].iloc[0]) if len(sub[sub["J_over_U"] == ju]) > 0 else np.nan for ju in ju_vals]
        errs = [[v - l for v, l in zip(vals, los)],
                [h - v for h, v in zip(his, vals)]]

        ax.bar(range(len(ju_vals)), vals, color=color, alpha=0.7, zorder=3)
        ax.errorbar(range(len(ju_vals)), vals, yerr=errs,
                    fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(range(len(ju_vals)))
        ax.set_xticklabels([f"{j:.2f}" for j in ju_vals], fontsize=8)
        ax.set_xlabel("J/U", fontsize=9)
        ax.set_title(f"{label}\n(L={L_ref}, τ={tau_ref})", fontsize=9)
        ax.set_ylabel("Gap vs subset mean", fontsize=8)

    fig.suptitle("Target robustness: $F_i$ advantage over all-subset mean", fontsize=10)
    fig.tight_layout()
    stem = str(outdir / "fig_target_robustness")
    for ext in [".pdf", ".png"]:
        fig.savefig(stem + ext, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {stem}.pdf/png")


# ===========================================================================
# REGIME SUMMARY
# ===========================================================================

def regime_summary(susc_rows, subset_rows, robust_rows):
    """Aggregate by J/U."""
    all_ju = sorted(set(r["J_over_U"] for r in susc_rows))
    summary = {}
    for ju in all_ju:
        s_rows = [r for r in susc_rows   if abs(r["J_over_U"] - ju) < 1e-6]
        sb_rows= [r for r in subset_rows if abs(r["J_over_U"] - ju) < 1e-6]
        r_rows = [r for r in robust_rows if abs(r["J_over_U"] - ju) < 1e-6]

        sp_clips  = [r["spearman_clip"]   for r in s_rows if not np.isnan(r["spearman_clip"])]
        sp_signs  = [r["spearman_signed"] for r in s_rows if not np.isnan(r["spearman_signed"])]
        pcts      = [r["percentile_clip"] for r in sb_rows]
        top_q     = [r["percentile_clip"] >= 75 for r in sb_rows]
        same_sign = [r["same_sign_clip_signed"] for r in r_rows]

        summary[f"{ju:.2f}"] = dict(
            J_over_U=ju,
            mean_spearman_clip=float(np.mean(sp_clips))   if sp_clips else None,
            mean_spearman_signed=float(np.mean(sp_signs)) if sp_signs else None,
            mean_percentile_clip=float(np.mean(pcts))     if pcts else None,
            median_percentile_clip=float(np.median(pcts)) if pcts else None,
            frac_top_quartile=float(np.mean(top_q))       if top_q else None,
            frac_same_sign_clip_signed=float(np.mean(same_sign)) if same_sign else None,
        )
    return summary


# ===========================================================================
# BURN-IN SENSITIVITY CHECK
# ===========================================================================

def test_burnin_sensitivity(L, JU_list, tau_list,
                            burnin_multipliers=(0.5, 1.0, 2.0),
                            base_burnin=BURN_IN):
    """L may be a single int or a list of ints."""
    if isinstance(L, (list, tuple)):
        all_rows = []
        for l in L:
            all_rows.extend(test_burnin_sensitivity(
                l, JU_list, tau_list, burnin_multipliers, base_burnin))
        return all_rows
    """
    Test whether positive-pocket and negative-regime conclusions depend on
    burn-in time.

    For each (L, J/U, tau, burnin_multiplier):
      - evolve ground state under baseline dephasing for t_burn * multiplier
      - compute F_i at that state, select top-k subset
      - run exhaustive subset ranking (clipped metric)
      - report gap_clip, gap_signed, gap_redist, percentile_clip, fi_subset

    Interpretation:
      - If positive pocket stays positive and high percentile across burn-ins:
        result is robust; add paragraph to paper.
      - If subset changes but percentile/gap stable:
        handle is robust at response level, not at site-label level.
      - If result flips: burn-in-dependent; reframe required.
    """
    U   = 1.0
    N   = filling(L)
    rows = []

    for ju in JU_list:
        J = ju * U
        basis   = build_basis(L, N, NMAX)
        idx_map = basis_index(basis)
        D       = len(basis)

        H        = build_hamiltonian(L, J, U, NMAX, basis, idx_map)
        n_ops    = [number_op(i, D, basis) for i in range(L)]
        n_diags  = np.array([np.diag(nop) for nop in n_ops], dtype=np.float64)
        n2_diags = n_diags ** 2

        liouv_base = build_liouvillian(H, n_ops, [GAMMA_BASE] * L)
        site_diss  = [_make_site_dissipator(n_ops[i], D) for i in range(L)]

        eigvals, eigvecs = np.linalg.eigh(H)
        psi0 = eigvecs[:, 0]
        rho0 = np.outer(psi0, psi0.conj())

        k           = k_sites(L)
        all_subsets = list(itertools.combinations(range(L), k))
        n_subsets   = len(all_subsets)
        assert n_subsets == comb(L, k)

        for mult in burnin_multipliers:
            t_burn   = base_burnin * mult
            rho_burn = evolve_rho(rho0, liouv_base, t_burn)
            rho_diag = np.real(np.diag(rho_burn))

            # Sanity: trace and N
            tr  = float(np.real(np.trace(rho_burn)))
            occ = _fast_expectations(rho_diag, n_diags)
            assert abs(tr - 1.0) < 1e-8,   f"trace={tr} at burn {t_burn}"
            assert abs(occ.sum() - N) < 1e-6, f"N err at burn {t_burn}"

            Fi            = _fast_variances(rho_diag, n_diags, n2_diags)
            occ_burn      = occ.copy()
            top_fi_subset = tuple(sorted(np.argsort(Fi)[-k:]))

            cond_lite = dict(rho_burn=rho_burn, liouv_base=liouv_base,
                             site_diss=site_diss, n_diags=n_diags,
                             L=L, Fi=Fi, occ_burn=occ_burn)

            for tau in tau_list:
                rho_base_t = evolve_rho(rho_burn, liouv_base, tau)
                occ_base_t = occ_from_rho(rho_base_t, cond_lite)

                vals_clip   = np.zeros(n_subsets)
                vals_signed = np.zeros(n_subsets)
                vals_redist = np.zeros(n_subsets)

                desc = (f"  burnin x{mult:.1f} L={L} J/U={ju:.2f} tau={tau:.0f}")
                for sidx, subset in enumerate(
                        tqdm(all_subsets, desc=desc, leave=False, ncols=80)):
                    rho_s  = evolve_with_extra(cond_lite, list(subset), tau)
                    occ_s  = occ_from_rho(rho_s, cond_lite)
                    diff_s = occ_burn - occ_s

                    vals_clip[sidx]   = sum(max(0.0, diff_s[i]) for i in subset)
                    vals_signed[sidx] = sum(diff_s[i] for i in subset)
                    vals_redist[sidx] = float(np.abs(occ_s - occ_burn).sum())

                fi_idx     = all_subsets.index(top_fi_subset)
                pct_clip   = float(100.0 * np.mean(vals_clip   <= vals_clip[fi_idx]))
                gap_clip   = float(vals_clip[fi_idx]   - vals_clip.mean())
                gap_signed = float(vals_signed[fi_idx] - vals_signed.mean())
                gap_redist = float(vals_redist[fi_idx] - vals_redist.mean())

                rows.append(dict(
                    L=L, N=N, J_over_U=ju, tau=tau,
                    burnin_multiplier=mult, t_burn=t_burn, k=k,
                    fi_subset=list(top_fi_subset),
                    Fi=Fi.tolist(),
                    gap_clip=gap_clip,
                    gap_signed=gap_signed,
                    gap_redist=gap_redist,
                    percentile_clip=pct_clip,
                    fi_val_clip=float(vals_clip[fi_idx]),
                    mean_clip=float(vals_clip.mean()),
                ))

    return rows


def print_burnin_summary(rows):
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)}
                       for r in rows])
    print("\n" + "=" * 75)
    print("BURN-IN SENSITIVITY CHECK")
    print("=" * 75)
    print(df[["L", "J_over_U", "tau", "burnin_multiplier",
              "gap_clip", "gap_signed", "gap_redist",
              "percentile_clip"]].to_string(index=False))

    print("\nSign stability (gap_clip) across burn-in multipliers:")
    for ju in sorted(df["J_over_U"].unique()):
        for L in sorted(df["L"].unique()):
            for tau in sorted(df["tau"].unique()):
                sub = df[(df["J_over_U"] == ju) & (df["L"] == L) &
                         (df["tau"] == tau)].sort_values("burnin_multiplier")
                if len(sub) < 2:
                    continue
                signs  = sub["gap_clip"].apply(np.sign).values
                stable = len(set(signs)) == 1
                vals   = sub["gap_clip"].values
                print(f"  L={L} J/U={ju:.2f} tau={tau}: "
                      f"gaps={[f'{v:+.5f}' for v in vals]}  "
                      f"sign_stable={'YES' if stable else 'NO'}")


# ===========================================================================
# NMAX TRUNCATION CHECK
# ===========================================================================

def test_nmax_truncation(L, JU_list, tau_list, nmax_list=(3, 4)):
    """
    Check whether qualitative results (regime ordering, top-Fi subset rank)
    are stable under nmax truncation.

    For each (L, J/U, nmax): compute Fi, top-Fi subset, exhaustive percentile.
    Report whether nmax=4 preserves sign pattern and top-quartile ranking.
    """
    rows = []

    for nmax in nmax_list:
        for ju in JU_list:
            U   = 1.0
            J   = ju * U
            N   = filling(L)
            basis   = build_basis(L, N, nmax)
            idx_map = basis_index(basis)
            D       = len(basis)

            H      = build_hamiltonian(L, J, U, nmax, basis, idx_map)
            n_ops  = [number_op(i, D, basis) for i in range(L)]
            n_diags  = np.array([np.diag(nop) for nop in n_ops], dtype=np.float64)
            n2_diags = n_diags ** 2

            liouv_base = build_liouvillian(H, n_ops, [GAMMA_BASE] * L)
            site_diss  = [_make_site_dissipator(n_ops[i], D) for i in range(L)]

            eigvals, eigvecs = np.linalg.eigh(H)
            psi0     = eigvecs[:, 0]
            rho0     = np.outer(psi0, psi0.conj())
            rho_burn = evolve_rho(rho0, liouv_base, BURN_IN)
            rho_diag = np.real(np.diag(rho_burn))

            Fi       = _fast_variances(rho_diag, n_diags, n2_diags)
            occ_burn = _fast_expectations(rho_diag, n_diags)
            k        = k_sites(L)

            top_fi_subset = tuple(sorted(np.argsort(Fi)[-k:]))
            all_subsets   = list(itertools.combinations(range(L), k))
            n_subsets     = len(all_subsets)

            # Make a lightweight cond dict for evolve_with_extra
            cond = dict(rho_burn=rho_burn, liouv_base=liouv_base,
                        site_diss=site_diss, n_diags=n_diags, L=L,
                        Fi=Fi, occ_burn=occ_burn)

            for tau in tau_list:
                rho_base_t = evolve_rho(rho_burn, liouv_base, tau)
                occ_base_t = occ_from_rho(rho_base_t, cond)

                vals_clip = np.zeros(n_subsets)
                desc = f"  nmax={nmax} J/U={ju:.2f} tau={tau:.0f}"
                for sidx, subset in enumerate(tqdm(all_subsets, desc=desc,
                                                   leave=False, ncols=80)):
                    rho_s = evolve_with_extra(cond, list(subset), tau)
                    occ_s = occ_from_rho(rho_s, cond)
                    diff  = occ_burn - occ_s
                    vals_clip[sidx] = sum(max(0.0, diff[i]) for i in subset)

                fi_idx     = all_subsets.index(top_fi_subset)
                percentile = float(100.0 * np.mean(vals_clip <= vals_clip[fi_idx]))
                fi_val     = float(vals_clip[fi_idx])
                mean_val   = float(vals_clip.mean())

                rows.append(dict(
                    L=L, N=N, nmax=nmax, J_over_U=ju, tau=tau, k=k,
                    D=D, n_subsets=n_subsets,
                    fi_subset=list(top_fi_subset),
                    Fi=Fi.tolist(),
                    fi_val_clip=fi_val,
                    mean_clip=mean_val,
                    gap_clip=fi_val - mean_val,
                    percentile_clip=percentile,
                    top_fi_site=int(np.argmax(Fi)),
                ))

    return rows


def print_nmax_summary(rows):
    """Print compact comparison table nmax=3 vs nmax=4."""
    df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, list)}
                       for r in rows])
    print("\n" + "=" * 70)
    print("NMAX TRUNCATION CHECK  (L=6, tau=3)")
    print("=" * 70)
    sub = df[df["tau"] == 3].sort_values(["J_over_U", "nmax"])
    print(sub[["nmax", "J_over_U", "D", "percentile_clip", "gap_clip",
               "top_fi_site"]].to_string(index=False))
    print()
    # Sign stability: does nmax=3 and nmax=4 agree on beneficial/harmful?
    for ju in sorted(df["J_over_U"].unique()):
        for tau in sorted(df["tau"].unique()):
            vals = {}
            for nmax in sorted(df["nmax"].unique()):
                row = df[(df["J_over_U"] == ju) & (df["tau"] == tau) &
                         (df["nmax"] == nmax)]
                if len(row):
                    vals[nmax] = row.iloc[0]["gap_clip"]
            if len(vals) == 2:
                agree = (np.sign(vals[3]) == np.sign(vals[4]))
                print(f"  J/U={ju:.2f} tau={tau}: "
                      f"gap nmax3={vals[3]:+.5f}  nmax4={vals[4]:+.5f}  "
                      f"sign_agree={'YES' if agree else 'NO'}")


# ===========================================================================
# MAIN
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="BH paper hardening tests")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="L=6 only, J/U={0.12,0.40}, tau={1,3}")
    mode.add_argument("--full",  action="store_true",
                      help="L=6,7,8, all J/U, all tau (paper conditions)")
    p.add_argument("--delta-gamma", nargs="+", type=float,
                   default=[0.1],
                   metavar="DG",
                   help="finite-difference step(s) for susceptibility test (default: 0.1)")
    p.add_argument("--outdir", type=Path,
                   default=OUTDIR,
                   help="output directory")
    p.add_argument("--no-figures", action="store_true",
                   help="skip figure generation")
    p.add_argument("--skip-sanity", action="store_true",
                   help="skip sanity checks (faster)")
    p.add_argument("--no-test2", action="store_true",
                   help="skip exhaustive subset ranking (saves time if only susceptibility needed)")
    p.add_argument("--burnin-check", action="store_true",
                   help="run burn-in sensitivity: 0.5x,1x,2x at L=6,8 J/U=0.12,0.40")
    p.add_argument("--burnin-multipliers", nargs="+", type=float,
                   default=[0.5, 1.0, 2.0],
                   help="burn-in multipliers to test (default: 0.5 1.0 2.0)")
    p.add_argument("--burnin-L", nargs="+", type=int, default=[6, 8],
                   help="L values for burn-in check (default: 6 8)")
    p.add_argument("--burnin-JU", nargs="+", type=float, default=[0.12, 0.40],
                   help="J/U values for burn-in check (default: 0.12 0.40)")
    p.add_argument("--burnin-tau", nargs="+", type=int, default=[3],
                   help="tau values for burn-in check (default: 3)")
    p.add_argument("--nmax-check", action="store_true",
                   help="run nmax=3 vs nmax=4 truncation check at L=8")
    p.add_argument("--nmax-list", nargs="+", type=int, default=[3, 4],
                   help="nmax values to compare in truncation check (default: 3 4)")
    return p.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        L_list   = [6]
        JU_list  = [0.12, 0.40]
        tau_list = [1, 3]
    else:
        # --full or default
        L_list   = [6, 7, 8]
        JU_list  = [0.12, 0.20, 0.30, 0.40]
        tau_list = [1, 2, 3]

    delta_gammas = args.delta_gamma

    print("=" * 60)
    print("BH HARDENING TESTS")
    print(f"  L={L_list}  J/U={JU_list}  tau={tau_list}")
    print(f"  delta_gamma={delta_gammas}")
    print(f"  outdir={outdir}")
    print("=" * 60)

    all_susc   = []
    all_subset = []
    all_robust = []

    conditions = list(itertools.product(L_list, JU_list))

    for L, ju in tqdm(conditions, desc="Conditions", ncols=80):
        print(f"\n{'─'*50}")
        print(f"  Building condition: L={L}, J/U={ju:.2f}")
        cond = build_condition(L, ju)

        if not args.skip_sanity:
            sanity_check(cond)

        # --- TEST 1 ---
        print(f"  [Test 1] Susceptibility  (delta_gamma={delta_gammas})")
        rows1 = test_susceptibility(cond, tau_list, delta_gammas)
        all_susc.extend(rows1)

        # --- TEST 2 ---
        if not args.no_test2:
            k = k_sites(L)
            n_sub = comb(L, k)
            print(f"  [Test 2] Exhaustive subsets: C({L},{k})={n_sub}")
            rows2 = test_exhaustive_subsets(cond, tau_list)
            all_subset.extend(rows2)

            # --- TEST 3 (shares subset evaluations; re-runs internally for simplicity) ---
            print(f"  [Test 3] Target robustness")
            rows3 = test_target_robustness(cond, tau_list)
            all_robust.extend(rows3)

    # ---------------------------------------------------------------------------
    # Save CSVs (drop heavy list columns)
    # ---------------------------------------------------------------------------
    def drop_lists(rows):
        out = []
        for r in rows:
            out.append({k: v for k, v in r.items() if not isinstance(v, list)})
        return out

    df_susc   = pd.DataFrame(drop_lists(all_susc))
    df_subset = pd.DataFrame(drop_lists(all_subset))
    df_robust = pd.DataFrame(drop_lists(all_robust))

    df_susc.to_csv(outdir / "susceptibility_results.csv",   index=False)
    df_subset.to_csv(outdir / "subset_ranking_results.csv", index=False)
    df_robust.to_csv(outdir / "target_robustness_results.csv", index=False)

    print(f"\n  [saved] susceptibility_results.csv  ({len(df_susc)} rows)")
    print(f"  [saved] subset_ranking_results.csv  ({len(df_subset)} rows)")
    print(f"  [saved] target_robustness_results.csv ({len(df_robust)} rows)")

    # ---------------------------------------------------------------------------
    # Regime summary
    # ---------------------------------------------------------------------------
    if all_susc and all_subset and all_robust:
        summary = regime_summary(all_susc, all_subset, all_robust)
        with open(outdir / "summary_hardening.json", "w") as f:
            json.dump(summary, f, indent=2, default=float)
        print(f"  [saved] summary_hardening.json")

        print("\n" + "=" * 60)
        print("REGIME SUMMARY")
        print("=" * 60)
        for ju_str, s in summary.items():
            print(f"\n  J/U = {ju_str}")
            print(f"    mean Spearman(F_i, chi_clip)   = {s['mean_spearman_clip']}")
            print(f"    mean Spearman(F_i, chi_signed) = {s['mean_spearman_signed']}")
            print(f"    mean percentile rank (clip)    = {s['mean_percentile_clip']:.1f}%")
            print(f"    fraction top quartile          = {s['frac_top_quartile']:.2f}")
            print(f"    frac same sign clip/signed     = {s['frac_same_sign_clip_signed']:.2f}")

    # ---------------------------------------------------------------------------
    # Figures
    # ---------------------------------------------------------------------------
    if not args.no_figures:
        print("\n  Generating figures...")
        if all_susc:
            fig_susceptibility_scatter(all_susc, outdir)
        if all_subset:
            fig_subset_rank_heatmap(all_subset, outdir)
        if all_robust:
            fig_target_robustness(all_robust, outdir)

    # ---------------------------------------------------------------------------
    # NMAX truncation check
    # ---------------------------------------------------------------------------
    # ---------------------------------------------------------------------------
    # BURN-IN SENSITIVITY CHECK
    # ---------------------------------------------------------------------------
    if args.burnin_check:
        print(f"\n{'─'*50}")
        print(f"  [burn-in check] L={args.burnin_L}  J/U={args.burnin_JU}  "
              f"tau={args.burnin_tau}  multipliers={args.burnin_multipliers}")
        burnin_rows = test_burnin_sensitivity(
            L=args.burnin_L,
            JU_list=args.burnin_JU,
            tau_list=args.burnin_tau,
            burnin_multipliers=args.burnin_multipliers,
        )
        df_burnin = pd.DataFrame(
            [{k: v for k, v in r.items() if not isinstance(v, list)}
             for r in burnin_rows]
        )
        df_burnin.to_csv(outdir / "burnin_sensitivity_results.csv", index=False)
        print(f"  [saved] burnin_sensitivity_results.csv")
        print_burnin_summary(burnin_rows)

    if args.nmax_check:
        # nmax truncation is only binding when N > nmax for some site, i.e. N >= nmax+1.
        # At half-filling N=L//2: L=6 N=3, L=7 N=3 are unaffected (max occupation=N<=3).
        # L=8 N=4 IS affected: nmax=3 excludes states with a site holding 4 bosons (D=322→330).
        # We run L=8 only; L=6/7 checks would be trivially identical and misleading.
        print(f"\n{'─'*50}")
        print(f"  [nmax check] L=8 N=4 (binding case), nmax={args.nmax_list}, "
              f"J/U={{0.12,0.30,0.40}}, tau={{1,2,3}}")
        print(f"  Note: L=6,7 (N=3) are unaffected by nmax 3→4 (same basis D=56,84).")
        nmax_rows = test_nmax_truncation(
            L=8,
            JU_list=[0.12, 0.30, 0.40],
            tau_list=[1, 2, 3],
            nmax_list=args.nmax_list,
        )
        # Save
        df_nmax = pd.DataFrame(
            [{k: v for k, v in r.items() if not isinstance(v, list)} for r in nmax_rows]
        )
        df_nmax.to_csv(outdir / "nmax_truncation_results.csv", index=False)
        print(f"  [saved] nmax_truncation_results.csv")
        print_nmax_summary(nmax_rows)
        # Persist to summary
        nmax_sign_stable = all(
            np.sign(
                df_nmax[(df_nmax["J_over_U"] == ju) & (df_nmax["tau"] == tau) &
                        (df_nmax["nmax"] == args.nmax_list[0])]["gap_clip"].values[0]
            ) == np.sign(
                df_nmax[(df_nmax["J_over_U"] == ju) & (df_nmax["tau"] == tau) &
                        (df_nmax["nmax"] == args.nmax_list[-1])]["gap_clip"].values[0]
            )
            for ju in [0.12, 0.30, 0.40]
            for tau in [1, 2, 3]
            if len(df_nmax[(df_nmax["J_over_U"] == ju) & (df_nmax["tau"] == tau) &
                           (df_nmax["nmax"] == args.nmax_list[0])]) > 0
            and len(df_nmax[(df_nmax["J_over_U"] == ju) & (df_nmax["tau"] == tau) &
                            (df_nmax["nmax"] == args.nmax_list[-1])]) > 0
        )
        print(f"\n  nmax sign stability (all conditions): {nmax_sign_stable}")

    print("\n✓ All hardening tests complete.")
    print(f"  Outputs: {outdir}")


if __name__ == "__main__":
    main()
