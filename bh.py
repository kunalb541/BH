"""
bh.py  (AWS-optimized, v2)
==========================
Bose-Hubbard causal-handle paper package.
Generates all data, figures, and LaTeX-ready tables via exact Lindblad evolution.

Performance notes (v2 vs v1)
-----------------------------
* RandomLinearOperator path: avoids copying the ~150-400 MB sparse Liouvillian
  for every random trial (was the dominant cost for L>=8).
* Diagonal fast expectations: O(L·D) instead of O(L·D²) for ⟨n_i⟩ / Fᵢ.
* Vectorised bootstrap: single NumPy call, no Python loop over 1000 resamples.
* Per-condition checkpointing: safe resume on spot-instance preemption.
* Clean worker init: BLAS pinning in initializer, not in the parent.
* CLI flags: --pilot, --resume, --l-list, --tau-list, --workers, --no-figures.

Targeting L=9 on m5.large/xlarge class instances.
L=10 (D≈1902) is memory-feasible with LinearOperator but expm_multiply will be
~35× slower than L=9 per call; budget accordingly.

Scientific invariants
---------------------
* Exact Lindblad evolution: unchanged (expm_multiply, no Trotter/stochastic).
* Causal-test definition: unchanged (targeted vs matched-budget random, 95% CI).
* Bootstrap semantics: unchanged (trial-level resampling, 1000 resamples).
* Particle number conservation: unchanged.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply, LinearOperator
from tqdm import tqdm

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

ROOT     = os.path.abspath(os.path.dirname(__file__))
OUT      = os.path.join(ROOT, "outputs")
DATA_DIR = os.path.join(OUT, "data")
FIG_DIR  = os.path.join(OUT, "figures")
TAB_DIR  = os.path.join(OUT, "tables")
LOG_DIR  = os.path.join(OUT, "logs")
CKPT_DIR = os.path.join(OUT, "checkpoints")     # NEW: per-condition results

for d in [OUT, DATA_DIR, FIG_DIR, TAB_DIR, LOG_DIR, CKPT_DIR]:
    os.makedirs(d, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)

# Dense Liouvillian threshold: D=56 (L=6) → 157 MB ok; D=322 (L=8) → 172 GB.
_SPARSE_D_THRESHOLD = 100


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def savefig(fig, stem):
    for ext in [".pdf", ".png"]:
        fig.savefig(os.path.join(FIG_DIR, stem + ext), dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


# ---------------------------------------------------------------------------
# FOCK SPACE BASIS
# ---------------------------------------------------------------------------

def build_basis(L, N, nmax):
    """All Fock states |n1,...,nL⟩ with sum=N, 0≤nᵢ≤nmax."""
    states = []
    for combo in itertools.product(range(nmax + 1), repeat=L):
        if sum(combo) == N:
            states.append(combo)
    return sorted(states)


def basis_index(basis):
    return {state: idx for idx, state in enumerate(basis)}


# ---------------------------------------------------------------------------
# OPERATORS
# ---------------------------------------------------------------------------

def number_op(site, D, basis):
    op = np.zeros((D, D), dtype=np.float64)
    for idx, state in enumerate(basis):
        op[idx, idx] = state[site]
    return op


def build_hamiltonian(L, J, U, nmax, basis, idx_map):
    D = len(basis)
    H = np.zeros((D, D), dtype=np.float64)

    for state_idx, state in enumerate(basis):
        for site in range(L):
            ni = state[site]
            H[state_idx, state_idx] += 0.5 * U * ni * (ni - 1)

    for site in range(L - 1):
        for state_idx, state in enumerate(basis):
            ni, nj = state[site], state[site + 1]
            if nj > 0 and ni < nmax:
                new = list(state)
                new[site] = ni + 1; new[site + 1] = nj - 1
                t = tuple(new)
                if t in idx_map:
                    H[idx_map[t], state_idx] += -J * np.sqrt((ni + 1) * nj)
            if ni > 0 and nj < nmax:
                new = list(state)
                new[site] = ni - 1; new[site + 1] = nj + 1
                t = tuple(new)
                if t in idx_map:
                    H[idx_map[t], state_idx] += -J * np.sqrt(ni * (nj + 1))
    return H


# site_expectations / site_variances kept for backward compatibility and tests.
def site_expectations(rho, n_ops):
    return np.array([np.real(np.trace(nop @ rho)) for nop in n_ops])


def site_variances(rho, n_ops, n2_ops):
    means = site_expectations(rho, n_ops)
    sq    = np.array([np.real(np.trace(n2op @ rho)) for n2op in n2_ops])
    return sq - means**2


# ---------------------------------------------------------------------------
# LINDBLAD SUPEROPERATOR  (unchanged from v1)
# ---------------------------------------------------------------------------

def build_liouvillian(H, L_ops, gammas):
    """Lindblad superoperator in row-major vec convention."""
    D = H.shape[0]
    if D > _SPARSE_D_THRESHOLD:
        return _build_liouvillian_sparse(H, L_ops, gammas, D)
    I   = np.eye(D)
    sup = -1j * (np.kron(H, I) - np.kron(I, H.T))
    for Lk, gk in zip(L_ops, gammas):
        LdL  = Lk.conj().T @ Lk
        sup += gk * (np.kron(Lk, Lk.conj())
                     - 0.5 * np.kron(LdL, I)
                     - 0.5 * np.kron(I, LdL.T))
    return sup


def _build_liouvillian_sparse(H, L_ops, gammas, D):
    H_sp  = sp.csr_matrix(H.astype(complex))
    I_sp  = sp.eye(D, format="csr", dtype=complex)
    sup   = -1j * (sp.kron(H_sp, I_sp, format="csr")
                   - sp.kron(I_sp, H_sp.T, format="csr"))
    for Lk, gk in zip(L_ops, gammas):
        Lk_sp = sp.csr_matrix(Lk.astype(complex))
        LdL   = Lk_sp.conj().T @ Lk_sp
        sup   = sup + gk * (sp.kron(Lk_sp, Lk_sp.conj(), format="csr")
                             - 0.5 * sp.kron(LdL, I_sp, format="csr")
                             - 0.5 * sp.kron(I_sp, LdL.T, format="csr"))
    return sup.tocsr()


def _make_site_dissipator(Lk, D):
    """Single-site dephasing dissipator in superoperator form."""
    if D > _SPARSE_D_THRESHOLD:
        Lk_sp = sp.csr_matrix(Lk.astype(complex))
        LdL   = Lk_sp.conj().T @ Lk_sp
        I_sp  = sp.eye(D, format="csr", dtype=complex)
        return (sp.kron(Lk_sp, Lk_sp.conj(), format="csr")
                - 0.5 * sp.kron(LdL, I_sp, format="csr")
                - 0.5 * sp.kron(I_sp, LdL.T, format="csr")).tocsr()
    else:
        I   = np.eye(D)
        LdL = Lk.conj().T @ Lk
        return (np.kron(Lk, Lk.conj())
                - 0.5 * np.kron(LdL, I)
                - 0.5 * np.kron(I, LdL.T))


def evolve_rho(rho, liouvillian, tau):
    """Exact Lindblad time evolution.  Accepts sparse matrix or LinearOperator.

    Passes traceA to expm_multiply when available, suppressing scipy's
    performance warning and allowing better step-count selection.
    """
    D   = rho.shape[0]
    vec = rho.flatten()
    # traceA for expm_multiply is the trace of the scaled operator (A = liouvillian*tau)
    if sp.issparse(liouvillian):
        traceA = float(tau * np.real(liouvillian.diagonal().sum()))
    elif hasattr(liouvillian, "_trace"):
        traceA = float(tau * liouvillian._trace)
    else:
        traceA = None
    vec_t = expm_multiply(liouvillian * tau, vec, traceA=traceA)
    return vec_t.reshape(D, D)


# ---------------------------------------------------------------------------
# OPTIMISED HELPERS  (new in v2)
# ---------------------------------------------------------------------------

def _make_additive_op(base, addons):
    """LinearOperator for (base + Σ addons) without materialising the sum.

    Key optimisation for L≥8: avoids copying the large sparse Liouvillian
    once per random trial.  The result is passed directly to evolve_rho.

    matvec  computes (base + Σ addons) @ v   — exact.
    rmatvec computes (base + Σ addons).T @ v — used only for expm_multiply's
    one-norm estimation; an approximation here is fine for correctness.

    .T on a CSR matrix returns a CSC view sharing the same data buffer (no copy).
    """
    n        = base.shape[0]
    base_T   = base.T                   # CSC view, O(1), no data copy
    addons_T = [a.T for a in addons]    # idem

    def matvec(v):
        w = base.dot(v)
        for a in addons:
            w += a.dot(v)
        return w

    def rmatvec(v):
        w = base_T.dot(v)
        for a in addons_T:
            w += a.dot(v)
        return w

    op = LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec, dtype=base.dtype)
    # Pre-compute trace for evolve_rho → expm_multiply(traceA=...) performance hint.
    # For sparse matrices, .diagonal() is O(nnz_diag), no full materialisation.
    op._trace = float(np.real(
        base.diagonal().sum() + sum(float(np.real(a.diagonal().sum())) for a in addons)
    ))
    return op


def _fast_expectations(rho_diag, n_diags):
    """⟨nᵢ⟩ using pre-extracted operator diagonals.  O(L·D) not O(L·D²).

    Args:
        rho_diag : (D,) real array — np.real(np.diag(rho))
        n_diags  : (L, D) real array — row i is np.diag(n_ops[i])
    """
    return n_diags @ rho_diag          # (L,)


def _fast_variances(rho_diag, n_diags, n2_diags):
    """Fᵢ = ⟨nᵢ²⟩ − ⟨nᵢ⟩².  O(L·D)."""
    means = n_diags  @ rho_diag        # (L,)
    sq    = n2_diags @ rho_diag        # (L,)
    return sq - means ** 2


def _bootstrap_ci(diffs, n_boot, rng, lo=2.5, hi=97.5):
    """Vectorised bootstrap CI — single NumPy call, no Python loop."""
    n   = len(diffs)
    idx = rng.integers(0, n, size=(n_boot, n))  # (n_boot, n)
    boot = diffs[idx].mean(axis=1)              # (n_boot,)
    return float(np.percentile(boot, lo)), float(np.percentile(boot, hi))


# ---------------------------------------------------------------------------
# CHECKPOINTING  (new in v2)
# ---------------------------------------------------------------------------

def _ckpt_path(L, N, J_over_U):
    return os.path.join(CKPT_DIR, f"L{L}_N{N}_JU{J_over_U:.4f}.json")


def _load_ckpt(L, N, J_over_U):
    p = _ckpt_path(L, N, J_over_U)
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None


def _save_ckpt(res):
    """Write result immediately; safe for spot-instance preemption."""
    with open(_ckpt_path(res["L"], res["N"], res["J_over_U"]), "w") as f:
        json.dump(res, f, default=float)


# ---------------------------------------------------------------------------
# EXPERIMENT  (optimised run_single_condition)
# ---------------------------------------------------------------------------

def run_single_condition(L, N, nmax, J_over_U, gamma_base, gamma_extra,
                         tau_list, n_trials, burn_in_time, seed,
                         verbose=True, n_boot=1000):
    """Run one (L, J/U) condition.  Returns same dict structure as v1.

    Optimisations vs v1
    -------------------
    1. Random-trial Liouvillians via LinearOperator — no 150-400 MB copy per trial.
    2. ⟨nᵢ⟩ / Fᵢ from diagonal pre-extraction — O(L·D) not O(L·D²).
    3. Vectorised bootstrap — single NumPy call.
    4. Writes checkpoint to disk immediately on completion.
    """
    U = 1.0
    J = J_over_U * U

    basis   = build_basis(L, N, nmax)
    idx_map = basis_index(basis)
    D       = len(basis)

    H     = build_hamiltonian(L, J, U, nmax, basis, idx_map)
    n_ops = [number_op(i, D, basis) for i in range(L)]

    # Pre-extract number-operator diagonals for O(L·D) expectation values.
    n_diags  = np.array([np.diag(nop) for nop in n_ops], dtype=np.float64)  # (L, D)
    n2_diags = n_diags ** 2                                                   # (L, D)

    gammas_base = [gamma_base] * L
    liouv_base  = build_liouvillian(H, n_ops, gammas_base)

    # Per-site dissipators — pre-scaled by gamma_extra so the trial loop is alloc-free.
    site_diss        = [_make_site_dissipator(n_ops[i], D) for i in range(L)]
    site_diss_scaled = [gamma_extra * d for d in site_diss]   # L matrices, precomputed

    # Targeted Liouvillian materialised once (used only 3 times, one per tau).
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0    = eigvecs[:, 0]
    rho0    = np.outer(psi0, psi0.conj())
    rho_burn = evolve_rho(rho0, liouv_base, burn_in_time)

    rho_burn_diag = np.real(np.diag(rho_burn))
    Fi       = _fast_variances(rho_burn_diag, n_diags, n2_diags)
    k        = max(1, int(np.ceil(L / 3)))
    selected = np.argsort(Fi)[-k:][::-1].tolist()

    # Materialise targeted Liouvillian once (small fixed cost).
    liouv_tgt = liouv_base + sum(site_diss_scaled[s] for s in selected)
    if sp.issparse(liouv_tgt):
        liouv_tgt = liouv_tgt.tocsr()

    rng         = np.random.default_rng(seed)
    occ_before  = _fast_expectations(rho_burn_diag, n_diags)

    results = []

    for tau in tau_list:
        rho_tgt      = evolve_rho(rho_burn, liouv_tgt, tau)
        rho_tgt_diag = np.real(np.diag(rho_tgt))
        occ_tgt      = _fast_expectations(rho_tgt_diag, n_diags)
        loss_tgt     = sum(max(0.0, occ_before[s] - occ_tgt[s]) for s in selected)
        delta_tgt    = occ_tgt - occ_before

        diffs = np.empty(n_trials)
        mech  = []

        it = range(n_trials)
        if verbose:
            it = tqdm(it, desc=f"  L={L} J/U={J_over_U:.2f} tau={tau:.0f}",
                      leave=False, ncols=80)

        for trial_idx in it:
            rsites = rng.choice(L, size=k, replace=False).tolist()

            # KEY OPTIMISATION: LinearOperator for sparse (avoids large matrix copy);
            # direct sum for dense (sparse=False → Padé expm is faster than Krylov).
            addons  = [site_diss_scaled[s] for s in rsites]  # list refs, no alloc
            if sp.issparse(liouv_base):
                op_rnd = _make_additive_op(liouv_base, addons)
            else:
                op_rnd = liouv_base + sum(addons)
            rho_rnd = evolve_rho(rho_burn, op_rnd, tau)

            rho_rnd_diag = np.real(np.diag(rho_rnd))
            occ_rnd  = _fast_expectations(rho_rnd_diag, n_diags)
            loss_rnd = sum(max(0.0, occ_before[s] - occ_rnd[s]) for s in rsites)

            diffs[trial_idx] = loss_tgt - loss_rnd
            mech.append({
                "delta_tgt": delta_tgt.tolist(),
                "delta_rnd": (occ_rnd - occ_before).tolist(),
                "selected":  selected,
                "random":    rsites,
            })

        mean_diff          = float(np.mean(diffs))
        ci_lo, ci_hi = _bootstrap_ci(diffs, n_boot, rng)   # vectorised

        results.append({
            "L": L, "J_over_U": J_over_U, "tau": tau,
            "mean_diff": mean_diff, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "n_trials": n_trials, "selected": selected, "selected_k": k,
            "Fi": Fi.tolist(), "mechanism": mech,
            "trial_diffs": diffs.tolist(),
        })

    out = {"L": L, "N": N, "J_over_U": J_over_U, "D": D, "k": k,
           "selected": selected, "Fi": Fi.tolist(), "results": results}
    _save_ckpt(out)
    return out


# ---------------------------------------------------------------------------
# MULTIPROCESSING  (worker init + wrapper)
# ---------------------------------------------------------------------------

def _worker_init():
    """Called once in each spawn worker before any task runs.
    Pins BLAS to 1 thread to avoid oversubscription across workers.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"


def _condition_worker(args):
    return run_single_condition(*args)


# ---------------------------------------------------------------------------
# RUN ORCHESTRATION
# ---------------------------------------------------------------------------

# Recommended workers by instance family (conservative, memory-aware):
#   m5.large   (2  vCPU,  8 GB): L≤8 → 2, L=9 → 1-2
#   m5.xlarge  (4  vCPU, 16 GB): L≤8 → 4, L=9 → 2-4
#   m5.2xlarge (8  vCPU, 32 GB): L≤8 → 8, L=9 → 4-6
_DEFAULT_WORKERS = min(4, os.cpu_count() or 4)


def run_all(cfg, n_workers=None, resume=False, verbose_workers=False):
    """Build condition list, skip checkpointed ones if resume=True, run pool."""
    n_workers = n_workers or _DEFAULT_WORKERS

    conditions, skipped = [], []
    sc = cfg["SEED"]
    for L in cfg["L_LIST"]:
        N = L // 2
        for ju in cfg["J_OVER_U_LIST"]:
            sc += 1
            if resume and _load_ckpt(L, N, ju) is not None:
                skipped.append((L, N, ju))
                continue
            conditions.append((
                L, N, cfg["NMAX"], ju,
                cfg["GAMMA_BASE"], cfg["GAMMA_EXTRA"],
                cfg["TAU_LIST"], cfg["N_TRIALS"],
                cfg["BURN_IN_TIME"], sc,
                verbose_workers,
                cfg.get("N_BOOT", 1000),
            ))

    total = len(conditions) + len(skipped)
    print(f"\nConditions: {total} total | {len(skipped)} from checkpoint | "
          f"{len(conditions)} to run | {n_workers} workers\n")

    if skipped:
        print("  Resuming from checkpoint:", ", ".join(f"L={L} J/U={ju:.2f}"
              for L, N, ju in skipped))

    # Reload checkpointed results (in original order).
    all_res_ckpt = {}
    for L, N, ju in skipped:
        res = _load_ckpt(L, N, ju)
        all_res_ckpt[(L, ju)] = res

    # Pin BLAS in the parent too (for any single-process fallback).
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"

    new_res = []
    if conditions:
        n_w = min(n_workers, len(conditions))
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_w, initializer=_worker_init) as pool:
            # tqdm wraps imap_unordered so progress shows as conditions complete.
            for res in tqdm(
                pool.imap_unordered(_condition_worker, conditions),
                total=len(conditions),
                desc="Conditions",
                ncols=80,
                unit="cond",
            ):
                new_res.append(res)

    # Merge and sort into canonical order.
    all_res_dict = {**all_res_ckpt,
                    **{(r["L"], r["J_over_U"]): r for r in new_res}}

    all_res = []
    for L in cfg["L_LIST"]:
        for ju in cfg["J_OVER_U_LIST"]:
            if (L, ju) in all_res_dict:
                all_res.append(all_res_dict[(L, ju)])

    return all_res


# ---------------------------------------------------------------------------
# SUMMARY PRINT
# ---------------------------------------------------------------------------

def print_summary(all_res):
    for res in all_res:
        print(f"\nL={res['L']}, J/U={res['J_over_U']:.2f}, D={res['D']}")
        for r in res["results"]:
            tag = "PASS" if r["ci_lo"] > 0 else "FAIL"
            print(f"  tau={r['tau']:.0f}: diff={r['mean_diff']:.4f} "
                  f"CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] {tag}")


# ---------------------------------------------------------------------------
# TABLES  (unchanged from v1)
# ---------------------------------------------------------------------------

def make_tables(all_res, cfg):
    rows6 = [r for res in all_res if res["L"] == 6 for r in res["results"]]
    df6   = pd.DataFrame(rows6)
    df6.to_csv(os.path.join(DATA_DIR, "results_L6.csv"), index=False)

    lines = [r"\begin{tabular}{l ccc}", r"\toprule",
             r"$J/U$ & $\tau=1$ & $\tau=2$ & $\tau=3$ \\", r"\midrule"]
    for ju in cfg["J_OVER_U_LIST"]:
        cells = []
        for tau in cfg["TAU_LIST"]:
            row = df6[(abs(df6["J_over_U"] - ju) < 0.001) &
                      (abs(df6["tau"] - tau) < 0.01)]
            if len(row) == 1:
                r = row.iloc[0]
                cells.append(f"${r['mean_diff']:.3f}$ $[{r['ci_lo']:.3f},"
                              f"\\,{r['ci_hi']:.3f}]$")
            else:
                cells.append("---")
        lines.append(f"{ju:.2f} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB_DIR, "table_main.tex"), "w") as f:
        f.write("\n".join(lines))

    rob_lines = [r"\begin{tabular}{l cc}", r"\toprule",
                 r"Condition & $L=6$ & $L=7$ \\", r"\midrule"]
    for ju in [0.30, 0.40]:
        cells = []
        for Lval in [6, 7]:
            matching = [r for res in all_res if res["L"] == Lval
                        for r in res["results"] if abs(r["J_over_U"] - ju) < 0.001]
            if matching:
                pooled = float(np.mean([r["mean_diff"] for r in matching]))
                lo     = float(np.mean([r["ci_lo"]     for r in matching]))
                hi     = float(np.mean([r["ci_hi"]     for r in matching]))
                cells.append(f"${pooled:.3f}$ $[{lo:.3f},\\,{hi:.3f}]$")
            else:
                cells.append("---")
        rob_lines.append(f"$J/U={ju:.2f}$ & " + " & ".join(cells) + r" \\")
    rob_lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB_DIR, "table_robust.tex"), "w") as f:
        f.write("\n".join(rob_lines))

    summary = [{"L": res["L"], "J_over_U": r["J_over_U"], "tau": r["tau"],
                "mean_diff": r["mean_diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]}
               for res in all_res for r in res["results"]]
    pd.DataFrame(summary).to_csv(os.path.join(TAB_DIR, "all_results.csv"), index=False)


# ---------------------------------------------------------------------------
# FIGURES  (unchanged from v1)
# ---------------------------------------------------------------------------

def make_figures(all_res, cfg):
    L6  = [r for res in all_res if res["L"] == 6 for r in res["results"]]
    if L6:
        df  = pd.DataFrame(L6)[["J_over_U", "tau", "mean_diff"]]
        pivot = df.pivot(index="J_over_U", columns="tau", values="mean_diff")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                    cbar_kws={"label": "Local-loss diff (targeted − random)"})
        ax.set_xlabel(r"$\tau$"); ax.set_ylabel("$J/U$")
        ax.set_title("Local-loss difference across regimes ($L=6$)")
        savefig(fig, "fig1_main_heatmap")

    rob = [{"L": res["L"], "J_over_U": r["J_over_U"], "tau": r["tau"],
            "mean_diff": r["mean_diff"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]}
           for res in all_res for r in res["results"]
           if any(abs(r["J_over_U"] - ju) < 0.001 for ju in [0.30, 0.40])]
    if rob:
        df_rob = pd.DataFrame(rob)
        Lvals  = sorted(df_rob["L"].unique())
        colors = {Lv: f"C{i}" for i, Lv in enumerate(Lvals)}
        marks  = {Lv: m for Lv, m in zip(Lvals, ["o", "^", "s", "D"])}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, ju in zip(axes, [0.30, 0.40]):
            sub = df_rob[abs(df_rob["J_over_U"] - ju) < 0.001]
            for Lv in Lvals:
                ss = sub[sub["L"] == Lv]
                if ss.empty:
                    continue
                ax.errorbar(ss["tau"], ss["mean_diff"],
                            yerr=[ss["mean_diff"] - ss["ci_lo"],
                                  ss["ci_hi"] - ss["mean_diff"]],
                            fmt=marks[Lv], capsize=4, label=f"$L={Lv}$",
                            color=colors[Lv], markersize=7, linewidth=1.5)
            ax.axhline(0, color="gray", lw=0.6, ls=":")
            ax.set_xlabel(r"$\tau$"); ax.set_title(f"$J/U = {ju}$"); ax.legend()
        axes[0].set_ylabel("Local-loss diff")
        fig.tight_layout()
        savefig(fig, "fig2_robustness")

    mech_res = next((r for res in all_res if res["L"] == 6
                     for r in res["results"]
                     if abs(r["J_over_U"] - 0.30) < 0.01
                     and abs(r["tau"] - 2.0) < 0.1), None)
    if mech_res is not None:
        Lm   = 6; sel = mech_res["selected"]; nt = len(mech_res["mechanism"])
        d_tgt = np.mean([md["delta_tgt"] for md in mech_res["mechanism"]], axis=0)
        d_rnd = np.mean([md["delta_rnd"] for md in mech_res["mechanism"]], axis=0)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(Lm); w = 0.35
        ax.bar(x - w/2, d_tgt, w, label="Targeted", color="C0", alpha=0.8)
        ax.bar(x + w/2, d_rnd, w, label="Random",   color="C1", alpha=0.8)
        for s in sel:
            ax.axvspan(s - 0.5, s + 0.5, alpha=0.12, color="blue")
        ax.axhline(0, color="gray", lw=0.6, ls=":")
        ax.set_xlabel("Site index"); ax.set_ylabel(r"$\Delta\langle n_i \rangle$")
        ax.set_title(r"Site-resolved occupation change ($J/U=0.30$, $L=6$, $\tau=2$)")
        ax.set_xticks(x); ax.legend()
        savefig(fig, "fig3_mechanism")


# ---------------------------------------------------------------------------
# ADJUDICATION  (unchanged from v1)
# ---------------------------------------------------------------------------

def write_adjudication(all_res):
    lines = ["# BH Adjudication Note\n\n"]
    for res in all_res:
        for r in res["results"]:
            ok = r["ci_lo"] > 0
            lines.append(
                f"- L={res['L']}, J/U={r['J_over_U']:.2f}, tau={r['tau']:.0f}: "
                f"diff={r['mean_diff']:.4f} CI=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}] "
                f"{'PASS' if ok else 'FAIL'}\n")
    with open(os.path.join(LOG_DIR, "adjudication.md"), "w") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# CLI + MAIN
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="BH causal-handle simulation (AWS-optimised)")
    p.add_argument("--pilot", action="store_true",
                   help="Smoke test: L={6,7}, tau=2 only, 20 trials.")
    p.add_argument("--l-list", nargs="+", type=int, default=None,
                   help="Chain lengths to run (default: 6 7 8).")
    p.add_argument("--ju-list", nargs="+", type=float, default=None,
                   help="J/U values (default: 0.12 0.20 0.30 0.40).")
    p.add_argument("--tau-list", nargs="+", type=float, default=None,
                   help="Measurement horizons (default: 1 2 3).")
    p.add_argument("--trials", type=int, default=None,
                   help="Random trials per condition (default: 100).")
    p.add_argument("--resume", action="store_true",
                   help="Skip conditions already in outputs/checkpoints/.")
    p.add_argument("--workers", type=int, default=None,
                   help=f"Worker processes (default: {_DEFAULT_WORKERS}).")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation.")
    p.add_argument("--no-tables", action="store_true",
                   help="Skip table generation.")
    return p.parse_args()


def main():
    args = parse_args()
    t0   = time.time()
    print("Running BH causal-handle simulation (v2 AWS-optimised)...\n")

    # --- Build configuration ---
    cfg = {
        "SEED":          20260325,
        "NMAX":          3,
        "GAMMA_BASE":    0.1,
        "GAMMA_EXTRA":   0.5,
        "BURN_IN_TIME":  5.0,
        "N_TRIALS":      100,
        "N_BOOT":        1000,
        "TAU_LIST":      [1.0, 2.0, 3.0],
        "J_OVER_U_LIST": [0.12, 0.20, 0.30, 0.40],
        "L_LIST":        [6, 7, 8],
    }

    if args.pilot:
        cfg.update({"L_LIST": [6, 7], "TAU_LIST": [2.0],
                    "N_TRIALS": 20, "N_BOOT": 200})
        print("  [PILOT MODE] L={6,7}, tau=2, 20 trials\n")

    # CLI overrides
    if args.l_list:   cfg["L_LIST"]        = args.l_list
    if args.ju_list:  cfg["J_OVER_U_LIST"] = args.ju_list
    if args.tau_list: cfg["TAU_LIST"]      = args.tau_list
    if args.trials:   cfg["N_TRIALS"]      = args.trials

    save_json(cfg, os.path.join(DATA_DIR, "config.json"))

    # --- Run ---
    all_res = run_all(cfg, n_workers=args.workers, resume=args.resume)
    print_summary(all_res)

    # --- Outputs ---
    if not args.no_tables:
        make_tables(all_res, cfg)
    if not args.no_figures:
        make_figures(all_res, cfg)
    write_adjudication(all_res)

    print(f"\nDone. {time.time() - t0:.0f}s")
    print(f"Outputs: {OUT}")


if __name__ == "__main__":
    main()
