### Targeted Dephasing as a Causal Handle on Local Occupation Loss in an Open BH Chain

**Kunal Bhatia** — Independent Researcher, Heidelberg, Germany  
ORCID: [0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

Companion code for the paper submitted to *Physical Review A*.

---

## What this paper does

This paper asks a precise question: in a 1D open Bose–Hubbard chain under uniform Lindblad dephasing, do sites with high **local occupation variance** F_i = ⟨n_i²⟩ − ⟨n_i⟩² serve as **causal handles** for future local occupation loss?

**Handle** (after Woodward 2003): a variable is a causal handle on an outcome if targeted intervention at that variable produces a measurably different outcome than matched-budget random intervention.

**Protocol:**
1. Evolve the ground state under uniform baseline dephasing (γ = 0.1) for a burn-in period t_burn = 5 to build up spatially heterogeneous fluctuation structure.
2. Compute F_i on the post-burn-in state; select the top-k sites (k = ⌈L/3⌉).
3. **Targeted arm**: apply extra dephasing γ_extra = 0.5 at the k high-F_i sites.
4. **Random arm**: apply the same total extra budget (k × γ_extra) to k randomly chosen sites (100 independent draws).
5. Compare local occupation loss ∑_{i∈S} max(0, ⟨n_i⟩_t − ⟨n_i⟩_{t+τ}) at the intervention sites.

All evolution is **exact Lindblad** (no approximations, no Trotter decomposition, no stochastic unravelling). The jump operator L_i = n_i is a pure dephasing operator that preserves all local and global occupation expectations at all times.

---

## Key results

Three regimes are identified across the tested parameter range:

| Regime | J/U | Result (L ∈ {6,7,8,9}, τ ∈ {1,2,3}) |
|--------|-----|--------------------------------------|
| **Negative** | 0.12 | Targeted produces **less** loss than random at all sizes and horizons |
| **Crossover band** | ≈ 0.18–0.24 | Size- and horizon-dependent; onset shifts upward with L, downward with τ |
| **Positive pocket** | ≥ 0.30 | Targeted > random at **all** τ and **all** L — 95% CI strictly above zero |

**Crossover-band onset** (smallest J/U at which 95% CI excludes zero):

| L | τ=2 | τ=3 |
|---|-----|-----|
| 6 | ≤ 0.20 | ≤ 0.20 |
| 7 | > 0.20 | ≤ 0.20 |
| 8 | 0.20 | 0.18 |
| 9 | 0.24 | 0.22 |

Both trends are monotone across the ladder: onset coupling increases with L (harder to activate at larger sizes) and decreases with τ (longer horizon helps).

The spatial response is consistent with **nonlocal redistribution**: perturbing high-F_i sites reduces loss at those sites but increases it at non-selected sites — opposite of a naive local-instability picture.

---

## System parameters

| Parameter | Value |
|-----------|-------|
| Model | 1D Bose–Hubbard, open BC |
| Filling | Half-filling: N = ⌊L/2⌋ |
| n_max | 3 per site |
| L (primary sweep) | 6, 7, 8, 9 |
| N | 3 (L=6,7) / 4 (L=8,9) |
| Hilbert space D | 56 (L=6), 84 (L=7), 322 (L=8), 486 (L=9) |
| Liouvillian dim | D² = 3136, 7056, 103684, 236196 |
| Matrix storage | Dense (D ≤ 84), Sparse CSR (D ≥ 322) |
| Baseline dephasing γ | 0.1 |
| Extra dephasing γ_extra | 0.5 |
| Burn-in time t_burn | 5.0 (units of ℏ/U) |
| Time horizons τ | 1, 2, 3 |
| Random trials per condition | 100 |
| Bootstrap resamples | 1000 |
| Random seed | 20260325 |

---

## Repository structure

```
BH/
├── bh.py                    # Complete simulation package (Hilbert space, Lindblad evolution,
│                            # intervention protocol, figures, tables)
├── paper.tex                # Manuscript (REVTeX 4.2 / PRA format)
├── refs.bib                 # BibTeX references (17 entries)
├── paper.pdf                # Compiled manuscript
└── outputs/
    ├── checkpoints/         # Per-condition JSON checkpoints (resume on interruption)
    ├── data/
    │   ├── config.json      # Simulation configuration
    │   └── results_L6.csv   # Raw results for L=6
    ├── figures/
    │   ├── fig1_main_heatmap.pdf    # Heatmap of loss difference across J/U and τ (L=6)
    │   ├── fig2_robustness.pdf      # Robustness across chain lengths and τ
    │   └── fig3_mechanism.pdf       # Site-resolved occupation change (J/U=0.30, L=6, τ=2)
    └── tables/
        ├── table_main.tex           # Main results table (L=6, all J/U × τ)
        ├── table_robust.tex         # Robustness table (L=6,7 at J/U=0.30,0.40)
        └── all_results.csv          # All results across L, J/U, τ
```

---

## Reproducing the results

### Requirements

- Python ≥ 3.9
- numpy, scipy, pandas, matplotlib, seaborn, tqdm
- LaTeX with REVTeX 4.2 (`texlive-publishers` or MacTeX)

```bash
pip install numpy scipy pandas matplotlib seaborn tqdm
```

### Run the simulation

```bash
# Primary sweep: L=6,7,8,9 × J/U={0.12,0.20,0.30,0.40} × τ={1,2,3}
python bh.py --l-list 6 7 8 9 --ju-list 0.12 0.20 0.30 0.40 --tau-list 1 2 3 --workers 4

# Crossover boundary sweep (Phase C): adds fine J/U resolution at L=8,9
python bh.py --l-list 8 9 --ju-list 0.16 0.18 0.22 0.24 0.26 0.28 \
             --tau-list 2 3 --workers 4 --resume
```

Per-condition checkpoints are written to `outputs/checkpoints/` — safe to interrupt and resume with `--resume`.

**Expected runtimes** (4-core laptop, single-threaded BLAS per worker):

| Chain length | D | Time per condition |
|---|---|---|
| L=6 | 56 | ~3 min |
| L=7 | 84 | ~15 min |
| L=8 | 322 | ~45 min |
| L=9 | 486 | ~3 hr |

### Compile the paper

```bash
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

---

## Code overview (`bh.py`)

The entire simulation is self-contained in a single file:

| Function | Purpose |
|----------|---------|
| `build_basis(L, N, nmax)` | Enumerate Fock basis states in fixed-N sector |
| `number_op(site, D, basis)` | Diagonal number operator for site i |
| `build_hamiltonian(L, J, U, basis, idx_map)` | Bose–Hubbard Hamiltonian (tunneling + interaction) |
| `build_liouvillian(H, L_ops, gammas)` | Lindblad superoperator in row-major vec form |
| `evolve_rho(rho, liouvillian, tau)` | Exact time evolution via `expm_multiply` |
| `_make_additive_op(base, addons)` | Sparse LinearOperator for per-trial Liouvillian (avoids large matrix copy) |
| `run_single_condition(...)` | Full experiment for one (L, J/U) condition |
| `make_tables(...)` | Generate LaTeX-ready tables |
| `make_figures(...)` | Generate PDF/PNG figures |

**Key implementation notes:**
- `expm_multiply` is exact to floating-point precision (Al-Mohy & Higham 2011) — not an approximation.
- For L=6,7 (D ≤ 84): dense matrices, direct Padé/Krylov path.
- For L=8,9 (D ≥ 322): sparse CSR Liouvillian; per-trial modified operator built as a `LinearOperator` to avoid allocating a new sparse matrix per trial.
- Vectorised bootstrap (1000 resamples, single NumPy call per condition).
- `multiprocessing.get_context("spawn")` with `imap_unordered` for parallel conditions.

---

## What is not claimed

- That high-F_i targeting is **optimal** — only that it beats matched-budget random targeting in the positive-pocket regime (J/U ≥ 0.30).
- That the effect holds at all J/U — it is reversed at J/U = 0.12 and size-dependent in the crossover band J/U ≈ 0.18–0.24.
- That results extend to the **thermodynamic limit** — exact Lindblad evolution is feasible through L = 9 at these parameters; larger sizes require approximate methods (e.g., MPO Lindblad).
- That the nonlocal redistribution mechanism is fully understood — the paper shows the spatial pattern is consistent with it, not that it proves it.

---

## License

MIT — see [LICENSE](LICENSE).
