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

### Clean chain (symmetric model)

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
| 9 | 0.24 | 0.20 |

Both trends are monotone across the ladder: onset coupling increases with L (harder to activate at larger sizes) and decreases with τ (longer horizon helps).

### Strong disorder: fi genuinely outperforms geo

Under on-site disorder μ_i ~ Uniform(−μ_max, +μ_max), spatial symmetry breaks and the high-F_i selector (fi) separates from the geometric-center selector (geo):

| μ_max | J/U | τ | fi mean | geo mean | fi − geo |
|-------|-----|---|---------|---------|---------|
| 0.50 | 0.40 | 3 | +0.031 | +0.022 | **+0.009** |
| 1.00 | 0.40 | 3 | +0.043 | +0.028 | **+0.015** |
| 2.00 | 0.40 | 3 | +0.050 | +0.033 | **+0.017** |

Effect is monotone in μ_max, τ, and J/U. At weak disorder (μ_max ≤ 0.20), fi ≈ geo as expected from symmetry.

### Variance information is independent of disorder amplitude

A new **disorder-amplitude selector** (dis_amp, selects top-k sites by |μ_i|) controls for the hypothesis that fi wins merely by tracking which sites have the strongest disorder. Key finding: fi and dis_amp select different sites in **58–79% of realizations** at strong disorder (μ_max ≥ 0.50), confirming that F_i carries genuinely independent variance information beyond raw disorder amplitude.

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
│                            # intervention protocol, all experiments, figures, tables)
├── run_all.sh               # AWS batch script — Priority A + extended campaign
├── test_bh.py               # Core physics regression tests (pytest)
├── test_new_experiments.py  # Selector sweep, dis_amp, inhomogeneous, gamma-scan tests
├── paper.tex                # Manuscript (REVTeX 4.2 / PRA format)
├── refs.bib                 # BibTeX references
├── paper.pdf                # Compiled manuscript
└── outputs/                 # Generated outputs (not tracked in git)
    ├── checkpoints/         # Per-condition JSON checkpoints (resume on interruption)
    ├── data/                # config.json, results CSVs
    ├── figures/             # PDF/PNG figures
    └── tables/              # LaTeX tables
```

---

## Reproducing the results

### Requirements

```bash
pip install numpy scipy pandas matplotlib seaborn tqdm
```

### Run the tests first

```bash
# Core physics tests (Hilbert space, Hamiltonian, Lindblad, F_i)
pytest test_bh.py -v

# Selector sweep, dis_amp, inhomogeneous, gamma-scan tests
python test_new_experiments.py
```

### Primary sweep (clean chain)

```bash
# L=6,7,8,9 × J/U={0.12,0.20,0.30,0.40} × τ={1,2,3}
python bh.py --l-list 6 7 8 9 --ju-list 0.12 0.20 0.30 0.40 --tau-list 1 2 3 --workers 4
```

### Disorder + selector experiments (AWS-scale)

```bash
# Selector sweep — all 9 selectors including dis_amp/dis_anti
python bh.py --selector-sweep --l-list 6 --ju-list 0.20 0.30 0.40 --tau-list 1 2 3 \
  --disorder-strengths 0.10 0.20 0.30 0.50 1.00 2.00 \
  --disorder-realizations 50 --dis-workers 4 --resume

# Strong disorder realizations (fi, geo, shell-perm)
python bh.py --disorder --l-list 6 --ju-list 0.20 0.30 0.40 --tau-list 1 2 3 \
  --disorder-strengths 0.50 1.00 2.00 --disorder-realizations 50 --dis-workers 4 --resume

python bh.py --shell-perm --l-list 6 --ju-list 0.20 0.30 0.40 --tau-list 1 2 3 \
  --disorder-strengths 0.50 1.00 2.00 --disorder-realizations 50 --dis-workers 4 --resume

# Inhomogeneous chain (deterministic asymmetry, single realization per condition)
python bh.py --inhomogeneous --l-list 6 --ju-list 0.20 0.30 0.40 --tau-list 1 2 3 \
  --inhom-tilts 0.5 1.0 2.0 --inhom-patterns tilt step --resume

# Gamma scan (robustness to γ_extra)
python bh.py --gamma-scan --l-list 6 --ju-list 0.30 0.40 --tau-list 1 2 3 \
  --disorder-strengths 0.10 --disorder-realizations 50 \
  --gamma-scan-values 0.1 0.2 0.5 1.0 2.0 --dis-workers 4 --resume
```

Per-condition checkpoints are written to `outputs/checkpoints/` — safe to interrupt and resume with `--resume`.

**Expected runtimes per realization** (4-core laptop):

| Chain length | D | Time per realization |
|---|---|---|
| L=6 | 56 | ~2–3 min |
| L=7 | 84 | ~5–8 min |

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
| `basis_index(basis)` | Dict mapping state tuple → row index |
| `number_op(site, D, basis)` | Diagonal number operator for site i |
| `build_hamiltonian(L, J, U, nmax, basis, idx_map)` | Bose–Hubbard Hamiltonian (tunneling + interaction) |
| `build_liouvillian(H, L_ops, gammas)` | Lindblad superoperator in row-major vec form; dissipator stored as diagonal 1D array |
| `evolve_rho(rho, liouvillian, tau)` | Exact time evolution via `expm_multiply` |
| `site_expectations(rho, n_ops)` | ⟨n_i⟩ for all sites |
| `site_variances(rho, n_ops, n2_ops)` | F_i = ⟨n_i²⟩ − ⟨n_i⟩² for all sites |
| `run_selector_sweep_realization(...)` | Full 9-selector protocol for one disorder realization |
| `run_disorder_realization(...)` | fi vs geo for one disorder realization |
| `run_inhomogeneous_experiment(...)` | Deterministic asymmetry experiment |
| `run_gamma_scan_experiment(...)` | γ_extra robustness scan |
| `make_tables(...)` | Generate LaTeX-ready tables |
| `make_figures(...)` | Generate PDF/PNG figures |

**Key implementation notes:**
- `expm_multiply` is exact to floating-point precision (Al-Mohy & Higham 2011).
- Dissipator diagonal: for L_i = n_i, the D²×D² dissipator is diagonal in the superoperator basis. Stored as a 25 KB 1D array rather than a 150 MB dense matrix — critical for L=6,7 without OOM.
- Per-realization checkpoint resume: each condition writes a JSON file; `--resume` skips existing files.
- Bootstrap: 1000 resamples, vectorised NumPy, per condition.
- `multiprocessing.get_context("spawn")` with `imap_unordered` for parallel disorder realizations.

**Selectors compared in `run_selector_sweep_realization`:**

| Selector | Sites chosen |
|----------|-------------|
| `fi` | Top-k by F_i = ⟨n_i²⟩ − ⟨n_i⟩² |
| `geo` | Geometric center k sites |
| `maxn` | Top-k by ⟨n_i⟩ |
| `minn` | Bottom-k by ⟨n_i⟩ |
| `bdy` | Boundary (outermost k) |
| `anti` | Bottom-k by F_i (inverted fi) |
| `gen` | Top-k by generator action \|⟨[L_i, H]⟩\| |
| `dis_amp` | Top-k by \|μ_i\| (disorder amplitude) |
| `dis_anti` | Bottom-k by \|μ_i\| (inverted dis_amp) |

`dis_amp` and `dis_anti` are controls that test whether fi's advantage over random is merely due to tracking the strongest disorder sites. Finding fi ≠ dis_amp in majority of strong-disorder realizations confirms variance carries independent information.

---

## What is not claimed

- That high-F_i targeting is **optimal** — only that it beats matched-budget random targeting in the positive-pocket regime.
- That the effect holds at all J/U — it reverses at J/U = 0.12 and is size-dependent near J/U ≈ 0.18–0.24.
- That results extend to the **thermodynamic limit** — exact Lindblad is feasible through L = 9; larger sizes require approximate methods (e.g., MPO Lindblad).
- That fi is independent of geometry in the **clean, symmetric chain** — at zero disorder, reflection symmetry forces F_i to peak at geometric-center sites, so fi = geo identically. Breaking symmetry (disorder or inhomogeneous potential) is required for separation.
- That dis_amp is the only alternative explanation — other amplitude-based proxies may exist. dis_amp is a strong representative test.

---

## License

MIT — see [LICENSE](LICENSE).
