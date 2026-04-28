### Local number variance as a causal handle in the one-dimensional Bose–Hubbard chain

**Kunal Bhatia** — Independent Researcher, Heidelberg, Germany  
ORCID: [0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

Companion code for the paper submitted to *Physical Review A*.

---

## What this paper does

In a 1D open Bose–Hubbard chain under uniform Lindblad dephasing, sites develop spatially heterogeneous local occupation variance F_i = ⟨n_i²⟩ − ⟨n_i⟩². This paper tests whether targeting additional dephasing at the highest-F_i sites produces measurably more local occupation loss than applying the same dephasing budget to randomly chosen sites.

**Protocol:**
1. Evolve the ground state under uniform baseline dephasing (γ = 0.1) for burn-in t_burn = 5 to build heterogeneous fluctuation structure.
2. Compute F_i; select the top-k sites (k = ⌈L/3⌉).
3. **Targeted arm**: apply extra dephasing γ_extra = 0.5 at the k high-F_i sites.
4. **Random arm**: apply the same total budget (k × γ_extra) to k randomly chosen sites (100 independent draws).
5. Compare local occupation loss ∑_{i∈S} max(0, ⟨n_i⟩_t − ⟨n_i⟩_{t+τ}) at the intervention sites.

All evolution is **exact Lindblad** (no approximations, no Trotter decomposition, no stochastic unravelling).

---

## Key results

### Primary sweep (clean chain, L ∈ {6, 7, 8})

Three regimes across J/U:

| Regime | J/U | Result |
|--------|-----|--------|
| **Negative** | 0.12 | Targeted produces *less* loss than random — 95% CI strictly below zero at all tested L, τ |
| **Crossover** | ≈ 0.20 | Size- and horizon-sensitive; not robustly positive or negative |
| **Positive pocket** | ≥ 0.30 | Targeted > random — 95% CI strictly above zero at all tested L, τ |

Best effects at J/U = 0.40, τ = 3, growing with system size:

| L | mean advantage |
|---|---------------|
| 6 | +0.047 |
| 7 | +0.068 |
| 8 | +0.072 |

### Disorder selector sweep (L=6, 9 selectors, 50 realizations)

At strong disorder (μ_max = 2.0, J/U = 0.4, τ = 3), fi ranks first among all tested selectors:

| Selector | Mean advantage |
|----------|---------------|
| **fi** (variance) | **+0.049** |
| maxn | +0.034 |
| geo (center) | +0.032 |
| gen (generator) | +0.028 |
| dis_amp | +0.023 |
| bdy | +0.010 |
| dis_anti | +0.005 |
| anti-fi | +0.002 |
| minn | ≈ 0 |

The fi−geo gap grows monotonically with μ_max and J/U. dis_amp ranks well below fi, confirming F_i carries independent information beyond raw disorder amplitude.

### Shell-matched permutation controls

fi beats all within-shell permutations by +0.002 to +0.006 at τ=3, J/U ≥ 0.30, establishing that F_i carries within-shell information beyond shell geometry.

### Deterministic inhomogeneous (tilt) chain

In a tilted chain where F_i ≠ geo by construction, fi > geo with a gap of +0.032 to +0.054 at J/U = 0.40, τ = 3.

### Gamma scan

The advantage peaks near γ_extra ≈ 0.5–1.0 and collapses at very high rates. The paper's choice γ_extra = 0.5 is near-optimal.

### Exhaustive subset ranking (hardening)

All C(L,k) intervention subsets evaluated exactly (C(6,2)=15, C(7,3)=35, C(8,3)=56):

| L | J/U | τ=1 | τ=2 | τ=3 |
|---|-----|-----|-----|-----|
| 6 | 0.30 | 100% | 100% | 100% |
| 6 | 0.40 | 100% | 100% | 100% |
| 7 | 0.30 | 100% | 100% | 100% |
| 7 | 0.40 | 100% | 100% | 100% |
| 8 | 0.30 | 100% | 100% | 100% |
| 8 | 0.40 |  98% |  98% |  98% |

In the positive pocket (J/U ≥ 0.30) the F_i-selected subset is globally optimal or tied at every tested (L, τ), except L=8, J/U=0.40 where it is 98th percentile (1 of 56 subsets ties/beats it). Negative regime: 21–40th percentile. Crossover: 29–87th percentile.

### Target robustness

In every positive-pocket condition: signed gap > clipped gap, absolute gap > 0, redistribution gap > 0. The positive-part clipping is conservative — removing it yields a larger advantage.

---

## System parameters

| Parameter | Value |
|-----------|-------|
| Model | 1D Bose–Hubbard, open BC |
| Filling | Half-filling: N = ⌊L/2⌋ |
| n_max | 3 per site |
| L (primary sweep) | 6, 7, 8 |
| N | 3 (L=6,7) / 4 (L=8) |
| Hilbert space D | 56 (L=6), 84 (L=7), 322 (L=8) |
| Liouvillian dim | D² = 3136, 7056, 103684 |
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
├── bh.py                    # Complete simulation package
├── bh_hardening.py          # Exhaustive subset ranking + target robustness tests
├── run_all.sh               # AWS batch script
├── test_bh.py               # Core physics regression tests (pytest)
├── test_new_experiments.py  # Selector sweep, dis_amp, inhomogeneous, gamma-scan tests
├── cover_letter.txt         # PRA submission cover letter
├── paper.tex                # Manuscript (REVTeX 4.2 / PRA format)
├── refs.bib                 # BibTeX references
├── paper.pdf                # Compiled manuscript
└── outputs/                 # Generated outputs (not tracked in git)
    ├── bh_hardening/        # Exhaustive subset + robustness CSVs, summary JSON, figures
    ├── checkpoints/         # Per-condition JSON checkpoints (resume on interruption)
    ├── data/                # config.json, results CSVs
    ├── figures/             # PDF/PNG figures (fig1–fig5)
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
# L=6,7,8 × J/U={0.12,0.20,0.30,0.40} × τ={1,2,3}
python bh.py --l-list 6 7 8 --ju-list 0.12 0.20 0.30 0.40 --tau-list 1 2 3 --workers 4
```

### Disorder + selector experiments

```bash
# Selector sweep — all 9 selectors including dis_amp/dis_anti
python bh.py --selector-sweep --l-list 6 --ju-list 0.20 0.30 0.40 --tau-list 1 2 3 \
  --disorder-strengths 0.10 0.20 0.30 0.50 1.00 2.00 \
  --disorder-realizations 50 --dis-workers 4 --resume

# Disorder realizations (fi vs geo)
python bh.py --disorder --l-list 6 --ju-list 0.20 0.30 0.40 --tau-list 1 2 3 \
  --disorder-strengths 0.50 1.00 2.00 --disorder-realizations 50 --dis-workers 4 --resume

# Shell-matched permutation controls
python bh.py --shell-perm --l-list 6 --ju-list 0.20 0.30 0.40 --tau-list 1 2 3 \
  --disorder-strengths 0.10 0.20 0.30 --disorder-realizations 50 --dis-workers 4 --resume

# Inhomogeneous chain (deterministic tilt)
python bh.py --inhomogeneous --l-list 6 --ju-list 0.20 0.30 0.40 --tau-list 1 2 3 \
  --inhom-tilts 0.5 1.0 2.0 --inhom-patterns tilt step --resume

# Gamma scan
python bh.py --gamma-scan --l-list 6 --ju-list 0.30 0.40 --tau-list 1 2 3 \
  --disorder-strengths 0.10 --disorder-realizations 50 \
  --gamma-scan-values 0.1 0.2 0.5 1.0 2.0 --dis-workers 4 --resume
```

Per-condition checkpoints are written to `outputs/checkpoints/` — safe to interrupt and resume with `--resume`.

**Expected runtimes per realization** (4-core laptop):

| L | D | Time per realization |
|---|---|---|
| 6 | 56 | ~2–3 min |
| 7 | 84 | ~5–8 min |

### Compile the paper

```bash
pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

---

## Code overview (`bh.py`)

| Function | Purpose |
|----------|---------|
| `build_basis(L, N, nmax)` | Enumerate Fock basis states in fixed-N sector |
| `basis_index(basis)` | Dict mapping state tuple → row index |
| `number_op(site, D, basis)` | Diagonal number operator for site i |
| `build_hamiltonian(L, J, U, nmax, basis, idx_map)` | Bose–Hubbard Hamiltonian |
| `build_liouvillian(H, L_ops, gammas)` | Lindblad superoperator; dissipator stored as diagonal 1D array |
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
- Dissipator diagonal: for L_i = n_i, the D²×D² dissipator is diagonal. Stored as a 1D array — avoids OOM at L=6,7.
- Parallel disorder realizations use `concurrent.futures.ProcessPoolExecutor` with `as_completed` (deadlock-free).
- Per-realization checkpoint resume: each condition writes a JSON file; `--resume` skips existing files.
- Bootstrap: 1000 resamples, vectorised NumPy, per condition.

**Selectors compared in `run_selector_sweep_realization`:**

| Selector | Sites chosen |
|----------|-------------|
| `fi` | Top-k by F_i = ⟨n_i²⟩ − ⟨n_i⟩² |
| `geo` | Geometric center k sites |
| `maxn` | Top-k by ⟨n_i⟩ |
| `minn` | Bottom-k by ⟨n_i⟩ |
| `bdy` | Boundary (outermost k) |
| `anti` | Bottom-k by F_i |
| `gen` | Top-k by generator action \|⟨[L_i, H]⟩\| |
| `dis_amp` | Top-k by \|μ_i\| (disorder amplitude) |
| `dis_anti` | Bottom-k by \|μ_i\| |

---

## What is not claimed

- That high-F_i targeting is **optimal** among all possible selectors.
- That the effect holds at all J/U — it reverses at J/U = 0.12 and is transitional near J/U ≈ 0.20.
- That results extend to the **thermodynamic limit** — exact Lindblad is feasible through L = 8; larger sizes require approximate methods (e.g., MPO Lindblad).
- That fi is independent of geometry in the **clean, symmetric chain** — reflection symmetry forces F_i to peak at geometric-center sites, so fi = geo identically there. Symmetry breaking (disorder or inhomogeneous potential) is required for separation.
- That dis_amp is the only alternative explanation — it is a strong representative control, not an exhaustive one.

---

## License

MIT — see [LICENSE](LICENSE).
