# Targeted Dephasing as a Causal Handle on Local Occupation Loss in an Open Bose–Hubbard Chain

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

All evolution is **exact Lindblad** (no approximations). The jump operator L_i = n_i is a pure dephasing operator: it preserves all local and global occupation expectations at all times.

---

## Key results

| Regime | J/U | Result (L=6 and L=7) |
|--------|-----|----------------------|
| Negative | 0.12 | Targeted produces **less** loss than random at all τ |
| Transitional | 0.20 | Positive at L=6, not robust at L=7 |
| **Positive pocket** | **0.30, 0.40** | **Targeted > random at all τ ∈ {1,2,3}, both L** |

The spatial response is consistent with **nonlocal redistribution**: perturbing high-F_i sites reduces loss at those sites but increases it at non-selected sites. This is the opposite of a naive local-instability picture.

---

## System parameters

| Parameter | Value |
|-----------|-------|
| Model | 1D Bose–Hubbard, open BC |
| Filling | Half-filling: N = ⌊L/2⌋ |
| n_max | 3 per site |
| L | 6, 7 |
| N | 3 (both chain lengths) |
| Hilbert space D | 56 (L=6), 84 (L=7) |
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
├── refs.bib                 # BibTeX references (16 entries)
├── build.sh                 # Build script: runs bh.py, then compiles paper.tex
├── test_bh.py               # Regression tests (pytest)
├── paper.pdf                # Compiled manuscript
└── outputs/
    ├── data/
    │   ├── config.json      # Simulation configuration
    │   └── results_L6.csv   # Raw results for L=6
    ├── figures/
    │   ├── fig1_main_heatmap.pdf    # Heatmap of loss difference (L=6)
    │   ├── fig2_robustness.pdf      # Robustness across L and τ
    │   └── fig3_mechanism.pdf       # Site-resolved occupation change
    └── tables/
        ├── table_main.tex           # Main results table (L=6)
        ├── table_robust.tex         # Robustness table (L=6 vs L=7)
        └── all_results.csv          # All results across L, J/U, τ
```

---

## Reproducing the results

### Requirements

- Python ≥ 3.9
- numpy, scipy, pandas, matplotlib, seaborn, tqdm
- LaTeX with REVTeX 4.2 (`texlive-publishers` or MacTeX), latexmk
- pytest (for tests)

Install dependencies:

```bash
pip install numpy scipy pandas matplotlib seaborn tqdm pytest
```

### Run everything

```bash
bash build.sh
```

This will:
1. Run `bh.py` to regenerate all data, figures, and tables (~5–15 minutes depending on hardware).
2. Compile `paper.tex` to `paper.pdf`.

### Run tests only

```bash
pytest test_bh.py -v
```

Tests cover: Hilbert space dimension, Hamiltonian hermiticity, Lindblad trace preservation, particle number conservation, F_i computation correctness, and intervention budget equality.

### Run simulation only

```bash
source /path/to/your/venv/bin/activate  # activate your Python environment
python bh.py
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
| `evolve_rho(rho, liouvillian, tau)` | Exact time evolution via scipy expm_multiply |
| `site_expectations(rho, n_ops)` | Site-resolved ⟨n_i⟩ |
| `site_variances(rho, n_ops, n2_ops)` | Site-resolved F_i = ⟨n_i²⟩ − ⟨n_i⟩² |
| `run_single_condition(...)` | Full experiment for one (L, J/U) condition |
| `make_tables(...)` | Generate LaTeX-ready tables |
| `make_figures(...)` | Generate PDF/PNG figures |

---

## What is not claimed

- That high-F_i targeting is **optimal** — only that it beats matched-budget random targeting in the positive-pocket regime.
- That the effect holds at all J/U (it fails at J/U = 0.12 and is only transitional at J/U = 0.20).
- That the results extend to the thermodynamic limit — the exact Lindblad approach is limited to L ≤ 7 at these parameters.
- That the nonlocal redistribution mechanism is fully understood — the paper shows the spatial pattern is consistent with it, not that it proves it.

---

## License

MIT — see [LICENSE](LICENSE).
