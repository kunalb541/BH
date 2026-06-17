# Variance as an intervention-relative control selector in an open Bose–Hubbard chain

**Kunal Bhatia** — Independent Researcher, Heidelberg, Germany
ORCID: [0009-0007-4447-6325](https://orcid.org/0009-0007-4447-6325)

Companion code and data for an exact-Lindblad study of the 1D open Bose–Hubbard chain. The
repository now hosts **two related manuscripts plus a supporting reanalysis**; all simulation is
exact (no Trotter, stochastic unravelling, or mean field), restricted to small chains ($L\le 8$).

---

## Two tracks

| Track | Manuscript | Question | Status |
|---|---|---|---|
| **1. Predictive** | `paper.tex` / `paper.pdf` | Does targeting the highest-variance sites with extra dephasing change future occupation loss more than matched-budget random targeting? | submitted to *Physical Review A* (under review) |
| **2. Mechanism** | `paper_v2.tex` / `paper_v2.pdf` | *Why* does it work, and does the operative control variable depend on the intervention class? | in preparation |
| (supporting) | `bh_reanalysis/` | Observer-geometry / principal-angle reanalysis of the same outputs | internal, pre-registered |

`paper_v2.tex` is the current mechanism manuscript; `paper.tex` is preserved as the version under
review. `paper_expanded_may6.tex` is an earlier expanded draft kept only as reference material.

---

## The result in brief (Track 2)

For a selected site set $S$ define the **directed self-drain** $D_S(\tau)=-\sum_{i\in S}\Delta\langle\hat n_i\rangle(\tau)$
($D_S>0$ ⇒ the selected sites lose occupation). Because the dephasing dissipator conserves each
$\langle\hat n_i\rangle$ exactly, particle-number continuity makes $D_S$ the **time-integrated net
outward current** of $S$. The findings:

- The handle works only in the **directed self-draining** regime; the burn-in current divergence
  of $S$ predicts it (Spearman $\rho\approx0.9$ at $L=6,7,8$), and the $J/U$ crossover is a
  **current reversal** (inward/fill → outward/drain).
- The earlier "redistribution susceptibility" explanation is **non-discriminating** ($\rho\approx0$
  with success) and is retired; geometry is a correlate, not fundamental.
- The operative selector is **intervention-relative**:

| intervention | conserves $N$? | coherent? | **operative selector** |
|---|:--:|:--:|:--:|
| dephasing | yes | no | **$F_i$** (reads pre-existing current) |
| detuning | yes | yes | **$F_i$** (imposes current) |
| local loss | no | no | **$\langle n_i\rangle$** (direct removal) |
| bond hopping mod. | yes | yes | site-level $F_i$ (not bond) |

The boundary between the two selectors is **transport-modulation vs. particle-removal**
(equivalently, particle-number conservation).

The original predictive thesis (Track 1) — top-$F_i$ targeting beats matched random in the
"positive pocket" $J/U\ge0.30$, robust across $L=6,7,8$ and the disorder/tilt/shell-permutation
controls — remains intact and is the content of `paper.tex`.

---

## Repository map

Everything lives at the top level (flat) plus `bh_reanalysis/` and `outputs/`. Grouped by purpose:

**Simulation engine**
- `bh.py` — exact-Lindblad Bose–Hubbard engine and the full experiment battery (primary sweep,
  disorder/selector sweep, shell-perm, inhomogeneous tilt, gamma scan). CLI-driven (`--help`).
- `bh_hardening.py` — hardening battery: exhaustive subset ranking, susceptibility, burn-in
  sensitivity, $n_{\max}$ truncation check. Defines `build_condition`/`evolve_with_extra` reused below.

**Mechanism program (Track 2) — each writes CSVs to `outputs/mechanism_pilot/`**
- `mechanism_pilot.py [L]` — response-kernel handle + size scaling ($L=6,7,8$).
- `current_mechanism.py [L]`, `current_symbreak.py` — current/continuity diagnostic ($C_S^{\rm burn}\!\to\!D_S$).
- `symbreak_diag.py tilt|disorder` — geometry separation under symmetry breaking (dephasing).
- `detune_probe.py`, `symbreak_detune.py tilt|disorder`, `finish_detune_disorder.py` — coherent detuning.
- `stage0_loss.py`, `loss_pilot.py`, `loss_symbreak.py` — multi-$N$ local loss (machinery + pilots).
- `bond_pilot.py` — per-bond-$J$ hopping modulation (site-vs-bond control).
- `mechanism_parity_check.py` — sparse vs dense Liouvillian parity ($\sim10^{-15}$).
- `make_paper_assets.py` — regenerates `paper_v2` figures, tables, and `key_numbers.json` from the CSVs.

**Manuscripts & bibliography**
- `paper.tex`/`paper.pdf` (Track 1), `paper_v2.tex`/`paper_v2.pdf` (Track 2),
  `paper_expanded_may6.tex` (reference draft), `refs.bib`, `cover_letter.txt`.

**Records (source-of-truth, human-readable)**
- `CLAIM_EVIDENCE_MAP.md` — **start here for reproducibility**: every paper_v2 claim → figure/table → source CSV → verified value.
- `MECHANISM_STATE.md` — full chronological mechanism log (steps 1–4; what is earned/retired).
- `CLAIM_STATUS.md` — one-page milestone. `PAPER_OUTLINE.md` — rewrite plan.
- `LOSS_PILOT_PREREG.md`, `BOND_HOPPING_PREREG.md` — pre-registrations.

**Tests & build**
- `test_bh.py`, `test_new_experiments.py` — unit tests (run first).
- `build.sh` — compile the papers. `run_all.sh` — the original AWS/EC2 batch campaign script.

**Data (`outputs/`)**
- `mechanism_pilot/` — Track-2 result CSVs (+ `paper_v2/` figures, tables, `key_numbers.json`).
- `bh_hardening/` — hardening CSVs/figures. `checkpoints/` — per-condition sweep checkpoints
  (L8/L9 tracked; the rest reproducible via `bh.py --resume`). `figures/`, `tables/`, `data/` — Track-1 paper assets.

---

## Reproducing the results

Requirements: Python 3.11+, `numpy scipy pandas matplotlib tqdm`. Run all commands **from the repo
root** (scripts import `bh.py`/`bh_hardening.py`); prefix with `PYTHONPATH=.` if needed.

```bash
python3 -m pytest test_bh.py test_new_experiments.py -q   # tests first
```

**Track 2 (mechanism) — fast, exact, laptop-scale.** Each line writes the CSV the paper draws from:

| Result (paper_v2) | Command | Output CSV |
|---|---|---|
| Handle + size scaling | `python3 mechanism_pilot.py 6` (and `7`, `8`) | `pilot_results_L{6,7,8}.csv` |
| Current/continuity mechanism | `python3 current_mechanism.py 6` (and `7`,`8`) | `current_mech_L{6,7,8}.csv` |
| Current under symmetry breaking | `python3 current_symbreak.py` | `current_symbreak_{tilt,disorder}.csv` |
| Geometry separation (dephasing) | `python3 symbreak_diag.py tilt` / `disorder` | `symbreak_{tilt,disorder}.csv` |
| Detuning (clean + sign) | `python3 detune_probe.py` | `detune_probe_L6.csv` |
| Detuning (symmetry-broken) | `python3 symbreak_detune.py tilt` / `disorder 10` | `symbreak_detune_{tilt,disorder}.csv` |
| Loss (validate → pilots) | `python3 stage0_loss.py`; `loss_pilot.py`; `loss_symbreak.py` | `loss_pilot_L6.csv`, `loss_symbreak_mini.csv` |
| Bond control | `python3 bond_pilot.py` | `bond_pilot_L6.csv` |
| Figures + tables + key numbers | `python3 make_paper_assets.py` | `outputs/paper_v2/…` |

Then compile: `pdflatex paper_v2 && bibtex paper_v2 && pdflatex paper_v2 && pdflatex paper_v2`.
The mechanism pilots run in seconds–minutes each; sparse/dense agreement is checked by
`mechanism_parity_check.py`. The loss disorder sweep self-checkpoints per realization
(`finish_detune_disorder.py` shows the resumable pattern).

**Track 1 (predictive paper).** The primary sweep, disorder/selector, shell-perm, inhomogeneous,
and gamma-scan experiments are driven by `bh.py` (see `--help`); the hardening battery (exhaustive
ranking, susceptibility, burn-in, $n_{\max}$) by `bh_hardening.py`. The full campaign was run on
AWS/EC2 (`run_all.sh`) with per-condition checkpoints synced to S3 and pulled into
`outputs/checkpoints/`.

### System parameters (both tracks)
1D Bose–Hubbard, open BC, half-filling $N=\lfloor L/2\rfloor$, $n_{\max}=3$; $L\in\{6,7,8\}$
($D=56,84,322$); baseline $\gamma=0.1$, $\gamma_{\rm extra}=0.5$, burn-in $t=5$, horizons
$\tau\in\{1,2,3\}$; dense for $D\le84$, sparse CSR above (mechanism scripts force sparse for
$L\ge7$). Seed `20260325`.

---

## What is and is not claimed
- **Earned:** the intervention-relative selector map above, the directed-self-drain/current
  mechanism, and its size support to $L=8$ — all from exact Lindblad evolution, with every
  headline number independently re-verified (see `CLAIM_EVIDENCE_MAP.md`).
- **Not claimed:** the thermodynamic limit, larger $L$, a universal control law, all intervention
  types, particle injection, or laboratory realization. The $J/U\approx0.20$ feature is a
  finite-size dynamical crossover, not the Mott–superfluid transition.
- The observer-geometry reanalysis's T1 "predictive–causal dissociation" is **retired** as a
  finding (a self-correlation artifact; honest rate $\sim0.52$, not $\sim0.90$); only its
  thermalization-convergence (T4) and scoring-rule-fork (T5) outcomes are kept as supporting.

## Branches & provenance
`main` is the integrated state. `pre-integration-2026-06-06` tags the pre-merge restore point.
`bh-reanalysis-archive` holds the reanalysis history; `claude/happy-yalow-455c61` is the original
(orphaned) worktree branch that also contains the expanded draft.

## License
See `LICENSE`.
