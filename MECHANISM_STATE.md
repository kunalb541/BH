# BH mechanism state — locked 2026-06-06

Status memo for the Bose–Hubbard PRA-rejection rescue. This locks the claim status
**before** any paper rewrite or Step-3 (Liouvillian) run. Exact Lindblad throughout.
Sign convention: **D_S ≡ −Σ_{i∈S} Δn_i(τ)**, so **D_S > 0 ⟺ the selected sites
drain/lose occupation** ("directed self-drain"); D_S ≤ 0 ⟺ they hold/fill.

## 1. Old mechanism — RETIRED
"F_i works because it tracks redistribution susceptibility" is **true but
non-discriminating**, and therefore not the explanation. Spearman(F_i, total
single-site response norm) is high (≈0.5–1.0) in **every** regime, *including where
the intervention fails*, and its correlation with handle success is ≈ 0:
−0.03 (L6), +0.09 (L7), −0.26 (L8), +0.02 (disorder). Keep only as a
necessary-not-sufficient fact.

Also retired: "transport freezes at low J/U" (falsified — bond coherence is ~flat,
0.55→0.65 across the whole range).

## 2. New mechanism — EARNED
The variance-targeted dephasing handle works **iff the selected sites are in a
directed self-draining response regime (D_S > 0)**; it fails when the dephasing
response is diffuse or in-flowing (D_S ≤ 0). D_S is the operative response variable.
F_i identifies the sites participating in that directed self-draining response.

**The control parameter is directed self-drain, not J/U.** J/U is one route to it
(the clean-chain crossover ≈ 0.20); an imposed tilt or strong disorder produce D_S>0
at all couplings, shifting/removing the crossover.

## 3. Evidence (three independent supports)
- **Clean L=6,7,8:** size-supported; D_S onsets near the original J/U≈0.20 crossover;
  D_S predicts handle with |ρ| ≈ 0.9 across the crossover; reproduces the paper's
  published exhaustive percentiles (e.g. L8 J/U=0.20 → 34/68/71).
- **Tilt (deterministic geometry separation):** mean overlap(F_i,geo)=0.29 yet
  pct_fi=96.5 vs geo=63.7 (pocket, F_i≥geo in 97%); F_i-set self-drains (+0.028),
  geo-set does not (−0.005); gradient makes D_S active at all J/U.
- **Disorder (random geometry separation, 10 real × μ_max{0.5,1,2}):** overlap=0.52;
  pct_fi=95.4 vs geo=72.1; where overlap<0.5, pct_fi=95.6 vs geo=37.3 (F_i≥geo 100%);
  D_S predicts handle (ρ=+0.65 overall, +0.78 within-realization); crossover
  interpolates from clean-like (μ=0.5) to always-on (μ=2.0).
- F_i is **not dethroned** by any response-kernel selector tried (best in ~90%).

## 4. Scope / limitations
Exact Lindblad dephasing, small open 1D Bose–Hubbard chains (L ≤ 8, nmax=3,
half-filling), clean + deterministic-tilt + random-disorder. **Not** the
thermodynamic limit. **No** Liouvillian-spectral / current-mode explanation yet
(the *why* of the D_S onset is still open).

## 5. Paper consequences
- Rewrite the mechanism section around **directed self-drain D_S**, not redistribution
  susceptibility.
- Retire the May-6 reanalysis **T1** (predictive-causal dissociation): it is a
  self-correlation artifact (predictor `dn_rnd_mean` is the target → wins at r=1 in
  52/52 cells); honest dissociation rate ≈ 0.52, not 0.90.
- Keep the observer-geometry reanalysis only as **secondary** support (earned: T4
  thermalization convergence, T5 scoring-rule fork).
- **Do not reuse** the expanded May-6 `paper.tex` (1301 lines; preserved on branch
  `claude/happy-yalow-455c61`). The submission `paper.tex` on `main` (930 lines) is
  untouched.

## 6. Next optional experiment
Liouvillian / current-mode explanation of **why** D_S turns on (which slow
non-stationary mode activates; J-scaling of the directed current via
d⟨n_i⟩/dt ∝ J·Im⟨a†_i a_{i+1}⟩). This is explanatory deepening — **not required** to
rescue the central mechanism, which is already earned.

## Artifacts (this branch)
- `mechanism_pilot.py` — clean-chain response-kernel pilot, L parametrized
  (sparse path forced for L≥7; exact, parity-checked).
- `symbreak_diag.py` — tilt + disorder geometry-separation diagnostic.
- `mechanism_parity_check.py` — confirms sparse==dense Liouvillian to ~1e-15.
- `outputs/mechanism_pilot/*.csv` — pilot_results_L{6,7,8}.csv, symbreak_{tilt,disorder}.csv.
