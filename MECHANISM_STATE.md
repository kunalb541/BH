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

## 6. Step 3A — current / continuity explanation (EARNED, 2026-06-06)
Directed self-drain **is** the time-integrated net outward particle current across the
selected set's boundary. Continuity d⟨n_i⟩/dt = ⟨Ĵ_{i-1,i}⟩ − ⟨Ĵ_{i,i+1}⟩ holds exactly
(∫ outward current reproduces D_S to ~1e-6), with the Hermitian bond-current operator
Ĵ_{i,i+1} = −iJ(a†_i a_{i+1} − a†_{i+1} a_i) = 2J Im⟨a†_i a_{i+1}⟩.

The **burn-in current divergence** of the F_i-set, C_S_burn = Σ_{i∈S}(⟨Ĵ_{i,i+1}⟩−⟨Ĵ_{i-1,i}⟩),
predicts D_S and the handle:
- clean L=6,7,8: ρ(C_S_burn, D_S) = 0.91/0.91/0.89; ρ(C_S_burn, handle) = 0.94/0.95/0.92.
- tilt: ρ(C_S_burn, D_S) = +0.70 (handle ρ uninformative — saturated, no fail regime).
- disorder: ρ(C_S_burn, D_S) = +0.75; ρ(C_S_burn, handle) = +0.72.

**The crossover is a current reversal:** at low J/U the F_i-set carries net *inward*
current and fills (C_S_burn<0, D_S<0, handle fails); in the pocket it flips to *outward*
current and drains. J/U, tilt, and disorder are three routes to the same selected-site
current structure — the control parameter is the directed current, not J/U.
Caveats: predictor weaker under symmetry breaking (0.70–0.75 vs 0.89–0.91 clean); the
robust signal is set-level (per-site burn-in current vs single-site R_ii is anti-correlated).

## 7. Step 3B — local detuning: mechanism generalizes beyond dephasing (2026-06-06)
Coherent, N-conserving intervention `H → H + μ_extra Σ_{i∈S} n_i` (baseline dephasing
unchanged), μ0 scan {0.1, 0.5, 1.0}. Clean L=6:
- **Sign knob:** +μ drains (D_S>0), −μ fills (D_S<0) — clean reversal. D_S scales with μ0
  (0.10/0.33/0.45). Magnitudes ~5–50× the dephasing D_S (detuning is a *directed push*,
  not a throttle).
- **Current mechanism generalizes:** ρ(C_S_burn, handle) = +0.90, ρ(C_S_burn, D_S) = +0.61,
  ρ(F_i, single-site drain) = +0.83.
- **Caveat:** clean chain has F_i = geo, so this proves the current/self-drain *law*
  generalizes to coherent control, but NOT yet that *F_i-beyond-geometry* generalizes.
  The detune-optimal selector slightly beats F_i (75 vs 50 percentile) — F_i is a
  good-but-imperfect proxy under coherent control.

Upgraded statement: directed self-drain is a **general local-transport-control** mechanism
— dephasing throttles the current, detuning biases it — and in both the burn-in outward-
current structure predicts which sites respond.

## 8. Step 3C — symmetry-broken detuning: F_i beats geometry under coherent control (2026-06-06)
Detuning intervention under tilt + disorder (μ_extra=+0.5; −0.5 sign check). Completes the
intervention × symmetry-breaking matrix:

| Intervention | Clean | Tilt | Disorder |
|---|---|---|---|
| Dephasing | pass | pass | pass |
| Detuning  | pass | pass | pass |

- Tilt (72 rows, overlap 0.29): pocket pct_fi=68.5 vs geo=44.6 (F_i≥geo 89%); at overlap<0.5, 60.8 vs 43.1 (83%).
- Disorder (360 rows, overlap 0.52): pocket pct_fi=70.9 vs geo=58.8 (79%); at overlap<0.5, 49.8 vs geo 34.6 (71%); D_S fi>geo.
- Sign reversal persists under breaking (+μ drains, −μ fills).
- F_i tracks single-site detuning drain ρ=0.92–0.95; not dethroned by the detune-optimal selector.
- **Predictor nuance:** C_S_burn predicts the response but more weakly for detuning
  (tilt 0.46, disorder 0.55) than dephasing (0.70–0.91) — detuning *imposes* its own current
  rather than *reading* the pre-existing one.

**CONCLUSION:** F_i is a **general local-transport-control selector** — beyond geometry, for
both dissipative dephasing and coherent detuning. The current/continuity law holds for both;
only the *predictor* differs (dephasing reads the pre-existing burn-in current; detuning
manufactures it).

## 9. Next science (fork — none run yet)
- **A — amplitude damping / local loss** (N-changing; requires a multi-N Hilbert-space
  extension): does the mechanism survive a dissipative intervention that changes particle
  number? Strongest remaining physics test.
- **B — Liouvillian modes:** which relaxation mode carries the self-drain/current channel.
- **C — larger L (trajectories / MPO):** later, not now.

## Artifacts (this branch)
- `mechanism_pilot.py` — clean-chain response-kernel pilot, L parametrized
  (sparse path forced for L≥7; exact, parity-checked).
- `symbreak_diag.py` — tilt + disorder geometry-separation diagnostic.
- `mechanism_parity_check.py` — confirms sparse==dense Liouvillian to ~1e-15.
- `current_mechanism.py` — Step 3A continuity/current diagnostic (clean L).
- `current_symbreak.py` — Step 3A current diagnostic under tilt + disorder.
- `detune_probe.py` — Step 3B coherent local-detuning probe (clean L=6).
- `symbreak_detune.py` — Step 3C detuning under tilt + disorder; `finish_detune_disorder.py`
  resumable finisher (per-realization checkpoint; ~30-min env process cap).
- `outputs/mechanism_pilot/*.csv` — pilot_results_L{6,7,8}.csv, symbreak_{tilt,disorder}.csv,
  current_mech_L{6,7,8}.csv, current_symbreak_{tilt,disorder}.csv, detune_probe_L6.csv,
  symbreak_detune_{tilt,disorder}.csv.
