# BH Reanalysis Battery — Pre-registration

**Status:** LOCKED at write time. Any post-hoc edits will be appended in a `## Amendments` section with date, reason, and what specifically was changed. The locked predictions and thresholds in §§ 3–8 are immutable once code starts.

**Lock timestamp:** 2026-05-06 (file creation date, see git log).

**Author:** Kunal Bhatia (research subject) + Claude Code (analyst).

**Working directory:** `/Users/kunalbhatia/Desktop/Research/BH/.claude/worktrees/happy-yalow-455c61/`

**Inputs:** existing simulation outputs in `outputs/` (no new simulations).

**Companion brief:** "BH Reanalysis Battery" (user message, 2026-05-06). Where this pre-reg deviates from the brief, the deviation is justified in §1.4 and reflected in the per-test sections. The brief is the spec of intent; this document is the spec of what is actually testable on existing data, locked before code.

---

## 1. Context and data scope

### 1.1 Original paper, in two sentences

In a 1D open Bose–Hubbard chain under uniform Lindblad dephasing, sites develop spatially heterogeneous local occupation variance F_i = ⟨n_i²⟩−⟨n_i⟩² after a burn-in. The original paper showed that targeted additional dephasing at the top-k F_i sites produces measurably more local occupation loss than matched-budget random-site interventions, with a regime boundary near J/U ≈ 0.20–0.30 below which the effect vanishes or inverts.

### 1.2 What this reanalysis adds

The reanalysis applies a framework apparatus (predictive vs. causal observer dissociation; subspace alignment between observers; scoring-rule forks) to ask whether the BH data shows structure beyond what the paper claimed. None of this changes the paper's empirical claims; it tests whether the framework's structural language picks up further regularities in the data.

### 1.3 Existing data (verified by inspection of files)

Per (L, J/U, τ) cell:

* `Fi[L]`: pre-intervention local occupation variance at every site (the same F_i used to select targets).
* `selected[k]`: the targeted-arm site indices (top-k F_i with `k = ⌈L/3⌉`).
* `delta_tgt[L]`: per-site change in mean occupation Δ⟨n_i⟩ for the targeted intervention. Deterministic for a fixed cell, identical across the 100 trial entries.
* `delta_rnd[L]` per trial × 100 trials: per-site change in mean occupation Δ⟨n_i⟩ for random matched-budget controls. Each trial picks a different `random[k]` site set.
* `random[k]` per trial × 100 trials: which sites the random arm hit.
* `mean_diff`, `ci_lo`, `ci_hi`: aggregated targeted-vs-random gap with bootstrap CI.

Cells available:

* L=6, J/U ∈ {0.12, 0.20, 0.30, 0.40}, τ ∈ {1, 2, 3}: 12 cells. Source: `outputs/data/results_L6.csv` (mechanism + trial_diffs columns).
* L=8, J/U ∈ {0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30}, τ ∈ {2, 3}: 16 cells. Source: `outputs/checkpoints/L8_N4_JU*.json` (note: τ=1 not stored at L=8).
* L=9, J/U ∈ {0.12, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.40}, τ ∈ {2, 3}: 20 cells. Source: `outputs/checkpoints/L9_N4_JU*.json` (τ=1 not stored).

Total: **48 cells** of trial-level data.

Hardening tables in `outputs/bh_hardening/` (susceptibility, burn-in sensitivity, n_max truncation) provide derived statistics; per-trial data sits in the checkpoints / results_L6.

### 1.4 What the data does NOT contain (and brief deviations)

The brief assumed access to several observables that the saved data does not provide:

| Observable in brief | Available? | Deviation |
|---|---|---|
| F_i per site | YES | T1, T2, T4 use F_i. |
| ⟨n_i⟩ per site (mean occupation, pre-intervention) | NO — only F_i is stored | T1 observer family is restricted to F_i variants and Δn variants. n_i absent. |
| Kinetic energy density per site | NO — requires density matrix, not saved | Removed from observer family. |
| Interaction energy density per site | NO — same | Removed. |
| Correlation function g_d(i,j) | NO — same | Removed. |
| Per-site Δ⟨n_i⟩, targeted and random arm | YES | T1, T5 use these. |
| Lindblad evolution at fine times (intermediate t < τ) | NO — only τ ∈ {1, 2, 3} | T4 reduced to a 3-point monotonicity test rather than a continuous trajectory. |
| Multiple intervention TYPES (injection, heating, phase imprint) | NO — only additional dephasing | T6 declared NOT TESTABLE on existing data. See §8. |
| γ_extra parameter sweep | NO — γ_extra fixed at 0.5 | The Fisher / sensitivity analyses for T2 and T3 use J/U as the single continuous parameter, supplemented by the per-trial `random` site indicator vector as an L-dim discrete parameter axis. |
| Density matrix snapshot | NO | Any test requiring a state vector is removed. |

These deviations are forced by the data, not by results. They are locked here before any test runs. The substitute formulations in §§3–8 use only what is in the saved files.

### 1.5 Discipline rules (carry over from cosmo_battery V1/V2/V3)

1. This pre-reg locks before any analysis code is written. Code lives in `bh_reanalysis/src/`. Pre-reg lives at `bh_reanalysis/bh_reanalysis_prereg.md`.
2. Synthetic gates (T2 and T4) must pass before real data is touched in those tests.
3. Per-module unit tests for loaders, observers, nulls, and scoring rules.
4. No threshold adjustment after seeing real-data results.
5. Locked failures stay failures.
6. Negative results reported honestly.
7. All six tests run, including T6's not-testable declaration.
8. Multi-agent cross-check before final summary: math, code, independent re-derivation, pre-reg consistency. Run in parallel.

---

## 2. Observer family (locked)

All observers below are vectors over sites, parameterized by cell (L, J/U, τ). The trial axis is treated as a stochastic ensemble where applicable.

### 2.1 F-class (pre-intervention)

| Symbol | Definition | Domain |
|---|---|---|
| `F_i` | Pre-intervention site variance F_i (raw) | per site, per cell |
| `F_rank_i` | Rank of site i in F descending order | per site, per cell |
| `F_centered_i` | F_i − mean_j F_j | per site, per cell |
| `F_zscore_i` | (F_i − mean_j F_j) / std_j F_j | per site, per cell |

Aggregates (scalar over sites): `F_sum`, `F_max`, `F_std`, `F_argmax`. (`F_argmax` is integer; treated separately.)

### 2.2 Δn-class (post-intervention response)

| Symbol | Definition | Domain |
|---|---|---|
| `dn_tgt_i` | delta_tgt[i], targeted-arm Δ⟨n_i⟩ | per site, per cell |
| `dn_rnd_mean_i` | mean over 100 trials of delta_rnd[i] | per site, per cell |
| `dn_rnd_std_i` | std over 100 trials of delta_rnd[i] | per site, per cell |
| `abs_dn_tgt_i` | \|delta_tgt[i]\| | per site, per cell |
| `abs_dn_rnd_mean_i` | mean over trials of \|delta_rnd[i]\| | per site, per cell |
| `redist_tgt_i` | max(0, −delta_tgt[i]) (positive-loss component) | per site, per cell |

Aggregates: `dn_tgt_sum_abs`, `dn_tgt_max_abs`, `dn_tgt_argmax_abs`, and the same for the random-arm mean.

### 2.3 Cross-class observers (used only in T1 and T5)

| Symbol | Definition |
|---|---|
| `gap_i` | \|delta_tgt[i]\| − mean_t \|delta_rnd_t[i]\| (per-site advantage of targeting) |
| `gap_signed_i` | delta_tgt[i] − mean_t delta_rnd_t[i] |
| `redist_gap_i` | redist_tgt[i] − mean_t redist_rnd_t[i] |

### 2.4 Class assignment for T2

For T2, the observers above partition into three classes (used for the within-class vs. across-class angle prediction):

* **F-class**: `F_i`, `F_rank_i`, `F_centered_i`, `F_zscore_i`. Aggregates excluded from the pairwise-angle test (they are 1-d).
* **dn-class**: `dn_tgt_i`, `dn_rnd_mean_i`, `abs_dn_tgt_i`, `abs_dn_rnd_mean_i`, `redist_tgt_i`. Aggregates excluded.
* **gap-class**: `gap_i`, `gap_signed_i`, `redist_gap_i`.

This gives 4 + 5 + 3 = 12 vector observers, yielding C(12,2) = 66 pairwise angles. The class boundaries above are locked here.

---

## 3. T1 — Predictive vs causal observer dissociation

### 3.1 Question

In each cell, does the observer that best PREDICTS the system's response disagree with the observer that best identifies the optimal CAUSAL intervention site?

### 3.2 Operational definitions

For each cell (L, J/U, τ):

**Predictive score** (per observer X over sites): the squared Pearson correlation r²(X_i, dn_rnd_mean_i) across sites, computed within the cell. Interpretation: how well does this observer's spatial pattern at burn-in predict the average random-arm response pattern? (We use the random arm as the reference because it is the unbiased baseline response without selector bias.)

**Causal score** (per intervention site i, single-site question): the per-site `gap_i = |delta_tgt[i]| − mean_t |delta_rnd_t[i]|`. The causal "winner" is `argmax_i gap_i` — i.e., the site at which targeting yields the largest per-site advantage over random.

**Predictive winner**: the observer X whose pattern X_i best matches the response pattern dn_rnd_mean_i (highest |r| across sites). Note this is an observer label, not a site.

**Causal winner**: the site index i with largest gap_i. This is a site index, not an observer label.

### 3.3 The dissociation question

We define dissociation in the cell as: *the site that the predictive winner observer would point at* (its own argmax_i) ≠ *the causal winner site*.

Concretely:

* Predictive winner observer X*: argmax over X in observer family of |corr(X_i, dn_rnd_mean_i)|.
* Predicted-best-site under X*: argmax_i X*_i.
* Causal-best-site: argmax_i gap_i.
* Cell counts as DISSOCIATED if predicted-best-site ≠ causal-best-site.

### 3.4 Locked prediction

**P1.1**: Across all 48 cells, dissociation rate ≥ 50%.

**Falsification F1.1**: dissociation rate ≤ 20% (i.e., predictive-best-site agrees with causal-best-site in ≥ 80% of cells).

### 3.5 Auxiliary, exploratory (NOT pre-registered)

Report (without thresholds) the median |r| of the predictive winner across cells, and the median gap of the causal winner. These quantify how strong the two channels are independently.

### 3.6 Sensitivity (declared in advance)

Re-run T1 using `dn_tgt_i` instead of `dn_rnd_mean_i` as the response target (with the targeted arm result substituted). If the pre-registered conclusion flips between these two response references, report the flip. Do not change the locked threshold.

---

## 4. T2 — Pairwise principal angle structure between observers

### 4.1 Question

If we treat each observer as a point in observation space, do same-class observers cluster (small principal angles) and across-class observers separate (large angles), beyond what a Haar-uniform random null would produce?

### 4.2 Operational construction (data-bound substitute for the brief's Fisher version)

The brief asked for Fisher matrices w.r.t. (target site, γ_extra, J/U). γ_extra is fixed in our data, so we use:

* The L-dimensional **site axis** within each cell (the observer's value at each site).
* The **cell axis** across (L, J/U, τ) cells. To make cells comparable across L, observers are computed per-cell separately, then **per-cell-normalized** by subtracting cell-wise mean and dividing by cell-wise std (z-score per cell). After per-cell z-scoring, observer values across cells of different L still live in different site-spaces.

To compute principal angles between observers across cells of different L, we restrict the angle computation to cells of the same L, then aggregate across L. For each L (6, 8, 9), for each pair of observers (X, Y):

1. Build the cell × site matrix M_X[c, i] = X_i in cell c (z-scored within cell).
2. Build M_Y similarly.
3. Compute the top-r leading right-singular subspaces of M_X and M_Y (r = 2).
4. Compute the principal angles between these two r-dim subspaces in site-space (dimension L; meaningful since L ≥ r + 1 for L ∈ {6, 8, 9}).

The pairwise angle for the (X, Y) pair at L is the larger of the two principal angles (a standard worst-case alignment metric).

The cell axis for L=6 has 12 cells (4 J/U × 3 τ); for L=8, 16 (8 × 2); for L=9, 20 (10 × 2). All cells per L are used.

### 4.3 Class definitions

Locked in §2.4: F-class (4 observers), dn-class (5 observers), gap-class (3 observers). Aggregate scalar observers are excluded from T2 because the principal-angle calculation requires vector observers.

Within-class pairs: C(4,2)+C(5,2)+C(3,2) = 6+10+3 = 19.
Across-class pairs: 4×5 + 4×3 + 5×3 = 20+12+15 = 47.
Total pairs: 66.

### 4.4 Locked prediction

**P2.1** (within < across): For each L in {6, 8, 9}, the median within-class principal angle is strictly less than the median across-class principal angle.

**P2.2** (magnitude floor): Within-class median angle < 0.7 rad; across-class median angle > 1.0 rad. (These are loosened from the brief's 0.3 / 1.0, because per-cell-normalized vectors in 6–9-dim site space have smaller dynamic range than the cosmo settings the brief was calibrated against. The 0.7 / 1.0 split keeps a clear separation while admitting the smaller ambient space.)

**P2.3** (KS distinguishability from Haar null): For each L, the empirical pairwise angle distribution differs from the Haar-uniform null on principal angles between random r=2 subspaces in R^L at p < 0.05 (KS test), with the empirical distribution showing a longer left tail (within-class observers cluster tighter than random).

**Falsification F2.1**: P2.1 fails (within ≥ across) for any L.

**Falsification F2.2**: P2.3 fails (KS p ≥ 0.05) for all three L values — i.e., the angle distribution is indistinguishable from random at every L.

**Falsification F2.3**: Within-class median angle exceeds across-class median angle at any L.

### 4.5 Haar null

For each L and each r=2, generate 5000 pairs of random orthonormal r-frames in R^L (uniform on the Stiefel manifold via QR of i.i.d. Gaussian L×r matrices), compute their principal angles, take the worst (larger) angle. Use this 5000-sample distribution as the null. KS test compares empirical 66 angles (per L) to the null distribution. Pre-registered seed `np.random.default_rng(20260506)` for null sampling.

### 4.6 Synthetic gate

Before running on real data: build synthetic observer matrices with known class structure (5 observers, sharing top-r subspace within class; 5 observers, disjoint across class). Verify the pipeline:

* Recovers within-class median angle < 0.05 rad on synthetic same-subspace observers.
* Recovers across-class median angle > 1.4 rad on synthetic disjoint-subspace observers.
* KS distinguishability p < 0.001 from Haar null on synthetic data.

If any of these fails, the pipeline is broken; do not touch real data until fixed. Synthetic gate code: `tests/test_synthetic_t2_classes.py`.

---

## 5. T3 — Principal direction shift across the J/U threshold

### 5.1 Question

The original paper's empirical regime boundary sits at J/U ≈ 0.20–0.30. Does the leading principal direction of the observer family shift sharply across this boundary in a way that explains why the protocol's behavior changes there?

### 5.2 Operational substitute for the brief's "band edge"

The brief asked for predicted band edges α_r/β_1 and α_1/β_r as functions of J/U. Computing those requires Fisher matrices with at least two parameter directions, which our data does not give cleanly (J/U is the only continuous parameter). We substitute a directly testable variant:

For each L ∈ {8, 9} (the two L's with finely-sampled J/U), and at each τ ∈ {2, 3}:

1. Build the J/U × site response matrix R[j, i] = `gap_i` at J/U = j-th sweep value, fixed L, fixed τ.
2. Compute the leading right-singular vector v_j of the matrix at J/U value j (via SVD; v_j is a unit L-vector over sites).
3. Compute the angle θ(j, j+1) = arccos(|⟨v_j, v_{j+1}⟩|) between consecutive J/U values.

A sharp principal-direction shift across the threshold is operationalized as:

θ_threshold = max over consecutive J/U pairs straddling J/U = 0.30 of arccos(|⟨v_j, v_{j+1}⟩|).
θ_offthreshold = median over consecutive J/U pairs not straddling J/U = 0.30 of the same quantity.

### 5.3 Locked prediction

**P3.1**: θ_threshold > θ_offthreshold for both (L=8, τ=2) and (L=9, τ=2). Specifically: θ_threshold > 2 × θ_offthreshold for at least one (L, τ) combination.

**P3.2** (peak location): The argmax over consecutive J/U pairs of θ falls within the J/U interval [0.24, 0.32] for at least one (L, τ) combination — i.e., the principal direction shift co-localizes with the empirical regime boundary.

**Falsification F3.1**: θ_threshold ≤ θ_offthreshold for both (L=8, τ=2) and (L=9, τ=2) — no shift visible at the boundary.

**Falsification F3.2**: argmax of θ across J/U pairs falls outside [0.20, 0.36] for both L=8 and L=9 — the principal-direction shift, if any, sits somewhere unrelated to the empirical threshold.

### 5.4 Sensitivity

Re-run with `redist_gap_i` and with `dn_tgt_i` in place of `gap_i`. Report whether the locked conclusion stabilizes across the three response observables. Do not adjust thresholds based on results.

---

## 6. T4 — Temporal evolution of misalignment band (3-timepoint coarse version)

### 6.1 Question (revised from brief)

The brief asked for monotonic angle shrinkage as the system thermalizes. Our data has only τ ∈ {1, 2, 3} (and τ=1 only at L=6). We test the coarse 3-point version: does the median pairwise angle between observers decrease monotonically from τ=1 to τ=2 to τ=3 at L=6?

For L=8 and L=9, only the 2-point comparison τ=2 vs τ=3 is available.

### 6.2 Operational construction

For each L, and at each available τ:

1. Build observer matrices as in T2 (cell-axis is just {fixed L, fixed J/U range, this τ}).
2. Compute all pairwise principal angles (66 pairs).
3. Take the median angle.

### 6.3 Locked prediction

**P4.1** (L=6, three points): median(τ=1) ≥ median(τ=2) ≥ median(τ=3), with strict inequality in at least one step (i.e., not a flat sequence).

**P4.2** (L=8 and L=9, two points each): median(τ=2) ≥ median(τ=3) at both L=8 and L=9.

**Falsification F4.1**: P4.1 fails — the L=6 sequence is non-monotonic in the wrong direction (e.g., median(τ=1) < median(τ=2)).

**Falsification F4.2**: Both P4.1 and P4.2 fail — no monotonic shrinkage at any L.

### 6.4 Synthetic gate

Build a synthetic observer family at three "thermalization stages" with known angle shrinkage (e.g., observers progressively more aligned at later stages). Verify the pipeline detects the monotonic shrinkage with median-angle separation > 0.2 rad between stages. Code: `tests/test_synthetic_t4_thermalization.py`.

---

## 7. T5 — Scoring-rule fork

### 7.1 Question

Different scoring rules (trace, op-norm, gap) for evaluating an intervention give different rankings of intervention sites. Does the optimal site disagree across rules in a non-trivial fraction of cells?

### 7.2 Operational construction

For each cell (L, J/U, τ), and considering single-site interventions only (which we proxy by per-site contributions of the actual k-site intervention; see §7.3 for the caveat):

* **Trace score per site i**: |delta_tgt[i]| − mean_t |delta_rnd_t[i]|. Equivalent to `gap_i`.
* **Op-norm score per site i**: max over sites j of |delta_tgt[j]| under "intervention at i" — but we lack per-single-site intervention data. Substitute: for site i, op-norm score is max_j |delta_tgt[j]| weighted by the indicator that i ∈ selected set. This is degenerate in the sense that for a fixed selected set the same maximum applies — so the op-norm rule reduces to a constant across i in the targeted set.

This degeneracy means the brief's three-rule fork is not cleanly testable on the existing intervention design (k > 1, fixed selected set per cell). We substitute three SCORING RULES that act on the per-site response vector and still yield meaningful different rankings:

**Revised three rules** (each gives a per-site score for SELECTING which single site to target if one were free to redo the intervention with k=1; computed retrospectively from the random-arm trials, where each trial hits a different site set):

For each site i, restrict to random-arm trials t in which `i ∈ random_t` (i.e., site i was hit in trial t). Let S_i = {trials in which i was hit}. Then:

* **Trace rule (R_trace)**: mean over t in S_i of (sum_j |delta_rnd_t[j]|). Total occupation movement when site i is part of the hit set.
* **Op-norm rule (R_opnorm)**: mean over t in S_i of (max_j |delta_rnd_t[j]|). Largest single-site movement.
* **Gap rule (R_gap)**: mean over t in S_i of |delta_rnd_t[i]| − mean over j ∉ random_t of |delta_rnd_t[j]|. Per-site advantage of having i in the hit set.

Each rule produces a ranking of L sites. Optimal site = argmax site under that rule.

### 7.3 Locked prediction

**P5.1** (disagreement rate): Across all 48 cells, the three rules' optimal-site choices disagree (i.e., not all three pick the same site) in ≥ 30% of cells.

**P5.2** (pairwise disagreement): At least one pair of rules (trace vs gap, opnorm vs gap, trace vs opnorm) disagrees on the optimal site in ≥ 20% of cells.

**Falsification F5.1**: All three rules agree on the optimal site in ≥ 90% of cells.

### 7.4 Sensitivity

Report the cell-level disagreement matrix. Do not change thresholds based on observed pattern.

---

## 8. T6 — Susceptibility hierarchy across intervention TYPES

### 8.1 Status: NOT TESTABLE on existing data

**Locked declaration**: The original brief asked for susceptibility ranking across multiple intervention types (dephasing, particle injection, phase imprinting, local heating). The existing simulation data contains a single intervention type — additional dephasing (γ_extra at the selected sites). Particle injection, phase imprinting, and heating were never simulated; the saved files contain no quantities computable for those intervention types.

T6 as originally framed therefore cannot be answered without new simulations.

### 8.2 Substitute (NOT pre-registered as a primary test, run only if useful)

A weak substitute exists: ranking observers by their susceptibility to the dephasing intervention at different scoring-rule variants (clip / signed / redist), using the existing `outputs/bh_hardening/susceptibility_results.csv`. This tests "rankings across scoring rules" rather than "rankings across intervention types." It is a different and weaker question. We will compute it as exploratory output but **not** lock a prediction or falsification on it. The "primary test" of T6 is declared NOT TESTABLE; that declaration is itself the locked outcome.

### 8.3 Recommended follow-up (not part of this battery)

For genuine T6, simulate the BH chain under at least one additional intervention type (e.g., local heating via amplitude damping, or per-site particle injection via a one-quanta source term) at the same (L, J/U, τ) cells. Then run the T6 protocol from the brief on the augmented dataset. This is OUT OF SCOPE for this reanalysis battery.

---

## 9. Synthetic gates (must pass before real data)

| Gate | Test file | Pass criteria |
|---|---|---|
| T2 angle pipeline correctness | `tests/test_synthetic_t2_classes.py` | Synthetic same-subspace observers: median within-angle < 0.05 rad. Disjoint synthetic observers: median across-angle > 1.4 rad. KS vs Haar p < 0.001. |
| T4 monotonic shrinkage detection | `tests/test_synthetic_t4_thermalization.py` | Synthetic 3-stage observer family with imposed shrinkage shows median(stage 1) − median(stage 3) > 0.2 rad. |

If a gate fails, fix the pipeline before touching real data. Real-data results computed before the gate passes are invalid.

---

## 10. Per-module unit tests (in addition to synthetic gates)

| Module | Test file | What it tests |
|---|---|---|
| `src/load_bh_data.py` | `tests/test_loaders.py` | Loader returns correct shapes and dtypes; F_i sums match summary table; mean_diff recomputed from delta_tgt and delta_rnd matches `mean_diff` column to numerical precision; cell counts match §1.3. |
| `src/observers.py` | `tests/test_observers.py` | Each observer is computed deterministically; class assignment matches §2.4; aggregate observers handle edge cases (constant input, single-site input). |
| `src/nulls.py` | `tests/test_nulls.py` | Haar null produces uniform Stiefel samples (mean cosine of random unit vectors near zero, std matches theory); seed is reproducible. |
| `src/scoring.py` | `tests/test_scoring.py` | Each scoring rule (trace, opnorm, gap) returns expected scores on small hand-built fixtures. |
| `src/principal_angles.py` | `tests/test_principal_angles.py` | Identical subspace gives angle 0; orthogonal subspace gives π/2; reconstructions stable under permutations. |

These tests must pass before any battery test runs. Failure of any of them invalidates results from any test that depends on the failing module.

---

## 11. Cross-check protocol (run before final summary)

After all six tests have results (including T6's not-testable declaration), spawn parallel agents:

* **Math agent**: re-verifies formulas in §§3–7 against the implementation. Specifically: T1 dissociation definition, T2 principal-angle formula and Haar null, T3 SVD direction shift, T5 conditional means.
* **Code agent**: audits `src/` against this pre-reg, looking for silent threshold drift, off-by-one in cell counting, or accidental observer-class reassignment.
* **Re-derivation agent**: independently re-computes the T1 dissociation rate and the T2 within-vs-across angle medians via a separate code path. Reports agreement / disagreement.
* **Pre-reg consistency agent**: verifies that the locked predictions in this document are the predictions tested in the summary, that no thresholds were adjusted, and that locked failures stayed failures.

Run agents in parallel (one message, multiple Agent calls). Surface all disagreements. Do not silently pick a side.

---

## 12. Output format

`bh_reanalysis/results/reanalysis_summary.md` must contain:

1. Pre-registered predictions table with PASS / FAIL / NOT TESTABLE for each of P1.1, P2.1, P2.2, P2.3, P3.1, P3.2, P4.1, P4.2, P5.1, P5.2, T6 status.
2. Detailed numerical results for each test.
3. Sensitivity notes where pre-registered.
4. Cross-check agent outputs and any disagreements.
5. Plain-language interpretation paragraph.
6. Explicit statement of what bounds the framework's reach in BH data, given the observed results.

---

## 13. Stopping conditions

Battery is complete when:

1. This pre-reg is locked (achieved at file creation).
2. Per-module unit tests pass.
3. Synthetic gates (T2 and T4) pass.
4. T1, T2, T3, T4, T5 run on real data; T6 status declared.
5. Multi-agent cross-check complete.
6. `reanalysis_summary.md` written with all locked predictions resolved.

---

## 14. What this earns (calibrated to the data scope)

* If T1 confirms ≥50% dissociation in BH, the framework gains a third substrate showing predictive-causal observer dissociation, alongside HPCM and Kuramoto. Limited by the F_i + Δn observer family (no energy/correlation observers due to data).
* If T2 shows clustered within-class vs spread across-class principal angles distinguishable from Haar null, the geometric tool's core signature (subspace structure tracking class membership) generalizes from cosmology to a dissipative quantum many-body system.
* If T3 shows a sharp principal-direction shift co-localized with the J/U regime boundary, the BH paper's central empirical result gets a structural correlate the original did not provide.
* T4 (3-point coarse) and T5 (scoring fork) are exploratory; outcomes either way are informative.
* T6 is NOT TESTABLE on existing data and that conclusion is locked here. If a follow-up runs additional intervention types, T6 can be re-tested from the brief's original specification.

If most of T1–T5 pass: the BH paper resubmission has new structural content the desk-rejected version did not.
If most fail: the framework's reach in BH is bounded to the original paper's empirical territory and that bound is documented honestly.

---

## 15. Amendments

### A1 — 2026-05-06 — Factual correction to §1.3 cell inventory

**What changed**: §1.3 declared L=9 has 20 cells (10 J/U × 2 τ) and total = 48 cells. The actual data has 24 L=9 cells (and 52 total). Specifically, four L=9 J/U values do store τ=1: J/U ∈ {0.12, 0.20, 0.30, 0.40} have τ ∈ {1, 2, 3} (= 12 cells). The other six L=9 J/U values {0.16, 0.18, 0.22, 0.24, 0.26, 0.28} have τ ∈ {2, 3} (= 12 cells). Total L=9 = 24 cells.

**Reason**: I miscounted the L=9 checkpoint contents at lock time. The error was caught by `tests/test_loaders.py::test_cell_counts_per_L_match_prereg` on first run, before any battery test was executed.

**Effect on predictions**: NONE. Tests T1–T5 use "all available cells" as input; the count is descriptive, not a threshold.

**Effect on T4**: Mild bonus — for L=9, the four J/U values with τ ∈ {1, 2, 3} stored allow an additional 3-point monotonic-shrinkage check restricted to those four J/U values. This is reported as auxiliary, not pre-registered:
* Pre-registered T4 P4.1 stays as defined (L=6 only).
* Pre-registered T4 P4.2 stays as defined (L=8 and L=9 at τ ∈ {2, 3}, using all available J/U values).
* Auxiliary (NOT pre-registered): three-point check at L=9 restricted to J/U ∈ {0.12, 0.20, 0.30, 0.40}, reported alongside but not weighted in pass/fail.

**Cell counts under correction**:
* L=6: 12 cells (4 J/U × 3 τ). Unchanged.
* L=8: 16 cells (8 J/U × 2 τ). Unchanged.
* L=9: 24 cells (4 J/U × 3 τ + 6 J/U × 2 τ). Corrected from 20.
* Total: **52 cells**, corrected from 48.

No thresholds in §§3–8 reference cell counts directly.

### A2 — 2026-05-06 — Operationalization of "longer left tail" in P2.3

**What changed**: §4.4 P2.3 says "with the empirical distribution showing a longer left tail." The phrase "longer left tail" is qualitative and needs an explicit operationalization to be testable as a boolean. The code (`src/observer_geometry.py`) operationalizes it as: empirical 25th percentile of pairwise angles (combined within + across) is strictly less than the Haar null's 25th percentile, i.e. `emp_q25 < null_q25`.

**Reason**: This was the operationalization implemented in the code at lock time, but the pre-reg text only stated the qualitative phrase. Surfaced by the code agent during the multi-agent cross-check. The operationalization is a faithful concretization of the locked phrase, not a tightening of the prediction.

**Effect on outcome**: NONE. All three L pass `emp_q25 < null_q25` and KS p < 0.05, so P2.3 passes either way under any reasonable operationalization of "longer left tail."

### A3 — 2026-05-06 — P3.2 acceptance band corrected to match pre-reg §5.3

**What changed**: The code originally used the wider band `[0.20, 0.36]` for the P3.2 acceptance check. Pre-reg §5.3 P3.2 explicitly locked the band as `[0.24, 0.32]` (with `[0.20, 0.36]` reserved for the F3.2 falsification check). Surfaced by the pre-reg consistency agent during cross-check. Corrected after results were computed.

**Reason**: Code-vs-spec drift caught by cross-check. Bringing code into compliance with the locked pre-reg.

**Effect on outcome**: NONE. The L=8 (τ=2) argmax pair [0.28, 0.30] and L=8 (τ=3) argmax pair [0.26, 0.28] both lie inside the stricter band [0.24, 0.32], so P3.2 passes under either band. The L=9 cases [0.3, 0.4] lie outside both bands. F3.2 also not falsified because the L=8 cases are inside the falsification band.
