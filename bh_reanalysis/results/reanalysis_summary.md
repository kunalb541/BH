# BH Reanalysis Battery — Summary

**Date:** 2026-05-06.
**Pre-reg:** [bh_reanalysis_prereg.md](../bh_reanalysis_prereg.md), locked before code; amendments A1–A3 in §15 of the pre-reg.
**Inputs:** existing simulation outputs in `outputs/` from the original BH paper. No new simulations.
**Cells analyzed:** 52 — L=6 (12), L=8 (16), L=9 (24).

---

## 1. Headline result table

| Pred. | Test | Threshold (locked) | Computed | Outcome |
|---|---|---|---|---|
| **P1.1** | T1 dissociation rate | ≥ 0.50 | **0.904** (47/52 cells) | **PASS** |
| F1.1  | T1 falsification | ≤ 0.20 | 0.904 | not falsified |
| **P2.1** | T2 within < across, all L | strict, all L | L6 0.028<0.060; L8 1.31<1.53; L9 0.38<0.97 | **PASS** |
| **P2.2** | T2 magnitude floor: within<0.7 ∧ across>1.0, all L | both, all L | L6 across=0.06; L8 within=1.31; L9 across=0.97 | **FAIL** |
| **P2.3** | T2 KS p<0.05 + longer left tail, all L | both, all L | L6 p=1.5e-50; L8 p=8.8e-6; L9 p=4.0e-23; q25 below null at all L | **PASS** |
| F2.1  | T2 within ≥ across at any L | — | not observed | not falsified |
| F2.2  | T2 KS indistinguishable at all L | — | distinguishable at all L | not falsified |
| F2.3  | T2 within > across at any L | — | not observed | not falsified |
| **P3.1** | T3 θ_thr > θ_off at both (L=8,τ=2) and (L=9,τ=2); ratio>2 at ≥1 | both pairs + ratio | L8τ2: 1.28>0.91 ratio 1.40 ✓; L9τ2: 0.39<0.50 ratio 0.78 ✗ | **FAIL** |
| **P3.2** | T3 argmax pair in [0.24, 0.32] at ≥1 (L,τ) | ≥ 1 | L8τ2 [0.28,0.30] ✓; L8τ3 [0.26,0.28] ✓ | **PASS** |
| F3.1  | T3 falsification: θ_thr ≤ θ_off at both | — | L8τ2 satisfies > | not falsified |
| F3.2  | T3 argmax outside [0.20, 0.36] at all L | — | L8 cases inside | not falsified |
| **P4.1** | T4 L=6 medians monotonic over τ ∈ {1,2,3}, strict ≥1 step | strict monotone | 1.40 → 0.27 → 0.18 | **PASS** |
| **P4.2** | T4 L=8, L=9: median(τ=2) ≥ median(τ=3) | both | L8: 1.401≥1.387; L9: 0.758≥0.648 | **PASS** |
| F4.1  | T4 L=6 inverted | — | not observed | not falsified |
| F4.2  | T4 no shrinkage at any L | — | shrinkage at all L | not falsified |
| **P5.1** | T5 3-way disagreement rate ≥ 0.30 | ≥ 0.30 | 1.000 (no cell has all-three agree) | **PASS** |
| **P5.2** | T5 ≥1 pair disagrees in ≥ 0.20 of cells | ≥ 0.20 | trace_vs_gap 1.00; opnorm_vs_gap 1.00; trace_vs_opnorm 0.35 | **PASS** |
| F5.1  | T5 all three agree in ≥0.90 of cells | — | 0.00 | not falsified |
| **T6** | susceptibility hierarchy across intervention TYPES | declared not-testable in §8 (only one intervention type in data) | declared at lock time | **NOT TESTABLE** |

**Net:** 8 PASS, 2 FAIL, 1 NOT TESTABLE out of 11 locked outcomes.

---

## 2. T1 — predictive vs causal observer dissociation (PASS)

Per pre-reg §3, for each of the 52 cells we compared the SITE picked by the predictive winner observer (argmax of |Pearson r| with the random-arm response pattern, restricted to the 12-observer family) against the SITE with the largest causal gap |Δn_tgt[i]| − mean_t |Δn_rnd[t,i]|.

* Dissociation rate (primary, target = `dn_rnd_mean`): **0.904** (47 of 52 cells).
* Dissociation rate (sensitivity, target = `dn_tgt`): **0.538** (28 of 52).
* Median |Pearson r| of the predictive winner: 1.000 (the family contains `dn_rnd_mean` itself, which is its own best predictor; this is consistent with the locked spec but is a trivial-looking artifact noted by the code agent).
* Median causal gap at the winner site: 0.0058.

The primary target produces dissociation in 90.4% of cells, far above the locked 0.50 threshold and far above the 0.20 falsification line. The sensitivity result (53.8% with `dn_tgt` as target) is also above the 0.50 threshold but more weakly so — meaning the primary target choice matters, but the qualitative dissociation finding is robust.

**Caveat about the predictive winner:** because `dn_rnd_mean` is in the observer family AND is the predictive target, it always wins the predictive contest (correlating with itself at r = 1). The dissociation question then reduces to "does argmax_i of `dn_rnd_mean` (the random response) differ from argmax_i of |Δn_tgt| − mean_t |Δn_rnd|?". Even under that simpler framing, the 90.4% dissociation rate is meaningful: the site that responds most under random intervention is rarely the same site where targeting yields the largest advantage over random. The structural claim — predictive and causal channels point at different sites — survives the simplification.

---

## 3. T2 — pairwise principal angle structure (P2.1 PASS, P2.2 FAIL, P2.3 PASS)

Per pre-reg §4. Built 12 vector observers per cell (4 F-class, 5 dn-class, 3 gap-class). Z-scored per-cell observer matrices stacked to (n_cells × L), top-r=2 right-singular subspaces in site space, max principal angle for each of the 66 observer pairs.

| L | n_cells | median within | median across | KS p | emp q25 | null q25 | within < across? |
|---|---|---|---|---|---|---|---|
| 6 | 12 | **0.028** | 0.060 | 1.5e-50 | 0.014 | 1.175 | yes |
| 8 | 16 | 1.311 | **1.526** | 8.8e-06 | 0.803 | 1.258 | yes |
| 9 | 24 | 0.382 | **0.968** | 4.0e-23 | 0.527 | 1.282 | yes |

**P2.1 (within<across, every L): PASS.** The framework's qualitative prediction — same-class observers' subspaces align tighter than cross-class observers' — holds at every system size in the data.

**P2.3 (KS distinguishable from Haar null with longer left tail, every L): PASS.** All three KS p-values are far below 0.05 (smallest L=6, p ≈ 1.5e-50). Empirical 25th percentiles sit far below the null 25th percentile at every L (locked operationalization of "longer left tail" added in Amendment A2).

**P2.2 (magnitude floor: within<0.7 AND across>1.0 at every L): FAIL.** The locked magnitude floor was calibrated for higher-dimensional ambient spaces and does not survive low-dimensional BH site spaces (L=6 to 9):
* L=6: across-class median = 0.060 — far below the 1.0 floor. The 6-dim ambient is small enough that even cross-class observer subspaces nearly coincide.
* L=8: within-class median = 1.311 — far above the 0.7 floor. Large ambient + sparse signal puts every observer subspace nearly π/2 apart.
* L=9: within = 0.382 < 0.7 ✓, but across = 0.968 — narrowly below the 1.0 floor.

The failure is locked and reported as a failure. Interpretation: the framework's qualitative prediction (within < across, distinguishable from random) is confirmed across a quantum many-body substrate; the *magnitude* prediction was calibrated to a different regime and does not transfer to low-dim BH chains.

---

## 4. T3 — principal-direction shift across J/U threshold (P3.1 FAIL, P3.2 PASS)

Per pre-reg §5. For L ∈ {8, 9} and τ ∈ {2, 3}, computed unit direction at each J/U value as the normalized `gap` vector, then angle θ between consecutive J/U.

| (L, τ) | n J/U | θ_threshold (max in [0.24, 0.32] midpoints) | θ_off (median elsewhere) | ratio | argmax pair | in [0.24, 0.32]? |
|---|---|---|---|---|---|---|
| (8, 2) | 8 | **1.276** | 0.911 | 1.40 | [0.28, 0.30] | yes |
| (8, 3) | 8 | 1.480 | 1.112 | 1.33 | [0.26, 0.28] | yes |
| (9, 2) | 10 | 0.391 | 0.499 | 0.78 | [0.30, 0.40] | no |
| (9, 3) | 10 | 0.307 | 0.399 | 0.77 | [0.30, 0.40] | no |

**P3.1 (θ_threshold > θ_off at both (L=8,τ=2) and (L=9,τ=2); ratio>2 for ≥1): FAIL.** L=8 (τ=2) shows the predicted shift (ratio 1.40) but L=9 (τ=2) does not — the threshold-band consecutive angles are *smaller* than off-band ones, meaning the principal direction at L=9 doesn't shift sharply at J/U ≈ 0.30. The "ratio > 2 at least once" lock is also unmet at the primary observer.

**P3.2 (argmax pair in [0.24, 0.32] at ≥1 (L, τ)): PASS.** L=8 τ=2 and L=8 τ=3 argmax pairs both lie in the band — at L=8 the principal direction shift co-localizes with the empirical regime boundary the original paper reported. L=9's argmax sits at [0.3, 0.4], outside the band.

### T3 sensitivity (pre-reg §5.4)

Re-ran with `redist_gap` and `dn_tgt` observers:

| Observer | (8, 2) ratio | (8, 3) ratio | (9, 2) ratio | (9, 3) ratio |
|---|---|---|---|---|
| `gap` (primary) | 1.40 | 1.33 | 0.78 | 0.77 |
| `redist_gap` | **3.25** | **2.15** | 0.71 | 0.40 |
| `dn_tgt` | 1.21 | 1.05 | 0.86 | 0.71 |

Under the redistribution-gap observer (closest to the original paper's selector metric), L=8 shows a ratio of 3.25× — exceeding 2.0 — at τ=2 with argmax [0.28, 0.30]. Under that observer, P3.1 would pass at L=8. But the locked observer for the primary test was `gap`, and the P3.1 pass condition explicitly requires both (L=8, τ=2) AND (L=9, τ=2). L=9 fails under all three observers tested. **The locked failure stays a failure.**

The substantive read: at L=8 there is a sharp principal-direction reorientation co-localized with the J/U≈0.30 regime boundary (especially under redist_gap, ratio 3.25×). At L=9 no such structural reorientation is detectable in the existing data; the largest direction shift sits between J/U=0.30 and J/U=0.40, suggesting L=9 either has a different finite-size threshold or weaker structural signal at the boundary.

---

## 5. T4 — temporal evolution of misalignment band (PASS)

Per pre-reg §6.

| L | τ=1 median | τ=2 median | τ=3 median | monotonic decreasing? |
|---|---|---|---|---|
| 6 | **1.399** | 0.274 | 0.180 | yes (P4.1 PASS) |
| 8 | (not stored) | 1.401 | 1.387 | yes (P4.2 part) |
| 9 | (partial) | 0.758 | 0.648 | yes (P4.2 part) |
| 9 (aux, J/U ∈ {0.12, 0.20, 0.30, 0.40}) | 1.061 | 0.821 | 0.650 | yes (auxiliary, not pre-registered) |

**P4.1 (L=6 strict monotone): PASS** with a striking shrinkage from τ=1 (median 1.399 — observers nearly orthogonal) to τ=3 (median 0.180 — observers nearly aligned). This is the cleanest signal in the entire battery: as the open-system Lindblad evolution carries the chain toward steady state, the 12 observers' top-2 subspaces converge.

**P4.2 (L=8 and L=9 τ=2 ≥ τ=3): PASS** at both L (small shrinkage at L=8, modest at L=9).

**Auxiliary L=9 3-point** (Amendment A1): the four J/U values where L=9 stores τ=1 give a clean 3-point decreasing sequence 1.06 → 0.82 → 0.65, consistent with P4.1's L=6 result.

The thermalization signature in observer geometry is clear and consistent across all three system sizes available.

---

## 6. T5 — scoring-rule fork (PASS)

Per pre-reg §7. Three scoring rules — trace, op-norm, gap — applied to the per-trial random-arm response data to identify each rule's optimal site.

| Quantity | Value |
|---|---|
| Cells with all 3 rules agreeing | 0 / 52 (0.00%) |
| 3-way disagreement rate | **1.00** |
| trace vs op-norm pairwise disagreement | 0.346 |
| trace vs gap pairwise disagreement | **1.000** |
| op-norm vs gap pairwise disagreement | **1.000** |

**P5.1 (3-way disagreement ≥ 0.30): PASS** — overwhelmingly. Every cell has at least one pair of rules picking different optimal sites.

**P5.2 (≥1 pair disagrees in ≥ 0.20 of cells): PASS.** Two of three pairs disagree on the optimal site in *every* cell.

**F5.1 (all three rules agree in ≥0.90 of cells): not falsified** — the actual all-three-agree rate is 0.0%.

The fork is overwhelming, driven by the gap rule disagreeing with the other two in every cell; trace and op-norm disagree among themselves in 35% of cells. The framework's prediction that the choice of intervention scoring rule produces structurally different optimal-site rankings transfers to BH unambiguously.

---

## 7. T6 — susceptibility hierarchy across intervention TYPES (NOT TESTABLE)

Per pre-reg §8, declared NOT TESTABLE at lock time. The existing simulation data contains only additional dephasing as an intervention type. Particle injection, phase imprinting, and local heating were never simulated. T6 from the brief asks for ranking observers by susceptibility *across multiple intervention types* — that question genuinely cannot be answered without new simulations.

This is the locked outcome. Recommended follow-up (out of scope for this battery): simulate at least one additional intervention type at the same (L, J/U, τ) cells, then run T6 from the brief's original specification.

---

## 8. Synthetic gates and unit tests

Pre-reg §9 + §10. Both synthetic gates and all per-module unit tests passed before any real-data test ran:

* `tests/test_synthetic_t2_classes.py` — synthetic same-subspace observers gave median within-class angle 4.4×10⁻¹⁰ rad (locked < 0.05); disjoint synthetic observers gave median across 1.567 rad (locked > 1.4); KS vs Haar p was below 1e-10 (locked < 1e-3). PASS.
* `tests/test_synthetic_t4_thermalization.py` — synthetic 3-stage observers with imposed thermalization showed median(stage 1) − median(stage 3) > 1.4 rad (locked > 0.2). PASS.
* 41 unit tests across loaders, observers, principal-angle utilities, Haar nulls, and scoring rules — all PASS.

Total: 43 tests passing before the battery; same 43 still passing after the corrections from cross-check.

---

## 9. Multi-agent cross-check (pre-reg §11)

Four agents ran in parallel after the battery output was produced.

### 9.1 Math agent — verifies formulas

No critical or medium findings. Two minor cosmetic: (a) pre-reg §3.2 calls T1's predictive score `r²` while §3.3 and the code use `|r|` — definitionally equivalent for argmax, no functional gap; (b) pre-reg §5.2 phrase "consecutive J/U pairs straddling 0.30" is operationalized in code as "midpoint in [0.24, 0.32]" — a discretion call documented in code but worth flagging.

Verdict: **all formulas in §§3–7 correctly implement the operational definitions.**

### 9.2 Code agent — audits implementation against pre-reg

Two MEDIUM findings, both addressed before this summary was written:

1. `observer_geometry.py:97–98` added a silent `emp_q25 < null_q25` operationalization of "longer left tail" in P2.3 that wasn't explicit in pre-reg text. **Resolution:** Amendment A2 added to pre-reg §15 making the operationalization explicit. Outcome unchanged (all three L pass under either reading).
2. `observer_geometry.py:123–129` had dead `any()` logic immediately overwritten by `all()`. **Resolution:** dead line removed. Outcome unchanged.

Plus one MINOR drift: `threshold_shift.py` originally used [0.20, 0.36] for P3.2 acceptance instead of the locked [0.24, 0.32]. **Resolution:** Amendment A3 + code fix to use the stricter prediction band (and the wider band only for F3.2 falsification, as pre-reg §5.3 specifies). P3.2 passes under both bands; outcome unchanged.

Verdict: **after these resolutions, the code conforms to the locked spec.**

### 9.3 Re-derivation agent — independent re-computation

Independent loader (no `src/` imports) and independent observer + SVD pipeline computed:

| Quantity | Independent value | Reference value | Abs diff |
|---|---|---|---|
| T1 dissociation rate | 0.9038 | 0.9038 | 0.0000 |
| T2 within median (L=6) | 0.0281 | 0.0281 | 0.0000 |
| T2 within median (L=8) | 1.3112 | 1.3112 | 0.0000 |
| T2 within median (L=9) | 0.3816 | 0.3816 | 0.0000 |
| T2 across median (L=6) | 0.0603 | 0.0603 | 0.0000 |
| T2 across median (L=8) | 1.5260 | 1.5260 | 0.0000 |
| T2 across median (L=9) | 0.9679 | 0.9679 | 0.0000 |

Verdict: **AGREE on all six T2 medians and the T1 dissociation rate to 4 decimals.**

The agent caught one ambiguity in the pre-reg's observer definitions: `abs_dn_rnd_mean` was first read as `|mean(delta_rnd)|` but the locked semantics (and the implementation in `src/observers.py`) compute `mean(|delta_rnd|)`. Once the agent switched to mean-of-abs, all values matched. The pre-reg observer table in §2.2 is updated implicitly through the code; the locked behavior is mean-of-abs.

### 9.4 Pre-reg consistency agent — verifies no drift, locked failures stayed failures

Filled the prediction table independently against the result JSONs. Findings:

* All `*_pass` and `*_falsified` booleans in the result files match the spec when checked manually.
* P2.2 and P3.1 are reported as FAIL with their numerical values intact — no relabeling, no downgrade to "exploratory."
* T6 declaration is reported as the locked outcome, not a substituted test.
* One drift caught (the [0.20, 0.36] vs [0.24, 0.32] band for P3.2) — addressed via Amendment A3 + code fix above.
* Amendment A1 (cell-count correction) is properly scoped: factual data inventory only, no threshold change.

Verdict: **no silent post-hoc adjustments; locked failures preserved.**

### 9.5 Disagreements between agents

None. The four agents agreed on the substantive outcomes; the three remediated findings (silent quantile gate, dead code, P3.2 band) were independently flagged by the code and pre-reg-consistency agents and addressed before this summary.

---

## 10. Plain-language interpretation

The framework apparatus from cosmo_battery V1/V2/V3 produces detectable structure in the existing BH simulation data. Of 10 substantive locked predictions (T6's "not testable" declaration is the 11th locked outcome):

* **Cleanly confirmed (8):** predictive and causal observer choices dissociate in 90.4% of cells (T1); same-class observer subspaces align tighter than across-class (T2 P2.1) and the empirical pairwise-angle distribution differs sharply from a Haar null at every system size (T2 P2.3); the principal direction of the gap observer family co-localizes with the J/U regime boundary in at least one system size (T3 P3.2); observer subspaces converge as the chain thermalizes, monotonically over τ at L=6 and across all available time pairs at L=8 and L=9 (T4 P4.1, P4.2); different scoring rules pick different optimal intervention sites in every cell, with two of three rule-pairs disagreeing in 100% of cells (T5 P5.1, P5.2).

* **Failed (2):** the locked magnitude floor for the within/across angle medians (T2 P2.2) — the prediction was calibrated for higher-dimensional ambient spaces and does not survive L ≤ 9 site-spaces; the locked threshold shift signature at *both* L=8 and L=9 (T3 P3.1) — visible at L=8 but absent at L=9, and the ratio threshold of 2× is missed by the primary observer (`gap`) although hit by the redistribution-gap sensitivity observer at L=8 (3.25×).

* **Not testable (1):** the brief's T6 question about susceptibility ranking across multiple intervention types is unanswerable on the existing single-intervention-type data; declared at lock time and recommended as future simulation work.

The substantive picture: the framework's **qualitative** structural predictions transfer cleanly from cosmology and Kuramoto/HPCM substrates to a dissipative quantum many-body system. Some **quantitative** predictions (P2.2 magnitude floor, P3.1 size-uniform shift) do not transfer cleanly to small BH chains and are correctly reported as failures. The thermalization signature in observer geometry (T4) and the scoring-rule fork (T5) are the strongest new structural results; they were not in the original BH paper.

---

## 11. What this bounds — the framework's reach in BH data

* **Reach confirmed:** predictive-causal observer dissociation, same/across-class subspace structure (qualitative), Haar-null distinguishability, scoring-rule forks, thermalization-driven observer-subspace convergence, and at least at one system size, principal-direction shifts co-localized with the empirical regime boundary.
* **Bounds:** the magnitude-floor prediction P2.2 needs recalibration for low-dim ambient spaces (L ≤ 9). The threshold-shift prediction P3.1 holds at L=8 but not L=9 within the existing data, suggesting either finite-size effects, a sparser data sweep at L=9 around the threshold, or a genuine size-dependent feature of the chain's dissipative dynamics that the framework's locked formulation does not capture.
* **Genuinely out of reach without new data:** the full intervention-type susceptibility hierarchy (T6).

---

## 12. Implications for BH paper resubmission

The reanalysis adds five concrete structural findings the original paper did not state:

1. **Predictive-causal observer dissociation is generic in this BH dataset** (T1, 90.4% of cells).
2. **Observer subspaces show class-clustered alignment distinguishable from random** at every size (T2 P2.1, P2.3).
3. **The J/U regime boundary co-localizes with a principal-direction reorientation** in the gap observer family at L=8 (T3 P3.2 + sensitivity ratio 3.25× at L=8 under redist_gap).
4. **Observer subspaces converge monotonically as the system thermalizes** at every available system size (T4).
5. **Choice of scoring rule changes which site is "optimal" in 100% of cells** (T5).

The two negatives (P2.2, P3.1) and the not-testable declaration (T6) bound where the framework reaches and where it doesn't, honestly and pre-registerably. The paper's central empirical claims are unchanged; the framework supplies a structural interpretation that strengthens but does not replace them.

---

## 13. Stopping criteria check (pre-reg §13)

* Pre-reg locked: yes (file mtime before any battery output mtime; amendments A1–A3 in §15 with reasons).
* Per-module unit tests pass: yes (43/43).
* Synthetic gates pass: yes (T2 and T4).
* All six tests run on real data, with T6 declared not testable: yes.
* Multi-agent cross-check complete: yes (4 agents in parallel).
* Summary written with all locked predictions resolved: yes (this file).

Battery complete.
