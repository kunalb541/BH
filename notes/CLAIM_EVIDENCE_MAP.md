# paper_v2 — claim-to-evidence map

Every headline claim in `paper_v2.tex` traced to its figure/table, the committed CSV it derives
from, the value, and its independent-verification status. Verification = 7-agent adversarial
workflow (`wf_ce7a0122-bff`): 6 agents recomputed each cluster directly from raw CSVs, 1
traceability critic. **All six recompute clusters returned `all_match: true`.** Numbers below are
the independently-recomputed values.

| # | Claim (paper) | Fig/Table | Source CSV | Verified value | Status |
|---|---|---|---|---|---|
| 1 | Handle onsets across the crossover; reproduces published 34/68/71 (L8, J/U=0.20) | Fig A, §3.1 | `pilot_results_L8.csv` | 33.9/67.9/71.4 → 34/68/71 (top-F_i = pct_geo/maxn col) | ✅ |
| 2 | Burn-in current divergence predicts handle, L=6,7,8 | Table T1, Fig A/B | `current_mech_L{6,7,8}.csv` | ρ(C_S_burn,D_S)=0.906/0.912/0.894; ρ(·,handle)=0.936/0.949/0.916 | ✅ |
| 3 | Crossover is a current reversal (C_S_burn changes sign) | Fig A(b) | `current_mech_L{6,7,8}.csv` | C_S_burn<0 low-J/U → >0 pocket | ✅ |
| 4 | D_S **is** integrated outward current (continuity) | §3.2, App. | `current_mech_*.csv` `cont_err` | finite-diff residual <6×10⁻⁵ (paper states <10⁻⁴) | ✅ (see note A) |
| 5 | Old redistribution-susceptibility is non-discriminating (ρ≈0 w/ success) | Table T2 | `pilot_results_L{6,7,8}.csv`, `symbreak_disorder.csv` | ρ(handle, F_i↔χ)=−0.03/+0.09/−0.26; disorder +0.02 (all p>0.3) | ✅ |
| 6 | F_i beats geometry under tilt & disorder (dephasing) | Table T3, Fig C | `symbreak_{tilt,disorder}.csv` | pocket 96.5/63.7 (tilt), 95.4/72.1 (dis); disjoint 95.6/37.3 | ✅ (note B) |
| 7 | Directed self-drain predicts handle (disorder) | §3.4 | `symbreak_disorder.csv` | ρ(pct_fi,D_S)=+0.65 (p=5e-45, n=360) | ✅ |
| 8 | Detuning: sign-controlled handle (+μ drains, −μ fills) | Fig D(a) | `detune_probe_L6.csv` | mean D_S τ=3: +0.33 (+μ) / −0.28 (−μ) | ✅ |
| 9 | Detuning beats geometry (tilt & disorder) | Table T3, Fig D(b) | `symbreak_detune_{tilt,disorder}.csv` | pocket 68.5/44.6 (tilt), 70.9/58.8 (dis); disjoint 49.8/34.6 | ✅ (note B) |
| 10 | Predictor bifurcation: dephasing reads (ρ≈0.7–0.9) vs detuning imposes (0.46–0.55) | §3.5 | `current_mech_*`, `current_symbreak_*`, `symbreak_detune_*` | deph 0.89–0.91 (clean), 0.70/0.75 (broken); detune 0.45/0.55 | ✅ |
| 11 | Loss is occupation-driven, not F_i | Table T4, Fig E | `loss_pilot_L6.csv`, `loss_symbreak_mini.csv` | clean ρ: ⟨n⟩+0.95 vs F_i+0.87; separated ⟨n⟩+0.998 vs F_i+0.73; maxn 99.6 vs F_i 84.9 | ✅ |
| 12 | Bond control does not beat site-level F_i | Table T5 | `bond_pilot_L6.csv` | induced best |ρ|=0.19 (none); total endpoint-F +0.87 ≥ coh +0.60, cur −0.83 | ✅ |
| 13 | Reanalysis: T4/T5 earned; T1 dissociation retired (artifact) | §3.8 | `bh_reanalysis/results/*.json` | T1 0.90→0.52 once self-predictor removed | ✅ (qualitative) |

## Downgraded / avoided claims (the critic's flags — NOT used in the paper)
- **A. "continuity exact to ~1e-6"** — that integral-demo number is script *stdout only*, not in any
  CSV (`current_mechanism.py` prints it; no column). The paper instead states the committed,
  CSV-backed finite-difference residual **<10⁻⁴** (`cont_err` max ≈ 5.5×10⁻⁵) and notes continuity
  is *analytically* exact. ✔ avoided.
- **"+0.78 within-realization"** — reproducible from `symbreak_disorder.csv` but **saturation-sensitive**
  (median +0.78, mean +0.54; pocket groups pin pct_fi at 100). The paper uses only the robust overall
  **+0.65**. ✔ avoided. (`MECHANISM_STATE.md`/memory corrected accordingly.)

## Note B — "disjoint" wording
"low-overlap" = `overlap_fi_geo < 0.5`; since overlap takes discrete values {0, 0.5, 1.0} in the
pocket, this is exactly the **disjoint** (overlap 0) subset. The paper says "fully disjoint /
most different sites," which is precise. (Using ≤0.5 would include the 0.5 bin and change the
numbers — the strict `<` is intended and stated.)

## Appendix-validation provenance
The appendix validation figures (sparse=dense to ~1e-15; γ_loss=0 ≡ fixed-N to 2e-16; per-bond-J
≡ scalar-J) come from committed, re-runnable **validation scripts** (`mechanism/mechanism_parity_check.py`,
`mechanism/stage0_loss.py`, `mechanism/bond_pilot.py` Stage 0), not result CSVs — standard for a
methods appendix and reproducible on demand. (All scripts live in `mechanism/`; run from repo root
with `PYTHONPATH=.`.)
