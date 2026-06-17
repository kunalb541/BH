# Paper rewrite — outline for approval (write prose only after sign-off)

Repo integrated on `main` (7354b83). Sources to mine: committed CSVs in
`outputs/mechanism_pilot/`, `MECHANISM_STATE.md`, `CLAIM_STATUS.md`, the old
`paper.tex` (methods/robustness prose, reusable), and `paper_expanded_may6.tex`
(reference — reuse only what is traceable to code/results; it contains thesis-derived
material flagged earlier as possibly unsupported).

## Central claim (the thesis of the rewrite)
In an exact, small open Bose–Hubbard chain, **local number variance F_i is the operative
control selector for N-conserving transport interventions (dephasing, detuning), while local
occupation ⟨n_i⟩ is the operative selector for the N-changing loss intervention** — geometry is
never fundamental. The handle works iff the targeted sites are in a *directed self-draining*
regime, which *is* the integrated outward particle current; the crossover is a current reversal.
**Description privilege is intervention-relative.**

## Title options (pick one)
1. *Local number variance is a transport-control selector for number-conserving interventions in an open Bose–Hubbard chain*
2. *Intervention-relative control selectors in an open Bose–Hubbard chain: variance, occupation, and the transport boundary*
3. *When is a local observable a control handle? An intervention-relative answer in an open Bose–Hubbard chain*

## Framing decision (recommend physics-first)
Lead with the **physics** (directed self-drain = outward current; the intervention map). Put the
"intervention-relative description privilege" reading in the Discussion as the conceptual payoff —
PRA referees want mechanism, not philosophy. (The ODD language is the *interpretation*, not the
result.)

## Section skeleton → content → tables/figures
**1. Introduction.** The question: can a local observable serve as a *targeted control handle* in
an open quantum system, and is the answer intervention-dependent? What the rejected version had
(a selector that works) vs what this adds (a mechanism + a map across intervention classes). Cite
open BH / dephasing / engineered-dissipation / quantum-Zeno literature (mine old `refs.bib`).

**2. Model and methods.** H + Lindblad; F_i; sign convention D_S = −Σ_{i∈S}Δn_i; bond current +
continuity; the four intervention families (dephasing, detuning, loss, bond-modulation); exact
fixed-N evolution and the validated multi-N space for loss; exhaustive subset ranking; symmetry
breaking (tilt/disorder). → *numerical-validation table* (parity 1e-15, Stage-0 gates, γ_loss=0≡fixed-N).

**3. Results.**
- **3.1 The handle and its regimes** (setup; recap positive pocket as a manifestation). → reuse old Fig (heatmap).
- **3.2 Mechanism: directed self-drain = outward current.** Continuity exact (~1e-6); C_S_burn predicts handle; current reversal at crossover; size scaling. → **Table T1** (predictor by L=6,7,8); **Fig A** (handle onset + C_S_burn vs J/U, current reversal); **Fig B** (C_S_burn↔D_S scatter).
- **3.3 The old explanation retired.** Redistribution-susceptibility is non-discriminating (ρ≈0 with success). → **Table T2**.
- **3.4 Beyond geometry (dephasing).** Tilt + disorder; F_i beats geo, esp. at low overlap. → **Table T3a**; **Fig C** (F_i vs geo, overlap<0.5 panel).
- **3.5 Generality: coherent detuning.** Sign-controlled handle; beats geometry; predictor bifurcation (read vs impose). → **Table T3b**; **Fig D** (sign reversal + beats-geo).
- **3.6 The boundary: particle loss.** Occupation wins (+0.998), not F_i; N-conservation boundary. → **Table T4**; **Fig E** (ΔN_total predictor bars: ⟨n⟩ vs F_i vs current vs geo).
- **3.7 Site vs bond control.** Bonds don't beat sites; site-level F_i is operative. → **Table T5**.
- **3.8 (Secondary) Observer-geometry reanalysis** as supporting structure: T4 thermalization convergence, T5 scoring-rule fork **earned**; T1 dissociation **retired** (self-correlation artifact, 0.90→0.52, stated honestly). → small table or appendix.

**4. Discussion.** The classification map (**Table T6**, the headline); transport-modulation vs
particle-removal boundary; the intervention-relative description-privilege reading; relation to
quantum control / Zeno / dissipation-engineering.

**5. Conclusion & outlook.** Optional next: Liouvillian modes (which mode carries the current),
larger L via MPO/trajectories. Scope/limitations (L≤8, exact only, no thermodynamic limit, no lab).

**Appendices / Methods.** Multi-N construction + Stage-0 gates; exhaustive enumeration; sparse-vs-dense
parity; nmax & burn-in robustness (reuse old paper's hardening); per-bond-J Hamiltonian.

## Figures to generate (new, from committed CSVs) — needed before/with prose
- Fig A: `current_mech_L{6,7,8}.csv` — handle pct & C_S_burn vs J/U (current reversal).
- Fig B: C_S_burn vs D_S scatter (per condition).
- Fig C: `symbreak_{tilt,disorder}.csv` — F_i vs geo (pocket + low-overlap).
- Fig D: `detune_probe_L6.csv` + `symbreak_detune_*.csv` — sign reversal + beats-geo.
- Fig E: `loss_pilot_L6.csv` + `loss_symbreak_mini.csv` — ΔN_total predictor bars.
- (Reuse old `outputs/figures/` heatmap/robustness where still valid.)
I'll write a `make_paper_figures.py` from the CSVs (no re-simulation needed).

## Retire / keep
- **Retire:** "F_i = redistribution susceptibility" as mechanism; "transport freezes at low J/U"; reanalysis T1 headline.
- **Keep/repurpose:** exact-Lindblad rigor, exhaustive enumeration, nmax/burn-in/γ-scan robustness (now supporting), the positive-pocket data (recast as directed-self-drain onset).

## Open decisions for you (before I write prose)
1. **Title** (1/2/3 above)?
2. **Framing**: physics-first with ODD in Discussion (recommended) — ok?
3. **Reanalysis (3.8)**: include as a short results subsection, push to an appendix, or omit?
4. **Target**: PRA resubmission framing, or a different venue?
5. **In-place vs new file** for the manuscript: I'll write `paper_v2.tex` and leave `paper.tex` until you're happy, then swap — ok?
