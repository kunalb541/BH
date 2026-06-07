# Pre-registration — amplitude-damping / local-loss pilot (Step 3D, DESIGN ONLY)

Status: **design/prereg, not run.** Locks the plan before writing multi-N code so the
analysis is honest. Approve before Stage 0.

## Question
Does the F_i local-transport-control selector generalize to an **N-changing dissipative**
channel (particle loss), or is it scoped to N-conserving controls (dephasing, detuning)?
This is the strongest remaining boundary test of the earned claim.

## Intervention
Local amplitude damping (loss) at the selected sites, applied after burn-in:
- jump operator `L_i = a_i`, dissipator `γ_loss(a_i ρ a_i† − ½{n_i, ρ})`
- extra loss rate `γ_loss_extra` on the selected set S; baseline dephasing γ=0.1 unchanged.

## The two design hazards (and how the prereg handles them)

**1. Multi-N Hilbert space (loss couples N→N−1).** The fixed-N sector no longer closes.
But loss only *decreases* N from N0=⌊L/2⌋=3, so the complete space is N∈{0,1,2,3}.
For L=6, nmax=3 the sector sizes are 56+21+6+1 = **D=84** — so the evolution is **EXACT
(no truncation error)**, and the cost is ~L=7-like (dense D²≈7056, or sparse via the
existing `_SPARSE_D_THRESHOLD` trick). H and the dephasing dissipator are block-diagonal
in N; `a_i` is the only off-block (N→N−1) piece. Burn-in is computed in the N=3 sector as
before, then embedded into the multi-N space (top block) for the intervention.

**2. Trivial-drain pitfall.** Loss *directly* removes particles from whatever site it acts
on, so on-site occupation loss at S is trivially large and is NOT a meaningful selector
test (the original kill test). **Resolution — use the non-trivial target:** total *system*
particle loss under matched budget. Physical hypothesis tied to the current mechanism: loss
at a **high-current (high-F_i) site is sustained** because the site is continuously
replenished by transport from neighbours, whereas loss at a low-current/edge site
**saturates** (drains locally, then starves). So:

> **Primary target: ΔN_total(τ) = N0 − ⟨N̂(τ)⟩**, comparing top-F_i vs matched-budget
> random/geo targeting. Hypothesis: high-F_i targeting gives **larger** ΔN_total
> (current-replenished sustained loss). Secondary: off-selected redistribution.

## Stage 0 — validation gates (must pass before any science)
1. Build multi-N basis (N=0..3) + operators `a_i`, `n_i`, H, dissipators.
2. Trace preservation |Tr ρ(τ) − 1| < 1e-9 and positivity (λ_min ≳ −1e-10) under loss.
3. ⟨N̂⟩ strictly **decreases** under uniform local loss; ⟨N̂⟩ **constant** when γ_loss=0.
4. **γ_loss_extra = 0 reproduces the fixed-N evolution** (occupations match old code to ~1e-10)
   — confirms the multi-N embedding is correct.
5. Continuity still holds for the Hamiltonian part; loss adds a local sink term
   d⟨n_i⟩/dt = (current divergence) − γ_loss⟨n_i⟩ — verify numerically.

## Pilot settings (L=6 first)
- L=6, N0=3, nmax=3, baseline γ=0.1, burn-in=5, J/U ∈ {0.12,0.20,0.30,0.40}, τ ∈ {1,2,3}
- selected sets: top-F_i, geo, exhaustive C(6,2)=15 subsets (cheap)
- `γ_loss_extra` scan: {0.02, 0.05, 0.1} (small first — keeps low-N leakage modest, though
  the space is exact regardless)
- clean chain first; tilt/disorder **only after** the clean pilot passes.

## Measurements
1. ΔN_total(τ) (primary), per selector.
2. per-site Δn_i; on-selected loss Σ_{i∈S}Δn_i; off-selected Σ_{j∉S}Δn_j.
3. handle percentile of top-F_i among all subsets, by **ΔN_total** (and by off-selected).
4. F_i vs geo percentile/gap; whether F_i beats geo (clean, then low-overlap under breaking).
5. whether C_S_burn (burn-in current divergence) predicts ΔN_total — tests the
   "current-replenished sustained loss" hypothesis directly.
6. populations leaking to N=2,1,0 (report; sanity, not truncation).

## Kill tests
- If ΔN_total is essentially **independent of which sites are targeted** → loss is a trivial
  drain, not a selector test (report as such; do not claim generalization).
- If **geometry beats F_i**, the F_i selector does **not** generalize to N-changing loss.
- If the top-F_i advantage vanishes across all γ_loss → result is **N-conserving-channel
  scoped** (precise, publishable narrowing).
- If F_i beats geo AND C_S_burn predicts ΔN_total → **promote**: F_i is a broad local-control
  selector spanning incoherent dephasing, coherent detuning, and particle-loss dissipation.

## Decision / scope guard
Earned scope stays: exact Lindblad, small open 1D BH (L≤8), dephasing + detuning + (pending)
loss, clean/tilt/disorder. Do **not** claim thermodynamic limit, universal control law, all
interventions/regimes, large L, injection, or lab experiments.

## Cost
Multi-N D=84 ≈ L=7 cost; clean pilot ~few min sparse. Tilt/disorder would hit the same
~30-min env process cap → use the per-realization checkpoint/resume pattern from
`finish_detune_disorder.py`.

---
**Next action after approval:** implement Stage 0 + validation gates only; report gate
results before running the science pilot.
