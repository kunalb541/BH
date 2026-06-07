# Pre-registration — bond / hopping-modulation pilot (Step 4, DESIGN ONLY)

Status: **design/prereg, not run.** Tests whether the BH transport-control handle is
fundamentally **bond-level** (current lives on bonds) rather than site-level — i.e. whether
site-F_i is a *projection* of a deeper bond/current object.

## Question
Current is a bond observable, yet every handle so far (dephasing, detuning, loss) is a *site*
control. If we control **bonds directly** (modulate J), does a **bond-intrinsic selector beat
the site-F_i-derived choice**, and does direct bond control act more cleanly than site control?

## Intervention
Coherent, N-conserving hopping modulation on a selected bond set B, during post-burn evolution:
`J_b → J_b(1 + δ)`  for b ∈ B  (i.e. add `−Jδ(a†_b a_{b+1} + h.c.)` to H).
Uses exact **fixed-N** Lindblad machinery (reuses Steps 1–3; no multi-N).
- New machinery = a **per-bond-J Hamiltonian** (current `build_hamiltonian` takes scalar J).
  Stage-0 sanity (must pass before science): per-bond-J H with all J equal == scalar-J H to
  1e-12; δ=0 reproduces the no-intervention evolution.

## Primary target (per your decision)
For selected bond set B, endpoint set `E(B) = {i : i touches a selected bond}`:
> **D_E = −Σ_{i∈E(B)} Δn_i**  (positive = endpoints drain). Mirrors site-level D_S.

Secondary / diagnostic (NOT primary — avoid tautology): global redistribution norm
Σ_j|Δn_j|; integrated current change across selected bonds; off-endpoint response.

## Selectors (rank bonds; pick top k_bonds = 2; exhaustive C(L−1,2)=10 for L=6)
1. **endpoint-F sum** F_b + F_{b+1}  — the site-F_i-derived bond choice ("bonds adjacent to high-F_i"); the thing to beat
2. **bond coherence** C_b = ⟨a†_b a_{b+1} + h.c.⟩  — bond-intrinsic
3. **bond current magnitude** |I_b| = |⟨−iJ(a†_b a_{b+1} − h.c.)⟩|  — bond-intrinsic (may be sign-fragile)
4. **endpoint occupation sum** ⟨n_b⟩ + ⟨n_{b+1}⟩  — the confound control (the loss-lesson)
5. **random / exhaustive** bond subsets

## System
L=6 first; N=3; nmax=3; γ=0.1; burn-in=5; J/U ∈ {0.12,0.20,0.30,0.40}; τ ∈ {1,2,3};
δ ∈ {+0.1,+0.2,+0.5} (test −δ if cheap, for sign behaviour). Exhaustive over the 10 bond pairs.

## Comparisons / decisive metrics
- **Selector test (rigorous core):** Spearman(D_E, Σ selector) across the 10 bond subsets for
  each selector; head-to-head percentile of each selector's chosen bond set by D_E.
- **Bond-level promotion:** does selector 2 or 3 beat selector 1 (endpoint-F)?
- **Occupation control:** does selector 4 (endpoint occupation) explain D_E instead?
- **Bond-vs-site control:** qualitative/structural only — δ (fractional) and μ_extra/γ_extra
  (energy/rate) are different units, so compare *structure* (does bond control drive endpoint
  drain at all, and in the same regime pattern), not absolute matched magnitude.
- **Sign:** +δ (stronger hopping) vs −δ should behave sensibly and oppositely if coherent-flow-driven.

## Kill tests
- All bond selectors tie → bond-level claim not earned.
- Endpoint occupation (selector 4) dominates → bond control is occupation-like, not transport-like.
- Site-F_i-adjacent (selector 1) beats all intrinsic bond selectors → keep the **site-level**
  description (F_i is the object, not the bond).
- Bond coherence or |I_b| consistently beats endpoint-F → **promote to bond-level handle**
  (F_i is a site-projection of the bond/current handle).
- No structured endpoint drain under modulation → do not claim bond control.

## Scope guard
Earned scope unchanged: exact Lindblad, small open 1D BH (L≤8), N-conserving controls.
Do not claim thermodynamic limit, larger L, or universal control law from this pilot.

## Cost
Fixed-N D=56, exhaustive over 10 bond pairs × 4 J/U × 3 τ × 3 δ ≈ small (<2 min, no env-cap risk).

---
**Expected (hypothesis, not assumed):** bond coherence or endpoint-F sum should rank well;
|I_b| may be sign-fragile. The high-value upgrade would be selector 2/3 beating selector 1 →
*F_i is a site-level proxy for a bond-level current handle.*
**Next action after approval:** Stage-0 per-bond-J sanity, then the pilot; report before claims.
