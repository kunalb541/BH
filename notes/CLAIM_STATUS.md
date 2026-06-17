# BH — claim status (milestone, 2026-06-06)

> **In exact small open Bose–Hubbard chains, local number variance F_i is an earned selector
> for N-conserving transport-control interventions, while local occupation ⟨n_i⟩ is the earned
> selector for N-changing loss; description privilege is intervention-relative.**

This closes the BH *discovery* phase. The mechanism map is trusted; the paper is not yet written.
Full detail and numbers: [`MECHANISM_STATE.md`](MECHANISM_STATE.md). All work on branch
`bh-mechanism` (head 8452221); submission `paper.tex` on `main` untouched throughout.

## The result (one table)
| intervention | conserves N? | winning description | mechanism |
|---|---|---|---|
| dephasing | yes | **F_i** | reads / throttles pre-existing current |
| detuning  | yes | **F_i** | imposes coherent current |
| local loss | no | **⟨n_i⟩** | direct particle removal |

Boundary = **transport-modulation vs particle-removal**. Geometry is a correlate, never the
fundamental selector.

## What is earned
- Old "F_i ≈ redistribution susceptibility" explanation **retired** (non-discriminating, ρ≈0).
- New mechanism: the handle works iff the targeted sites are net-draining; **directed self-drain
  = integrated outward particle current** (continuity exact to ~1e-6).
- Size-supported clean chain **L=6,7,8** (reproduces the paper's published percentiles).
- **Geometry separation** under tilt and disorder (F_i beats geo where they differ).
- **Dephasing and detuning both pass** (clean/tilt/disorder) → F_i is a general *N-conserving
  transport-control* selector, not a dephasing trick.
- **Loss fails meaningfully**: occupation-driven, confirmed under symmetry breaking
  (Spearman(ΔN_total, Σ⟨n_i⟩)=+0.998 vs ΣF_i=+0.73 where selectors separate).
- Multi-N loss machinery **Stage-0 validated** (γ_loss=0 ≡ fixed-N to 2e-16).
- Everything committed and traceable to code + CSVs; reanalysis preserved; T1 artifact retired.

## Scope / NOT claimed
Exact Lindblad, small open 1D Bose–Hubbard (L≤8, nmax=3, half-filling); dephasing + detuning +
loss controls; clean/tilt/disorder. **Not** claimed: thermodynamic limit, larger L, universal
quantum-control law, all intervention types, injection, real lab experiments.

## ODD significance
The same physical system selects **different useful descriptions depending on the allowed
intervention class** — F_i for transport-control handles, ⟨n_i⟩ for loss handles. An empirical,
falsifiable instance of intervention-relative description privilege.

## Status: PAUSED (discovery phase complete)
Optional, deliberately **not** run (would risk turning a sharp map into a messy search):
Liouvillian-mode deepening, larger L. Paper architecture is the natural next move when desired.
