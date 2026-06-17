#!/usr/bin/env python3
"""
Stage 0 — validation gates for the multi-N amplitude-damping / local-loss machinery.
NO science pilot. Builds the multi-N space and checks gates A-E. L=6.

Multi-N space: all Fock states with total N in {0,1,2,3} (loss only lowers N from N0=3),
each site <= nmax=3 -> EXACT (no truncation). For L=6 this is D = 56+21+6+1 = 84.
"""
import itertools, time
import numpy as np

import bh
import bh_hardening as bhh
from bh import (basis_index, number_op, build_hamiltonian, build_liouvillian,
                evolve_rho, site_expectations)

bh._SPARSE_D_THRESHOLD = 60          # D=84 -> sparse path (fast, exact)
import scipy.sparse as sp

L, NMAX, N0, U, GAMMA = 6, 3, 3, 1.0, 0.1
JU = 0.30
ok_all = True
def gate(name, cond, detail=""):
    global ok_all; ok_all = ok_all and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))


def make_multi_basis(L, Ntot, nmax):
    return sorted(s for s in itertools.product(range(nmax + 1), repeat=L) if sum(s) <= Ntot)

def annihilation_op(site, basis, idx):
    D = len(basis); A = np.zeros((D, D))
    for s, st in enumerate(basis):
        ni = st[site]
        if ni > 0:
            t = list(st); t[site] = ni - 1
            j = idx.get(tuple(t))
            if j is not None:
                A[j, s] += np.sqrt(ni)
    return A

def occ(rho, n_ops):
    return np.array([np.real(np.trace(n @ rho)) for n in n_ops])

def Ntot_exp(rho, Nop):
    return float(np.real(np.trace(Nop @ rho)))


print("=" * 84)
print("STAGE 0 — multi-N local-loss validation (L=6)")
print("=" * 84)

# ---- build multi-N basis + operators ----
mb = make_multi_basis(L, N0, NMAX); midx = basis_index(mb); D = len(mb)
n_ops = [number_op(i, D, mb) for i in range(L)]
a_ops = [annihilation_op(i, mb, midx) for i in range(L)]
J = JU * U
H = build_hamiltonian(L, J, U, NMAX, mb, midx)
Nop = sum(n_ops)

print("\n-- Gate A: basis / operator sanity --")
gate("multi-N dimension", D == 84, f"D={D} (expect 56+21+6+1=84)")
# a_i lowers N by 1
sectorN = np.array([sum(s) for s in mb])
a_lowers = True
for i in range(L):
    rows, cols = np.nonzero(a_ops[i])
    a_lowers &= all(sectorN[r] == sectorN[c] - 1 for r, c in zip(rows, cols))
gate("a_i maps N -> N-1", a_lowers)
gate("n_i diagonal & nonneg", all(np.allclose(n, np.diag(np.diag(n))) and np.diag(n).min() >= 0 for n in n_ops))
# H conserves N (block-diagonal in N)
hr, hc = np.nonzero(np.abs(H) > 1e-12)
gate("H conserves N (block-diagonal)", all(sectorN[r] == sectorN[c] for r, c in zip(hr, hc)))
# a_i^dag a_i == n_i
gate("a_i^dag a_i == n_i", all(np.allclose(a_ops[i].conj().T @ a_ops[i], n_ops[i]) for i in range(L)))

print("\n-- Gate B: Liouvillian sanity --")
liouv_deph = build_liouvillian(H, n_ops, [GAMMA] * L)                      # dephasing only
gloss = 0.05
liouv_loss = build_liouvillian(H, n_ops + a_ops, [GAMMA] * L + [gloss] * L)  # + uniform loss
# test state: maximally mixed within N=3 block embedded? use a fixed-N burn-in (built later);
# here use a simple normalized state in N=3
cond = bhh.build_condition(L, JU)                  # fixed-N (N=3) machinery, D=56
fb = cond["basis"]; emap = [midx[s] for s in fb]   # embed fixed-N -> multi-N
def embed(rho_fixed):
    R = np.zeros((D, D), dtype=complex)
    R[np.ix_(emap, emap)] = rho_fixed
    return R
rho0 = embed(cond["rho_burn"])
r_t = evolve_rho(rho0, liouv_loss.tocsr() if sp.issparse(liouv_loss) else liouv_loss, 1.0)
gate("trace preserved under loss", abs(np.trace(r_t) - 1.0) < 1e-8, f"|Tr-1|={abs(np.trace(r_t)-1):.1e}")
gate("Hermitian under loss", np.allclose(r_t, r_t.conj().T, atol=1e-9))
gate("positive (lambda_min >= -1e-9)", np.linalg.eigvalsh((r_t + r_t.conj().T) / 2).min() > -1e-9,
     f"lmin={np.linalg.eigvalsh((r_t+r_t.conj().T)/2).min():.1e}")

print("\n-- Gate C: fixed-N equivalence (gamma_loss = 0 must reproduce old fixed-N evolution) --")
liouv_base_fixed = cond["liouv_base"]
maxerr = 0.0
for tau in (1, 2, 3):
    occ_fixed = site_expectations(evolve_rho(cond["rho_burn"], liouv_base_fixed, tau), cond["n_ops"])
    occ_multi = occ(evolve_rho(rho0, liouv_deph, tau), n_ops)   # dephasing only, no loss
    maxerr = max(maxerr, float(np.max(np.abs(occ_fixed - occ_multi))))
gate("multi-N (no loss) == fixed-N occupations", maxerr < 1e-8, f"max|Δ⟨n_i⟩|={maxerr:.1e}")
# and N stays at 3 with no loss
n_noloss = Ntot_exp(evolve_rho(rho0, liouv_deph, 3.0), Nop)
gate("N conserved when gamma_loss=0", abs(n_noloss - N0) < 1e-8, f"⟨N⟩={n_noloss:.8f}")

print("\n-- Gate D: loss response sanity --")
# uniform loss: <N> monotonically decreasing
Ns = [Ntot_exp(evolve_rho(rho0, liouv_loss, t), Nop) for t in (0.0, 1.0, 2.0, 3.0)]
gate("uniform loss: <N> monotonically decreases", all(x > y for x, y in zip(Ns, Ns[1:])),
     "⟨N⟩(t)=" + ", ".join(f"{x:.4f}" for x in Ns))
# selected-site loss gives finite total loss
Fi = cond["Fi"]; k = bhh.k_sites(L)
S_fi = sorted(int(s) for s in np.argsort(Fi)[-k:])
def dN_total(sites, gl, tau=3.0):
    liouv = build_liouvillian(H, n_ops + [a_ops[i] for i in sites], [GAMMA] * L + [gl] * len(sites))
    return N0 - Ntot_exp(evolve_rho(rho0, liouv, tau), Nop)
dN_fi = dN_total(S_fi, 0.05)
gate("selected-site loss -> finite ΔN_total", dN_fi > 1e-6, f"ΔN_total(top-F_i)={dN_fi:.5f}")
# linear in small gamma_loss
small = {gl: dN_total(S_fi, gl) for gl in (0.01, 0.02, 0.04)}
ratios = [small[0.02] / small[0.01], small[0.04] / small[0.02]]
gate("linear response at small gamma_loss (ratios ~2)", all(1.6 < r < 2.4 for r in ratios),
     f"ratios={ratios[0]:.2f}, {ratios[1]:.2f}")
# selector-variation preview (trivial-drain check): does ΔN_total depend on WHICH sites?
center = (L - 1) / 2.0
S_geo = sorted(range(L), key=lambda i: abs(i - center))[:k]
S_edge = [0, L - 1]
dN_geo, dN_edge = dN_total(S_geo, 0.05), dN_total(S_edge, 0.05)
spread = max(dN_fi, dN_geo, dN_edge) - min(dN_fi, dN_geo, dN_edge)
print(f"     [preview] ΔN_total: top-F_i={dN_fi:.5f}  geo={dN_geo:.5f}  edge={dN_edge:.5f}  spread={spread:.5f}")
gate("ΔN_total VARIES with target set (not a trivial drain)", spread > 1e-4,
     "if spread~0 the science pilot would be uninformative")

print("\n-- Gate E: runtime / cost --")
t0 = time.time(); _ = build_liouvillian(H, n_ops + a_ops, [GAMMA] * L + [0.05] * L); t_build = time.time() - t0
t0 = time.time(); _ = evolve_rho(rho0, liouv_loss, 3.0); t_ev = time.time() - t0
print(f"     D={D}  Liouvillian {'sparse' if sp.issparse(liouv_loss) else 'dense'}  "
      f"build={t_build:.2f}s  one evolve(tau=3)={t_ev:.2f}s")
print(f"     est. clean pilot (4 J/U x 3 tau x ~24 evolves) ~ {4*3*24*t_ev/60:.1f} min")

print("\n" + "=" * 84)
print(f"STAGE 0 OVERALL: {'ALL GATES PASS' if ok_all else 'SOME GATES FAILED'}")
print("=" * 84)
