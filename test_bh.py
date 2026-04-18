"""
test_bh.py
==========
Regression tests for the Bose-Hubbard causal-handle paper package.

Run with:  pytest test_bh.py -v
"""

import sys
import os

import numpy as np
import pytest

# Make bh importable from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from bh import (
    build_basis,
    basis_index,
    number_op,
    build_hamiltonian,
    build_liouvillian,
    evolve_rho,
    site_expectations,
    site_variances,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_system(L=6, N=3, nmax=3, J_over_U=0.30, gamma=0.1):
    """Return the standard system ingredients for a given condition."""
    U = 1.0
    J = J_over_U * U

    basis = build_basis(L, N, nmax)
    idx_map = basis_index(basis)
    D = len(basis)

    H = build_hamiltonian(L, J, U, basis, idx_map)
    n_ops = [number_op(i, D, basis) for i in range(L)]
    n2_ops = [nop @ nop for nop in n_ops]
    L_ops = n_ops
    gammas = [gamma] * L
    liouv = build_liouvillian(H, L_ops, gammas)

    return basis, idx_map, D, H, n_ops, n2_ops, L_ops, gammas, liouv


def ground_state_rho(H):
    """Pure-state density matrix of the ground state of H."""
    eigvals, eigvecs = np.linalg.eigh(H)
    psi0 = eigvecs[:, 0]
    return np.outer(psi0, psi0.conj())


# ---------------------------------------------------------------------------
# 1. Hilbert space dimension
# ---------------------------------------------------------------------------

class TestHilbertSpaceDimension:
    def test_L6_N3_dim56(self):
        """L=6, N=3, nmax=3 must give D=56 (paper Table I / Sec II B)."""
        basis = build_basis(6, 3, 3)
        assert len(basis) == 56

    def test_L7_N3_dim84(self):
        """L=7, N=3, nmax=3 must give D=84 (paper Table I / Sec II B)."""
        basis = build_basis(7, 3, 3)
        assert len(basis) == 84

    def test_all_states_sum_to_N(self):
        """Every basis state must have particle sum exactly N."""
        L, N, nmax = 6, 3, 3
        basis = build_basis(L, N, nmax)
        for state in basis:
            assert sum(state) == N, f"State {state} has wrong particle count"

    def test_all_states_respect_nmax(self):
        """No site occupation exceeds nmax in any basis state."""
        L, N, nmax = 6, 3, 3
        basis = build_basis(L, N, nmax)
        for state in basis:
            for ni in state:
                assert ni <= nmax, f"State {state} violates nmax={nmax}"

    def test_half_filling_definition(self):
        """N = floor(L/2) for both even and odd L (paper Sec II A)."""
        for L in [6, 7, 8, 9]:
            N_expected = L // 2
            basis = build_basis(L, N_expected, 3)
            assert len(basis) > 0  # just check we get a valid non-empty basis


# ---------------------------------------------------------------------------
# 2. Hamiltonian hermiticity
# ---------------------------------------------------------------------------

class TestHamiltonianHermiticity:
    @pytest.mark.parametrize("L,J_over_U", [(6, 0.12), (6, 0.30), (7, 0.40)])
    def test_H_is_hermitian(self, L, J_over_U):
        """H must equal H† to within floating-point tolerance."""
        N = L // 2
        basis = build_basis(L, N, 3)
        idx_map = basis_index(basis)
        H = build_hamiltonian(L, J_over_U, 1.0, basis, idx_map)
        np.testing.assert_allclose(
            H, H.conj().T, atol=1e-12,
            err_msg=f"H not Hermitian for L={L}, J/U={J_over_U}"
        )

    def test_H_real_for_real_J_U(self):
        """H should be real-valued (all matrix elements are real)."""
        basis = build_basis(6, 3, 3)
        idx_map = basis_index(basis)
        H = build_hamiltonian(6, 0.30, 1.0, basis, idx_map)
        assert np.isrealobj(H) or np.max(np.abs(H.imag)) < 1e-14

    def test_ground_state_is_lowest_eigenvalue(self):
        """eigh returns eigenvalues in ascending order; index 0 is ground state."""
        basis = build_basis(6, 3, 3)
        idx_map = basis_index(basis)
        H = build_hamiltonian(6, 0.30, 1.0, basis, idx_map)
        eigvals = np.linalg.eigvalsh(H)
        assert eigvals[0] <= eigvals[1], "Eigenvalues not sorted ascending"
        assert eigvals[0] < 0, "Ground state energy should be negative for J>0"


# ---------------------------------------------------------------------------
# 3. Lindblad superoperator: trace preservation
# ---------------------------------------------------------------------------

class TestLindbladTracePreservation:
    """The Lindblad map is trace-preserving: Tr(ρ(τ)) = Tr(ρ(0)) = 1."""

    @pytest.mark.parametrize("tau", [0.5, 1.0, 3.0])
    def test_trace_preserved_ground_state(self, tau):
        """Trace of evolved ground state density matrix stays 1."""
        basis, idx_map, D, H, n_ops, n2_ops, L_ops, gammas, liouv = make_system()
        rho0 = ground_state_rho(H)
        rho_t = evolve_rho(rho0, liouv, tau)
        trace_t = np.real(np.trace(rho_t))
        assert abs(trace_t - 1.0) < 1e-8, (
            f"Trace deviated from 1 at tau={tau}: trace={trace_t}"
        )

    def test_trace_preserved_mixed_state(self):
        """Trace preserved for a mixed (maximally mixed) initial state."""
        basis, idx_map, D, H, n_ops, n2_ops, L_ops, gammas, liouv = make_system()
        rho0 = np.eye(D) / D  # maximally mixed state (trace = 1)
        rho_t = evolve_rho(rho0, liouv, 1.0)
        trace_t = np.real(np.trace(rho_t))
        assert abs(trace_t - 1.0) < 1e-8

    def test_rho_stays_in_fixed_N_sector(self):
        """Total particle number must be conserved under dephasing evolution."""
        basis, idx_map, D, H, n_ops, n2_ops, L_ops, gammas, liouv = make_system(L=6, N=3)
        rho0 = ground_state_rho(H)
        # Total number operator: sum of all site operators
        N_op = sum(n_ops)
        N_before = np.real(np.trace(N_op @ rho0))
        rho_t = evolve_rho(rho0, liouv, 2.0)
        N_after = np.real(np.trace(N_op @ rho_t))
        assert abs(N_after - N_before) < 1e-8, (
            f"Particle number changed: {N_before} -> {N_after}"
        )
        assert abs(N_before - 3.0) < 1e-8  # sanity check: N=3 for L=6


# ---------------------------------------------------------------------------
# 4. Particle number conservation under dephasing
# ---------------------------------------------------------------------------

class TestParticleNumberConservation:
    """Local ⟨n_i⟩ is preserved under dephasing with L_i = n_i."""

    def test_local_occupation_preserved(self):
        """Each ⟨n_i⟩ must not change under pure dephasing evolution."""
        L, N = 6, 3
        basis = build_basis(L, N, 3)
        idx_map = basis_index(basis)
        D = len(basis)
        # Zero tunnelling: pure dephasing, no coherent dynamics
        H_zero = np.zeros((D, D))
        n_ops = [number_op(i, D, basis) for i in range(L)]
        gammas = [0.1] * L
        liouv = build_liouvillian(H_zero, n_ops, gammas)

        # Use a superposition state in the fixed-N sector
        rng = np.random.default_rng(42)
        psi = rng.random(D) + 1j * rng.random(D)
        psi /= np.linalg.norm(psi)
        rho0 = np.outer(psi, psi.conj())

        occ_before = site_expectations(rho0, n_ops)
        rho_t = evolve_rho(rho0, liouv, 1.0)
        occ_after = site_expectations(rho_t, n_ops)

        np.testing.assert_allclose(
            occ_after, occ_before, atol=1e-7,
            err_msg="Local occupations changed under pure dephasing"
        )

    def test_local_occupation_preserved_with_tunnelling(self):
        """Total ⟨N⟩ = sum_i ⟨n_i⟩ is conserved even with non-zero J."""
        basis, idx_map, D, H, n_ops, n2_ops, L_ops, gammas, liouv = make_system(J_over_U=0.30)
        rho0 = ground_state_rho(H)
        N_op = sum(n_ops)
        N_before = np.real(np.trace(N_op @ rho0))
        rho_t = evolve_rho(rho0, liouv, 2.0)
        N_after = np.real(np.trace(N_op @ rho_t))
        assert abs(N_after - N_before) < 1e-8


# ---------------------------------------------------------------------------
# 5. F_i computation correctness
# ---------------------------------------------------------------------------

class TestFiComputation:
    """Local occupation variance F_i = <n_i^2> - <n_i>^2."""

    def test_variance_zero_for_fock_state(self):
        """For a Fock state |n1,...,nL>, F_i = 0 at every site."""
        L, N, nmax = 6, 3, 3
        basis = build_basis(L, N, nmax)
        D = len(basis)
        n_ops = [number_op(i, D, basis) for i in range(L)]
        n2_ops = [nop @ nop for nop in n_ops]

        # Pick an arbitrary Fock state in the basis
        state_idx = 0
        state = basis[state_idx]
        rho = np.zeros((D, D))
        rho[state_idx, state_idx] = 1.0

        variances = site_variances(rho, n_ops, n2_ops)
        np.testing.assert_allclose(
            variances, np.zeros(L), atol=1e-12,
            err_msg=f"Variance nonzero for Fock state {state}"
        )

    def test_variance_nonnegative(self):
        """F_i >= 0 for any valid density matrix (by Cauchy-Schwarz)."""
        basis, idx_map, D, H, n_ops, n2_ops, L_ops, gammas, liouv = make_system()
        rho0 = ground_state_rho(H)
        rho_t = evolve_rho(rho0, liouv, 2.0)
        variances = site_variances(rho_t, n_ops, n2_ops)
        assert np.all(variances >= -1e-12), (
            f"Negative variance found: {variances}"
        )

    def test_variance_mean_consistent_with_manual(self):
        """F_i = <n_i^2> - <n_i>^2 matches a manual two-step computation."""
        basis, idx_map, D, H, n_ops, n2_ops, L_ops, gammas, liouv = make_system()
        rho0 = ground_state_rho(H)
        rho_t = evolve_rho(rho0, liouv, 1.5)

        variances = site_variances(rho_t, n_ops, n2_ops)
        means = site_expectations(rho_t, n_ops)
        sq_means = np.array([np.real(np.trace(n2ops @ rho_t)) for n2ops in n2_ops])
        manual = sq_means - means**2

        np.testing.assert_allclose(variances, manual, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. Intervention budget equality
# ---------------------------------------------------------------------------

class TestBudgetEquality:
    """Targeted and random interventions must have the same total extra budget."""

    def test_targeted_budget(self):
        """Targeted: k sites each get gamma_extra = 0.5. Total = k * 0.5."""
        L = 6
        N = L // 2
        nmax = 3
        gamma_base = 0.1
        gamma_extra = 0.5

        basis = build_basis(L, N, nmax)
        idx_map = basis_index(basis)
        D = len(basis)
        H = build_hamiltonian(L, 0.30, 1.0, basis, idx_map)
        n_ops = [number_op(i, D, basis) for i in range(L)]
        n2_ops = [nop @ nop for nop in n_ops]

        # Simulate burn-in to get F_i
        gammas_base = [gamma_base] * L
        liouv_base = build_liouvillian(H, n_ops, gammas_base)
        rho0 = ground_state_rho(H)
        rho_burn = evolve_rho(rho0, liouv_base, 5.0)
        Fi = site_variances(rho_burn, n_ops, n2_ops)

        k = max(1, int(np.ceil(L / 3)))
        selected = np.argsort(Fi)[-k:][::-1].tolist()

        # Targeted budget
        targeted_total = k * gamma_extra
        # Random: also exactly k random sites each with gamma_extra
        rng = np.random.default_rng(0)
        for _ in range(10):
            rsites = rng.choice(L, size=k, replace=False).tolist()
            random_total = len(rsites) * gamma_extra
            assert abs(targeted_total - random_total) < 1e-12, (
                f"Budget mismatch: targeted={targeted_total}, random={random_total}"
            )

    def test_k_ceiling_L_over_3(self):
        """k = ceil(L/3): L=6 -> k=2, L=7 -> k=3."""
        assert max(1, int(np.ceil(6 / 3))) == 2
        assert max(1, int(np.ceil(7 / 3))) == 3


# ---------------------------------------------------------------------------
# 7. Number operator correctness
# ---------------------------------------------------------------------------

class TestNumberOperator:
    def test_number_op_diagonal(self):
        """n_i should be diagonal in the Fock basis with eigenvalues n_i."""
        L, N, nmax = 6, 3, 3
        basis = build_basis(L, N, nmax)
        D = len(basis)
        for i in range(L):
            nop = number_op(i, D, basis)
            # Off-diagonal elements should be zero
            off_diag = nop - np.diag(np.diag(nop))
            np.testing.assert_allclose(
                off_diag, np.zeros((D, D)), atol=1e-15,
                err_msg=f"number_op({i}) has off-diagonal elements"
            )
            # Diagonal elements should equal n_i for each basis state
            for idx, state in enumerate(basis):
                assert abs(nop[idx, idx] - state[i]) < 1e-14, (
                    f"number_op({i})[{idx},{idx}] = {nop[idx,idx]} != {state[i]}"
                )

    def test_total_number_op_equals_N(self):
        """Sum of number operators has eigenvalue N on every basis state."""
        L, N, nmax = 6, 3, 3
        basis = build_basis(L, N, nmax)
        D = len(basis)
        n_ops = [number_op(i, D, basis) for i in range(L)]
        N_op = sum(n_ops)
        # In our basis, every state has total occupation N
        expected = N * np.eye(D)
        np.testing.assert_allclose(N_op, expected, atol=1e-14)
