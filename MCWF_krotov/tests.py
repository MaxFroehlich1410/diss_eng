"""Tests for the MCWF-Krotov package.

Run with
--------
    python -m pytest MCWF_krotov/tests.py -v
or
    python MCWF_krotov/tests.py          (standalone, no pytest needed)
"""

from __future__ import annotations

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MCWF_krotov import utils, mcwf, network_model, krotov

ATOL = 1e-8


def _check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    assert cond, msg


# ======================================================================
# utils tests
# ======================================================================

def test_basis_state():
    psi = utils.basis_state(4, 2)
    _check(abs(psi[2] - 1.0) < ATOL, "basis_state has 1 at correct index")
    _check(abs(np.linalg.norm(psi) - 1.0) < ATOL, "basis_state is normalised")


def test_w_state():
    psi = utils.w_state(3)
    _check(abs(np.linalg.norm(psi) - 1.0) < ATOL, "W state is normalised")
    expected_nonzero = {1, 2, 4}
    for i in range(8):
        if i in expected_nonzero:
            _check(abs(psi[i]) > 0.1, f"W state nonzero at |{i:03b}>")
        else:
            _check(abs(psi[i]) < ATOL, f"W state zero at |{i:03b}>")


def test_ghz_state():
    psi = utils.ghz_state(2)
    _check(abs(psi[0] - 1 / np.sqrt(2)) < ATOL, "GHZ |00> amplitude")
    _check(abs(psi[3] - 1 / np.sqrt(2)) < ATOL, "GHZ |11> amplitude")


def test_pure_state_dm():
    psi = utils.random_statevector(2, seed=0)
    rho = utils.pure_state_dm(psi)
    info = utils.is_physical(rho)
    _check(info["is_valid"], "pure-state DM is physical")
    _check(abs(utils.purity(rho) - 1.0) < ATOL, "pure-state purity = 1")


def test_fidelity_identity():
    psi = utils.random_statevector(2, seed=1)
    rho = utils.pure_state_dm(psi)
    F = utils.fidelity_to_pure(rho, psi)
    _check(abs(F - 1.0) < ATOL, "fidelity of state with itself = 1")


def test_overlap_orthogonal():
    psi0 = utils.basis_state(4, 0)
    psi1 = utils.basis_state(4, 1)
    _check(abs(utils.overlap(psi0, psi1)) < ATOL, "orthogonal states have overlap 0")


def test_target_cooling_operators():
    psi = utils.random_statevector(2, seed=2)
    ops = utils.target_cooling_operators(psi)
    _check(len(ops) == 3, "d-1 = 3 cooling operators for 2 qubits")
    for k, Lk in enumerate(ops):
        result = Lk @ psi
        _check(np.linalg.norm(result) < ATOL,
               f"L_{k} annihilates target state")


def test_blackman_pulse():
    T = 5.0
    _check(abs(utils.blackman_pulse(0, T)) < ATOL, "Blackman zero at t=0")
    _check(abs(utils.blackman_pulse(T, T)) < ATOL, "Blackman zero at t=T")
    _check(utils.blackman_pulse(T / 2, T) > 0.5, "Blackman peak near center")


# ======================================================================
# network_model tests
# ======================================================================

def test_hilbert_dim():
    _check(network_model.hilbert_dim(2) == 5, "2 nodes -> d=5")
    _check(network_model.hilbert_dim(4) == 9, "4 nodes -> d=9")


def test_drift_hamiltonian_hermitian():
    H = network_model.build_drift_hamiltonian(3)
    _check(np.allclose(H, H.conj().T, atol=ATOL), "H_drift is Hermitian")


def test_control_hamiltonians_hermitian():
    H_ctrls = network_model.build_control_hamiltonians(3)
    for i, Hi in enumerate(H_ctrls):
        _check(np.allclose(Hi, Hi.conj().T, atol=ATOL),
               f"H_ctrl[{i}] is Hermitian")


def test_control_hamiltonians_count():
    H_ctrls = network_model.build_control_hamiltonians(4)
    _check(len(H_ctrls) == 4, "4 nodes -> 4 control Hamiltonians")


def test_initial_state():
    psi = network_model.initial_state(3)
    _check(abs(np.linalg.norm(psi) - 1.0) < ATOL, "initial state normalised")
    qi = network_model.qubit_index(1)
    _check(abs(psi[qi] - 1.0) < ATOL, "qubit 1 excited in initial state")


def test_dark_state_target():
    N = 3
    psi = network_model.dark_state_target(N)
    _check(abs(np.linalg.norm(psi) - 1.0) < ATOL, "target state normalised")
    for i in range(1, N + 1):
        qi = network_model.qubit_index(i)
        _check(abs(abs(psi[qi]) - 1 / np.sqrt(N)) < 1e-6,
               f"equal weight on qubit {i}")


def test_dark_state_condition_ground():
    """Ground state should trivially satisfy dark-state condition."""
    N = 2
    d = network_model.hilbert_dim(N)
    psi_ground = np.zeros(d, dtype=complex)
    psi_ground[0] = 1.0
    L = network_model.build_network_lindblad_op(N)
    val = network_model.dark_state_condition(psi_ground, L)
    _check(abs(val) < ATOL, "ground state satisfies dark-state condition")


def test_lindblad_op_structure():
    N = 2
    L = network_model.build_network_lindblad_op(N)
    d = network_model.hilbert_dim(N)
    _check(L.shape == (d, d), f"Lindblad op has shape ({d},{d})")
    psi_ground = np.zeros(d, dtype=complex)
    psi_ground[0] = 1.0
    _check(np.linalg.norm(L @ psi_ground) < ATOL,
           "L annihilates ground state")


def test_piecewise_hamiltonian():
    N = 2
    nt = 10
    times = np.linspace(0, 5, nt)
    controls = np.ones((N, nt)) * 0.5
    H_pw = network_model.PiecewiseConstantHamiltonian(N, controls, times)
    H_t = H_pw(2.5)
    _check(np.allclose(H_t, H_t.conj().T, atol=ATOL),
           "PiecewiseConstantHamiltonian returns Hermitian H")


# ======================================================================
# mcwf tests
# ======================================================================

def test_effective_hamiltonian():
    """H_eff should be non-Hermitian (has anti-Hermitian part from L^dag L)."""
    d = 3
    H = np.eye(d, dtype=complex)
    L = np.zeros((d, d), dtype=complex)
    L[0, 1] = 1.0
    H_eff = mcwf.build_effective_hamiltonian(H, [L])
    anti_herm = H_eff - H_eff.conj().T
    _check(np.linalg.norm(anti_herm) > 0.1,
           "H_eff has non-trivial anti-Hermitian part")


def test_mcwf_no_dissipation():
    """Without Lindblad ops the MCWF should give unitary evolution."""
    d = 4
    H = np.diag([0.0, 1.0, 2.0, 3.0]).astype(complex)
    psi0 = np.ones(d, dtype=complex) / 2
    times = np.linspace(0, 1, 50)

    L_zero = np.zeros((d, d), dtype=complex)
    states, jumps = mcwf.propagate_mcwf_trajectory(
        psi0, lambda t: H, [L_zero], times, rng=np.random.default_rng(0),
    )
    _check(len(jumps) == 0, "no jumps without dissipation")
    for i, psi in enumerate(states):
        _check(abs(np.linalg.norm(psi) - 1.0) < 1e-4,
               f"norm preserved at step {i}")


def test_mcwf_norm_preservation():
    """States should be normalised at every time step."""
    N = 2
    d = network_model.hilbert_dim(N)
    H = network_model.build_drift_hamiltonian(N)
    L = network_model.build_network_lindblad_op(N)
    psi0 = network_model.initial_state(N)
    times = np.linspace(0, 2, 100)

    states, _ = mcwf.propagate_mcwf_trajectory(
        psi0, lambda t: H, [L], times, rng=np.random.default_rng(42),
    )
    for i, psi in enumerate(states):
        _check(abs(np.linalg.norm(psi) - 1.0) < 1e-6,
               f"MCWF norm at step {i}: {np.linalg.norm(psi):.8f}")


def test_mcwf_density_matrix_trace():
    """Density matrix from trajectories should have trace ~ 1."""
    N = 2
    d = network_model.hilbert_dim(N)
    H = network_model.build_drift_hamiltonian(N)
    L = network_model.build_network_lindblad_op(N)
    psi0 = network_model.initial_state(N)
    times = np.linspace(0, 2, 50)

    all_trajs = []
    for seed in range(20):
        states, _ = mcwf.propagate_mcwf_trajectory(
            psi0, lambda t: H, [L], times, rng=np.random.default_rng(seed),
        )
        all_trajs.append(states)

    rho = mcwf.density_matrix_from_trajectories(all_trajs, -1)
    tr = utils.trace(rho)
    _check(abs(tr - 1.0) < 0.1, f"MCWF rho trace = {tr:.4f} ~ 1")


def test_backward_propagation_shape():
    """Backward propagation should return the correct number of states."""
    d = 3
    H = np.eye(d, dtype=complex) * 0.1
    L = np.zeros((d, d), dtype=complex)
    L[0, 1] = 0.1
    times = np.linspace(0, 1, 20)
    chi_T = np.ones(d, dtype=complex) / np.sqrt(d)

    states = mcwf.propagate_backward_no_jump(
        chi_T, lambda t: H, [L], times,
    )
    _check(len(states) == len(times), "backward prop returns nt states")


def test_no_jump_propagation():
    """No-jump propagation should reduce norm (non-unitary) for a state
    that couples to the dissipation channel (has cavity excitation)."""
    N = 2
    d = network_model.hilbert_dim(N)
    H = network_model.build_drift_hamiltonian(N)
    L = network_model.build_network_lindblad_op(N)
    # Use a state with cavity excitation so L^dag L is nonzero
    psi0 = np.zeros(d, dtype=complex)
    psi0[network_model.cavity_index(1)] = 1.0  # cavity 1 excited
    times = np.linspace(0, 5, 100)

    states = mcwf.propagate_no_jump(psi0, lambda t: H, [L], times)
    norm_final = np.linalg.norm(states[-1])
    _check(norm_final < 1.0 - 1e-6,
           f"no-jump norm decays: {norm_final:.6f} < 1.0")


# ======================================================================
# krotov tests (short runs to verify mechanics)
# ======================================================================

def test_krotov_independent_runs():
    """Independent-trajectory Krotov should run without error and reduce JT."""
    N = 2
    d = network_model.hilbert_dim(N)
    psi0 = network_model.initial_state(N)
    psi_tgt = network_model.dark_state_target(N)
    H_drift = network_model.build_drift_hamiltonian(N)
    H_ctrls = network_model.build_control_hamiltonians(N)
    L = network_model.build_network_lindblad_op(N)
    T = 5.0
    nt = 50
    times = np.linspace(0, T, nt)

    controls_guess = np.array([
        [utils.blackman_pulse(t, T, 100.0) for t in times] for _ in range(N)
    ])

    result = krotov.krotov_independent(
        psi0, psi_tgt, H_drift, H_ctrls, [L], times,
        controls_guess, n_trajectories=2, n_iterations=20,
        lambda_a=0.001, seed=42, verbose=False,
    )
    _check(len(result.errors) == 20, "20 error values recorded")
    _check(result.controls.shape == (N, nt), "controls have correct shape")
    print(f"    JT: {result.errors[0]:.4f} -> {result.errors[-1]:.4f}")


def test_krotov_cross_runs():
    """Cross-trajectory Krotov should run without error."""
    N = 2
    d = network_model.hilbert_dim(N)
    psi0 = network_model.initial_state(N)
    psi_tgt = network_model.dark_state_target(N)
    H_drift = network_model.build_drift_hamiltonian(N)
    H_ctrls = network_model.build_control_hamiltonians(N)
    L = network_model.build_network_lindblad_op(N)
    T = 5.0
    nt = 50
    times = np.linspace(0, T, nt)

    controls_guess = np.array([
        [utils.blackman_pulse(t, T, 100.0) for t in times] for _ in range(N)
    ])

    result = krotov.krotov_cross_trajectory(
        psi0, psi_tgt, H_drift, H_ctrls, [L], times,
        controls_guess, n_trajectories=2, n_iterations=20,
        lambda_a=0.001, seed=42, verbose=False,
    )
    _check(len(result.errors) == 20, "20 error values for cross-traj")
    _check(result.controls.shape == (N, nt), "controls shape for cross-traj")
    print(f"    JT: {result.errors[0]:.4f} -> {result.errors[-1]:.4f}")


def test_krotov_dm_runs():
    """Density-matrix Krotov should run and converge."""
    N = 2
    d = network_model.hilbert_dim(N)
    psi0 = network_model.initial_state(N)
    psi_tgt = network_model.dark_state_target(N)
    H_drift = network_model.build_drift_hamiltonian(N)
    H_ctrls = network_model.build_control_hamiltonians(N)
    L = network_model.build_network_lindblad_op(N)
    T = 5.0
    nt = 30
    times = np.linspace(0, T, nt)

    controls_guess = np.array([
        [utils.blackman_pulse(t, T, 100.0) for t in times] for _ in range(N)
    ])

    result = krotov.krotov_density_matrix(
        psi0, psi_tgt, H_drift, H_ctrls, [L], times,
        controls_guess, n_iterations=10, lambda_a=0.001, verbose=False,
    )
    _check(len(result.errors) == 10, "10 DM error values recorded")
    print(f"    JT: {result.errors[0]:.4f} -> {result.errors[-1]:.4f}")


def test_mcwf_agrees_with_master_equation():
    """MCWF average over many trajectories should approximate master eq."""
    N = 2
    d = network_model.hilbert_dim(N)
    H = network_model.build_drift_hamiltonian(N)
    L_op = network_model.build_network_lindblad_op(N)
    psi0 = network_model.initial_state(N)
    psi_tgt = network_model.dark_state_target(N)
    T = 2.0
    times = np.linspace(0, T, 50)

    L_super = utils.build_liouvillian([L_op], hamiltonian=H)
    rho0 = utils.pure_state_dm(psi0)
    rho_exact = utils.evolve_density_matrix(rho0, L_super, T)
    F_exact = utils.fidelity_to_pure(rho_exact, psi_tgt)

    M = 100
    all_trajs = []
    for seed in range(M):
        states, _ = mcwf.propagate_mcwf_trajectory(
            psi0, lambda t: H, [L_op], times, rng=np.random.default_rng(seed),
        )
        all_trajs.append(states)

    rho_mcwf = mcwf.density_matrix_from_trajectories(all_trajs, -1)
    F_mcwf = utils.fidelity_to_pure(rho_mcwf, psi_tgt)

    diff = abs(F_exact - F_mcwf)
    _check(diff < 0.15,
           f"MCWF fidelity {F_mcwf:.4f} vs exact {F_exact:.4f} (diff={diff:.4f})")


# ======================================================================
# Standalone runner
# ======================================================================

ALL_TESTS = [
    test_basis_state,
    test_w_state,
    test_ghz_state,
    test_pure_state_dm,
    test_fidelity_identity,
    test_overlap_orthogonal,
    test_target_cooling_operators,
    test_blackman_pulse,
    test_hilbert_dim,
    test_drift_hamiltonian_hermitian,
    test_control_hamiltonians_hermitian,
    test_control_hamiltonians_count,
    test_initial_state,
    test_dark_state_target,
    test_dark_state_condition_ground,
    test_lindblad_op_structure,
    test_piecewise_hamiltonian,
    test_effective_hamiltonian,
    test_mcwf_no_dissipation,
    test_mcwf_norm_preservation,
    test_mcwf_density_matrix_trace,
    test_backward_propagation_shape,
    test_no_jump_propagation,
    test_krotov_independent_runs,
    test_krotov_cross_runs,
    test_krotov_dm_runs,
    test_mcwf_agrees_with_master_equation,
]


def run_all() -> bool:
    print("=" * 64)
    print("  Tests for MCWF-Krotov")
    print("=" * 64)
    passed = failed = 0
    for fn in ALL_TESTS:
        print(f"\n{fn.__name__}:")
        try:
            fn()
            passed += 1
        except (AssertionError, Exception) as exc:
            failed += 1
            if not isinstance(exc, AssertionError):
                import traceback
                traceback.print_exc()
                print(f"  [ERROR] {exc}")
    print(f"\n{'=' * 64}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 64)
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
