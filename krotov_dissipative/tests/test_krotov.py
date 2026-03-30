"""Tests for the Krotov optimisation algorithm."""

import sys
import numpy as np

sys.path.insert(0, "/home/user/diss_eng")
from krotov_dissipative.liouville import (
    pure_state_dm, fidelity_pure, dissipator_superop,
    build_liouvillian_from_amplitudes, vectorize, unvectorize,
)
from krotov_dissipative.krotov import run_krotov, analyse_steady_state, KrotovConfig
from krotov_dissipative.operators import random_pure_state

ATOL = 1e-6


def _check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    assert cond, msg


def test_krotov_known_cooling_2qubit():
    """Krotov with target-cooling operators should converge for 2 qubits.

    Use the known-optimal operators L_k = |psi*><psi_k^perp| and
    verify that Krotov finds high-fidelity controls.
    """
    d = 4  # 2 qubits
    psi_target = random_pure_state(d, seed=100)
    psi_target /= np.linalg.norm(psi_target)

    # Target-cooling operators
    from scipy.linalg import null_space
    orth = null_space(psi_target.conj().reshape(1, -1))
    lindblad_ops = [np.outer(psi_target, orth[:, k].conj()) for k in range(orth.shape[1])]

    # Initial state: computational basis |0>
    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0
    rho0 = pure_state_dm(psi0)

    config = KrotovConfig(
        n_qubits=2, T=5.0, N_t=30, max_iter=50,
        lambda_reg=0.5, tol=1e-8, u_init=0.5, verbose=False
    )
    result = run_krotov(rho0, psi_target, lindblad_ops, config)

    _check(result.fidelities[-1] > 0.99,
           f"Final fidelity = {result.fidelities[-1]:.6f} > 0.99")
    _check(result.fidelities[-1] > result.fidelities[0],
           "Fidelity improved over iterations")


def test_krotov_monotonic_improvement():
    """Krotov should show (approximate) monotonic improvement in fidelity.

    Due to discretisation, strict monotonicity may be violated slightly,
    but the overall trend must be improving.
    """
    d = 4
    psi_target = random_pure_state(d, seed=200)
    from scipy.linalg import null_space
    orth = null_space(psi_target.conj().reshape(1, -1))
    lindblad_ops = [np.outer(psi_target, orth[:, k].conj()) for k in range(orth.shape[1])]

    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0
    rho0 = pure_state_dm(psi0)

    config = KrotovConfig(
        n_qubits=2, T=5.0, N_t=30, max_iter=30,
        lambda_reg=1.0, tol=1e-10, u_init=0.1, verbose=False
    )
    result = run_krotov(rho0, psi_target, lindblad_ops, config)

    # Check that fidelity at end > fidelity at start
    _check(result.fidelities[-1] > result.fidelities[0] + 0.01,
           f"F improved: {result.fidelities[0]:.4f} -> {result.fidelities[-1]:.4f}")

    # Check approximate monotonicity: allow small violations
    n_violations = 0
    for i in range(1, len(result.fidelities)):
        if result.fidelities[i] < result.fidelities[i-1] - 1e-4:
            n_violations += 1
    frac = n_violations / (len(result.fidelities) - 1) if len(result.fidelities) > 1 else 0
    _check(frac < 0.2,
           f"Monotonicity violations: {n_violations}/{len(result.fidelities)-1} "
           f"({frac:.1%}) < 20%")


def test_krotov_with_random_operators():
    """Krotov with random Lindblad operators should still improve fidelity."""
    d = 4
    K = 5
    psi_target = random_pure_state(d, seed=300)
    rng = np.random.default_rng(301)
    lindblad_ops = [(rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
                    / np.sqrt(2 * d) for _ in range(K)]

    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0
    rho0 = pure_state_dm(psi0)

    config = KrotovConfig(
        n_qubits=2, T=5.0, N_t=30, max_iter=40,
        lambda_reg=0.5, tol=1e-10, u_init=0.1, verbose=False
    )
    result = run_krotov(rho0, psi_target, lindblad_ops, config)

    _check(result.fidelities[-1] > result.fidelities[0],
           f"Fidelity improved with random ops: "
           f"{result.fidelities[0]:.4f} -> {result.fidelities[-1]:.4f}")


def test_analyse_steady_state_known():
    """Analyse steady state for known target-cooling operators."""
    d = 4
    psi_target = random_pure_state(d, seed=400)
    from scipy.linalg import null_space
    orth = null_space(psi_target.conj().reshape(1, -1))
    lindblad_ops = [np.outer(psi_target, orth[:, k].conj()) for k in range(orth.shape[1])]

    # Use uniform rates (known to work for target cooling)
    N_t = 20
    K = len(lindblad_ops)
    controls = np.full((N_t, K), 1.0)

    analysis = analyse_steady_state(psi_target, lindblad_ops, controls)

    _check(analysis["steady_state_fidelity"] > 0.99,
           f"SS fidelity = {analysis['steady_state_fidelity']:.6f} > 0.99")
    _check(analysis["spectral_gap"] > 0.01,
           f"Spectral gap = {analysis['spectral_gap']:.6f} > 0.01")
    _check(analysis["is_unique_steady_state"],
           "Target is unique steady state")


ALL_TESTS = [
    test_krotov_known_cooling_2qubit,
    test_krotov_monotonic_improvement,
    test_krotov_with_random_operators,
    test_analyse_steady_state_known,
]

if __name__ == "__main__":
    print("=" * 60)
    print("  Tests: Krotov optimisation")
    print("=" * 60)
    passed = failed = 0
    for fn in ALL_TESTS:
        print(f"\n{fn.__name__}:")
        try:
            fn()
            passed += 1
        except (AssertionError, Exception) as exc:
            failed += 1
            import traceback; traceback.print_exc()
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
