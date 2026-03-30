"""Tests for forward and backward propagation."""

import sys
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, "/home/user/diss_eng")
from krotov_dissipative.liouville import (
    vectorize, unvectorize, dissipator_superop, adjoint_superop,
    build_liouvillian_from_amplitudes, apply_dissipator,
    pure_state_dm, trace_dm, is_physical, fidelity_pure,
)
from krotov_dissipative.propagation import (
    propagate_step, forward_propagation, backward_propagation,
    forward_propagation_sequential,
)

ATOL = 1e-8


def _check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    assert cond, msg


def _make_test_system(d=4, K=2, seed=10):
    """Create a small test system with random Lindblad operators."""
    rng = np.random.default_rng(seed)
    lindblad_ops = [(rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)))
                    / np.sqrt(2 * d) for _ in range(K)]
    S_ops = [dissipator_superop(L) for L in lindblad_ops]
    psi0 = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi0 /= np.linalg.norm(psi0)
    rho0 = pure_state_dm(psi0)
    return lindblad_ops, S_ops, rho0, psi0


def test_single_step_trace_preservation():
    """A single propagation step preserves trace."""
    lindblad_ops, S_ops, rho0, _ = _make_test_system()
    d = rho0.shape[0]
    u = np.array([0.5, 0.3])
    L = build_liouvillian_from_amplitudes(S_ops, u)
    dt = 0.1

    rho_vec = vectorize(rho0)
    rho_vec_new = propagate_step(rho_vec, L, dt)
    rho_new = unvectorize(rho_vec_new, d)
    tr = trace_dm(rho_new)
    _check(abs(tr - 1.0) < ATOL, f"Tr after step = {tr:.10f} ~ 1")


def test_single_step_physicality():
    """Evolved state remains a valid density matrix."""
    lindblad_ops, S_ops, rho0, _ = _make_test_system()
    d = rho0.shape[0]
    u = np.array([0.5, 0.3])
    L = build_liouvillian_from_amplitudes(S_ops, u)

    rho_vec = vectorize(rho0)
    rho_vec_new = propagate_step(rho_vec, L, 0.5)
    rho_new = unvectorize(rho_vec_new, d)
    _check(is_physical(rho_new), "rho after step is physical")


def test_forward_propagation_trace():
    """Full forward propagation preserves trace at all steps."""
    lindblad_ops, S_ops, rho0, _ = _make_test_system()
    N_t = 20
    K = len(lindblad_ops)
    controls = np.full((N_t, K), 0.3)
    dt = 0.1

    states = forward_propagation(rho0, S_ops, controls, dt)
    _check(len(states) == N_t + 1, f"Got {len(states)} states == {N_t + 1}")

    for i, rho in enumerate(states):
        tr = trace_dm(rho)
        _check(abs(tr - 1.0) < ATOL, f"Tr(rho[{i}]) = {tr:.10f} ~ 1")


def test_forward_propagation_matches_manual():
    """Forward propagation matches manual step-by-step expm."""
    lindblad_ops, S_ops, rho0, _ = _make_test_system()
    d = rho0.shape[0]
    N_t = 5
    K = len(lindblad_ops)
    rng = np.random.default_rng(20)
    controls = rng.uniform(0, 1, (N_t, K))
    dt = 0.1

    states = forward_propagation(rho0, S_ops, controls, dt)

    # Manual propagation
    rho_vec = vectorize(rho0)
    for j in range(N_t):
        L = build_liouvillian_from_amplitudes(S_ops, controls[j])
        rho_vec = expm(L * dt) @ rho_vec

    rho_manual = unvectorize(rho_vec, d)
    _check(
        np.allclose(states[-1], rho_manual, atol=ATOL),
        "Forward propagation matches manual expm"
    )


def test_backward_propagation_adjoint():
    """Backward propagation uses adjoint Liouvillian correctly."""
    lindblad_ops, S_ops, rho0, _ = _make_test_system()
    d = rho0.shape[0]
    N_t = 5
    K = len(lindblad_ops)
    controls = np.full((N_t, K), 0.3)
    dt = 0.1

    rng = np.random.default_rng(30)
    psi_t = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi_t /= np.linalg.norm(psi_t)
    chi_T = pure_state_dm(psi_t)

    costates = backward_propagation(chi_T, S_ops, controls, dt)
    _check(len(costates) == N_t + 1, f"Got {len(costates)} costates == {N_t + 1}")
    _check(np.allclose(costates[N_t], chi_T, atol=ATOL),
           "chi(T) = rho_target")

    # Verify: chi(t_j) = exp(L^dag dt) chi(t_{j+1})
    for j in range(N_t - 1, -1, -1):
        L = build_liouvillian_from_amplitudes(S_ops, controls[j])
        L_adj = adjoint_superop(L)
        chi_j_manual = unvectorize(
            expm(L_adj * dt) @ vectorize(costates[j + 1]), d
        )
        _check(
            np.allclose(costates[j], chi_j_manual, atol=ATOL),
            f"chi[{j}] matches manual adjoint propagation"
        )


def test_zero_controls_no_evolution():
    """With zero controls, the state should not change."""
    lindblad_ops, S_ops, rho0, _ = _make_test_system()
    N_t = 10
    K = len(lindblad_ops)
    controls = np.zeros((N_t, K))
    dt = 0.1

    states = forward_propagation(rho0, S_ops, controls, dt)
    _check(
        np.allclose(states[-1], rho0, atol=ATOL),
        "Zero controls -> no evolution"
    )


def test_sequential_forward_produces_valid_states():
    """Sequential forward propagation with Krotov update produces physical states."""
    lindblad_ops, S_ops, rho0, psi0 = _make_test_system()
    d = rho0.shape[0]
    K = len(lindblad_ops)
    N_t = 10
    dt = 0.1

    # Target state
    rng = np.random.default_rng(40)
    psi_t = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi_t /= np.linalg.norm(psi_t)
    rho_target = pure_state_dm(psi_t)

    # Old controls and costates
    controls_ref = np.full((N_t, K), 0.3)
    costates = backward_propagation(rho_target, S_ops, controls_ref, dt)

    lambda_reg = np.ones(K) * 10.0  # large lambda for stability

    states_new, controls_new = forward_propagation_sequential(
        rho0, S_ops, costates, controls_ref, lambda_reg, dt, lindblad_ops
    )

    _check(len(states_new) == N_t + 1, f"Got {len(states_new)} states")
    _check(controls_new.shape == (N_t, K), f"Controls shape = {controls_new.shape}")

    for i, rho in enumerate(states_new):
        _check(is_physical(rho, atol=1e-6),
               f"rho[{i}] is physical after sequential forward")

    # Controls should be non-negative
    _check(np.all(controls_new >= -1e-12),
           "All controls are non-negative")


ALL_TESTS = [
    test_single_step_trace_preservation,
    test_single_step_physicality,
    test_forward_propagation_trace,
    test_forward_propagation_matches_manual,
    test_backward_propagation_adjoint,
    test_zero_controls_no_evolution,
    test_sequential_forward_produces_valid_states,
]

if __name__ == "__main__":
    print("=" * 60)
    print("  Tests: Propagation")
    print("=" * 60)
    passed = failed = 0
    for fn in ALL_TESTS:
        print(f"\n{fn.__name__}:")
        try:
            fn()
            passed += 1
        except (AssertionError, Exception) as exc:
            failed += 1
            if not isinstance(exc, AssertionError):
                import traceback; traceback.print_exc()
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
