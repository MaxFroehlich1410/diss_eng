"""Tests for the gate-circuit Krotov method.

Run with
--------
    python -m pytest MCWF_krotov/tests_gate_circuit.py -v
or
    python MCWF_krotov/tests_gate_circuit.py
"""

from __future__ import annotations

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MCWF_krotov import gate_circuit_krotov as gck
from MCWF_krotov import utils

ATOL = 1e-8


def _check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    assert cond, msg


# ======================================================================
# Generator tests
# ======================================================================

def test_ry_generator_hermitian():
    G = gck.ry_generator(0, 2)
    _check(np.allclose(G, G.conj().T, atol=ATOL), "Ry generator is Hermitian")


def test_rz_generator_hermitian():
    G = gck.rz_generator(1, 2)
    _check(np.allclose(G, G.conj().T, atol=ATOL), "Rz generator is Hermitian")


def test_rx_generator_hermitian():
    G = gck.rx_generator(0, 2)
    _check(np.allclose(G, G.conj().T, atol=ATOL), "Rx generator is Hermitian")


def test_ry_gate_is_unitary():
    """exp(-i theta G) should be unitary."""
    from scipy.linalg import expm
    G = gck.ry_generator(0, 2)
    theta = 0.7
    U = expm(-1j * theta * G)
    _check(np.allclose(U @ U.conj().T, np.eye(4), atol=ATOL), "Ry gate is unitary")


def test_rz_rotation():
    """Rz(pi) on |+> should give |->."""
    from scipy.linalg import expm
    G = gck.rz_generator(0, 1)
    U = expm(-1j * np.pi * G)
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    result = U @ plus
    minus = np.array([1, -1], dtype=complex) / np.sqrt(2)
    _check(abs(abs(np.vdot(result, minus)) - 1.0) < 1e-6,
           "Rz(pi)|+> = |-> (up to phase)")


def test_cnot_generator_hermitian():
    G = gck.cnot_zx_generator(0, 1, 2)
    _check(np.allclose(G, G.conj().T, atol=ATOL), "CNOT generator is Hermitian")


# ======================================================================
# Gate application tests
# ======================================================================

def test_apply_gate_preserves_norm():
    d = 4
    psi = utils.random_statevector(2, seed=0)
    G = gck.ry_generator(0, 2)
    gate = gck.GateLayer(generator=G, theta=1.3, name="Ry_0")
    psi_out = gck.apply_gate(psi, gate)
    _check(abs(np.linalg.norm(psi_out) - 1.0) < ATOL,
           "gate preserves norm")


def test_apply_gate_identity():
    """theta=0 should give identity."""
    psi = utils.random_statevector(2, seed=1)
    G = gck.ry_generator(0, 2)
    gate = gck.GateLayer(generator=G, theta=0.0)
    psi_out = gck.apply_gate(psi, gate)
    _check(np.allclose(psi, psi_out, atol=ATOL),
           "theta=0 is identity")


def test_apply_gate_layer():
    psi = utils.random_statevector(2, seed=2)
    gates = [
        gck.GateLayer(gck.ry_generator(0, 2), 0.5),
        gck.GateLayer(gck.rz_generator(1, 2), 0.3),
    ]
    psi_out = gck.apply_gate_layer(psi, gates)
    _check(abs(np.linalg.norm(psi_out) - 1.0) < ATOL,
           "gate layer preserves norm")


# ======================================================================
# Hardware-efficient ansatz tests
# ======================================================================

def test_ansatz_structure():
    layers = gck.build_hardware_efficient_ansatz(3, 2)
    _check(len(layers) == 2, "2 layers created")
    n_gates_per_layer = 3 + 3 + 2  # Ry*3, Rz*3, CNOT*2
    _check(len(layers[0]) == n_gates_per_layer,
           f"{n_gates_per_layer} gates per layer")


def test_ansatz_generators_hermitian():
    layers = gck.build_hardware_efficient_ansatz(2, 1)
    for i, gate in enumerate(layers[0]):
        G = gate.generator
        _check(np.allclose(G, G.conj().T, atol=ATOL),
               f"ansatz gate {i} ({gate.name}) generator is Hermitian")


# ======================================================================
# MCWF dissipation step tests
# ======================================================================

def test_mcwf_step_preserves_norm():
    n = 2
    d = 2 ** n
    psi = utils.random_statevector(n, seed=3)
    ops = utils.amplitude_damping_operators(n)
    rng = np.random.default_rng(42)
    psi_out, _, _ = gck.mcwf_dissipation_step(psi, ops, 0.1, rng)
    _check(abs(np.linalg.norm(psi_out) - 1.0) < 1e-6,
           "MCWF step preserves norm")


def test_mcwf_step_no_ops():
    """With zero Lindblad ops (all zero matrices), state should barely change."""
    d = 4
    psi = utils.random_statevector(2, seed=4)
    L_zero = [np.zeros((d, d), dtype=complex)]
    rng = np.random.default_rng(0)
    psi_out, jumped, _ = gck.mcwf_dissipation_step(psi, L_zero, 0.1, rng)
    _check(not jumped, "no jump with zero Lindblad ops")
    _check(np.allclose(psi, psi_out, atol=1e-6),
           "state unchanged with zero dissipation")


# ======================================================================
# Krotov gate-circuit optimiser tests
# ======================================================================

def test_krotov_gate_circuit_runs():
    """Basic smoke test: should run without error."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = utils.w_state(n)
    layers = gck.build_hardware_efficient_ansatz(n, 2)
    ops = utils.target_cooling_operators(psi_tgt)
    rates = [0.5] * len(ops)
    lindblad_ops = [np.sqrt(r) * L for r, L in zip(rates, ops)]

    result = gck.krotov_gate_circuit(
        psi0, psi_tgt, layers, lindblad_ops,
        dissipation_dt=0.1, n_trajectories=2, n_iterations=5,
        lambda_a=1.0, seed=42, verbose=False,
    )
    _check(len(result.errors) == 5, "5 error values")
    _check(len(result.gate_params) == 2, "2 layers of params")
    print(f"    JT: {result.errors[0]:.4f} -> {result.errors[-1]:.4f}")


def test_krotov_gate_reduces_error():
    """With enough iterations, error should decrease."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)  # Bell-like
    layers = gck.build_hardware_efficient_ansatz(n, 3)
    ops = utils.target_cooling_operators(psi_tgt)
    lindblad_ops = [0.3 * L for L in ops]

    result = gck.krotov_gate_circuit(
        psi0, psi_tgt, layers, lindblad_ops,
        dissipation_dt=0.05, n_trajectories=2, n_iterations=30,
        lambda_a=0.5, seed=42, verbose=False,
    )
    _check(result.errors[-1] < result.errors[0],
           f"error decreased: {result.errors[0]:.4f} -> {result.errors[-1]:.4f}")


def test_krotov_gate_with_amp_damping():
    """Should work with hardware-realistic amplitude damping."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = utils.w_state(n)
    layers = gck.build_hardware_efficient_ansatz(n, 2)
    ops = utils.amplitude_damping_operators(n)
    lindblad_ops = [0.1 * L for L in ops]

    result = gck.krotov_gate_circuit(
        psi0, psi_tgt, layers, lindblad_ops,
        dissipation_dt=0.05, n_trajectories=2, n_iterations=10,
        lambda_a=1.0, seed=42, verbose=False,
    )
    _check(len(result.errors) == 10, "10 iterations with amp damping")
    print(f"    JT (amp damp): {result.errors[0]:.4f} -> {result.errors[-1]:.4f}")


def test_evaluate_circuit_dm():
    """DM evaluation should give physical density matrix."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = utils.w_state(n)
    layers = gck.build_hardware_efficient_ansatz(n, 1)
    ops = utils.target_cooling_operators(psi_tgt)
    lindblad_ops = [0.3 * L for L in ops]

    result = gck.evaluate_circuit_dm(
        psi0, psi_tgt, layers, lindblad_ops, dissipation_dt=0.1,
    )
    _check(abs(result["trace"] - 1.0) < 0.05, f"trace = {result['trace']:.4f}")
    _check(result["fidelity"] >= 0.0, f"fidelity = {result['fidelity']:.4f}")
    _check(result["fidelity"] <= 1.0, f"fidelity <= 1")


def test_no_dissipation_pure_unitary():
    """Without dissipation, circuit should be pure unitary."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = utils.w_state(n)
    layers = gck.build_hardware_efficient_ansatz(n, 2)

    L_zero = [np.zeros((d, d), dtype=complex)]

    result = gck.evaluate_circuit_dm(
        psi0, psi_tgt, layers, L_zero, dissipation_dt=0.0,
    )
    _check(abs(result["purity"] - 1.0) < 0.01,
           f"purity = {result['purity']:.6f} ~ 1 (no dissipation)")


# ======================================================================
# Standalone runner
# ======================================================================

ALL_TESTS = [
    test_ry_generator_hermitian,
    test_rz_generator_hermitian,
    test_rx_generator_hermitian,
    test_ry_gate_is_unitary,
    test_rz_rotation,
    test_cnot_generator_hermitian,
    test_apply_gate_preserves_norm,
    test_apply_gate_identity,
    test_apply_gate_layer,
    test_ansatz_structure,
    test_ansatz_generators_hermitian,
    test_mcwf_step_preserves_norm,
    test_mcwf_step_no_ops,
    test_krotov_gate_circuit_runs,
    test_krotov_gate_reduces_error,
    test_krotov_gate_with_amp_damping,
    test_evaluate_circuit_dm,
    test_no_dissipation_pure_unitary,
]


def run_all() -> bool:
    print("=" * 64)
    print("  Tests for gate-circuit Krotov")
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
