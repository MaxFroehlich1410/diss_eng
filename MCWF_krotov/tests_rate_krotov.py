"""Tests for the extended Krotov optimiser with Lindblad rate controls.

Tests cover:
  - DissipationBasis construction and interface
  - MCWF step with explicit rates
  - Gates-only mode (should match original behaviour)
  - Rates-only mode
  - Joint gates + rates mode
  - DM evaluation with per-layer rates
  - Non-negativity enforcement on rates

Run with
--------
    python -m pytest MCWF_krotov/tests_rate_krotov.py -v
or
    python MCWF_krotov/tests_rate_krotov.py
"""

from __future__ import annotations

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MCWF_krotov import rate_krotov as rk
from MCWF_krotov import gate_circuit_krotov as gck
from MCWF_krotov import utils

ATOL = 1e-8


def _check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    assert cond, msg


# ======================================================================
# DissipationBasis tests
# ======================================================================

def test_build_basis_amp_damp_only():
    db = rk.build_dissipation_basis(
        2, 3, include_amp_damp=True, include_dephasing=False,
        include_zz=False, init_rate=0.2,
    )
    _check(db.n_ops == 2, f"2 AD ops for 2 qubits, got {db.n_ops}")
    _check(db.rates.shape == (3, 2), f"rates shape (3,2), got {db.rates.shape}")
    _check(np.allclose(db.rates, 0.2), "all rates initialised to 0.2")


def test_build_basis_full():
    db = rk.build_dissipation_basis(
        3, 2, include_amp_damp=True, include_dephasing=True,
        include_zz=True, init_rate=0.1,
    )
    n_ad = 3
    n_deph = 3
    n_zz = 3  # C(3,2) = 3
    _check(db.n_ops == n_ad + n_deph + n_zz,
           f"expected {n_ad+n_deph+n_zz} ops, got {db.n_ops}")


def test_effective_ops_scaling():
    """effective_ops should scale bare operators by sqrt(gamma)."""
    db = rk.build_dissipation_basis(
        2, 1, include_amp_damp=True, include_dephasing=False,
        include_zz=False, init_rate=0.25,
    )
    eff = db.effective_ops(0)
    for i, (L_eff, L_bare) in enumerate(zip(eff, db.bare_ops)):
        expected = np.sqrt(0.25) * L_bare
        _check(np.allclose(L_eff, expected, atol=ATOL),
               f"effective op {i} scaled correctly")


def test_rates_shape_matches_layers():
    db = rk.build_dissipation_basis(2, 5, init_rate=0.0)
    _check(db.n_layers == 5, "n_layers matches")


# ======================================================================
# MCWF step with rates
# ======================================================================

def test_mcwf_step_rates_preserves_norm():
    n = 2
    psi = utils.random_statevector(n, seed=10)
    bare_ops = utils.amplitude_damping_operators(n)
    LdL_bare = [L.conj().T @ L for L in bare_ops]
    rates = np.array([0.5, 0.3])
    rng = np.random.default_rng(42)
    psi_out, _, _ = rk._mcwf_step_rates(
        psi, bare_ops, LdL_bare, rates, 0.1, rng)
    _check(abs(np.linalg.norm(psi_out) - 1.0) < 1e-10,
           f"norm preserved: {np.linalg.norm(psi_out):.12f}")


def test_mcwf_step_zero_rates_identity():
    """Zero rates should leave the state unchanged."""
    n = 2
    psi = utils.random_statevector(n, seed=11)
    bare_ops = utils.amplitude_damping_operators(n)
    LdL_bare = [L.conj().T @ L for L in bare_ops]
    rates = np.array([0.0, 0.0])
    rng = np.random.default_rng(42)
    psi_out, jumped, _ = rk._mcwf_step_rates(
        psi, bare_ops, LdL_bare, rates, 0.1, rng)
    _check(not jumped, "no jump with zero rates")
    _check(np.allclose(psi, psi_out, atol=1e-10),
           "state unchanged with zero rates")


# ======================================================================
# Gates-only mode (backward compatibility)
# ======================================================================

def test_gates_only_reduces_error():
    """Gates-only mode should reduce error like the original optimiser."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)

    layers = gck.build_hardware_efficient_ansatz(n, 3)
    db = rk.build_dissipation_basis(
        n, len(layers),
        include_amp_damp=True, include_dephasing=False,
        include_zz=False, init_rate=0.3,
    )

    result = rk.krotov_joint(
        psi0, psi_tgt, layers, db,
        dissipation_dt=0.05,
        n_trajectories=4, n_iterations=30,
        lambda_gates=0.5, lambda_rates=1.0,
        optimize_gates=True, optimize_rates=False,
        seed=42, verbose=False,
    )
    _check(result.fidelities[-1] > result.fidelities[0],
           f"gates-only: F improved {result.fidelities[0]:.4f} -> "
           f"{result.fidelities[-1]:.4f}")
    _check(np.allclose(result.final_rates, db.rates),
           "rates unchanged in gates-only mode")


# ======================================================================
# Rates-only mode
# ======================================================================

def test_rates_only_modifies_rates():
    """Rates-only mode should change the rates while leaving gates fixed."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = utils.w_state(n)

    layers = gck.build_hardware_efficient_ansatz(n, 2)
    initial_thetas = [[g.theta for g in layer] for layer in layers]

    db = rk.build_dissipation_basis(
        n, len(layers),
        include_amp_damp=True, include_dephasing=True,
        include_zz=False, init_rate=0.1,
    )
    rates_before = db.rates.copy()

    result = rk.krotov_joint(
        psi0, psi_tgt, layers, db,
        dissipation_dt=0.1,
        n_trajectories=4, n_iterations=20,
        lambda_gates=1.0, lambda_rates=0.5,
        optimize_gates=False, optimize_rates=True,
        seed=42, verbose=False,
    )

    final_thetas = [[g.theta for g in layer] for layer in layers]
    for li in range(len(layers)):
        for gi in range(len(layers[li])):
            _check(abs(final_thetas[li][gi] - initial_thetas[li][gi]) < 1e-15,
                   f"gate L{li}G{gi} unchanged in rates-only mode")

    rates_changed = not np.allclose(result.final_rates, rates_before)
    _check(rates_changed, "rates were modified in rates-only mode")


def test_rates_nonnegative():
    """Rates must remain >= 0 after optimisation."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = utils.w_state(n)

    layers = gck.build_hardware_efficient_ansatz(n, 2)
    db = rk.build_dissipation_basis(
        n, len(layers),
        include_amp_damp=True, include_dephasing=True,
        include_zz=False, init_rate=0.01,
    )

    result = rk.krotov_joint(
        psi0, psi_tgt, layers, db,
        dissipation_dt=0.1,
        n_trajectories=4, n_iterations=30,
        lambda_gates=1.0, lambda_rates=0.3,
        optimize_gates=False, optimize_rates=True,
        seed=42, verbose=False,
    )
    _check(np.all(result.final_rates >= 0),
           f"all rates >= 0 (min = {result.final_rates.min():.6f})")


# ======================================================================
# Joint mode
# ======================================================================

def test_joint_mode_runs():
    """Joint mode should run without error and return valid result."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = utils.w_state(n)

    layers = gck.build_hardware_efficient_ansatz(n, 2)
    db = rk.build_dissipation_basis(
        n, len(layers),
        include_amp_damp=True, include_dephasing=True,
        include_zz=False, init_rate=0.1,
    )

    result = rk.krotov_joint(
        psi0, psi_tgt, layers, db,
        dissipation_dt=0.05,
        n_trajectories=4, n_iterations=10,
        lambda_gates=0.5, lambda_rates=0.5,
        optimize_gates=True, optimize_rates=True,
        seed=42, verbose=False,
    )
    _check(len(result.errors) == 10, "10 error values")
    _check(len(result.rate_history) == 11,
           "rate_history has n_iterations+1 entries")
    _check(result.final_rates.shape == db.rates.shape,
           "final_rates has correct shape")
    _check(np.all(result.final_rates >= 0), "rates non-negative")


def test_joint_reduces_error():
    """Joint mode should reduce error."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)

    layers = gck.build_hardware_efficient_ansatz(n, 3)
    db = rk.build_dissipation_basis(
        n, len(layers),
        include_amp_damp=True, include_dephasing=True,
        include_zz=False, init_rate=0.1,
    )

    result = rk.krotov_joint(
        psi0, psi_tgt, layers, db,
        dissipation_dt=0.05,
        n_trajectories=4, n_iterations=40,
        lambda_gates=0.3, lambda_rates=0.5,
        optimize_gates=True, optimize_rates=True,
        seed=42, verbose=False,
    )
    _check(result.fidelities[-1] > result.fidelities[0],
           f"joint: F improved {result.fidelities[0]:.4f} -> "
           f"{result.fidelities[-1]:.4f}")


def test_joint_with_zz_and_clipping():
    """Joint mode with ZZ operators and update clipping."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = utils.ghz_state(n)

    layers = gck.build_hardware_efficient_ansatz(n, 2)
    db = rk.build_dissipation_basis(
        n, len(layers),
        include_amp_damp=True, include_dephasing=True,
        include_zz=True, init_rate=0.05,
    )

    result = rk.krotov_joint(
        psi0, psi_tgt, layers, db,
        dissipation_dt=0.05,
        n_trajectories=4, n_iterations=15,
        lambda_gates=0.3, lambda_rates=0.5,
        optimize_gates=True, optimize_rates=True,
        max_gate_step=0.1, max_rate_step=0.05,
        seed=42, verbose=False,
    )
    _check(len(result.errors) == 15, "15 iterations with ZZ + clipping")
    _check(np.all(result.final_rates >= 0), "rates non-negative with ZZ")


# ======================================================================
# DM evaluation with rates
# ======================================================================

def test_dm_rates_physical():
    """DM evaluation should produce a physical density matrix."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = utils.w_state(n)

    layers = gck.build_hardware_efficient_ansatz(n, 2)
    db = rk.build_dissipation_basis(
        n, len(layers),
        include_amp_damp=True, include_dephasing=True,
        include_zz=False, init_rate=0.1,
    )

    result = rk.evaluate_circuit_dm_rates(
        psi0, psi_tgt, layers, db, dissipation_dt=0.1,
    )
    _check(abs(result["trace"] - 1.0) < 0.05,
           f"trace = {result['trace']:.4f}")
    _check(0.0 <= result["fidelity"] <= 1.0,
           f"fidelity = {result['fidelity']:.4f}")
    _check(result["purity"] <= 1.0 + 1e-6,
           f"purity = {result['purity']:.4f}")


def test_dm_zero_rates_pure():
    """With zero rates, the circuit should remain pure unitary."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = utils.w_state(n)

    layers = gck.build_hardware_efficient_ansatz(n, 2)
    db = rk.build_dissipation_basis(
        n, len(layers),
        include_amp_damp=True, include_dephasing=True,
        include_zz=False, init_rate=0.0,
    )

    result = rk.evaluate_circuit_dm_rates(
        psi0, psi_tgt, layers, db, dissipation_dt=0.1,
    )
    _check(abs(result["purity"] - 1.0) < 1e-6,
           f"purity = {result['purity']:.8f} ~ 1 (zero rates)")


def test_dm_after_optimisation():
    """DM evaluation after optimisation should be consistent with MCWF."""
    n = 2
    d = 2 ** n
    psi0 = utils.basis_state(d, 0)
    psi_tgt = utils.w_state(n)

    layers = gck.build_hardware_efficient_ansatz(n, 2)
    db = rk.build_dissipation_basis(
        n, len(layers),
        include_amp_damp=True, include_dephasing=False,
        include_zz=False, init_rate=0.2,
    )

    result = rk.krotov_joint(
        psi0, psi_tgt, layers, db,
        dissipation_dt=0.1,
        n_trajectories=4, n_iterations=20,
        lambda_gates=0.5, lambda_rates=0.5,
        optimize_gates=True, optimize_rates=True,
        seed=42, verbose=False,
    )

    dm_result = rk.evaluate_circuit_dm_rates(
        psi0, psi_tgt, layers, db, dissipation_dt=0.1,
    )
    _check(abs(dm_result["fidelity"] - result.fidelities[-1]) < 0.3,
           f"DM fidelity {dm_result['fidelity']:.4f} roughly consistent "
           f"with MCWF {result.fidelities[-1]:.4f}")


# ======================================================================
# Standalone runner
# ======================================================================

ALL_TESTS = [
    test_build_basis_amp_damp_only,
    test_build_basis_full,
    test_effective_ops_scaling,
    test_rates_shape_matches_layers,
    test_mcwf_step_rates_preserves_norm,
    test_mcwf_step_zero_rates_identity,
    test_gates_only_reduces_error,
    test_rates_only_modifies_rates,
    test_rates_nonnegative,
    test_joint_mode_runs,
    test_joint_reduces_error,
    test_joint_with_zz_and_clipping,
    test_dm_rates_physical,
    test_dm_zero_rates_pure,
    test_dm_after_optimisation,
]


def run_all() -> bool:
    print("=" * 64)
    print("  Tests for extended Krotov (rate optimisation)")
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
    print(f"  Results: {passed} passed, {failed} failed out of {passed+failed}")
    print("=" * 64)
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
