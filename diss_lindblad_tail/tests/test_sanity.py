"""Sanity checks for the diss_lindblad package.

Run with
--------
    python -m pytest tests/test_sanity.py -v
or
    python tests/test_sanity.py          (standalone, no pytest needed)
"""

from __future__ import annotations

import sys
import numpy as np

# Allow running from the package root directory without installing.
sys.path.insert(0, ".")

from diss_lindblad import density_matrix as dm
from diss_lindblad import lindblad
from diss_lindblad import circuits
from diss_lindblad.experiment import run, ExperimentConfig

ATOL = 1e-8


# ======================================================================
# Helper for standalone execution
# ======================================================================

def _check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    assert cond, msg


# ======================================================================
# Tests
# ======================================================================

def test_pure_state_dm():
    """A pure-state DM must have trace 1, be Hermitian, and purity 1."""
    psi = circuits.random_statevector(2, seed=0)
    rho = dm.pure_state_dm(psi)
    info = dm.is_physical(rho)
    _check(info["is_valid"], "pure-state DM is a valid density matrix")
    _check(abs(dm.purity(rho) - 1.0) < ATOL, "pure-state purity = 1")


def test_unitary_preserves_dm():
    """Unitary evolution preserves trace, hermiticity, and purity."""
    psi = circuits.random_statevector(2, seed=1)
    rho = dm.pure_state_dm(psi)
    U = circuits.extend_to_unitary(circuits.random_statevector(2, seed=2))
    rho2 = dm.apply_unitary(rho, U)
    info = dm.is_physical(rho2)
    _check(info["is_valid"], "rho after unitary is still physical")
    _check(abs(dm.purity(rho2) - 1.0) < ATOL, "purity preserved under unitary")


def test_no_dissipation_preserves_state():
    """With gamma=0 and no Hamiltonian the Liouvillian is zero -> rho unchanged."""
    n = 2
    psi = circuits.random_statevector(n, seed=3)
    rho = dm.pure_state_dm(psi)
    ops = lindblad.target_cooling_operators(psi)
    rates = np.zeros(len(ops))
    L = lindblad.build_liouvillian(ops, rates)
    rho_t = lindblad.evolve(rho, L, t=5.0)
    _check(np.allclose(rho, rho_t, atol=ATOL), "rho unchanged when gamma = 0")


def test_trace_preservation():
    """Lindblad evolution must preserve Tr(rho) = 1 at all times."""
    n = 2
    psi_target = circuits.random_statevector(n, seed=4)
    psi_init = circuits.random_statevector(n, seed=5)
    rho = dm.pure_state_dm(psi_init)
    ops = lindblad.target_cooling_operators(psi_target)
    L = lindblad.build_liouvillian(ops, rates=np.ones(len(ops)))
    for t in [0.0, 0.1, 1.0, 5.0, 20.0]:
        rho_t = lindblad.evolve(rho, L, t)
        tr = float(np.real(dm.trace(rho_t)))
        _check(abs(tr - 1.0) < ATOL, f"Tr(rho(t={t})) = {tr:.12f} ~ 1")


def test_physicality_during_evolution():
    """rho(t) must remain a valid density matrix at all times."""
    n = 2
    psi = circuits.random_statevector(n, seed=6)
    rho = dm.pure_state_dm(circuits.random_statevector(n, seed=7))
    ops = lindblad.target_cooling_operators(psi)
    L = lindblad.build_liouvillian(ops)
    for t in np.linspace(0, 10, 20):
        rho_t = lindblad.evolve(rho, L, t)
        info = dm.is_physical(rho_t)
        _check(
            info["is_valid"],
            f"rho(t={t:.1f}) is physical  (min_eig={info['min_eigenvalue']:.2e})",
        )


def test_cooling_reaches_target():
    """Cooling dissipation must drive rho -> |psi*><psi*| as t -> inf."""
    n = 2
    psi_target = circuits.random_statevector(n, seed=8)
    psi_init = circuits.random_statevector(n, seed=9)
    rho = dm.pure_state_dm(psi_init)
    ops = lindblad.target_cooling_operators(psi_target)
    L = lindblad.build_liouvillian(ops, rates=np.ones(len(ops)) * 2.0)
    rho_inf = lindblad.evolve(rho, L, t=50.0)
    fid = dm.fidelity_to_pure(rho_inf, psi_target)
    _check(abs(fid - 1.0) < 1e-6, f"F(rho(t->inf), psi*) = {fid:.8f} ~ 1")


def test_fidelity_increases_with_cooling():
    """After an approximate circuit, cooling must increase fidelity."""
    cfg = ExperimentConfig(
        n_qubits=3,
        target_name="random",
        seed=42,
        gate_fraction=0.5,
        dissipation_type="cooling",
        gamma=1.0,
        t_max=5.0,
        n_time_steps=20,
    )
    result = run(cfg)
    _check(
        result.fidelity_final > result.fidelity_initial + 1e-4,
        f"Fidelity increased: {result.fidelity_initial:.6f} -> {result.fidelity_final:.6f}",
    )


def test_circuit_roundtrip():
    """The full (untruncated) circuit must reproduce the target state."""
    n = 2
    psi = circuits.random_statevector(n, seed=10)
    exact_circ = circuits.build_exact_circuit(psi)
    U = circuits.circuit_to_unitary(exact_circ)
    d = 2 ** n
    e0 = np.zeros(d, dtype=complex)
    e0[0] = 1.0
    psi_out = U @ e0
    fid = abs(np.vdot(psi, psi_out)) ** 2
    _check(abs(fid - 1.0) < 1e-6, f"Full circuit fidelity = {fid:.8f} ~ 1")


def test_amplitude_damping_trace_preservation():
    """Amplitude damping must also preserve trace."""
    n = 2
    rho = dm.pure_state_dm(circuits.random_statevector(n, seed=11))
    ops = lindblad.amplitude_damping_operators(n)
    L = lindblad.build_liouvillian(ops, rates=np.ones(len(ops)) * 0.5)
    for t in [0.0, 1.0, 10.0]:
        rho_t = lindblad.evolve(rho, L, t)
        tr = float(np.real(dm.trace(rho_t)))
        _check(abs(tr - 1.0) < ATOL, f"Amp-damp Tr(rho(t={t})) = {tr:.12f} ~ 1")


def test_dephasing_trace_preservation():
    """Dephasing must preserve trace."""
    n = 2
    rho = dm.pure_state_dm(circuits.random_statevector(n, seed=12))
    ops = lindblad.dephasing_operators(n)
    L = lindblad.build_liouvillian(ops, rates=np.ones(len(ops)) * 0.3)
    for t in [0.0, 1.0, 10.0]:
        rho_t = lindblad.evolve(rho, L, t)
        tr = float(np.real(dm.trace(rho_t)))
        _check(abs(tr - 1.0) < ATOL, f"Dephasing Tr(rho(t={t})) = {tr:.12f} ~ 1")


def test_incremental_vs_direct_evolution():
    """evolve_trajectory (incremental) must agree with independent evolve calls."""
    n = 2
    psi = circuits.random_statevector(n, seed=13)
    rho = dm.pure_state_dm(circuits.random_statevector(n, seed=14))
    ops = lindblad.target_cooling_operators(psi)
    L = lindblad.build_liouvillian(ops)
    times = np.linspace(0, 3.0, 11)

    snapshots = lindblad.evolve_trajectory(rho, L, times)
    for i, t in enumerate(times):
        rho_direct = lindblad.evolve(rho, L, t)
        _check(
            np.allclose(snapshots[i], rho_direct, atol=1e-6),
            f"Incremental == direct at t={t:.2f}",
        )


# ======================================================================
# Standalone runner
# ======================================================================

ALL_TESTS = [
    test_pure_state_dm,
    test_unitary_preserves_dm,
    test_no_dissipation_preserves_state,
    test_trace_preservation,
    test_physicality_during_evolution,
    test_cooling_reaches_target,
    test_fidelity_increases_with_cooling,
    test_circuit_roundtrip,
    test_amplitude_damping_trace_preservation,
    test_dephasing_trace_preservation,
    test_incremental_vs_direct_evolution,
]


def run_all() -> bool:
    """Execute every check.  Returns True when all pass."""
    print("=" * 64)
    print("  Sanity checks for diss_lindblad")
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
                print(f"  [ERROR] {exc}")
    print(f"\n{'=' * 64}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 64)
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
