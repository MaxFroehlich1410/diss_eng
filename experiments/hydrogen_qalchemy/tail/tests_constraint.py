"""Correctness / acceptance tests for the constrained dissipation tail.

Run with
--------
    python -m pytest tail/tests_constraint.py -v
or
    python tail/tests_constraint.py          (standalone)

Tests cover n=2 and n=3 qubits and finish in ~30 s total.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np

from tail.metrics import trace_dm
from tail.constraint_dissipation import (
    ConstraintConfig,
    ConstraintDissipation,
    _kraus_amplitude_damping,
    _kraus_dephasing,
    _kraus_depolarizing,
    _apply_1q_kraus,
)
from tail.fit_constraint_to_target import (
    fit_to_teacher,
    generate_training_states,
)

ATOL = 1e-9


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _random_state(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return psi / np.linalg.norm(psi)


def _random_dm(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def _check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    assert cond, msg


# ------------------------------------------------------------------
# Test 1: 1-qubit channels are CPTP
# ------------------------------------------------------------------

def test_one_qubit_channels_cptp():
    """After many steps of each 1q channel, trace ~ 1 and eigenvalues >= 0."""
    n = 2
    d = 4

    for ch_name, kraus_fn in [
        ("amp_damp", _kraus_amplitude_damping),
        ("dephasing", _kraus_dephasing),
        ("depolarizing", _kraus_depolarizing),
    ]:
        for qubit in range(n):
            gamma = 1.5
            dt = 0.1
            kraus = kraus_fn(gamma, dt)

            # Start with a random density matrix.
            rho = _random_dm(d, seed=100 + qubit)

            for _ in range(50):
                rho = _apply_1q_kraus(rho, kraus, qubit, n)

            tr = trace_dm(rho)
            _check(
                abs(tr - 1.0) < 1e-8,
                f"{ch_name} qubit={qubit}: trace={tr:.10f} ~ 1",
            )
            eigvals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
            _check(
                np.all(eigvals > -1e-10),
                f"{ch_name} qubit={qubit}: min eigval={eigvals.min():.2e} >= 0",
            )


# ------------------------------------------------------------------
# Test 2: ConstraintDissipation preserves trace and positivity
# ------------------------------------------------------------------

def test_constraint_step_cptp():
    """The composed step is CPTP: trace ~ 1, eigvals >= 0."""
    for n in [2, 3]:
        d = 2 ** n
        cfg = ConstraintConfig(
            n_qubits=n,
            allow_2q=True,
            gamma_max=2.0,
            enable_amp_damp=True,
            enable_dephasing=True,
            enable_depolarizing=True,
        )
        model = ConstraintDissipation(cfg)
        # Set moderate rates.
        params = np.full(model.num_params(), 0.5)
        model.unpack_params(params)

        rho = _random_dm(d, seed=200 + n)
        dt = 0.1
        for _ in range(30):
            rho = model.step(rho, dt)

        tr = trace_dm(rho)
        _check(
            abs(tr - 1.0) < 1e-7,
            f"n={n}: composed step trace = {tr:.10f}",
        )
        eigvals = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
        _check(
            np.all(eigvals > -1e-8),
            f"n={n}: min eigval = {eigvals.min():.2e}",
        )


# ------------------------------------------------------------------
# Test 3: teacher fidelity matches analytic formula (sanity)
# ------------------------------------------------------------------

def test_teacher_fidelity_formula():
    """Teacher snapshots reproduce F(t) = 1 - (1-F0) exp(-gamma t)."""
    from tail.fit_constraint_to_target import run_teacher_fidelities

    for n in [2, 3]:
        d = 2 ** n
        psi = _random_state(d, seed=300 + n)
        rho = _random_dm(d, seed=400 + n)
        gamma = 1.5
        tmax = 5.0
        steps = 25

        fids = run_teacher_fidelities(rho, psi, gamma, tmax, steps)
        times = np.linspace(0, tmax, steps + 1)
        f0 = fids[0]
        fids_exact = 1.0 - (1.0 - f0) * np.exp(-gamma * times)
        max_err = np.max(np.abs(fids - fids_exact))
        _check(
            max_err < 1e-9,
            f"n={n}: teacher fidelity matches formula (err={max_err:.2e})",
        )


# ------------------------------------------------------------------
# Test 4: fitting reduces loss vs random parameters
# ------------------------------------------------------------------

def test_fit_reduces_loss_small_n():
    """On n=2, fitting should achieve lower loss than random init."""
    n = 2
    d = 2 ** n
    psi = _random_state(d, seed=500)

    cfg = ConstraintConfig(
        n_qubits=n,
        allow_2q=True,
        gamma_max=2.0,
        enable_amp_damp=True,
        enable_dephasing=True,
    )

    rho_train = generate_training_states(
        d, n_states=3, psi_target=psi, seed=501,
    )

    # Compute loss at random init.
    model_rand = ConstraintDissipation(cfg)
    rng = np.random.default_rng(502)
    rand_params = rng.uniform(0.0, cfg.gamma_max, model_rand.num_params())
    model_rand.unpack_params(rand_params)

    from tail.fit_constraint_to_target import (
        _loss_fidelity,
        run_teacher_fidelities,
    )

    gamma_t = 1.0
    tmax = 3.0
    steps = 10
    teacher_fids = [
        run_teacher_fidelities(rho, psi, gamma_t, tmax, steps)
        for rho in rho_train
    ]
    loss_rand = _loss_fidelity(
        model_rand, teacher_fids, rho_train, psi, tmax, steps,
    )

    # Fit.
    result = fit_to_teacher(
        psi, rho_train, cfg,
        gamma_teacher=gamma_t, tmax=tmax, steps=steps,
        loss="fidelity", max_iter=80, seed=503, verbose=False,
    )
    loss_fit = result["best_loss"]

    _check(
        loss_fit < loss_rand,
        f"n={n}: fitted loss ({loss_fit:.4e}) < random loss ({loss_rand:.4e})",
    )


# ------------------------------------------------------------------
# Test 5: constraints are respected
# ------------------------------------------------------------------

def test_constraints_respected():
    """All rate parameters must lie in [0, gamma_max]."""
    n = 3
    cfg = ConstraintConfig(
        n_qubits=n,
        allow_2q=True,
        gamma_max=1.5,
        enable_amp_damp=True,
        enable_dephasing=True,
        enable_depolarizing=True,
    )
    model = ConstraintDissipation(cfg)

    # Attempt to set out-of-range values.
    bad_params = np.full(model.num_params(), 5.0)
    bad_params[:3] = -1.0
    model.unpack_params(bad_params)
    p = model.pack_params()

    # All rate params should be in [0, gamma_max].
    for name, start, count in model._layout:
        if name != "ancilla_reset":
            block = p[start:start + count]
            _check(
                np.all(block >= 0.0) and np.all(block <= cfg.gamma_max),
                f"{name}: all rates in [0, {cfg.gamma_max}]  "
                f"(range [{block.min():.4f}, {block.max():.4f}])",
            )


# ------------------------------------------------------------------
# Test 6: amplitude damping drives |1> -> |0>
# ------------------------------------------------------------------

def test_amp_damp_drives_to_ground():
    """Amplitude damping on every qubit should drive toward |0...0>."""
    n = 2
    d = 4
    cfg = ConstraintConfig(
        n_qubits=n,
        allow_2q=False,
        gamma_max=5.0,
        enable_amp_damp=True,
        enable_dephasing=False,
    )
    model = ConstraintDissipation(cfg)
    params = np.full(model.num_params(), 3.0)
    model.unpack_params(params)

    # Start in |11>
    rho = np.zeros((d, d), dtype=complex)
    rho[-1, -1] = 1.0

    rhos = model.run_trajectory(rho, tmax=5.0, steps=50)
    rho_final = rhos[-1]
    # Should be close to |00><00|.
    _check(
        float(np.real(rho_final[0, 0])) > 0.99,
        f"Amp damp: <00|rho|00> = {float(np.real(rho_final[0, 0])):.4f} > 0.99",
    )


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

ALL_TESTS = [
    test_one_qubit_channels_cptp,
    test_constraint_step_cptp,
    test_teacher_fidelity_formula,
    test_fit_reduces_loss_small_n,
    test_constraints_respected,
    test_amp_damp_drives_to_ground,
]


def run_all() -> bool:
    print("=" * 64)
    print("  Constrained dissipation  --  acceptance tests")
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
    ok = run_all()
    sys.exit(0 if ok else 1)
