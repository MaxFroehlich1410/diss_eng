"""Correctness / acceptance tests for the target-cooling tail.

Run with
--------
    python -m pytest tail/tests.py -v
or
    python tail/tests.py                  (standalone -- no pytest needed)

All tests work for n_qubits = 3..5 and finish in < 1 s each.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the hydrogen_qalchemy directory is importable.
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import numpy as np

from tail.metrics import fidelity_to_pure, trace_dm, purity
from tail.target_cooling import (
    apply_target_cooling_step,
    run_target_cooling_trajectory,
)

ATOL = 1e-10


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _random_state(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return psi / np.linalg.norm(psi)


def _random_dm(d: int, seed: int) -> np.ndarray:
    """Random full-rank density matrix."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def _check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    assert cond, msg


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def test_target_state_is_fixed_point():
    """If rho = |psi><psi|, the tail must leave it unchanged."""
    for n in [3, 4, 5]:
        d = 2 ** n
        psi = _random_state(d, seed=n)
        rho = np.outer(psi, psi.conj())

        times, fids, trs, purs = run_target_cooling_trajectory(
            rho, psi, gamma=2.0, tmax=10.0, steps=20,
        )
        _check(
            np.allclose(fids, 1.0, atol=ATOL),
            f"n={n}: fidelity stays 1 when starting at target",
        )
        _check(
            np.allclose(trs, 1.0, atol=ATOL),
            f"n={n}: trace stays 1 at target",
        )
        _check(
            np.allclose(purs, 1.0, atol=ATOL),
            f"n={n}: purity stays 1 at target",
        )


def test_trace_preservation():
    """Tr(rho(t)) must stay == 1 throughout the tail."""
    for n in [3, 4, 5]:
        d = 2 ** n
        psi = _random_state(d, seed=10 + n)
        rho = _random_dm(d, seed=20 + n)

        _, _, trs, _ = run_target_cooling_trajectory(
            rho, psi, gamma=1.5, tmax=8.0, steps=40,
        )
        _check(
            np.allclose(trs, 1.0, atol=1e-9),
            f"n={n}: trace preserved  (max dev = {np.max(np.abs(trs - 1)):.2e})",
        )


def test_fidelity_matches_exact_formula():
    r"""F(t) must equal  1 - (1 - F_0) exp(-gamma t)  for every t."""
    for n in [3, 4, 5]:
        d = 2 ** n
        psi = _random_state(d, seed=30 + n)
        rho = _random_dm(d, seed=40 + n)

        gamma = 1.0 + 0.5 * n
        tmax = 6.0
        steps = 30

        times, fids, _, _ = run_target_cooling_trajectory(
            rho, psi, gamma=gamma, tmax=tmax, steps=steps,
        )
        f0 = fids[0]
        fids_exact = 1.0 - (1.0 - f0) * np.exp(-gamma * times)

        max_err = np.max(np.abs(fids - fids_exact))
        _check(
            max_err < 1e-9,
            f"n={n}: fidelity matches analytic formula  (max err = {max_err:.2e})",
        )


def test_fidelity_is_monotonic():
    """Fidelity during the tail must be non-decreasing."""
    for n in [3, 4, 5]:
        d = 2 ** n
        psi = _random_state(d, seed=50 + n)
        rho = _random_dm(d, seed=60 + n)

        _, fids, _, _ = run_target_cooling_trajectory(
            rho, psi, gamma=1.0, tmax=5.0, steps=50,
        )
        diffs = np.diff(fids)
        _check(
            np.all(diffs >= -ATOL),
            f"n={n}: fidelity monotonically non-decreasing  "
            f"(min step = {np.min(diffs):.2e})",
        )


def test_purity_metric_frobenius():
    """Sanity: our O(d^2) purity matches the naive Tr(rho^2)."""
    for n in [3, 4]:
        d = 2 ** n
        rho = _random_dm(d, seed=70 + n)
        p_fast = purity(rho)
        p_naive = float(np.real(np.trace(rho @ rho)))
        _check(
            abs(p_fast - p_naive) < 1e-12,
            f"n={n}: Frobenius purity == Tr(rho^2)  (diff = {abs(p_fast - p_naive):.2e})",
        )


def test_cooling_reaches_target():
    """After long enough time, rho must converge to |psi><psi|."""
    for n in [3, 4, 5]:
        d = 2 ** n
        psi = _random_state(d, seed=80 + n)
        rho = _random_dm(d, seed=90 + n)

        _, fids, _, _ = run_target_cooling_trajectory(
            rho, psi, gamma=2.0, tmax=50.0, steps=100,
        )
        _check(
            abs(fids[-1] - 1.0) < 1e-8,
            f"n={n}: fidelity -> 1  (final = {fids[-1]:.10f})",
        )


# ------------------------------------------------------------------
# Runner
# ------------------------------------------------------------------

ALL_TESTS = [
    test_target_state_is_fixed_point,
    test_trace_preservation,
    test_fidelity_matches_exact_formula,
    test_fidelity_is_monotonic,
    test_purity_metric_frobenius,
    test_cooling_reaches_target,
]


def run_all() -> bool:
    print("=" * 64)
    print("  Target-cooling tail  --  acceptance tests")
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
    ok = run_all()
    sys.exit(0 if ok else 1)
