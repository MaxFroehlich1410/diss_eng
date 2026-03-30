"""Tests for operator basis construction."""

import sys
import numpy as np

sys.path.insert(0, "/home/user/diss_eng")
from krotov_dissipative.operators import (
    pauli_basis, single_qubit_operators, nearest_neighbour_operators,
    physical_operator_basis, random_operators, gell_mann_basis,
    ghz_state, w_state, random_pure_state, maximally_mixed_state,
)

ATOL = 1e-10


def _check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    assert cond, msg


def test_pauli_basis_count():
    """n-qubit Pauli basis has d^2 - 1 elements (excluding identity)."""
    for n in [1, 2, 3]:
        ops = pauli_basis(n, include_identity=False)
        d = 2 ** n
        expected = d * d - 1
        _check(len(ops) == expected, f"n={n}: {len(ops)} Paulis == {expected}")


def test_pauli_basis_traceless():
    """All non-identity Paulis are traceless."""
    ops = pauli_basis(2)
    for i, op in enumerate(ops):
        tr = np.trace(op)
        _check(abs(tr) < ATOL, f"Pauli {i}: Tr = {tr:.2e} ~ 0")


def test_pauli_basis_hermitian():
    """All Pauli strings are Hermitian."""
    ops = pauli_basis(2)
    for i, op in enumerate(ops):
        _check(np.allclose(op, op.conj().T, atol=ATOL),
               f"Pauli {i} is Hermitian")


def test_single_qubit_operators_count():
    """3 operators per qubit (sigma+, sigma-, Z)."""
    for n in [2, 3, 4]:
        ops = single_qubit_operators(n)
        _check(len(ops) == 3 * n, f"n={n}: {len(ops)} single-qubit ops == {3*n}")


def test_single_qubit_operators_dimension():
    """Each operator is d x d."""
    n = 3
    d = 2 ** n
    ops = single_qubit_operators(n)
    for op in ops:
        _check(op.shape == (d, d), f"shape = {op.shape} == ({d},{d})")


def test_nearest_neighbour_count():
    """3 operators per pair, n-1 pairs for chain topology."""
    for n in [2, 3, 4]:
        ops = nearest_neighbour_operators(n)
        expected = 3 * (n - 1)
        _check(len(ops) == expected, f"n={n}: {len(ops)} nn ops == {expected}")


def test_gell_mann_count():
    """d^2 - 1 Gell-Mann matrices for SU(d)."""
    for d in [2, 4, 8]:
        ops = gell_mann_basis(d)
        expected = d * d - 1
        _check(len(ops) == expected, f"d={d}: {len(ops)} Gell-Mann == {expected}")


def test_gell_mann_hermitian():
    """Gell-Mann matrices are Hermitian."""
    ops = gell_mann_basis(4)
    for i, op in enumerate(ops):
        _check(np.allclose(op, op.conj().T, atol=ATOL),
               f"Gell-Mann {i} is Hermitian")


def test_gell_mann_traceless():
    """Gell-Mann matrices are traceless."""
    ops = gell_mann_basis(4)
    for i, op in enumerate(ops):
        tr = np.trace(op)
        _check(abs(tr) < ATOL, f"Gell-Mann {i}: Tr = {tr:.2e} ~ 0")


def test_gell_mann_orthogonality():
    """Tr(G_a G_b) = 2 delta_{ab}."""
    d = 4
    ops = gell_mann_basis(d)
    for i in range(len(ops)):
        for j in range(i, len(ops)):
            inner = np.real(np.trace(ops[i] @ ops[j]))
            if i == j:
                _check(abs(inner - 2.0) < ATOL,
                       f"Tr(G_{i} G_{i}) = {inner:.6f} ~ 2")
            else:
                _check(abs(inner) < ATOL,
                       f"Tr(G_{i} G_{j}) = {inner:.2e} ~ 0")


def test_ghz_state():
    """GHZ state is normalised and has correct amplitudes."""
    psi = ghz_state(3)
    _check(abs(np.linalg.norm(psi) - 1.0) < ATOL, "GHZ is normalised")
    d = 8
    _check(abs(psi[0]) > 0.5, "GHZ has amplitude on |000>")
    _check(abs(psi[-1]) > 0.5, "GHZ has amplitude on |111>")


def test_random_pure_state():
    """Random state is normalised."""
    psi = random_pure_state(16, seed=42)
    _check(abs(np.linalg.norm(psi) - 1.0) < ATOL, "Random state is normalised")


def test_maximally_mixed():
    """Maximally mixed state has trace 1 and purity 1/d."""
    d = 8
    rho = maximally_mixed_state(d)
    tr = np.real(np.trace(rho))
    pur = np.real(np.trace(rho @ rho))
    _check(abs(tr - 1.0) < ATOL, f"Tr(I/d) = {tr:.6f} ~ 1")
    _check(abs(pur - 1.0/d) < ATOL, f"Purity = {pur:.6f} ~ {1.0/d:.6f}")


ALL_TESTS = [
    test_pauli_basis_count,
    test_pauli_basis_traceless,
    test_pauli_basis_hermitian,
    test_single_qubit_operators_count,
    test_single_qubit_operators_dimension,
    test_nearest_neighbour_count,
    test_gell_mann_count,
    test_gell_mann_hermitian,
    test_gell_mann_traceless,
    test_gell_mann_orthogonality,
    test_ghz_state,
    test_random_pure_state,
    test_maximally_mixed,
]

if __name__ == "__main__":
    print("=" * 60)
    print("  Tests: Operator basis construction")
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
                print(f"  [ERROR] {exc}")
    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed out of {passed + failed}")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
