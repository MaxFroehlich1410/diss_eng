"""Tests for Liouville-space utilities."""

import sys
import numpy as np

sys.path.insert(0, "/home/user/diss_eng")
from krotov_dissipative.liouville import (
    vectorize, unvectorize, apply_dissipator, dissipator_superop,
    hamiltonian_superop, adjoint_superop, build_liouvillian_from_amplitudes,
    fidelity_pure, trace_dm, purity, pure_state_dm, is_physical,
)

ATOL = 1e-10


def _check(cond, msg):
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {msg}")
    assert cond, msg


def test_vectorize_roundtrip():
    """vec(unvec(v)) == v and unvec(vec(A)) == A."""
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))
    v = vectorize(A)
    A2 = unvectorize(v, 4)
    _check(np.allclose(A, A2, atol=ATOL), "vectorize/unvectorize roundtrip")


def test_dissipator_trace_preservation():
    """Tr(D[L](rho)) = 0 for any L, rho."""
    rng = np.random.default_rng(1)
    d = 4
    L = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    # Make a valid density matrix
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())

    D_rho = apply_dissipator(L, rho)
    tr = np.trace(D_rho)
    _check(abs(tr) < ATOL, f"Tr(D[L](rho)) = {tr:.2e} ~ 0")


def test_dissipator_superop_matches_matrix_form():
    """S_L @ vec(rho) == vec(D[L](rho))."""
    rng = np.random.default_rng(2)
    d = 4
    L = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())

    D_rho = apply_dissipator(L, rho)
    S = dissipator_superop(L)
    D_rho_super = unvectorize(S @ vectorize(rho), d)

    _check(
        np.allclose(D_rho, D_rho_super, atol=ATOL),
        "Superoperator D[L] matches matrix-form D[L]"
    )


def test_hamiltonian_superop_trace_preservation():
    """Tr(-i[H, rho]) = 0."""
    rng = np.random.default_rng(3)
    d = 4
    H = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    H = (H + H.conj().T) / 2  # Hermitian
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())

    S_H = hamiltonian_superop(H)
    comm_rho = unvectorize(S_H @ vectorize(rho), d)
    tr = np.trace(comm_rho)
    _check(abs(tr) < ATOL, f"Tr(-i[H, rho]) = {tr:.2e} ~ 0")


def test_adjoint_superop_inner_product():
    """<A, S B> == <S^dag A, B> in Hilbert-Schmidt inner product."""
    rng = np.random.default_rng(4)
    d = 4
    L = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    S = dissipator_superop(L)
    S_adj = adjoint_superop(S)

    A_vec = rng.standard_normal(d * d) + 1j * rng.standard_normal(d * d)
    B_vec = rng.standard_normal(d * d) + 1j * rng.standard_normal(d * d)

    lhs = np.vdot(A_vec, S @ B_vec)
    rhs = np.vdot(S_adj @ A_vec, B_vec)
    _check(abs(lhs - rhs) < ATOL, f"<A, S B> == <S^dag A, B>: diff = {abs(lhs-rhs):.2e}")


def test_liouvillian_from_amplitudes():
    """L = sum_k u_k S_k computed correctly."""
    rng = np.random.default_rng(5)
    d = 4
    K = 3
    Ls = [rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d)) for _ in range(K)]
    S_ops = [dissipator_superop(L) for L in Ls]
    u = rng.uniform(0, 1, K)

    L_total = build_liouvillian_from_amplitudes(S_ops, u)
    L_manual = sum(u[k] * S_ops[k] for k in range(K))
    _check(np.allclose(L_total, L_manual, atol=ATOL), "Liouvillian from amplitudes is correct")


def test_pure_state_dm_properties():
    """Pure state DM has trace 1, is Hermitian, purity 1."""
    rng = np.random.default_rng(6)
    psi = rng.standard_normal(4) + 1j * rng.standard_normal(4)
    psi /= np.linalg.norm(psi)
    rho = pure_state_dm(psi)
    _check(abs(trace_dm(rho) - 1.0) < ATOL, "Tr(|psi><psi|) = 1")
    _check(abs(purity(rho) - 1.0) < ATOL, "Purity = 1 for pure state")
    _check(is_physical(rho), "|psi><psi| is physical")
    fid = fidelity_pure(rho, psi)
    _check(abs(fid - 1.0) < ATOL, "F(|psi><psi|, psi) = 1")


def test_dissipator_superop_preserves_hermiticity():
    """D[L](rho) is Hermitian when rho is Hermitian."""
    rng = np.random.default_rng(7)
    d = 4
    L = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())

    D_rho = apply_dissipator(L, rho)
    _check(
        np.allclose(D_rho, D_rho.conj().T, atol=ATOL),
        "D[L](rho) is Hermitian"
    )


ALL_TESTS = [
    test_vectorize_roundtrip,
    test_dissipator_trace_preservation,
    test_dissipator_superop_matches_matrix_form,
    test_hamiltonian_superop_trace_preservation,
    test_adjoint_superop_inner_product,
    test_liouvillian_from_amplitudes,
    test_pure_state_dm_properties,
    test_dissipator_superop_preserves_hermiticity,
]

if __name__ == "__main__":
    print("=" * 60)
    print("  Tests: Liouville-space utilities")
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
