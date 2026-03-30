r"""Liouville-space utilities for open quantum systems.

Column-stacking vectorisation convention:

    vec(A rho B) = (B^T \otimes A) vec(rho)

Key objects
-----------
* ``vectorize`` / ``unvectorize`` -- reshape between (d,d) and (d^2,)
* ``dissipator_superop`` -- build the d^2 x d^2 superoperator for D[L]
* ``adjoint_superop`` -- Hermitian adjoint of a superoperator (L -> L^dag)
* ``apply_dissipator`` -- D[L](rho) in matrix form (no vectorisation)
"""

from __future__ import annotations

import numpy as np


# -------------------------------------------------------------------
# Vectorisation
# -------------------------------------------------------------------

def vectorize(rho: np.ndarray) -> np.ndarray:
    """Column-stacking vectorisation: rho (d,d) -> vec (d^2,)."""
    return rho.flatten(order="F")


def unvectorize(vec: np.ndarray, d: int) -> np.ndarray:
    """Inverse of vectorize: vec (d^2,) -> rho (d,d)."""
    return vec.reshape((d, d), order="F")


# -------------------------------------------------------------------
# Dissipator in matrix form (no vectorisation)
# -------------------------------------------------------------------

def apply_dissipator(L: np.ndarray, rho: np.ndarray) -> np.ndarray:
    r"""Compute D[L](rho) = L rho L^dag - 1/2 {L^dag L, rho}.

    Parameters
    ----------
    L : ndarray (d, d)
        Lindblad operator.
    rho : ndarray (d, d)
        Density matrix.

    Returns
    -------
    ndarray (d, d)
    """
    Ldag = L.conj().T
    LdL = Ldag @ L
    return L @ rho @ Ldag - 0.5 * (LdL @ rho + rho @ LdL)


# -------------------------------------------------------------------
# Superoperator construction
# -------------------------------------------------------------------

def dissipator_superop(L: np.ndarray) -> np.ndarray:
    r"""Build the d^2 x d^2 superoperator for D[L].

    D[L](rho) = L rho L^dag - 1/2 {L^dag L, rho}

    In Liouville space (column-stacking):
        S_L = conj(L) \otimes L - 1/2 I \otimes L^dag L - 1/2 (L^dag L)^T \otimes I
    """
    d = L.shape[0]
    eye = np.eye(d, dtype=complex)
    Ldag = L.conj().T
    LdL = Ldag @ L
    S = (np.kron(L.conj(), L)
         - 0.5 * np.kron(eye, LdL)
         - 0.5 * np.kron(LdL.T, eye))
    return S


def hamiltonian_superop(H: np.ndarray) -> np.ndarray:
    r"""Build the superoperator for -i[H, rho].

    In Liouville space: -i(I \otimes H - H^T \otimes I).
    """
    d = H.shape[0]
    eye = np.eye(d, dtype=complex)
    return -1j * (np.kron(eye, H) - np.kron(H.T, eye))


def adjoint_superop(S: np.ndarray) -> np.ndarray:
    r"""Hermitian adjoint of a superoperator.

    For the column-stacking convention, the adjoint w.r.t. the
    Hilbert-Schmidt inner product <A, B> = Tr(A^dag B) is simply
    the conjugate transpose of the superoperator matrix.
    """
    return S.conj().T


# -------------------------------------------------------------------
# Full Liouvillian from amplitudes
# -------------------------------------------------------------------

def build_liouvillian_from_amplitudes(
    dissipator_superops: list[np.ndarray],
    amplitudes: np.ndarray,
    hamiltonian_superop_matrix: np.ndarray | None = None,
) -> np.ndarray:
    r"""Build L = sum_k u_k S_k  (+ optional Hamiltonian part).

    Parameters
    ----------
    dissipator_superops : list of ndarray (d^2, d^2)
        Pre-computed dissipator superoperators S_k = superop(D[L_k]).
    amplitudes : ndarray (K,)
        Control amplitudes u_k (must be >= 0 for physical rates).
    hamiltonian_superop_matrix : ndarray (d^2, d^2), optional
        Superoperator for -i[H, .] (time-independent Hamiltonian).

    Returns
    -------
    ndarray (d^2, d^2)
        Full Liouvillian.
    """
    d2 = dissipator_superops[0].shape[0]
    liouvillian = np.zeros((d2, d2), dtype=complex)
    for u_k, S_k in zip(amplitudes, dissipator_superops):
        liouvillian += u_k * S_k
    if hamiltonian_superop_matrix is not None:
        liouvillian += hamiltonian_superop_matrix
    return liouvillian


# -------------------------------------------------------------------
# Density matrix utilities
# -------------------------------------------------------------------

def fidelity_pure(rho: np.ndarray, psi: np.ndarray) -> float:
    """Fidelity F = <psi|rho|psi> for a pure target state."""
    psi = np.asarray(psi, dtype=complex).ravel()
    return float(np.real(psi.conj() @ rho @ psi))


def trace_dm(rho: np.ndarray) -> float:
    """Trace of a density matrix."""
    return float(np.real(np.trace(rho)))


def purity(rho: np.ndarray) -> float:
    """Purity Tr(rho^2)."""
    return float(np.real(np.trace(rho @ rho)))


def pure_state_dm(psi: np.ndarray) -> np.ndarray:
    """Build |psi><psi|."""
    psi = np.asarray(psi, dtype=complex).ravel()
    return np.outer(psi, psi.conj())


def is_physical(rho: np.ndarray, atol: float = 1e-8) -> bool:
    """Check trace ~ 1, Hermitian, positive semidefinite."""
    tr = float(np.real(np.trace(rho)))
    hermitian = np.allclose(rho, rho.conj().T, atol=atol)
    eigs = np.linalg.eigvalsh(rho)
    psd = float(eigs[0]) >= -atol
    return abs(tr - 1.0) < atol and hermitian and psd
