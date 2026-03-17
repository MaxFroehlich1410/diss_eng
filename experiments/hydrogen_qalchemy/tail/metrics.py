"""Density-matrix diagnostics that work on raw numpy arrays.

Every function here is O(d^2) or better -- no matrix-matrix products.
"""

from __future__ import annotations

import numpy as np


def fidelity_to_pure(rho: np.ndarray, psi: np.ndarray) -> float:
    r"""Fidelity of a density matrix to a pure target state.

    .. math:: F = \langle\psi|\rho|\psi\rangle

    Computed as ``Re(psi^dag @ rho @ psi)`` via a single matrix-vector
    product followed by a dot product: O(d^2).
    """
    v = rho @ psi                           # O(d^2)
    return float(np.real(psi.conj() @ v))   # O(d)


def trace_dm(rho: np.ndarray) -> float:
    """Tr(rho) -- O(d)."""
    return float(np.real(np.trace(rho)))


def purity(rho: np.ndarray) -> float:
    r"""Tr(rho^2).

    Implemented with the Frobenius norm to avoid a full matrix-matrix
    product: ``Tr(rho^2) = sum_{ij} |rho_{ij}|^2``.
    This is O(d^2), versus O(d^3) for ``trace(rho @ rho)``.
    """
    return float(np.real(np.vdot(rho, rho)))


def sanitize_density_matrix(
    rho: np.ndarray,
    *,
    fix_hermiticity: bool = True,
    fix_trace: bool = True,
    clip_negative_eigs: bool = False,
) -> np.ndarray:
    """In-place-safe enforcement of density-matrix constraints.

    Parameters
    ----------
    fix_hermiticity : bool
        Replace rho by (rho + rho^dag) / 2.
    fix_trace : bool
        Rescale so that Tr(rho) = 1.
    clip_negative_eigs : bool
        Diagonalise, clip eigenvalues to >= 0, reconstruct.
        This is O(d^3) and should only be used as a last-resort repair.
    """
    rho = rho.copy()
    if fix_hermiticity:
        rho = 0.5 * (rho + rho.conj().T)
    if clip_negative_eigs:
        eigvals, eigvecs = np.linalg.eigh(rho)
        eigvals = np.clip(eigvals, 0.0, None)
        rho = (eigvecs * eigvals) @ eigvecs.conj().T
    if fix_trace:
        tr = np.trace(rho)
        if abs(tr) > 1e-15:
            rho /= tr
    return rho
