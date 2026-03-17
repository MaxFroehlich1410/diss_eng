"""Density-matrix utilities: construction, channel application, and fidelity.

All functions operate on *dense* numpy arrays and use the standard
column-stacking vectorisation convention where needed:

    vec(A rho B) = (B^T kron A) vec(rho).
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def pure_state_dm(psi: np.ndarray) -> np.ndarray:
    r"""Convert a pure state |psi> to the density matrix |psi><psi|."""
    psi = np.asarray(psi, dtype=complex).ravel()
    return np.outer(psi, psi.conj())


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

def apply_unitary(rho: np.ndarray, U: np.ndarray) -> np.ndarray:
    r"""Apply a unitary channel:  rho -> U rho U^dagger."""
    return U @ rho @ U.conj().T


# ---------------------------------------------------------------------------
# Measures
# ---------------------------------------------------------------------------

def fidelity_to_pure(rho: np.ndarray, psi: np.ndarray) -> float:
    r"""Fidelity of a density matrix *rho* w.r.t. a pure state |psi>:

        F = <psi| rho |psi>.

    For a pure target this equals the squared overlap |<psi|phi>|^2 when
    rho = |phi><phi|.
    """
    psi = np.asarray(psi, dtype=complex).ravel()
    return float(np.real(psi.conj() @ rho @ psi))


def trace(rho: np.ndarray) -> complex:
    """Return Tr(rho)."""
    return np.trace(rho)


def purity(rho: np.ndarray) -> float:
    """Return Tr(rho^2)."""
    return float(np.real(np.trace(rho @ rho)))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def is_physical(rho: np.ndarray, atol: float = 1e-8) -> dict:
    """Check whether *rho* is a valid density matrix.

    Returns a dict with fields:
        trace, hermitian, positive_semidefinite, min_eigenvalue, is_valid.
    """
    tr = float(np.real(np.trace(rho)))
    hermitian = bool(np.allclose(rho, rho.conj().T, atol=atol))
    eigenvalues = np.linalg.eigvalsh(rho)
    min_eig = float(eigenvalues[0])
    psd = bool(min_eig >= -atol)
    valid = abs(tr - 1.0) < atol and hermitian and psd
    return {
        "trace": tr,
        "hermitian": hermitian,
        "positive_semidefinite": psd,
        "min_eigenvalue": min_eig,
        "is_valid": valid,
    }


# ---------------------------------------------------------------------------
# Vectorisation helpers  (column-stacking / "vec" convention)
# ---------------------------------------------------------------------------

def vectorize(rho: np.ndarray) -> np.ndarray:
    """Column-stacking vectorisation:  vec(rho)."""
    return rho.flatten(order="F")


def unvectorize(vec: np.ndarray, d: int) -> np.ndarray:
    """Reconstruct a d x d matrix from its column-stacked vectorisation."""
    return vec.reshape((d, d), order="F")
