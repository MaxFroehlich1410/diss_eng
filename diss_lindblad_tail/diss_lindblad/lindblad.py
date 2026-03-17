r"""Lindblad master equation: Liouvillian construction and time evolution.

This module implements the dissipator in Liouville (superoperator) space
using the column-stacking vectorisation convention.  The evolution is
computed deterministically via matrix exponentiation -- no stochastic
trajectories.

Master equation (with optional Hamiltonian H)
----------------------------------------------

    d rho / dt  =  -i [H, rho]  +  sum_k  gamma_k  D[L_k](rho)

where

    D[L](rho) = L rho L^dag  -  1/2 { L^dag L, rho }.

Vectorised form
---------------

    d |rho>> / dt  =  Liouvillian  |rho>>

with

    Liouvillian  =  -i ( I (x) H  -  H^T (x) I )
        + sum_k gamma_k [ conj(L_k) (x) L_k
                          - 1/2  I (x) L_k^dag L_k
                          - 1/2  (L_k^dag L_k)^T (x) I ]

where (x) denotes the Kronecker product and ``conj`` is element-wise
complex conjugation (NOT the conjugate-transpose).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, null_space


# ---------------------------------------------------------------------------
# Liouvillian construction
# ---------------------------------------------------------------------------

def build_liouvillian(
    lindblad_ops: list[np.ndarray],
    rates: np.ndarray | list[float] | None = None,
    hamiltonian: np.ndarray | None = None,
) -> np.ndarray:
    """Build the Liouvillian superoperator in Liouville space.

    Parameters
    ----------
    lindblad_ops : list of ndarray, each shape (d, d)
        Lindblad (jump) operators L_k.
    rates : array-like of length len(lindblad_ops), optional
        Dissipation rates gamma_k.  Defaults to all ones.
    hamiltonian : ndarray (d, d), optional
        System Hamiltonian H.  If *None*, no coherent part is added.

    Returns
    -------
    ndarray, shape (d^2, d^2)
        Complex Liouvillian superoperator.
    """
    if not lindblad_ops:
        raise ValueError("At least one Lindblad operator is required.")

    d = lindblad_ops[0].shape[0]
    d2 = d * d
    eye = np.eye(d, dtype=complex)

    if rates is None:
        rates = np.ones(len(lindblad_ops))
    rates = np.asarray(rates, dtype=float)

    liouvillian = np.zeros((d2, d2), dtype=complex)

    # -- coherent part:  -i ( I (x) H  -  H^T (x) I ) --
    if hamiltonian is not None:
        H = np.asarray(hamiltonian, dtype=complex)
        liouvillian += -1j * (np.kron(eye, H) - np.kron(H.T, eye))

    # -- dissipative part --
    for gamma_k, L_k in zip(rates, lindblad_ops):
        L_k = np.asarray(L_k, dtype=complex)
        LdL = L_k.conj().T @ L_k                        # L^dag L
        liouvillian += gamma_k * (
            np.kron(L_k.conj(), L_k)                     # conj(L) (x) L
            - 0.5 * np.kron(eye, LdL)                    # 1/2  I (x) L^dag L
            - 0.5 * np.kron(LdL.T, eye)                  # 1/2  (L^dag L)^T (x) I
        )

    return liouvillian


# ---------------------------------------------------------------------------
# Time evolution
# ---------------------------------------------------------------------------

def evolve(
    rho: np.ndarray,
    liouvillian: np.ndarray,
    t: float,
) -> np.ndarray:
    """Evolve a density matrix under a Liouvillian for time *t*.

    Computes  rho(t) = unvec( expm(L t) . vec(rho) ).
    """
    d = rho.shape[0]
    rho_vec = rho.flatten(order="F")
    propagator = expm(liouvillian * t)
    rho_t_vec = propagator @ rho_vec
    return rho_t_vec.reshape((d, d), order="F")


def evolve_trajectory(
    rho: np.ndarray,
    liouvillian: np.ndarray,
    times: np.ndarray,
) -> list[np.ndarray]:
    """Evolve rho and return a list of density matrices at each time.

    *times* must be a sorted 1-D array starting from 0.  Uses incremental
    propagation (single matrix exponentiation for dt) to avoid recomputing
    the propagator from scratch at every step.
    """
    d = rho.shape[0]
    times = np.asarray(times, dtype=float)

    if len(times) == 0:
        return []

    rho_vec = rho.flatten(order="F")
    snapshots: list[np.ndarray] = [rho_vec.reshape((d, d), order="F").copy()]

    if len(times) == 1:
        return snapshots

    # Uniform spacing assumed (linspace).  Compute a single propagator.
    dt = times[1] - times[0]
    propagator = expm(liouvillian * dt)

    for _ in range(1, len(times)):
        rho_vec = propagator @ rho_vec
        snapshots.append(rho_vec.reshape((d, d), order="F").copy())

    return snapshots


# ---------------------------------------------------------------------------
# Standard Lindblad operator sets
# ---------------------------------------------------------------------------

def target_cooling_operators(psi_target: np.ndarray) -> list[np.ndarray]:
    r"""Lindblad operators that drive *any* initial state toward |psi_target>.

    Returns d-1 operators  L_k = |psi*><psi_k^perp|  where
    {|psi_k^perp>} is an orthonormal basis for the orthogonal complement
    of |psi*>.

    With equal rates gamma the unique steady state is |psi*><psi*|.
    """
    psi = np.asarray(psi_target, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    d = len(psi)

    # Orthogonal complement via null space of <psi| (row vector psi^dag).
    orth = null_space(psi.conj().reshape(1, -1))       # shape (d, d-1)

    ops = []
    for k in range(orth.shape[1]):
        L_k = np.outer(psi, orth[:, k].conj())         # |psi*><psi_k^perp|
        ops.append(L_k)
    return ops


def amplitude_damping_operators(n_qubits: int) -> list[np.ndarray]:
    r"""Single-qubit amplitude-damping operators sigma^- = |0><1| on each qubit.

    These model standard T_1 decay, embedded in the full 2^n Hilbert space.
    """
    d = 2 ** n_qubits
    ops = []
    for q in range(n_qubits):
        sigma_minus = np.zeros((d, d), dtype=complex)
        for i in range(d):
            if (i >> q) & 1:                            # qubit q is |1>
                j = i ^ (1 << q)                        # flip to |0>
                sigma_minus[j, i] = 1.0
        ops.append(sigma_minus)
    return ops


def dephasing_operators(n_qubits: int) -> list[np.ndarray]:
    """Single-qubit dephasing (Z) operators on each qubit."""
    d = 2 ** n_qubits
    ops = []
    for q in range(n_qubits):
        Z_q = np.zeros((d, d), dtype=complex)
        for i in range(d):
            Z_q[i, i] = -1.0 if (i >> q) & 1 else 1.0
        ops.append(Z_q)
    return ops
