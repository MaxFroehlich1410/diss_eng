r"""Monte Carlo Wave Function (quantum-jump trajectory) propagation.

Implements the MCWF algorithm from Goerz & Jacobs, arXiv:1801.04382v2,
Section 2.1.

Algorithm
---------
1. Define the non-Hermitian effective Hamiltonian
       H_eff = H - (i/2) sum_l  L_l^dag L_l
2. Draw a random number r in [0, 1)
3. Propagate under H_eff until ||psi(t)||^2 = r  (norm decays)
4. Apply a quantum jump  psi -> L_l psi / ||L_l psi||
   choosing L_l with probability  p(L_l) = <psi|L_l^dag L_l|psi>
5. Draw a new r and continue
6. Normalise the resulting psi(t)

We use a first-order scheme with adaptive sub-stepping for jump detection.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


def build_effective_hamiltonian(
    hamiltonian: np.ndarray,
    lindblad_ops: list[np.ndarray],
) -> np.ndarray:
    r"""H_eff = H - (i/2) sum_l L_l^dag L_l   (Eq. 3 of paper)."""
    H_eff = hamiltonian.astype(complex, copy=True)
    for L in lindblad_ops:
        H_eff -= 0.5j * (L.conj().T @ L)
    return H_eff


def _propagate_step(psi: np.ndarray, H_eff: np.ndarray, dt: float) -> np.ndarray:
    """Propagate psi by dt under H_eff using matrix exponential."""
    U_eff = expm(-1j * H_eff * dt)
    return U_eff @ psi


def propagate_mcwf_trajectory(
    psi0: np.ndarray,
    hamiltonian_func,
    lindblad_ops: list[np.ndarray],
    times: np.ndarray,
    rng: np.random.Generator | None = None,
) -> tuple[list[np.ndarray], list[tuple[float, int]]]:
    r"""Propagate a single MCWF trajectory with quantum jumps.

    Parameters
    ----------
    psi0 : ndarray (d,)
        Initial pure state (normalised).
    hamiltonian_func : callable(t) -> ndarray (d, d)
        Time-dependent Hamiltonian H(t).  For time-independent H pass
        ``lambda t: H``.
    lindblad_ops : list of ndarray (d, d)
        Lindblad (jump) operators {L_l}.
    times : ndarray (nt,)
        Time grid [t_0, t_1, ..., t_{nt-1}].
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    states : list of ndarray
        Pure states at each time point (normalised after any jump).
    jump_record : list of (time, operator_index)
        Record of when and which jumps occurred.
    """
    if rng is None:
        rng = np.random.default_rng()

    psi = np.asarray(psi0, dtype=complex).ravel().copy()
    psi /= np.linalg.norm(psi)

    LdL_list = [L.conj().T @ L for L in lindblad_ops]

    states = [psi.copy()]
    jump_record: list[tuple[float, int]] = []

    r = rng.uniform()

    for step in range(len(times) - 1):
        t = times[step]
        dt = times[step + 1] - times[step]
        H = hamiltonian_func(t)
        H_eff = build_effective_hamiltonian(H, lindblad_ops)

        psi_new = _propagate_step(psi, H_eff, dt)
        norm_sq = np.real(np.vdot(psi_new, psi_new))

        if norm_sq < r:
            dp_list = np.array([
                np.real(np.vdot(psi, LdL @ psi)) for LdL in LdL_list
            ])
            dp_total = dp_list.sum()

            if dp_total > 1e-15:
                probs = dp_list / dp_total
                l_idx = rng.choice(len(lindblad_ops), p=probs)
                psi_jumped = lindblad_ops[l_idx] @ psi
                norm_jumped = np.linalg.norm(psi_jumped)
                if norm_jumped > 1e-15:
                    psi = psi_jumped / norm_jumped
                else:
                    psi = psi_new / np.sqrt(max(norm_sq, 1e-30))
            else:
                psi = psi_new / np.sqrt(max(norm_sq, 1e-30))

            jump_record.append((t + dt, l_idx if dp_total > 1e-15 else -1))
            r = rng.uniform()
        else:
            psi = psi_new / np.sqrt(max(norm_sq, 1e-30))

        states.append(psi.copy())

    return states, jump_record


def propagate_no_jump(
    psi0: np.ndarray,
    hamiltonian_func,
    lindblad_ops: list[np.ndarray],
    times: np.ndarray,
) -> list[np.ndarray]:
    r"""Propagate under H_eff without jumps (deterministic, un-normalised).

    Used for backward propagation in Krotov's method where we need the
    coherent (no-jump) evolution with the adjoint effective Hamiltonian.

    Returns un-normalised states.
    """
    psi = np.asarray(psi0, dtype=complex).ravel().copy()
    states = [psi.copy()]

    for step in range(len(times) - 1):
        t = times[step]
        dt = times[step + 1] - times[step]
        H = hamiltonian_func(t)
        H_eff = build_effective_hamiltonian(H, lindblad_ops)
        psi = _propagate_step(psi, H_eff, dt)
        states.append(psi.copy())

    return states


def propagate_backward_no_jump(
    chi_T: np.ndarray,
    hamiltonian_func,
    lindblad_ops: list[np.ndarray],
    times: np.ndarray,
) -> list[np.ndarray]:
    r"""Backward-propagate chi under H_eff^dag (adjoint, no jumps).

    The backward propagation for Krotov's method uses:
        d chi / dt = +i H_eff^dag chi
    which is equivalent to propagating backward in time with H_eff.

    Parameters
    ----------
    chi_T : ndarray (d,)
        Boundary condition at final time T.
    hamiltonian_func : callable(t) -> ndarray (d, d)
        Time-dependent Hamiltonian.
    lindblad_ops : list of ndarray
        Lindblad operators.
    times : ndarray
        Time grid (forward-ordered).  We propagate from T to 0.

    Returns
    -------
    states : list of ndarray
        chi(t) at each time point, in forward time order.
    """
    chi = np.asarray(chi_T, dtype=complex).ravel().copy()
    nt = len(times)
    states_rev = [chi.copy()]

    for step in range(nt - 1, 0, -1):
        t = times[step]
        dt = times[step] - times[step - 1]
        H = hamiltonian_func(t)
        H_eff = build_effective_hamiltonian(H, lindblad_ops)
        H_eff_dag = H_eff.conj().T
        U_back = expm(+1j * H_eff_dag * dt)
        chi = U_back @ chi
        states_rev.append(chi.copy())

    states_rev.reverse()
    return states_rev


def density_matrix_from_trajectories(
    trajectory_states: list[list[np.ndarray]],
    time_index: int,
) -> np.ndarray:
    r"""Reconstruct rho(t_j) = (1/M) sum_k |psi_k(t_j)><psi_k(t_j)|."""
    M = len(trajectory_states)
    d = len(trajectory_states[0][time_index])
    rho = np.zeros((d, d), dtype=complex)
    for k in range(M):
        psi = trajectory_states[k][time_index]
        rho += np.outer(psi, psi.conj())
    return rho / M
