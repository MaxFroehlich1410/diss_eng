r"""Forward and backward propagation for Krotov's method.

Time discretisation
-------------------
Time grid: t_0 = 0, t_1, ..., t_{N_t} = T  with dt = T / N_t.

The state at time t_{j+1} is obtained from t_j by:

    |rho(t_{j+1})>> = exp(L[u(t_j)] dt) |rho(t_j)>>

where L[u] = sum_k u_k S_k is the Liouvillian built from the
control amplitudes u_k and pre-computed dissipator superoperators S_k.

Forward propagation
~~~~~~~~~~~~~~~~~~~
Starting from rho_0, propagate forward using the current controls.

Backward propagation (adjoint)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Starting from chi(T) = rho_target (gradient of fidelity), propagate
backward using the adjoint Liouvillian L^dag:

    |chi(t_j)>> = exp(L^dag[u(t_j)] dt) |chi(t_{j+1})>>

Note: the backward propagation uses exp(+L^dag dt) because the adjoint
equation is d chi/dt = -L^dag chi, and going backward in time from
t_{j+1} to t_j gives chi(t_j) = exp(L^dag dt) chi(t_{j+1}).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from .liouville import (
    vectorize, unvectorize, build_liouvillian_from_amplitudes,
    adjoint_superop,
)


def propagate_step(
    rho_vec: np.ndarray,
    liouvillian: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Propagate one time step: rho_vec -> exp(L dt) rho_vec."""
    propagator = expm(liouvillian * dt)
    return propagator @ rho_vec


def forward_propagation(
    rho0: np.ndarray,
    dissipator_superops: list[np.ndarray],
    controls: np.ndarray,
    dt: float,
    hamiltonian_superop: np.ndarray | None = None,
) -> list[np.ndarray]:
    r"""Full forward propagation storing states at each time step.

    Parameters
    ----------
    rho0 : ndarray (d, d)
        Initial density matrix.
    dissipator_superops : list of ndarray (d^2, d^2)
        Pre-computed dissipator superoperators.
    controls : ndarray (N_t, K)
        Control amplitudes at each time step.
    dt : float
        Time step size.
    hamiltonian_superop : ndarray (d^2, d^2), optional
        Time-independent Hamiltonian superoperator.

    Returns
    -------
    list of ndarray (d, d)
        Density matrices at times t_0, t_1, ..., t_{N_t}.
        Length = N_t + 1.
    """
    d = rho0.shape[0]
    N_t = controls.shape[0]

    rho_vec = vectorize(rho0)
    states = [rho0.copy()]

    for j in range(N_t):
        L = build_liouvillian_from_amplitudes(
            dissipator_superops, controls[j], hamiltonian_superop
        )
        rho_vec = propagate_step(rho_vec, L, dt)
        states.append(unvectorize(rho_vec.copy(), d))

    return states


def backward_propagation(
    chi_T: np.ndarray,
    dissipator_superops: list[np.ndarray],
    controls: np.ndarray,
    dt: float,
    hamiltonian_superop: np.ndarray | None = None,
) -> list[np.ndarray]:
    r"""Full backward propagation of the adjoint (costate).

    chi(T) = rho_target, then backward using exp(L^dag dt).

    Parameters
    ----------
    chi_T : ndarray (d, d)
        Terminal costate (= rho_target for fidelity objective).
    dissipator_superops : list of ndarray (d^2, d^2)
        Pre-computed dissipator superoperators.
    controls : ndarray (N_t, K)
        Control amplitudes at each time step (from the OLD iteration).
    dt : float
        Time step size.
    hamiltonian_superop : ndarray (d^2, d^2), optional
        Time-independent Hamiltonian superoperator.

    Returns
    -------
    list of ndarray (d, d)
        Costates at times t_0, ..., t_{N_t}.
        Length = N_t + 1.  Index j corresponds to time t_j.
    """
    d = chi_T.shape[0]
    N_t = controls.shape[0]

    chi_vec = vectorize(chi_T)
    # Store backward: costates[N_t] = chi_T, ..., costates[0] = chi(0)
    costates = [None] * (N_t + 1)
    costates[N_t] = chi_T.copy()

    for j in range(N_t - 1, -1, -1):
        L = build_liouvillian_from_amplitudes(
            dissipator_superops, controls[j], hamiltonian_superop
        )
        L_adj = adjoint_superop(L)
        chi_vec = propagate_step(chi_vec, L_adj, dt)
        costates[j] = unvectorize(chi_vec.copy(), d)

    return costates


def forward_propagation_sequential(
    rho0: np.ndarray,
    dissipator_superops: list[np.ndarray],
    costates_old: list[np.ndarray],
    controls_ref: np.ndarray,
    lambda_reg: np.ndarray,
    dt: float,
    lindblad_ops: list[np.ndarray],
    hamiltonian_superop: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    r"""Sequential forward propagation with Krotov control update.

    This is the core of Krotov's method: at each time step, first update
    the control using the old costate and the new (just-propagated)
    forward state, then propagate forward with the updated control.

    Update rule for each time step j:
        u_k^{new}(t_j) = u_k^{ref}(t_j) + (1/lambda_k) Re Tr(chi_j^dag D[L_k](rho_j^{new}))

    Parameters
    ----------
    rho0 : ndarray (d, d)
        Initial density matrix.
    dissipator_superops : list of ndarray (d^2, d^2)
        Pre-computed dissipator superoperators.
    costates_old : list of ndarray (d, d)
        Costates from the previous Krotov iteration.
    controls_ref : ndarray (N_t, K)
        Reference controls (from previous iteration).
    lambda_reg : ndarray (K,)
        Regularisation parameters (one per control channel).
    dt : float
        Time step size.
    lindblad_ops : list of ndarray (d, d)
        Lindblad operators L_k (for computing D[L_k](rho) in matrix form).
    hamiltonian_superop : ndarray (d^2, d^2), optional
        Time-independent Hamiltonian superoperator.

    Returns
    -------
    states_new : list of ndarray (d, d)
        Updated forward states. Length = N_t + 1.
    controls_new : ndarray (N_t, K)
        Updated controls.
    """
    d = rho0.shape[0]
    N_t = controls_ref.shape[0]
    K = controls_ref.shape[1]

    rho_vec = vectorize(rho0)
    states_new = [rho0.copy()]
    controls_new = np.empty_like(controls_ref)

    for j in range(N_t):
        rho_j = unvectorize(rho_vec, d)
        chi_j = costates_old[j]

        # --- Krotov control update ---
        for k in range(K):
            # D[L_k](rho) in matrix form
            L_k = lindblad_ops[k]
            Ldag = L_k.conj().T
            LdL = Ldag @ L_k
            D_rho = L_k @ rho_j @ Ldag - 0.5 * (LdL @ rho_j + rho_j @ LdL)

            # Gradient: Re Tr(chi^dag D[L_k](rho))
            grad = np.real(np.trace(chi_j.conj().T @ D_rho))
            controls_new[j, k] = controls_ref[j, k] + grad / lambda_reg[k]

        # Enforce non-negativity (physical rates)
        controls_new[j] = np.maximum(controls_new[j], 0.0)

        # --- Propagate with updated control ---
        L = build_liouvillian_from_amplitudes(
            dissipator_superops, controls_new[j], hamiltonian_superop
        )
        rho_vec = propagate_step(rho_vec, L, dt)
        states_new.append(unvectorize(rho_vec.copy(), d))

    return states_new, controls_new
