r"""Exact, efficient target-cooling Lindblad channel.

Physics
-------
We model the dissipative tail with jump operators

    L_k = sqrt(gamma) |psi_*><psi_k^perp|      k = 1 .. d-1

where {|psi_k^perp>} spans the orthogonal complement of the target
state |psi_*>.  These operators "cool" population from every orthogonal
direction into the target:

* |psi_*><psi_*| is the *unique* steady state.
* Fidelity obeys  F(t) = 1 - (1 - F_0) exp(-gamma t).

Why closed-form instead of a Liouvillian?
-----------------------------------------
Building the full Liouvillian requires a d^2 x d^2 superoperator matrix.
For 10 qubits (d = 1024) that means ~ 10^12 elements -- completely
infeasible.  Instead we exploit the *block structure* induced by the
rank-1 projector P = |psi_*><psi_*|.

Decompose rho into four blocks relative to P and P_perp = I - P:

    a_0 = Tr(P rho) = <psi|rho|psi>          (scalar -- fidelity)
    B_0 = P_perp rho P_perp                   (d x d)
    C_0 = P rho P_perp                        (d x d, rank <= 1)
    D_0 = P_perp rho P                        (d x d, rank <= 1)

Under the cooling channel each block evolves independently:

    a(t) = 1 - (1 - a_0) exp(-gamma t)
    B(t) = exp(-gamma t)     B_0
    C(t) = exp(-gamma t / 2) C_0
    D(t) = exp(-gamma t / 2) D_0

and  rho(t) = a(t) P + B(t) + C(t) + D(t).

Efficient O(d^2) implementation
-------------------------------
We never form P (d x d) explicitly as a stored matrix.  Instead we use:

    v = rho @ psi           (matrix-vector, O(d^2))
    a_0 = Re(psi^* . v)    (dot,           O(d))
    w = v.conj()            (Hermiticity:   O(d))

Then:

    rho_new  =  e^{-g} rho
             +  alpha  |psi><psi|
             +  beta   |psi><w|
             +  beta   |v><psi|

with  g = gamma dt,  g2 = g/2,  and

    alpha = (1 - e^{-g}) + 2 a_0 (e^{-g} - e^{-g/2})
    beta  = e^{-g/2} - e^{-g}

This is four O(d^2) operations (one scalar-matrix multiply, three outer
products) per time step -- no d x d matrix-matrix products at all.
"""

from __future__ import annotations

import numpy as np

from .metrics import fidelity_to_pure, trace_dm, purity


# -----------------------------------------------------------------------
# Core one-step map
# -----------------------------------------------------------------------

def apply_target_cooling_step(
    rho: np.ndarray,
    psi: np.ndarray,
    psi_conj: np.ndarray,
    gamma: float,
    dt: float,
) -> np.ndarray:
    r"""Apply one exact dt-step of target-cooling Lindblad evolution.

    Parameters
    ----------
    rho : ndarray (d, d)
        Current density matrix (must be Hermitian).
    psi : ndarray (d,)
        Target state vector (normalised).
    psi_conj : ndarray (d,)
        Pre-computed ``psi.conj()`` -- avoids repeated conjugation.
    gamma : float  > 0
        Dissipation rate.
    dt : float > 0
        Time step.

    Returns
    -------
    rho_new : ndarray (d, d)
    """
    # Key vectors / scalar  [O(d^2)]
    v = rho @ psi                               # rho |psi>
    a0 = np.real(psi_conj @ v)                  # <psi|rho|psi>

    # Decay factors
    g = gamma * dt
    exp_g = np.exp(-g)
    exp_g2 = np.exp(-g / 2.0)

    # Coefficients for the rank-structured update
    alpha = (1.0 - exp_g) + 2.0 * a0 * (exp_g - exp_g2)
    beta = exp_g2 - exp_g

    # Assemble rho_new: one scalar multiply + three outer products  [O(d^2)]
    rho_new = exp_g * rho
    rho_new += alpha * np.outer(psi, psi_conj)
    rho_new += beta * np.outer(psi, v.conj())   # |psi><w| where w = v* (Hermiticity)
    rho_new += beta * np.outer(v, psi_conj)     # |v><psi|

    return rho_new


# -----------------------------------------------------------------------
# Full trajectory
# -----------------------------------------------------------------------

def run_target_cooling_trajectory(
    rho0: np.ndarray,
    psi: np.ndarray,
    gamma: float,
    tmax: float,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Run the target-cooling dissipative tail and collect diagnostics.

    Parameters
    ----------
    rho0 : ndarray (d, d)
        Initial density matrix (output of the preparation circuit).
    psi : ndarray (d,)
        Target state vector (normalised).
    gamma : float > 0
        Uniform dissipation rate for all d-1 cooling operators.
    tmax : float > 0
        Total duration of the tail.
    steps : int > 0
        Number of time steps (the grid has steps + 1 points).

    Returns
    -------
    times      : ndarray (steps+1,)
    fidelities : ndarray (steps+1,)
    traces     : ndarray (steps+1,)
    purities   : ndarray (steps+1,)
    """
    psi = np.asarray(psi, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    psi_conj = psi.conj()

    times = np.linspace(0.0, tmax, steps + 1)
    dt = times[1] - times[0] if steps > 0 else 0.0

    fidelities = np.empty(steps + 1)
    traces = np.empty(steps + 1)
    purities = np.empty(steps + 1)

    rho = rho0.copy()
    for i in range(steps + 1):
        # Record diagnostics *before* stepping (so index 0 = initial state).
        fidelities[i] = fidelity_to_pure(rho, psi)
        traces[i] = trace_dm(rho)
        purities[i] = purity(rho)

        # Step (skip at the last index -- we only record there).
        if i < steps:
            rho = apply_target_cooling_step(rho, psi, psi_conj, gamma, dt)

    return times, fidelities, traces, purities
