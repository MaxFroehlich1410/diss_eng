r"""Core Krotov optimisation loop for dissipative dynamics.

Algorithm
---------
Given:
* Initial state rho_0
* Target state rho_* = |psi_*><psi_*|
* Lindblad operators {L_k}_{k=1}^K with pre-computed dissipator superoperators {S_k}
* Time grid 0 = t_0 < t_1 < ... < t_{N_t} = T
* Reference controls u_k^{ref}(t_j) (initialised, then updated each iteration)
* Regularisation lambdas lambda_k > 0

Iteration n -> n+1:
    1. Backward propagation with OLD controls u^{(n)}:
       chi(T) = rho_*
       chi(t_j) = exp(L^dag[u^{(n)}(t_j)] dt) chi(t_{j+1})

    2. Sequential forward propagation with NEW controls u^{(n+1)}:
       rho^{(n+1)}(0) = rho_0
       For j = 0, ..., N_t - 1:
           u_k^{(n+1)}(t_j) = u_k^{ref}(t_j) + (1/lambda_k) Re Tr(chi_j^dag D[L_k](rho_j^{(n+1)}))
           u_k^{(n+1)}(t_j) = max(0, u_k^{(n+1)}(t_j))  [physical rates]
           rho^{(n+1)}(t_{j+1}) = exp(L[u^{(n+1)}(t_j)] dt) rho^{(n+1)}(t_j)

    3. Set reference controls: u^{ref} <- u^{(n+1)}

Monotonic convergence is guaranteed by the Krotov improvement theorem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .liouville import (
    dissipator_superop, fidelity_pure, pure_state_dm, trace_dm, is_physical,
)
from .propagation import (
    backward_propagation, forward_propagation_sequential, forward_propagation,
)


@dataclass
class KrotovResult:
    """Container for Krotov optimisation results."""
    n_iterations: int
    fidelities: list[float]
    infidelities: list[float]
    controls_history: list[np.ndarray]
    final_controls: np.ndarray
    final_states: list[np.ndarray]
    converged: bool
    spectral_gap: float | None = None
    steady_state_fidelity: float | None = None


@dataclass
class KrotovConfig:
    """Configuration for Krotov optimisation."""
    n_qubits: int = 4
    T: float = 5.0
    N_t: int = 50
    max_iter: int = 100
    lambda_reg: float = 1.0
    tol: float = 1e-6
    u_init: float = 0.1
    verbose: bool = True


def run_krotov(
    rho0: np.ndarray,
    psi_target: np.ndarray,
    lindblad_ops: list[np.ndarray],
    config: KrotovConfig,
    hamiltonian: np.ndarray | None = None,
) -> KrotovResult:
    r"""Run Krotov optimisation for dissipative steady-state engineering.

    Parameters
    ----------
    rho0 : ndarray (d, d)
        Initial density matrix.
    psi_target : ndarray (d,)
        Target pure state vector.
    lindblad_ops : list of ndarray (d, d)
        Lindblad operators {L_k}.
    config : KrotovConfig
        Optimisation configuration.
    hamiltonian : ndarray (d, d), optional
        Time-independent system Hamiltonian.

    Returns
    -------
    KrotovResult
    """
    d = rho0.shape[0]
    K = len(lindblad_ops)
    dt = config.T / config.N_t
    rho_target = pure_state_dm(psi_target)

    # Pre-compute dissipator superoperators
    if config.verbose:
        print(f"Pre-computing {K} dissipator superoperators (d={d})...")
    S_ops = [dissipator_superop(L) for L in lindblad_ops]

    # Hamiltonian superoperator (optional)
    H_superop = None
    if hamiltonian is not None:
        from .liouville import hamiltonian_superop
        H_superop = hamiltonian_superop(hamiltonian)

    # Regularisation
    lambda_reg = np.full(K, config.lambda_reg)

    # Initial controls: uniform small positive values
    controls = np.full((config.N_t, K), config.u_init)

    # Storage
    fidelities = []
    infidelities = []
    controls_history = []

    # Initial forward propagation to get initial fidelity
    states = forward_propagation(rho0, S_ops, controls, dt, H_superop)
    fid = fidelity_pure(states[-1], psi_target)
    fidelities.append(fid)
    infidelities.append(1.0 - fid)

    if config.verbose:
        print(f"Initial fidelity: {fid:.8f}")
        print(f"Starting Krotov optimisation (max_iter={config.max_iter})...")

    converged = False

    for iteration in range(config.max_iter):
        # 1. Backward propagation with current controls
        costates = backward_propagation(
            rho_target, S_ops, controls, dt, H_superop
        )

        # 2. Sequential forward propagation with Krotov update
        states_new, controls_new = forward_propagation_sequential(
            rho0, S_ops, costates, controls, lambda_reg, dt,
            lindblad_ops, H_superop
        )

        # 3. Compute fidelity
        fid_new = fidelity_pure(states_new[-1], psi_target)
        fidelities.append(fid_new)
        infidelities.append(1.0 - fid_new)
        controls_history.append(controls.copy())

        # 4. Update reference controls
        controls = controls_new
        states = states_new

        if config.verbose and (iteration % 10 == 0 or iteration < 5):
            print(f"  iter {iteration:4d}: F = {fid_new:.8f}, "
                  f"1-F = {1-fid_new:.2e}")

        # 5. Check convergence
        if len(fidelities) >= 2:
            delta = abs(fidelities[-1] - fidelities[-2])
            if delta < config.tol and fidelities[-1] > 0.99:
                converged = True
                if config.verbose:
                    print(f"  Converged at iteration {iteration} "
                          f"(delta={delta:.2e}, F={fid_new:.8f})")
                break

    # Final analysis
    controls_history.append(controls.copy())

    result = KrotovResult(
        n_iterations=len(fidelities) - 1,
        fidelities=fidelities,
        infidelities=infidelities,
        controls_history=controls_history,
        final_controls=controls,
        final_states=states,
        converged=converged,
    )

    return result


def analyse_steady_state(
    psi_target: np.ndarray,
    lindblad_ops: list[np.ndarray],
    controls: np.ndarray,
    T_analyse: float = 50.0,
) -> dict:
    r"""Analyse the time-independent generator derived from optimised controls.

    Extracts time-averaged rates from the optimised control pulse and
    checks whether the resulting time-independent Lindblad generator
    has the target as unique steady state.

    Parameters
    ----------
    psi_target : ndarray (d,)
        Target state.
    lindblad_ops : list of ndarray (d, d)
        Lindblad operators.
    controls : ndarray (N_t, K)
        Optimised control amplitudes.
    T_analyse : float
        Long evolution time for steady-state check.

    Returns
    -------
    dict with keys:
        'avg_rates', 'spectral_gap', 'steady_state_fidelity',
        'eigenvalues_real', 'is_unique_steady_state'
    """
    from scipy.linalg import expm

    d = lindblad_ops[0].shape[0]
    K = len(lindblad_ops)
    rho_target = pure_state_dm(psi_target)

    # Time-averaged rates (average over all time steps)
    avg_rates = np.mean(controls, axis=0)

    # Build time-independent Liouvillian
    S_ops = [dissipator_superop(L) for L in lindblad_ops]
    from .liouville import build_liouvillian_from_amplitudes
    L_avg = build_liouvillian_from_amplitudes(S_ops, avg_rates)

    # Eigenvalue analysis
    eigenvalues = np.linalg.eigvals(L_avg)
    # Sort by real part (descending)
    idx = np.argsort(-np.real(eigenvalues))
    eigenvalues = eigenvalues[idx]
    reals = np.real(eigenvalues)

    # Steady state: eigenvalue closest to 0
    # Spectral gap: difference between 0 and second-largest real part
    zero_idx = np.argmin(np.abs(reals))
    sorted_reals = np.sort(reals)[::-1]

    # The largest eigenvalue should be ~0 (trace preservation)
    spectral_gap = -sorted_reals[1] if len(sorted_reals) > 1 else 0.0

    # Evolve maximally mixed state for long time to find steady state
    from .liouville import vectorize, unvectorize
    rho_mixed = np.eye(d, dtype=complex) / d
    rho_vec = vectorize(rho_mixed)
    propagator = expm(L_avg * T_analyse)
    rho_ss_vec = propagator @ rho_vec
    rho_ss = unvectorize(rho_ss_vec, d)

    ss_fidelity = fidelity_pure(rho_ss, psi_target)

    # Check multiple initial states
    rng = np.random.default_rng(123)
    fids_from_random = []
    for _ in range(5):
        psi_rand = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        psi_rand /= np.linalg.norm(psi_rand)
        rho_rand = np.outer(psi_rand, psi_rand.conj())
        rho_rand_vec = vectorize(rho_rand)
        rho_rand_ss = unvectorize(propagator @ rho_rand_vec, d)
        fids_from_random.append(fidelity_pure(rho_rand_ss, psi_target))

    is_unique = (ss_fidelity > 0.99 and
                 all(f > 0.99 for f in fids_from_random) and
                 spectral_gap > 1e-4)

    return {
        "avg_rates": avg_rates,
        "spectral_gap": spectral_gap,
        "steady_state_fidelity": ss_fidelity,
        "random_init_fidelities": fids_from_random,
        "eigenvalues_real": sorted_reals[:10],
        "is_unique_steady_state": is_unique,
    }
