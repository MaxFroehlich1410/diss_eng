r"""Krotov optimal control via quantum trajectories.

Implements the trajectory-based Krotov method from Goerz & Jacobs,
arXiv:1801.04382v2, Section 2.3.

Two variants
------------
1. Independent trajectories  (Eq. 13):
       Delta u_i(t) = S(t)/(M lambda) sum_k Im<chi_k^(0)|H_i|psi_k^(1)>

2. Cross-trajectory  (Eq. 15):
       Delta u_i(t) = S(t)/(M^2 lambda) sum_{k,k'}
           Im[ <xi_k^(0)|H_i|psi_k'^(1)> <psi_k'^(1)|xi_k^(0)> ]

Boundary conditions
-------------------
Independent:  chi_k(T) = tau_k |psi_tgt>     with tau_k = <psi_k(T)|psi_tgt>
Cross:        xi_k(T)  = |psi_tgt>            for all k

The backward propagation uses H_eff^dag (adjoint no-jump evolution).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
from scipy.linalg import expm

from .mcwf import (
    build_effective_hamiltonian,
    propagate_mcwf_trajectory,
    propagate_backward_no_jump,
    density_matrix_from_trajectories,
)
from .utils import (
    fidelity_to_pure, pure_state_dm, overlap, shape_function,
    build_liouvillian, evolve_density_matrix,
)


@dataclass
class KrotovResult:
    """Result container for a Krotov optimization run."""
    controls: np.ndarray           # (n_controls, nt)
    errors: list[float]            # JT per iteration (evaluated via exact DM)
    errors_functional: list[float] # JT as seen by the optimizer (trajectory-based)
    final_states: list[list[np.ndarray]] | None = None


def _exact_error_piecewise(
    psi0: np.ndarray,
    psi_tgt: np.ndarray,
    H_drift: np.ndarray,
    H_ctrls: list[np.ndarray],
    lindblad_ops: list[np.ndarray],
    controls: np.ndarray,
    times: np.ndarray,
) -> float:
    """Evaluate JT exactly by piecewise density-matrix propagation."""
    rho = pure_state_dm(psi0)
    for j in range(len(times) - 1):
        dt = times[j + 1] - times[j]
        H = H_drift.copy()
        for c in range(controls.shape[0]):
            H += controls[c, j] * H_ctrls[c]
        L_super = build_liouvillian(lindblad_ops, hamiltonian=H)
        rho = evolve_density_matrix(rho, L_super, dt)
    return 1.0 - fidelity_to_pure(rho, psi_tgt)


def _forward_propagate_trajectory(
    psi0: np.ndarray,
    H_drift: np.ndarray,
    H_ctrls: list[np.ndarray],
    controls: np.ndarray,
    lindblad_ops: list[np.ndarray],
    times: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[tuple[float, int]]]:
    """Forward-propagate one MCWF trajectory with given controls."""
    nt = len(times)

    def H_func(t):
        idx = np.searchsorted(times, t, side="right") - 1
        idx = np.clip(idx, 0, nt - 1)
        H = H_drift.copy()
        for c in range(controls.shape[0]):
            H += controls[c, idx] * H_ctrls[c]
        return H

    return propagate_mcwf_trajectory(psi0, H_func, lindblad_ops, times, rng)


def _backward_propagate(
    chi_T: np.ndarray,
    H_drift: np.ndarray,
    H_ctrls: list[np.ndarray],
    controls: np.ndarray,
    lindblad_ops: list[np.ndarray],
    times: np.ndarray,
) -> list[np.ndarray]:
    """Backward-propagate chi under H_eff^dag with given controls (no jumps)."""
    nt = len(times)

    def H_func(t):
        idx = np.searchsorted(times, t, side="right") - 1
        idx = np.clip(idx, 0, nt - 1)
        H = H_drift.copy()
        for c in range(controls.shape[0]):
            H += controls[c, idx] * H_ctrls[c]
        return H

    return propagate_backward_no_jump(chi_T, H_func, lindblad_ops, times)


def _forward_propagate_updated(
    psi0: np.ndarray,
    H_drift: np.ndarray,
    H_ctrls: list[np.ndarray],
    controls_old: np.ndarray,
    controls_new: np.ndarray,
    lindblad_ops: list[np.ndarray],
    times: np.ndarray,
    rng: np.random.Generator,
    jump_record: list[tuple[float, int]] | None = None,
) -> list[np.ndarray]:
    """Forward-propagate with updated controls.

    For Krotov, the forward propagation uses the *new* (updated) controls.
    The control at time t_j is updated sequentially as we go forward.

    We replicate the same jump pattern from the guess propagation for
    consistency within an iteration.
    """
    psi = psi0.copy()
    d = len(psi)
    states = [psi.copy()]

    jump_times = set()
    jump_ops = {}
    if jump_record:
        for (jt, jl) in jump_record:
            jump_times.add(jt)
            jump_ops[jt] = jl

    for step in range(len(times) - 1):
        t = times[step]
        dt = times[step + 1] - times[step]

        H = H_drift.copy()
        for c in range(controls_new.shape[0]):
            H += controls_new[c, step] * H_ctrls[c]

        H_eff = build_effective_hamiltonian(H, lindblad_ops)
        U_eff = expm(-1j * H_eff * dt)
        psi_new = U_eff @ psi

        t_next = times[step + 1]
        jumped = False
        for jt in jump_times:
            if abs(jt - t_next) < dt * 0.5:
                l_idx = jump_ops[jt]
                if 0 <= l_idx < len(lindblad_ops):
                    psi_jumped = lindblad_ops[l_idx] @ psi
                    norm_j = np.linalg.norm(psi_jumped)
                    if norm_j > 1e-15:
                        psi = psi_jumped / norm_j
                        jumped = True
                break

        if not jumped:
            norm = np.linalg.norm(psi_new)
            psi = psi_new / max(norm, 1e-30)

        states.append(psi.copy())

    return states


# ---------------------------------------------------------------------------
# Independent-trajectory Krotov  (Eq. 13)
# ---------------------------------------------------------------------------

def krotov_independent(
    psi0: np.ndarray,
    psi_target: np.ndarray,
    H_drift: np.ndarray,
    H_ctrls: list[np.ndarray],
    lindblad_ops: list[np.ndarray],
    times: np.ndarray,
    controls_guess: np.ndarray,
    n_trajectories: int = 2,
    n_iterations: int = 100,
    lambda_a: float = 1.0,
    shape_func=None,
    seed: int = 42,
    store_final_states: bool = False,
    liouvillian=None,
    verbose: bool = True,
) -> KrotovResult:
    r"""Krotov optimization using M independent trajectories (Eq. 13).

    Parameters
    ----------
    psi0 : ndarray (d,)
        Initial state.
    psi_target : ndarray (d,)
        Target state.
    H_drift : ndarray (d, d)
        Time-independent drift Hamiltonian.
    H_ctrls : list of ndarray (d, d)
        Control Hamiltonians H_i (one per control field).
    lindblad_ops : list of ndarray (d, d)
        Lindblad jump operators.
    times : ndarray (nt,)
        Time grid.
    controls_guess : ndarray (n_controls, nt)
        Initial guess for controls.
    n_trajectories : int
        Number of independent trajectories M.
    n_iterations : int
        Number of Krotov iterations.
    lambda_a : float
        Regularisation parameter (larger = smaller updates).
    shape_func : callable(t, T) -> float, optional
        Shape function S(t).  Default: no shaping (S=1).
    seed : int
        Random seed.
    store_final_states : bool
        Whether to store trajectory states from the last iteration.
    liouvillian : ndarray, optional
        Full Liouvillian for exact error evaluation.  If None, error is
        estimated from trajectories.
    verbose : bool

    Returns
    -------
    KrotovResult
    """
    rng = np.random.default_rng(seed)
    d = len(psi0)
    n_ctrl = controls_guess.shape[0]
    nt = len(times)
    T = times[-1]

    controls = controls_guess.copy()
    errors = []
    errors_func = []

    if shape_func is None:
        S_vals = np.ones(nt)
    else:
        S_vals = np.array([shape_func(t, T) for t in times])

    psi_tgt = psi_target / np.linalg.norm(psi_target)

    for iteration in range(n_iterations):
        # --- Step 1: Forward-propagate M trajectories with guess controls ---
        fwd_trajs = []
        jump_records = []
        for k in range(n_trajectories):
            states_k, jumps_k = _forward_propagate_trajectory(
                psi0, H_drift, H_ctrls, controls, lindblad_ops, times, rng,
            )
            fwd_trajs.append(states_k)
            jump_records.append(jumps_k)

        # --- Evaluate JT (Eq. 12) ---
        tau_list = []
        for k in range(n_trajectories):
            tau_k = np.vdot(psi_tgt, fwd_trajs[k][-1])
            tau_list.append(tau_k)
        JT_traj = 1.0 - np.mean([abs(tau) ** 2 for tau in tau_list])
        errors_func.append(float(np.real(JT_traj)))

        # Exact error via piecewise density-matrix propagation
        if liouvillian is not None:
            JT_exact = _exact_error_piecewise(
                psi0, psi_tgt, H_drift, H_ctrls, lindblad_ops, controls, times,
            )
            errors.append(float(JT_exact))
        else:
            rho_T = density_matrix_from_trajectories(fwd_trajs, -1)
            JT_approx = 1.0 - fidelity_to_pure(rho_T, psi_tgt)
            errors.append(float(JT_approx))

        if verbose and iteration % max(1, n_iterations // 20) == 0:
            print(f"  iter {iteration:4d}  JT = {errors[-1]:.6e}")

        # --- Step 2: Backward-propagate chi_k with boundary (Eq. 14) ---
        bwd_trajs = []
        for k in range(n_trajectories):
            chi_T = tau_list[k] * psi_tgt
            chi_states = _backward_propagate(
                chi_T, H_drift, H_ctrls, controls, lindblad_ops, times,
            )
            bwd_trajs.append(chi_states)

        # --- Step 3: Sequential forward prop with updated controls (Eq. 13) ---
        controls_new = controls.copy()

        fwd_trajs_new = []
        for k in range(n_trajectories):
            fwd_trajs_new.append([psi0.copy()])

        for j in range(nt - 1):
            for c in range(n_ctrl):
                delta_u = 0.0
                for k in range(n_trajectories):
                    psi_k = fwd_trajs_new[k][-1]
                    chi_k = bwd_trajs[k][j]
                    delta_u += np.imag(np.vdot(chi_k, H_ctrls[c] @ psi_k))
                delta_u *= S_vals[j] / (n_trajectories * lambda_a)
                controls_new[c, j] = controls[c, j] + delta_u

            t = times[j]
            dt = times[j + 1] - times[j]
            for k in range(n_trajectories):
                H = H_drift.copy()
                for c in range(n_ctrl):
                    H += controls_new[c, j] * H_ctrls[c]
                H_eff = build_effective_hamiltonian(H, lindblad_ops)
                U_eff = expm(-1j * H_eff * dt)
                psi_k = fwd_trajs_new[k][-1]
                psi_new = U_eff @ psi_k

                jumped = False
                for (jt, jl) in jump_records[k]:
                    if abs(jt - times[j + 1]) < dt * 0.5:
                        if 0 <= jl < len(lindblad_ops):
                            psi_jumped = lindblad_ops[jl] @ psi_k
                            nj = np.linalg.norm(psi_jumped)
                            if nj > 1e-15:
                                psi_new = psi_jumped / nj
                                jumped = True
                        break

                if not jumped:
                    norm = np.linalg.norm(psi_new)
                    psi_new = psi_new / max(norm, 1e-30)

                fwd_trajs_new[k].append(psi_new.copy())

        controls = controls_new

    final = None
    if store_final_states:
        final = fwd_trajs_new

    return KrotovResult(
        controls=controls,
        errors=errors,
        errors_functional=errors_func,
        final_states=final,
    )


# ---------------------------------------------------------------------------
# Cross-trajectory Krotov  (Eq. 15)
# ---------------------------------------------------------------------------

def krotov_cross_trajectory(
    psi0: np.ndarray,
    psi_target: np.ndarray,
    H_drift: np.ndarray,
    H_ctrls: list[np.ndarray],
    lindblad_ops: list[np.ndarray],
    times: np.ndarray,
    controls_guess: np.ndarray,
    n_trajectories: int = 2,
    n_iterations: int = 100,
    lambda_a: float = 1.0,
    shape_func=None,
    seed: int = 42,
    store_final_states: bool = False,
    liouvillian=None,
    verbose: bool = True,
) -> KrotovResult:
    r"""Krotov optimization using M cross-referenced trajectories (Eq. 15).

    The cross-trajectory update uses:
       Delta u_i(t) = S(t)/(M^2 lambda) sum_{k,k'}
           Im[ <xi_k|H_i|psi_k'> <psi_k'|xi_k> ]

    with xi_k(T) = |psi_tgt> for all k.
    """
    rng = np.random.default_rng(seed)
    d = len(psi0)
    n_ctrl = controls_guess.shape[0]
    nt = len(times)
    T = times[-1]

    controls = controls_guess.copy()
    errors = []
    errors_func = []

    if shape_func is None:
        S_vals = np.ones(nt)
    else:
        S_vals = np.array([shape_func(t, T) for t in times])

    psi_tgt = psi_target / np.linalg.norm(psi_target)

    for iteration in range(n_iterations):
        # --- Forward-propagate M trajectories ---
        fwd_trajs = []
        jump_records = []
        for k in range(n_trajectories):
            states_k, jumps_k = _forward_propagate_trajectory(
                psi0, H_drift, H_ctrls, controls, lindblad_ops, times, rng,
            )
            fwd_trajs.append(states_k)
            jump_records.append(jumps_k)

        # --- JT ---
        tau_list = [np.vdot(psi_tgt, fwd_trajs[k][-1]) for k in range(n_trajectories)]
        JT_traj = 1.0 - np.mean([abs(tau) ** 2 for tau in tau_list])
        errors_func.append(float(np.real(JT_traj)))

        if liouvillian is not None:
            JT_exact = _exact_error_piecewise(
                psi0, psi_tgt, H_drift, H_ctrls, lindblad_ops, controls, times,
            )
            errors.append(float(JT_exact))
        else:
            rho_T = density_matrix_from_trajectories(fwd_trajs, -1)
            JT_approx = 1.0 - fidelity_to_pure(rho_T, psi_tgt)
            errors.append(float(JT_approx))

        if verbose and iteration % max(1, n_iterations // 20) == 0:
            print(f"  iter {iteration:4d}  JT = {errors[-1]:.6e}")

        # --- Backward: xi_k(T) = |psi_tgt> for all k ---
        bwd_trajs = []
        for k in range(n_trajectories):
            xi_states = _backward_propagate(
                psi_tgt.copy(), H_drift, H_ctrls, controls, lindblad_ops, times,
            )
            bwd_trajs.append(xi_states)

        # --- Sequential forward with cross-trajectory update (Eq. 15) ---
        controls_new = controls.copy()
        M = n_trajectories
        M2 = M * M

        fwd_trajs_new = []
        for k in range(n_trajectories):
            fwd_trajs_new.append([psi0.copy()])

        for j in range(nt - 1):
            for c in range(n_ctrl):
                delta_u = 0.0
                for k in range(M):
                    xi_k = bwd_trajs[k][j]
                    for kp in range(M):
                        psi_kp = fwd_trajs_new[kp][-1]
                        bracket1 = np.vdot(xi_k, H_ctrls[c] @ psi_kp)
                        bracket2 = np.vdot(psi_kp, xi_k)
                        delta_u += np.imag(bracket1 * bracket2)
                delta_u *= S_vals[j] / (M2 * lambda_a)
                controls_new[c, j] = controls[c, j] + delta_u

            t = times[j]
            dt = times[j + 1] - times[j]
            for k in range(n_trajectories):
                H = H_drift.copy()
                for c in range(n_ctrl):
                    H += controls_new[c, j] * H_ctrls[c]
                H_eff = build_effective_hamiltonian(H, lindblad_ops)
                U_eff = expm(-1j * H_eff * dt)
                psi_k = fwd_trajs_new[k][-1]
                psi_new = U_eff @ psi_k

                jumped = False
                for (jt, jl) in jump_records[k]:
                    if abs(jt - times[j + 1]) < dt * 0.5:
                        if 0 <= jl < len(lindblad_ops):
                            psi_jumped = lindblad_ops[jl] @ psi_k
                            nj = np.linalg.norm(psi_jumped)
                            if nj > 1e-15:
                                psi_new = psi_jumped / nj
                                jumped = True
                        break

                if not jumped:
                    norm = np.linalg.norm(psi_new)
                    psi_new = psi_new / max(norm, 1e-30)

                fwd_trajs_new[k].append(psi_new.copy())

        controls = controls_new

    final = None
    if store_final_states:
        final = fwd_trajs_new

    return KrotovResult(
        controls=controls,
        errors=errors,
        errors_functional=errors_func,
        final_states=final,
    )


# ---------------------------------------------------------------------------
# Full density-matrix Krotov (for reference / validation, Eq. 10)
# ---------------------------------------------------------------------------

def krotov_density_matrix(
    psi0: np.ndarray,
    psi_target: np.ndarray,
    H_drift: np.ndarray,
    H_ctrls: list[np.ndarray],
    lindblad_ops: list[np.ndarray],
    times: np.ndarray,
    controls_guess: np.ndarray,
    n_iterations: int = 100,
    lambda_a: float = 1.0,
    shape_func=None,
    verbose: bool = True,
) -> KrotovResult:
    r"""Reference Krotov optimization using the full density matrix (Eq. 10).

    Update:
       Delta u_i(t) = S(t)/lambda Im Tr[ P^(0)(t)^dag  [H_i, rho^(1)(t)] ]
    """
    from .utils import build_liouvillian, evolve_density_matrix

    d = len(psi0)
    n_ctrl = controls_guess.shape[0]
    nt = len(times)
    T = times[-1]

    controls = controls_guess.copy()
    errors = []
    psi_tgt = psi_target / np.linalg.norm(psi_target)
    P_tgt = pure_state_dm(psi_tgt)

    if shape_func is None:
        S_vals = np.ones(nt)
    else:
        S_vals = np.array([shape_func(t, T) for t in times])

    for iteration in range(n_iterations):
        # --- Forward-propagate rho with guess controls ---
        rho_fwd = [pure_state_dm(psi0)]
        for j in range(nt - 1):
            dt = times[j + 1] - times[j]
            H = H_drift.copy()
            for c in range(n_ctrl):
                H += controls[c, j] * H_ctrls[c]
            L_super = build_liouvillian(lindblad_ops, hamiltonian=H)
            rho_next = evolve_density_matrix(rho_fwd[-1], L_super, dt)
            rho_fwd.append(rho_next)

        JT = 1.0 - fidelity_to_pure(rho_fwd[-1], psi_tgt)
        errors.append(float(JT))

        if verbose and iteration % max(1, n_iterations // 20) == 0:
            print(f"  iter {iteration:4d}  JT = {JT:.6e}")

        # --- Backward-propagate P with conjugate Lindbladian ---
        P_bwd = [None] * nt
        P_bwd[-1] = P_tgt.copy()
        for j in range(nt - 1, 0, -1):
            dt = times[j] - times[j - 1]
            H = H_drift.copy()
            for c in range(n_ctrl):
                H += controls[c, j] * H_ctrls[c]
            L_super = build_liouvillian(lindblad_ops, hamiltonian=H)
            L_adj = L_super.conj().T
            P_bwd[j - 1] = evolve_density_matrix(P_bwd[j], L_adj, dt)

        # --- Sequential forward with updated controls ---
        controls_new = controls.copy()
        rho_new = [pure_state_dm(psi0)]

        for j in range(nt - 1):
            for c in range(n_ctrl):
                comm = H_ctrls[c] @ rho_new[-1] - rho_new[-1] @ H_ctrls[c]
                val = np.trace(P_bwd[j].conj().T @ comm)
                delta_u = S_vals[j] / lambda_a * np.imag(val)
                controls_new[c, j] = controls[c, j] + delta_u

            dt = times[j + 1] - times[j]
            H = H_drift.copy()
            for c in range(n_ctrl):
                H += controls_new[c, j] * H_ctrls[c]
            L_super = build_liouvillian(lindblad_ops, hamiltonian=H)
            rho_next = evolve_density_matrix(rho_new[-1], L_super, dt)
            rho_new.append(rho_next)

        controls = controls_new

    return KrotovResult(
        controls=controls,
        errors=errors,
        errors_functional=errors,
    )
