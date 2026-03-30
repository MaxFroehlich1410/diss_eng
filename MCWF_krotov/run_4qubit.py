r"""4-qubit experiment: MCWF-Krotov for dissipative dark-state preparation.

Tests whether the MCWF-based Krotov method can find dissipative control
pulses to prepare the 4-node dark state
    |psi_tgt> = (1/2)(|e_1 g g g> + |g e_2 g g> + |g g e_3 g> + |g g g e_4>)
starting from |e_1 g g g>, in the cascade network model from
Goerz & Jacobs, arXiv:1801.04382v2.

We compare:
  1. Full density-matrix Krotov (reference)
  2. Independent-trajectory Krotov (M = 1, 2, 4)
  3. Cross-trajectory Krotov (M = 2, 4)

Run:
    python MCWF_krotov/run_4qubit.py
"""

from __future__ import annotations

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from MCWF_krotov import network_model, krotov, utils, mcwf


def run_experiment():
    print("=" * 70)
    print("  4-Qubit MCWF-Krotov Dark-State Preparation Experiment")
    print("=" * 70)

    N = 4
    d = network_model.hilbert_dim(N)
    print(f"\nNetwork: {N} nodes, Hilbert space dim = {d}")

    psi0 = network_model.initial_state(N)
    psi_tgt = network_model.dark_state_target(N)
    H_drift = network_model.build_drift_hamiltonian(N)
    H_ctrls = network_model.build_control_hamiltonians(N)
    L_op = network_model.build_network_lindblad_op(N)
    lindblad_ops = [L_op]

    T = 15.0
    nt = 80
    times = np.linspace(0, T, nt)
    n_iter = 100
    lambda_a = 0.001

    print(f"T = {T}, nt = {nt}, iterations = {n_iter}, lambda = {lambda_a}")

    controls_guess = np.array([
        [utils.blackman_pulse(t, T, amplitude=100.0) for t in times]
        for _ in range(N)
    ])

    L_super = utils.build_liouvillian(lindblad_ops, hamiltonian=H_drift)

    # --- 1. Density-matrix Krotov ---
    print("\n--- Density-Matrix Krotov (reference) ---")
    t0 = time.time()
    result_dm = krotov.krotov_density_matrix(
        psi0, psi_tgt, H_drift, H_ctrls, lindblad_ops, times,
        controls_guess.copy(), n_iterations=n_iter, lambda_a=lambda_a,
        verbose=True,
    )
    dt_dm = time.time() - t0
    print(f"  Time: {dt_dm:.1f}s")
    print(f"  Final JT = {result_dm.errors[-1]:.6e}")

    # --- 2. Independent trajectories ---
    for M in [1, 2, 4]:
        print(f"\n--- Independent Trajectories (M={M}) ---")
        t0 = time.time()
        result_ind = krotov.krotov_independent(
            psi0, psi_tgt, H_drift, H_ctrls, lindblad_ops, times,
            controls_guess.copy(), n_trajectories=M, n_iterations=n_iter,
            lambda_a=lambda_a, seed=42, verbose=True,
            liouvillian=L_super,
        )
        dt_ind = time.time() - t0
        print(f"  Time: {dt_ind:.1f}s")
        print(f"  Final JT (exact) = {result_ind.errors[-1]:.6e}")
        print(f"  Final JT (traj)  = {result_ind.errors_functional[-1]:.6e}")

    # --- 3. Cross trajectories ---
    for M in [2, 4]:
        print(f"\n--- Cross Trajectories (M={M}) ---")
        t0 = time.time()
        result_cross = krotov.krotov_cross_trajectory(
            psi0, psi_tgt, H_drift, H_ctrls, lindblad_ops, times,
            controls_guess.copy(), n_trajectories=M, n_iterations=n_iter,
            lambda_a=lambda_a, seed=42, verbose=True,
            liouvillian=L_super,
        )
        dt_cross = time.time() - t0
        print(f"  Time: {dt_cross:.1f}s")
        print(f"  Final JT (exact) = {result_cross.errors[-1]:.6e}")
        print(f"  Final JT (traj)  = {result_cross.errors_functional[-1]:.6e}")

    # --- Evaluate the best result via exact propagation ---
    print("\n--- Final evaluation ---")
    best_controls = result_dm.controls
    rho0 = utils.pure_state_dm(psi0)

    rho = rho0.copy()
    for j in range(nt - 1):
        dt = times[j + 1] - times[j]
        H = H_drift.copy()
        for c in range(N):
            H += best_controls[c, j] * H_ctrls[c]
        L_step = utils.build_liouvillian(lindblad_ops, hamiltonian=H)
        rho = utils.evolve_density_matrix(rho, L_step, dt)

    F_final = utils.fidelity_to_pure(rho, psi_tgt)
    dark_cond = network_model.dark_state_condition(psi_tgt, L_op)
    print(f"  Fidelity to target: {F_final:.6f}")
    print(f"  Dark-state condition for target: {dark_cond:.2e}")
    print(f"  Tr(rho_final): {utils.trace(rho):.8f}")
    print(f"  Purity(rho_final): {utils.purity(rho):.8f}")

    info = utils.is_physical(rho)
    print(f"  rho is physical: {info['is_valid']}")

    print("\n" + "=" * 70)
    print("  Experiment complete.")
    print("=" * 70)

    return {
        "dm_errors": result_dm.errors,
        "dm_controls": result_dm.controls,
        "fidelity": F_final,
    }


if __name__ == "__main__":
    run_experiment()
