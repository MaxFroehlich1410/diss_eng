r"""10-qubit hydrogen wavefunction state preparation with MCWF-Krotov.

Scales the MCWF-Krotov method to 10 qubits (d=1024) with adapted
hyperparameters:

  * lambda scaled down by ~sqrt(d_10/d_4) to compensate for the
    1/sqrt(d) decay of Krotov update magnitudes in larger Hilbert spaces
  * More circuit layers for sufficient expressivity relative to d
  * Rank-1 teacher operators (Rank1Op) for O(d) memory/compute
  * Sparse Lindblad operators for hardware channels

Configs:
  1. Baseline: truncated circuit
  2. Teacher tail (DM, analytical) — upper bound
  3. Krotov-only (no dissipation)
  4. MCWF-Krotov + reduced teacher (top-k rank-1 ops)
  5. MCWF-Krotov + amplitude damping (sparse, fixed rate)
  6. MCWF-Krotov + AD + dephasing + ZZ chain (sparse)
  7. MCWF-Krotov + AD + dephasing + ZZ all-to-all (sparse)

Run:
    python MCWF_krotov/run_hydrogen_10q.py
"""

from __future__ import annotations

import os
import sys
import time
import json
import numpy as np
from itertools import combinations

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BASE_DIR)

from MCWF_krotov import gate_circuit_krotov as gck
from MCWF_krotov import utils


def build_hydrogen_target(n_qubits, n=2, l=1, m=0):
    d = 2 ** n_qubits
    grid_side = int(np.sqrt(d))
    if grid_side * grid_side != d:
        grid_side = d
    try:
        sys.path.insert(0, os.path.join(BASE_DIR, "experiments", "hydrogen_qalchemy"))
        from hydrogen_wavefunction import compute_psi_xz_slice
        _, _, psi_grid, _ = compute_psi_xz_slice(
            n, l, m, grid_points=grid_side, extent_a_mu=15.0)
        psi = psi_grid.ravel()[:d].astype(complex)
    except Exception:
        rng = np.random.default_rng(hash((n, l, m)) % 2**32)
        psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    norm = np.linalg.norm(psi)
    if norm < 1e-15:
        psi = np.ones(d, dtype=complex); norm = np.linalg.norm(psi)
    return psi / norm


def build_truncated_circuit_state(psi_target, n_qubits, gate_fraction=0.5):
    from scipy.linalg import expm
    d = 2 ** n_qubits
    orth = np.linalg.svd(psi_target.conj().reshape(1, -1),
                         full_matrices=True)[2][1:]
    U = np.column_stack([psi_target, orth.T])
    eigvals, eigvecs = np.linalg.eig(U)
    log_U = sum(np.log(eigvals[i]) * np.outer(eigvecs[:, i],
                eigvecs[:, i].conj()) for i in range(d))
    U_trunc = expm(gate_fraction * log_U)
    e0 = np.zeros(d, dtype=complex); e0[0] = 1.0
    psi_init = U_trunc @ e0
    return psi_init / np.linalg.norm(psi_init)


def _fresh_ansatz(n_qubits, n_layers, seed=42, init_scale=0.05):
    """Build ansatz with all gates near zero (circuit ≈ identity)."""
    layers = gck.build_hardware_efficient_ansatz(
        n_qubits, n_layers, full_generators=(n_qubits <= 6))
    rng = np.random.default_rng(seed)
    for layer in layers:
        for gate in layer:
            gate.theta = rng.uniform(-init_scale, init_scale)
    return layers


def run_experiment():
    print("=" * 72)
    print("  10-Qubit Hydrogen MCWF-Krotov Experiment (corrected)")
    print("=" * 72)

    n_qubits = 10
    d = 2 ** n_qubits
    n_layers = 6
    n_iter_no_diss = 200
    n_iter_diss = 100
    n_traj = 8
    diss_dt = 0.1

    lam = 0.3
    max_step = 0.05

    gamma_tc = 0.3
    n_teacher_ops = 10
    gamma_ad = 0.05
    gamma_deph = 0.01
    gamma_zz = 0.01

    n_gates_per_layer = 3 * n_qubits - 1
    n_total_gates = n_layers * n_gates_per_layer

    print(f"\nSystem: {n_qubits} qubits, d = {d}")
    print(f"Ansatz: {n_layers} layers, {n_total_gates} gates")
    print(f"Krotov: {n_iter_no_diss}/{n_iter_diss} iters, {n_traj} traj, "
          f"lambda={lam}, max_step={max_step}")
    print(f"Teacher: {n_teacher_ops} rank-1 ops, gamma={gamma_tc}")
    print(f"Hardware: gamma_ad={gamma_ad}, gamma_deph={gamma_deph}, "
          f"gamma_zz={gamma_zz}")

    # --- Target ---
    print("\nBuilding hydrogen target ...")
    psi_target = build_hydrogen_target(n_qubits)

    print("Building truncated circuit state ...")
    t0 = time.time()
    psi_trunc = build_truncated_circuit_state(psi_target, n_qubits, 0.5)
    F_trunc = abs(np.vdot(psi_target, psi_trunc)) ** 2
    print(f"  Built in {time.time()-t0:.1f}s, baseline F = {F_trunc:.6f}")
    rho_trunc = utils.pure_state_dm(psi_trunc)

    results = {
        "n_qubits": n_qubits, "target": "H_2p",
        "n_layers": n_layers,
        "n_iter_no_diss": n_iter_no_diss,
        "n_iter_diss": n_iter_diss,
        "n_traj": n_traj,
        "lambda": lam, "dissipation_dt": diss_dt,
        "truncated_fidelity": float(F_trunc),
    }

    # --- Teacher tail (DM, analytical) ---
    print(f"\n--- Teacher tail (DM, ideal) ---")
    rho = rho_trunc.copy()
    psi_c = psi_target.conj()
    teacher_fids = [utils.fidelity_to_pure(rho, psi_target)]
    dt_t = 5.0 / 50
    for _ in range(50):
        v = rho @ psi_target
        a0 = np.real(psi_c @ v)
        g = 1.0 * dt_t
        eg, eg2 = np.exp(-g), np.exp(-g / 2)
        al = (1 - eg) + 2 * a0 * (eg - eg2)
        be = eg2 - eg
        rho = eg * rho + al * np.outer(psi_target, psi_c) \
            + be * np.outer(psi_target, v.conj()) + be * np.outer(v, psi_c)
        teacher_fids.append(utils.fidelity_to_pure(rho, psi_target))
    print(f"  Fidelity: {teacher_fids[-1]:.6f}")
    results["teacher_final_fidelity"] = float(teacher_fids[-1])

    # --- Krotov-only (no dissipation) ---
    import gc
    gc.collect()
    print(f"\n--- Krotov-only (no dissipation) ---")
    from scipy import sparse
    null_op = [sparse.csr_matrix((d, d), dtype=complex)]
    layers_k = _fresh_ansatz(n_qubits, n_layers, seed=42)
    t0 = time.time()
    res_k = gck.krotov_gate_circuit(
        psi_trunc, psi_target, layers_k, null_op,
        dissipation_dt=0.0, n_trajectories=n_traj,
        n_iterations=n_iter_no_diss, lambda_a=lam, seed=42, verbose=True,
        max_gate_step=max_step)
    dm_k = gck.evaluate_circuit_dm(
        psi_trunc, psi_target, layers_k, null_op, dissipation_dt=0.0)
    elapsed_k = time.time() - t0
    print(f"  Time: {elapsed_k:.1f}s, F(DM)={dm_k['fidelity']:.6f}")
    results["krotov_only_fidelity"] = float(dm_k["fidelity"])
    results["krotov_only_purity"] = float(dm_k["purity"])
    results["krotov_only_time"] = elapsed_k
    results["krotov_only_fidelities"] = [float(f) for f in res_k.fidelities]

    # --- MCWF-Krotov with reduced teacher (rank-1 ops) ---
    gc.collect()
    print(f"\n{'='*72}")
    print(f"--- MCWF-Krotov: Reduced teacher ({n_teacher_ops} rank-1 ops) ---")
    t0 = time.time()
    teacher_ops = utils.target_cooling_operators_rank1(
        psi_target, gamma=gamma_tc,
        max_ops=n_teacher_ops, ref_state=psi_trunc)
    print(f"  Built {len(teacher_ops)} Rank1Op operators in {time.time()-t0:.2f}s")

    layers_tc = _fresh_ansatz(n_qubits, n_layers, seed=42)
    t0 = time.time()
    res_tc = gck.krotov_gate_circuit(
        psi_trunc, psi_target, layers_tc, teacher_ops,
        dissipation_dt=diss_dt, n_trajectories=n_traj,
        n_iterations=n_iter_diss, lambda_a=lam, seed=42, verbose=True,
        max_gate_step=max_step)
    dm_tc = gck.evaluate_circuit_dm(
        psi_trunc, psi_target, layers_tc, teacher_ops,
        dissipation_dt=diss_dt)
    elapsed_tc = time.time() - t0
    print(f"  Time: {elapsed_tc:.1f}s, F(DM)={dm_tc['fidelity']:.6f}, "
          f"Purity={dm_tc['purity']:.4f}")
    results["teacher_rank1_fidelity"] = float(dm_tc["fidelity"])
    results["teacher_rank1_purity"] = float(dm_tc["purity"])
    results["teacher_rank1_time"] = elapsed_tc
    results["teacher_rank1_n_ops"] = len(teacher_ops)
    results["teacher_rank1_fidelities"] = [float(f) for f in res_tc.fidelities]

    # --- Hardware configs ---
    chain_edges = [(q, q + 1) for q in range(n_qubits - 1)]
    all2all_edges = list(combinations(range(n_qubits), 2))

    hw_configs = [
        ("amp_damping", "1q amplitude damping", lambda: (
            [np.sqrt(gamma_ad) * L
             for L in utils.amplitude_damping_operators_sparse(n_qubits)],
            n_qubits)),
        ("ad_deph_zz_chain", "AD+deph+ZZ (chain)", lambda: (
            [np.sqrt(gamma_ad) * L
             for L in utils.amplitude_damping_operators_sparse(n_qubits)]
            + [np.sqrt(gamma_deph) * L
               for L in utils.dephasing_operators_sparse(n_qubits)]
            + [np.sqrt(gamma_zz) * L
               for L in utils.zz_dephasing_operators_sparse(n_qubits,
                                                            chain_edges)],
            n_qubits + n_qubits + len(chain_edges))),
        ("ad_deph_zz_all2all", "AD+deph+ZZ (all-to-all)", lambda: (
            [np.sqrt(gamma_ad) * L
             for L in utils.amplitude_damping_operators_sparse(n_qubits)]
            + [np.sqrt(gamma_deph) * L
               for L in utils.dephasing_operators_sparse(n_qubits)]
            + [np.sqrt(gamma_zz) * L
               for L in utils.zz_dephasing_operators_sparse(n_qubits,
                                                            all2all_edges)],
            n_qubits + n_qubits + len(all2all_edges))),
    ]

    for cfg_name, cfg_label, build_fn in hw_configs:
        gc.collect()
        print(f"\n{'='*72}")
        print(f"--- MCWF-Krotov: {cfg_label} ---")
        lindblad_ops, n_ops = build_fn()
        print(f"  {n_ops} Lindblad operators")

        layers = _fresh_ansatz(n_qubits, n_layers, seed=42)
        t0 = time.time()
        result = gck.krotov_gate_circuit(
            psi_trunc, psi_target, layers, lindblad_ops,
            dissipation_dt=diss_dt, n_trajectories=n_traj,
            n_iterations=n_iter_diss, lambda_a=lam, seed=42, verbose=True,
            max_gate_step=max_step)
        dm_eval = gck.evaluate_circuit_dm(
            psi_trunc, psi_target, layers, lindblad_ops,
            dissipation_dt=diss_dt)
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s, F={dm_eval['fidelity']:.6f}, "
              f"P={dm_eval['purity']:.4f}")
        print(f"  vs baseline: {dm_eval['fidelity'] - F_trunc:+.6f}")

        results[f"{cfg_name}_n_ops"] = n_ops
        results[f"{cfg_name}_fidelity"] = float(dm_eval["fidelity"])
        results[f"{cfg_name}_purity"] = float(dm_eval["purity"])
        results[f"{cfg_name}_time"] = elapsed
        results[f"{cfg_name}_fidelities"] = [float(f) for f in result.fidelities]

    # --- Summary ---
    print("\n" + "=" * 72)
    print("  Summary: 10-Qubit Hydrogen State Preparation")
    print("=" * 72)
    hdr = f"  {'Config':<40s} {'F(DM)':>8s}  {'Purity':>8s}  {'#Ops':>5s}  {'Time':>7s}"
    print(hdr)
    print(f"  {'-'*40} {'-'*8}  {'-'*8}  {'-'*5}  {'-'*7}")
    print(f"  {'Truncated circuit':<40s} {F_trunc:8.6f}  {'1.0000':>8s}  "
          f"{'--':>5s}  {'--':>7s}")
    print(f"  {'Teacher tail (DM)':<40s} "
          f"{results['teacher_final_fidelity']:8.6f}  {'--':>8s}  "
          f"{'--':>5s}  {'--':>7s}")
    print(f"  {'Krotov-only (no dissipation)':<40s} "
          f"{results['krotov_only_fidelity']:8.6f}  "
          f"{results['krotov_only_purity']:8.4f}  {'0':>5s}  "
          f"{results['krotov_only_time']:6.0f}s")
    print(f"  {'Reduced teacher (Rank1Op)':<40s} "
          f"{results['teacher_rank1_fidelity']:8.6f}  "
          f"{results['teacher_rank1_purity']:8.4f}  "
          f"{results['teacher_rank1_n_ops']:5d}  "
          f"{results['teacher_rank1_time']:6.0f}s")
    for cfg_name, cfg_label, _ in hw_configs:
        fk, pk, nk, tk = (f"{cfg_name}_fidelity", f"{cfg_name}_purity",
                          f"{cfg_name}_n_ops", f"{cfg_name}_time")
        if fk in results:
            print(f"  {cfg_label:<40s} {results[fk]:8.6f}  "
                  f"{results[pk]:8.4f}  {results[nk]:5d}  "
                  f"{results[tk]:6.0f}s")
    print("=" * 72)

    out_path = os.path.join(os.path.dirname(__file__), "hydrogen_results_10q.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    run_experiment()
