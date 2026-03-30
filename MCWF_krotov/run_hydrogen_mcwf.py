r"""Hydrogen wavefunction state preparation with MCWF-Krotov gate circuits.

Extended experiment comparing dissipation channels of increasing expressivity.

For each constraint level, the dissipation *rates* are first optimised
(via ``fit_constraint_to_target``) to best approximate the ideal
target-cooling teacher.  Then the gate parameters are jointly optimised
via MCWF-Krotov.  This two-stage pipeline shows how fidelity scales
with the expressivity of the hardware-allowed dissipation.

Configs (increasing expressivity):
  - target_cooling : ideal teacher Lindblad ops  (oracle upper bound)
  - amp_damping    : 1q amplitude damping only
  - config_A       : 1q amp-damp + 1q dephasing
  - config_B       : A + 2q ZZ dephasing (chain connectivity)
  - config_C       : A + 2q ZZ dephasing (all-to-all connectivity)
  - config_D       : C + 1q & 2q ancilla-reset gadgets

Run:
    python MCWF_krotov/run_hydrogen_mcwf.py
"""

from __future__ import annotations

import sys
import os
import time
import json
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "experiments", "hydrogen_qalchemy"))

from MCWF_krotov import gate_circuit_krotov as gck
from MCWF_krotov import utils

from tail.fit_constraint_to_target import (
    fit_to_teacher,
    generate_training_states,
)
from tail.constraint_dissipation import (
    ConstraintConfig,
    ConstraintDissipation,
    _ancilla_reset_kraus,
    _ancilla_reset_kraus_2q,
)


# ---------------------------------------------------------------------------
# Hydrogen target / truncated-circuit helpers
# ---------------------------------------------------------------------------

def build_hydrogen_target(n_qubits: int, n: int = 2, l: int = 1, m: int = 0) -> np.ndarray:
    d = 2 ** n_qubits
    grid_side = int(np.sqrt(d))
    if grid_side * grid_side != d:
        grid_side = d
    try:
        from hydrogen_wavefunction import compute_psi_xz_slice
        _, _, psi_grid, _ = compute_psi_xz_slice(
            n, l, m, grid_points=grid_side, extent_a_mu=15.0,
        )
        psi = psi_grid.ravel()[:d].astype(complex)
    except Exception:
        rng = np.random.default_rng(hash((n, l, m)) % 2**32)
        psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    norm = np.linalg.norm(psi)
    if norm < 1e-15:
        psi = np.ones(d, dtype=complex)
        norm = np.linalg.norm(psi)
    return psi / norm


def build_truncated_circuit_state(psi_target, n_qubits, gate_fraction=0.5):
    from scipy.linalg import null_space, expm
    d = 2 ** n_qubits
    orth = null_space(psi_target.conj().reshape(1, -1))
    U = np.column_stack([psi_target, orth])
    eigvals, eigvecs = np.linalg.eig(U)
    log_U = sum(
        np.log(eigvals[i]) * np.outer(eigvecs[:, i], eigvecs[:, i].conj())
        for i in range(d)
    )
    U_trunc = expm(gate_fraction * log_U)
    e0 = np.zeros(d, dtype=complex); e0[0] = 1.0
    psi_init = U_trunc @ e0
    return psi_init / np.linalg.norm(psi_init)


# ---------------------------------------------------------------------------
# Extract MCWF operators from a fitted ConstraintDissipation model
# ---------------------------------------------------------------------------

def extract_mcwf_operators(
    model: ConstraintDissipation,
    n_qubits: int,
) -> tuple[list[np.ndarray], list[list[np.ndarray]] | None]:
    """Build Lindblad jump operators and Kraus channel sets from a fitted model.

    Returns (lindblad_ops, kraus_ops_sets_or_None).
    """
    d = 2 ** n_qubits
    lindblad_ops: list[np.ndarray] = []

    rates_ad = model._get_block("amp_damp")
    if len(rates_ad) > 0:
        bare_ad = utils.amplitude_damping_operators(n_qubits)
        for q in range(n_qubits):
            if rates_ad[q] > 1e-12:
                lindblad_ops.append(np.sqrt(rates_ad[q]) * bare_ad[q])

    rates_deph = model._get_block("dephasing")
    if len(rates_deph) > 0:
        bare_deph = utils.dephasing_operators(n_qubits)
        for q in range(n_qubits):
            if rates_deph[q] > 1e-12:
                lindblad_ops.append(np.sqrt(rates_deph[q]) * bare_deph[q])

    rates_zz = model._get_block("zz_deph")
    if len(rates_zz) > 0 and model.cfg.edges:
        bare_zz = utils.zz_dephasing_operators(n_qubits, model.cfg.edges)
        for k in range(len(rates_zz)):
            if rates_zz[k] > 1e-12:
                lindblad_ops.append(np.sqrt(rates_zz[k]) * bare_zz[k])

    kraus_sets: list[list[np.ndarray]] = []

    if model.cfg.allow_ancilla_reset:
        angles = model._get_block("ancilla_reset")
        n_per_q = model._ancilla_n_params
        for q in range(n_qubits):
            params = angles[q * n_per_q:(q + 1) * n_per_q]
            local_kraus = _ancilla_reset_kraus(params, q, n_qubits)
            K0 = utils.embed_operator(local_kraus[0], q, n_qubits)
            K1 = utils.embed_operator(local_kraus[1], q, n_qubits)
            kraus_sets.append([K0, K1])

    if (model.cfg.allow_2q
            and model.cfg.edges
            and getattr(model.cfg, "allow_2q_ancilla_reset", False)):
        angles = model._get_block("ancilla_reset_2q")
        n_per_e = model._anc2q_n_params
        for k, (qi, qj) in enumerate(model.cfg.edges):
            params = angles[k * n_per_e:(k + 1) * n_per_e]
            local_kraus = _ancilla_reset_kraus_2q(params, n_per_e)
            K0 = utils.embed_2q_operator(local_kraus[0], qi, qj, n_qubits)
            K1 = utils.embed_2q_operator(local_kraus[1], qi, qj, n_qubits)
            kraus_sets.append([K0, K1])

    if len(lindblad_ops) == 0:
        lindblad_ops.append(np.zeros((d, d), dtype=complex))

    return lindblad_ops, (kraus_sets if kraus_sets else None)


# ---------------------------------------------------------------------------
# Constraint configurations
# ---------------------------------------------------------------------------

def _make_configs(n_qubits):
    """Return a list of (name, label, ConstraintConfig) for each expressivity level."""
    configs = []

    # Config A: 1q amp-damp + dephasing
    configs.append(("config_A", "1q amp-damp + dephasing", ConstraintConfig(
        n_qubits=n_qubits,
        allow_2q=False,
        enable_amp_damp=True,
        enable_dephasing=True,
        enable_depolarizing=False,
    )))

    # Config B: A + ZZ chain
    configs.append(("config_B", "A + ZZ dephasing (chain)", ConstraintConfig(
        n_qubits=n_qubits,
        allow_2q=True,
        connectivity="chain",
        enable_amp_damp=True,
        enable_dephasing=True,
        enable_depolarizing=False,
    )))

    # Config C: A + ZZ all-to-all
    configs.append(("config_C", "A + ZZ dephasing (all2all)", ConstraintConfig(
        n_qubits=n_qubits,
        allow_2q=True,
        connectivity="all_to_all",
        enable_amp_damp=True,
        enable_dephasing=True,
        enable_depolarizing=False,
    )))

    # Config D: C + 1q ancilla-reset (2q ancilla disabled to keep param count tractable)
    configs.append(("config_D", "C + 1q ancilla-reset", ConstraintConfig(
        n_qubits=n_qubits,
        allow_2q=True,
        connectivity="all_to_all",
        enable_amp_damp=True,
        enable_dephasing=True,
        enable_depolarizing=False,
        allow_ancilla_reset=True,
        allow_2q_ancilla_reset=False,
    )))

    return configs


# ---------------------------------------------------------------------------
# Fresh ansatz
# ---------------------------------------------------------------------------

def _fresh_ansatz(n_qubits, n_layers, seed=42):
    layers = gck.build_hardware_efficient_ansatz(n_qubits, n_layers)
    rng = np.random.default_rng(seed)
    for layer in layers:
        for gate in layer:
            gate.theta = rng.uniform(-0.5, 0.5)
    return layers


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    print("=" * 72)
    print("  Hydrogen MCWF-Krotov: Extended Dissipation Expressivity Experiment")
    print("=" * 72)

    n_qubits = 4
    d = 2 ** n_qubits
    n_layers = 4
    n_iter = 100
    n_traj = 6
    lam = 0.3
    diss_dt = 0.1

    gamma_teacher = 1.0
    fit_tmax = 5.0
    fit_steps = 30
    fit_max_iter = 150
    n_train = 5

    print(f"\nSystem: {n_qubits} qubits, d = {d}")
    print(f"Ansatz: {n_layers} layers, {n_iter} Krotov iters, "
          f"{n_traj} trajectories, lambda={lam}")
    print(f"Rate fitting: {fit_max_iter} L-BFGS-B iters, "
          f"{n_train} training states, {fit_steps} steps")

    psi_target = build_hydrogen_target(n_qubits, n=2, l=1, m=0)
    psi_trunc = build_truncated_circuit_state(psi_target, n_qubits, gate_fraction=0.5)
    F_trunc = abs(np.vdot(psi_target, psi_trunc)) ** 2
    print(f"\nTarget: Hydrogen 2p (n=2, l=1, m=0)")
    print(f"Baseline truncated fidelity: {F_trunc:.6f}")

    rho_trunc = utils.pure_state_dm(psi_trunc)

    results = {
        "n_qubits": n_qubits,
        "target": "H_2p",
        "n_layers": n_layers,
        "n_iter": n_iter,
        "n_traj": n_traj,
        "lambda": lam,
        "dissipation_dt": diss_dt,
        "truncated_fidelity": F_trunc,
    }

    # --- Teacher tail (DM, ideal) ---
    print(f"\n--- Teacher: Global Target-Cooling Tail ---")
    rho = rho_trunc.copy()
    psi_conj = psi_target.conj()
    teacher_fids = [utils.fidelity_to_pure(rho, psi_target)]
    dt_t = fit_tmax / 50
    for _ in range(50):
        v = rho @ psi_target
        a0 = np.real(psi_conj @ v)
        g = gamma_teacher * dt_t
        eg, eg2 = np.exp(-g), np.exp(-g / 2.0)
        alpha = (1.0 - eg) + 2.0 * a0 * (eg - eg2)
        beta = eg2 - eg
        rho = eg * rho + alpha * np.outer(psi_target, psi_conj) \
            + beta * np.outer(psi_target, v.conj()) \
            + beta * np.outer(v, psi_conj)
        teacher_fids.append(utils.fidelity_to_pure(rho, psi_target))
    print(f"  Fidelity after tail: {teacher_fids[-1]:.6f}")
    results["teacher_final_fidelity"] = teacher_fids[-1]
    results["teacher_fidelities"] = [float(f) for f in teacher_fids]

    # --- MCWF-Krotov with ideal teacher ops ---
    print(f"\n--- MCWF-Krotov: Teacher Lindblad (ideal) ---")
    gamma_tc = 0.3
    ops_tc = utils.target_cooling_operators(psi_target)
    lindblad_tc = [np.sqrt(gamma_tc) * L for L in ops_tc]
    layers_tc = _fresh_ansatz(n_qubits, n_layers, seed=42)

    t0 = time.time()
    res_tc = gck.krotov_gate_circuit(
        psi_trunc, psi_target, layers_tc, lindblad_tc,
        dissipation_dt=diss_dt, n_trajectories=n_traj,
        n_iterations=n_iter, lambda_a=lam, seed=42, verbose=True,
    )
    dm_tc = gck.evaluate_circuit_dm(
        psi_trunc, psi_target, layers_tc, lindblad_tc, dissipation_dt=diss_dt,
    )
    elapsed_tc = time.time() - t0
    print(f"  Time: {elapsed_tc:.1f}s, F(DM)={dm_tc['fidelity']:.6f}, "
          f"Purity={dm_tc['purity']:.4f}")
    results["target_cooling_final_fidelity_dm"] = dm_tc["fidelity"]
    results["target_cooling_final_purity"] = dm_tc["purity"]
    results["target_cooling_elapsed"] = elapsed_tc
    results["target_cooling_fidelities"] = [float(f) for f in res_tc.fidelities]

    # --- MCWF-Krotov with amp-damping only (no fitting needed) ---
    print(f"\n--- MCWF-Krotov: 1q amp-damping only ---")
    gamma_ad = 0.2
    bare_ad = utils.amplitude_damping_operators(n_qubits)
    lindblad_ad = [np.sqrt(gamma_ad) * L for L in bare_ad]
    layers_ad = _fresh_ansatz(n_qubits, n_layers, seed=42)

    t0 = time.time()
    res_ad = gck.krotov_gate_circuit(
        psi_trunc, psi_target, layers_ad, lindblad_ad,
        dissipation_dt=diss_dt, n_trajectories=n_traj,
        n_iterations=n_iter, lambda_a=lam, seed=42, verbose=True,
    )
    dm_ad = gck.evaluate_circuit_dm(
        psi_trunc, psi_target, layers_ad, lindblad_ad, dissipation_dt=diss_dt,
    )
    elapsed_ad = time.time() - t0
    print(f"  Time: {elapsed_ad:.1f}s, F(DM)={dm_ad['fidelity']:.6f}, "
          f"Purity={dm_ad['purity']:.4f}")
    results["amp_damping_final_fidelity_dm"] = dm_ad["fidelity"]
    results["amp_damping_final_purity"] = dm_ad["purity"]
    results["amp_damping_elapsed"] = elapsed_ad
    results["amp_damping_fidelities"] = [float(f) for f in res_ad.fidelities]

    # --- Configs A-D: fit rates then MCWF-Krotov ---
    training_states = generate_training_states(
        d, n_train, psi_target, rho_circuit=rho_trunc, seed=0,
    )

    for cfg_name, cfg_label, cfg in _make_configs(n_qubits):
        print(f"\n{'='*72}")
        print(f"--- {cfg_label} ---")

        n_model_params = ConstraintDissipation(cfg).num_params()
        use_max_iter = max(fit_max_iter, min(400, n_model_params * 3))

        # Stage 1: fit rates to teacher
        print(f"  [Stage 1] Fitting dissipation rates "
              f"({n_model_params} params, {use_max_iter} iters) ...")
        t_fit0 = time.time()
        fit_res = fit_to_teacher(
            psi_target=psi_target,
            rho_train=training_states,
            cfg=cfg,
            gamma_teacher=gamma_teacher,
            tmax=fit_tmax,
            steps=fit_steps,
            loss="fidelity",
            max_iter=use_max_iter,
            seed=0,
            l2_weight=0.005,
            verbose=True,
            fit_steps=20,
        )
        fitted_model = fit_res["model"]
        fit_time = time.time() - t_fit0
        print(f"  Fit loss: {fit_res['best_loss']:.6e}, time: {fit_time:.1f}s")

        params = fitted_model.pack_params()
        n_params = fitted_model.num_params()
        print(f"  Fitted {n_params} params, "
              f"rate range [{params.min():.4f}, {params.max():.4f}]")

        # Dissipation-only evaluation (no Krotov, just fitted tail)
        rho_diss = rho_trunc.copy()
        dt_diss = fit_tmax / fit_steps
        fitted_model.prepare_step(dt_diss)
        for _ in range(fit_steps):
            rho_diss = fitted_model.cached_step(rho_diss)
        F_diss_only = utils.fidelity_to_pure(rho_diss, psi_target)
        print(f"  Dissipation-only F: {F_diss_only:.6f}")
        results[f"{cfg_name}_diss_only_fidelity"] = F_diss_only

        # Stage 2: extract MCWF operators
        lindblad_ops, kraus_sets = extract_mcwf_operators(fitted_model, n_qubits)
        n_lindblad = len(lindblad_ops)
        n_kraus = len(kraus_sets) if kraus_sets else 0
        print(f"  Lindblad ops: {n_lindblad}, Kraus sets: {n_kraus}")

        # Stage 3: run MCWF-Krotov
        print(f"  [Stage 2] MCWF-Krotov optimisation ...")
        layers = _fresh_ansatz(n_qubits, n_layers, seed=42)

        t0 = time.time()
        result = gck.krotov_gate_circuit(
            psi_trunc, psi_target, layers, lindblad_ops,
            dissipation_dt=diss_dt, n_trajectories=n_traj,
            n_iterations=n_iter, lambda_a=lam, seed=42,
            verbose=True, kraus_ops_sets=kraus_sets,
        )
        dm_eval = gck.evaluate_circuit_dm(
            psi_trunc, psi_target, layers, lindblad_ops,
            dissipation_dt=diss_dt, kraus_ops_sets=kraus_sets,
        )
        elapsed = time.time() - t0

        print(f"  Krotov time: {elapsed:.1f}s")
        print(f"  F(DM) = {dm_eval['fidelity']:.6f}, "
              f"Purity = {dm_eval['purity']:.4f}")
        print(f"  Improvement over baseline: "
              f"{dm_eval['fidelity'] - F_trunc:+.6f}")

        results[f"{cfg_name}_n_params"] = n_params
        results[f"{cfg_name}_n_lindblad"] = n_lindblad
        results[f"{cfg_name}_n_kraus_sets"] = n_kraus
        results[f"{cfg_name}_fit_loss"] = fit_res["best_loss"]
        results[f"{cfg_name}_fit_time"] = fit_time
        results[f"{cfg_name}_final_fidelity_dm"] = dm_eval["fidelity"]
        results[f"{cfg_name}_final_purity"] = dm_eval["purity"]
        results[f"{cfg_name}_elapsed"] = elapsed
        results[f"{cfg_name}_fidelities"] = [float(f) for f in result.fidelities]

    # --- Summary ---
    print("\n" + "=" * 72)
    print("  Summary: Fidelity vs Dissipation Expressivity")
    print("=" * 72)
    header = (f"  {'Config':<35s} {'F(DM)':>8s}  {'Purity':>8s}  "
              f"{'#Ops':>5s}  {'FitLoss':>10s}")
    print(header)
    print(f"  {'-'*35} {'-'*8}  {'-'*8}  {'-'*5}  {'-'*10}")

    print(f"  {'Truncated circuit':<35s} {F_trunc:8.6f}  {'1.0000':>8s}  "
          f"{'--':>5s}  {'--':>10s}")
    print(f"  {'Teacher tail (DM, ideal)':<35s} "
          f"{results['teacher_final_fidelity']:8.6f}  {'--':>8s}  "
          f"{'--':>5s}  {'--':>10s}")
    print(f"  {'Teacher Lindblad (MCWF-K)':<35s} "
          f"{results['target_cooling_final_fidelity_dm']:8.6f}  "
          f"{results['target_cooling_final_purity']:8.4f}  "
          f"{'15':>5s}  {'--':>10s}")
    print(f"  {'1q amp-damping (MCWF-K)':<35s} "
          f"{results['amp_damping_final_fidelity_dm']:8.6f}  "
          f"{results['amp_damping_final_purity']:8.4f}  "
          f"{'4':>5s}  {'--':>10s}")

    for cfg_name, cfg_label, _ in _make_configs(n_qubits):
        fk = f"{cfg_name}_final_fidelity_dm"
        pk = f"{cfg_name}_final_purity"
        nk = f"{cfg_name}_n_lindblad"
        kk = f"{cfg_name}_n_kraus_sets"
        flk = f"{cfg_name}_fit_loss"
        if fk in results:
            total_ops = results[nk] + results.get(kk, 0)
            print(f"  {cfg_label:<35s} {results[fk]:8.6f}  "
                  f"{results[pk]:8.4f}  {total_ops:5d}  "
                  f"{results[flk]:10.2e}")
    print("=" * 72)

    out_path = os.path.join(os.path.dirname(__file__), "hydrogen_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    run_experiment()
