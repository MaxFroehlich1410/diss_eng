r"""Hydrogen experiment: extended Krotov with Lindblad rate controls.

Compares three optimisation modes on the same state-preparation task:

  1. **gates_only** -- standard circuit Krotov (rates fixed, gates optimised)
  2. **rates_only** -- rate-Krotov (gates fixed at initial ansatz, rates optimised)
  3. **joint**      -- joint Krotov (both gates and rates optimised)

For each mode, several dissipation-basis configurations are tested:

  - 1q amp-damping only
  - 1q amp-damping + dephasing
  - 1q amp-damping + dephasing + 2q ZZ (all-to-all)
  - AD + teacher cooling (mixed)

Supports 4, 6, and 8 qubits via command-line argument:
    python MCWF_krotov/run_hydrogen_rate_krotov.py [n_qubits]
"""

from __future__ import annotations

import json
import os
import sys
import time
import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BASE_DIR)

from MCWF_krotov import gate_circuit_krotov as gck
from MCWF_krotov import rate_krotov as rk
from MCWF_krotov import utils


# ---------------------------------------------------------------------------
# Hydrogen target / truncated-circuit helpers (same as run_hydrogen_mcwf.py)
# ---------------------------------------------------------------------------

def build_hydrogen_target(n_qubits: int, n: int = 2, l: int = 1,
                          m: int = 0) -> np.ndarray:
    d = 2 ** n_qubits
    grid_side = int(np.sqrt(d))
    if grid_side * grid_side != d:
        grid_side = d
    try:
        sys.path.insert(
            0, os.path.join(BASE_DIR, "experiments", "hydrogen_qalchemy"))
        from hydrogen_wavefunction import compute_psi_xz_slice
        _, _, psi_grid, _ = compute_psi_xz_slice(
            n, l, m, grid_points=grid_side, extent_a_mu=15.0)
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
        for i in range(d))
    U_trunc = expm(gate_fraction * log_U)
    e0 = np.zeros(d, dtype=complex)
    e0[0] = 1.0
    psi_init = U_trunc @ e0
    return psi_init / np.linalg.norm(psi_init)


def _fresh_ansatz(n_qubits, n_layers, seed=42, full_generators=True):
    layers = gck.build_hardware_efficient_ansatz(
        n_qubits, n_layers, full_generators=full_generators)
    rng = np.random.default_rng(seed)
    for layer in layers:
        for gate in layer:
            gate.theta = rng.uniform(-0.5, 0.5)
    return layers


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(n_qubits: int = 4):
    d = 2 ** n_qubits

    # Scale hyperparameters with system size
    if n_qubits <= 4:
        n_layers, n_iter, n_traj = 4, 80, 6
        lam_g, lam_r = 0.3, 0.5
        max_gate_step, max_rate_step = 0.15, 0.08
        n_teacher_mixed = 4
        full_gen = True
    elif n_qubits <= 6:
        n_layers, n_iter, n_traj = 5, 80, 6
        lam_g, lam_r = 0.3, 0.5
        max_gate_step, max_rate_step = 0.10, 0.06
        n_teacher_mixed = 6
        full_gen = True
    else:  # 8+
        n_layers, n_iter, n_traj = 5, 80, 6
        lam_g, lam_r = 0.3, 0.5
        max_gate_step, max_rate_step = 0.08, 0.05
        n_teacher_mixed = 8
        full_gen = n_qubits <= 8

    diss_dt = 0.1
    init_rate = 0.1

    print("=" * 72)
    print(f"  Extended Krotov: Joint Gate + Rate Optimisation ({n_qubits} qubits)")
    print("=" * 72)

    print(f"\nSystem: {n_qubits} qubits, d = {d}")
    print(f"Ansatz: {n_layers} layers, {n_iter} Krotov iters, "
          f"{n_traj} trajectories")
    print(f"lambda_gates={lam_g}, lambda_rates={lam_r}")
    print(f"max_gate_step={max_gate_step}, max_rate_step={max_rate_step}")
    print(f"init_rate={init_rate}, dissipation_dt={diss_dt}")

    psi_target = build_hydrogen_target(n_qubits, n=2, l=1, m=0)
    psi_trunc = build_truncated_circuit_state(psi_target, n_qubits,
                                              gate_fraction=0.5)
    F_trunc = abs(np.vdot(psi_target, psi_trunc)) ** 2
    print(f"\nTarget: Hydrogen 2p (n=2, l=1, m=0)")
    print(f"Baseline truncated fidelity: {F_trunc:.6f}")

    results = {
        "n_qubits": n_qubits,
        "target": "H_2p",
        "n_layers": n_layers,
        "n_iter": n_iter,
        "n_traj": n_traj,
        "lambda_gates": lam_g,
        "lambda_rates": lam_r,
        "dissipation_dt": diss_dt,
        "init_rate": init_rate,
        "truncated_fidelity": F_trunc,
    }

    # Dissipation basis configurations
    # "custom" entries supply pre-built DissipationBasis (for teacher ops)
    basis_configs = [
        ("ad_only", "1q amp-damping", dict(
            include_amp_damp=True, include_dephasing=False, include_zz=False)),
        ("ad_deph", "1q amp-damp + dephasing", dict(
            include_amp_damp=True, include_dephasing=True, include_zz=False)),
        ("ad_deph_zz", "1q AD + deph + 2q ZZ (all2all)", dict(
            include_amp_damp=True, include_dephasing=True, include_zz=True)),
        ("mixed_teacher", "AD + teacher cooling (mixed)", "custom"),
    ]

    modes = [
        ("gates_only", "Gates-only Krotov",
         dict(optimize_gates=True, optimize_rates=False)),
        ("rates_only", "Rates-only Krotov",
         dict(optimize_gates=False, optimize_rates=True)),
        ("joint", "Joint gates+rates Krotov",
         dict(optimize_gates=True, optimize_rates=True)),
    ]

    def _build_mixed_teacher_basis(psi_tgt, n_qubits, n_layers, init_rate):
        """AD operators + a few target-cooling operators in one basis."""
        bare_ops: list[np.ndarray] = []
        names: list[str] = []
        for q, op in enumerate(utils.amplitude_damping_operators(n_qubits)):
            bare_ops.append(op)
            names.append(f"AD_q{q}")
        tc_ops = utils.target_cooling_operators(psi_tgt)
        n_teacher = min(n_teacher_mixed, len(tc_ops))
        for i in range(n_teacher):
            bare_ops.append(tc_ops[i])
            names.append(f"TC_{i}")
        K = len(bare_ops)
        rates = np.full((n_layers, K), init_rate, dtype=float)
        return rk.DissipationBasis(bare_ops=bare_ops, rates=rates, names=names)

    for basis_name, basis_label, basis_kwargs in basis_configs:
        print(f"\n{'='*72}")
        print(f"  Dissipation basis: {basis_label}")
        print(f"{'='*72}")

        if basis_kwargs == "custom":
            db_probe = _build_mixed_teacher_basis(
                psi_target, n_qubits, n_layers, init_rate)
        else:
            db_probe = rk.build_dissipation_basis(
                n_qubits, n_layers, init_rate=init_rate, **basis_kwargs)
        print(f"  Operators: {db_probe.n_ops}  "
              f"({', '.join(db_probe.names[:6])}{'...' if db_probe.n_ops > 6 else ''})")
        print(f"  Rate params per mode: {n_layers} layers x {db_probe.n_ops} "
              f"ops = {n_layers * db_probe.n_ops}")

        for mode_name, mode_label, mode_kwargs in modes:
            key = f"{basis_name}__{mode_name}"
            print(f"\n  --- {mode_label} ---")

            layers = _fresh_ansatz(n_qubits, n_layers, seed=42,
                                    full_generators=full_gen)
            if basis_kwargs == "custom":
                db = _build_mixed_teacher_basis(
                    psi_target, n_qubits, n_layers, init_rate)
            else:
                db = rk.build_dissipation_basis(
                    n_qubits, n_layers, init_rate=init_rate, **basis_kwargs)

            t0 = time.time()
            res = rk.krotov_joint(
                psi_trunc, psi_target, layers, db,
                dissipation_dt=diss_dt,
                n_trajectories=n_traj,
                n_iterations=n_iter,
                lambda_gates=lam_g,
                lambda_rates=lam_r,
                max_gate_step=max_gate_step,
                max_rate_step=max_rate_step,
                seed=42, verbose=True,
                **mode_kwargs,
            )
            elapsed = time.time() - t0

            dm_eval = rk.evaluate_circuit_dm_rates(
                psi_trunc, psi_target, layers, db,
                dissipation_dt=diss_dt,
            )

            F_mcwf = res.fidelities[-1]
            F_dm = dm_eval["fidelity"]
            pur = dm_eval["purity"]

            print(f"  Time: {elapsed:.1f}s")
            print(f"  F(MCWF) = {F_mcwf:.6f}, F(DM) = {F_dm:.6f}, "
                  f"Purity = {pur:.4f}")
            print(f"  Improvement over baseline: {F_dm - F_trunc:+.6f}")

            final_rates = res.final_rates
            rate_min = final_rates.min()
            rate_max = final_rates.max()
            rate_mean = final_rates.mean()
            rate_zeros = int(np.sum(final_rates < 1e-10))
            print(f"  Rates: min={rate_min:.4f}, max={rate_max:.4f}, "
                  f"mean={rate_mean:.4f}, #zero={rate_zeros}")

            # Per-operator-type rate summary
            for op_idx, name in enumerate(db.names):
                r_vals = final_rates[:, op_idx]
                print(f"    {name}: rates = "
                      f"[{', '.join(f'{v:.4f}' for v in r_vals)}]")

            results[f"{key}__fidelity_mcwf"] = float(F_mcwf)
            results[f"{key}__fidelity_dm"] = float(F_dm)
            results[f"{key}__purity"] = float(pur)
            results[f"{key}__elapsed"] = float(elapsed)
            results[f"{key}__fidelities"] = [float(f) for f in res.fidelities]
            results[f"{key}__rate_min"] = float(rate_min)
            results[f"{key}__rate_max"] = float(rate_max)
            results[f"{key}__rate_mean"] = float(rate_mean)
            results[f"{key}__rate_zeros"] = rate_zeros
            results[f"{key}__final_rates"] = final_rates.tolist()

    # --- Summary table ---
    print(f"\n{'='*72}")
    print("  Summary: Fidelity(DM) across modes and bases")
    print(f"{'='*72}")
    header = f"  {'Basis':<28s}"
    for mn, ml, _ in modes:
        header += f"  {ml:>18s}"
    print(header)
    print("  " + "-" * (28 + 3 * 20))

    for bn, bl, _ in basis_configs:
        row = f"  {bl:<28s}"
        for mn, _, _ in modes:
            key = f"{bn}__{mn}__fidelity_dm"
            val = results.get(key, float("nan"))
            row += f"  {val:18.6f}"
        print(row)

    print(f"\n  Baseline truncated F = {F_trunc:.6f}")
    print(f"{'='*72}")

    out_path = os.path.join(os.path.dirname(__file__),
                            f"hydrogen_results_rate_krotov_{n_qubits}q.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_experiment(nq)
