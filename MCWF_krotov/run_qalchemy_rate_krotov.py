r"""Q-Alchemy circuit + rate-only Krotov experiment.

Uses Q-Alchemy to obtain an approximate state-preparation circuit at various
``max_fidelity_loss`` levels (controlling circuit depth / fidelity trade-off),
then applies **rate-only Krotov** to optimise the interleaved Lindblad
dissipation rates and close the remaining fidelity gap.

The circuit gates are **frozen** -- only the dissipation rates are optimised.

Dissipation bases tested:
  - hw_only : 1q amplitude-damping + dephasing (hardware-realistic)
  - teacher : target-cooling operators only (ideal, for proof of concept)
  - mixed   : AD + a few target-cooling operators

Usage:
    export Q_ALCHEMY_API_KEY="..."
    python3 MCWF_krotov/run_qalchemy_rate_krotov.py [n_qubits]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "experiments", "hydrogen_qalchemy"))

from MCWF_krotov import gate_circuit_krotov as gck
from MCWF_krotov import rate_krotov as rk
from MCWF_krotov import utils


# ---------------------------------------------------------------------------
# Q-Alchemy helpers
# ---------------------------------------------------------------------------

def _get_qalchemy_circuit(statevector: np.ndarray, max_fidelity_loss: float,
                          api_key: str):
    """Call Q-Alchemy and return (qiskit_circuit, compiled_circuit)."""
    from qiskit import QuantumCircuit
    from q_alchemy.initialize import OptParams
    from q_alchemy.qiskit_integration import QAlchemyInitialize

    num_qubits = int(np.ceil(np.log2(len(statevector))))
    opt_params = OptParams(api_key=api_key, max_fidelity_loss=max_fidelity_loss)
    qc = QuantumCircuit(num_qubits)
    qc.append(QAlchemyInitialize(statevector, opt_params=opt_params), qc.qubits)
    compiled = qc.decompose()
    return compiled


def _circuit_output_state(circuit, n_qubits: int) -> np.ndarray:
    """Evolve |0...0> through a Qiskit circuit and return the statevector."""
    from qiskit.quantum_info import Statevector
    sv = Statevector.from_label("0" * n_qubits)
    sv = sv.evolve(circuit)
    return np.asarray(sv.data, dtype=complex)


# ---------------------------------------------------------------------------
# Hydrogen target
# ---------------------------------------------------------------------------

def _hydrogen_target(n_qubits: int, n: int = 2, l: int = 1,
                     m: int = 0) -> np.ndarray:
    d = 2 ** n_qubits
    grid_side = int(np.sqrt(d))
    if grid_side * grid_side != d:
        grid_side = d
    try:
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


# ---------------------------------------------------------------------------
# Dissipation basis builders
# ---------------------------------------------------------------------------

def _build_hw_basis(n_qubits, n_layers, init_rate=0.1):
    return rk.build_dissipation_basis(
        n_qubits, n_layers,
        include_amp_damp=True, include_dephasing=True,
        include_zz=False, init_rate=init_rate,
    )


def _build_teacher_basis(psi_target, n_qubits, n_layers, init_rate=0.1,
                         max_ops=None):
    tc_ops = utils.target_cooling_operators(psi_target)
    if max_ops is not None:
        tc_ops = tc_ops[:max_ops]
    names = [f"TC_{i}" for i in range(len(tc_ops))]
    K = len(tc_ops)
    rates = np.full((n_layers, K), init_rate, dtype=float)
    return rk.DissipationBasis(bare_ops=tc_ops, rates=rates, names=names)


def _build_mixed_basis(psi_target, n_qubits, n_layers, teacher_rate=0.1,
                       n_teacher=4, hw_rate=0.01):
    bare_ops: list[np.ndarray] = []
    names: list[str] = []
    n_hw = 0
    for q, op in enumerate(utils.amplitude_damping_operators(n_qubits)):
        bare_ops.append(op)
        names.append(f"AD_q{q}")
        n_hw += 1
    tc_ops = utils.target_cooling_operators(psi_target)
    for i in range(min(n_teacher, len(tc_ops))):
        bare_ops.append(tc_ops[i])
        names.append(f"TC_{i}")
    K = len(bare_ops)
    rates = np.zeros((n_layers, K), dtype=float)
    rates[:, :n_hw] = hw_rate
    rates[:, n_hw:] = teacher_rate
    return rk.DissipationBasis(bare_ops=bare_ops, rates=rates, names=names)


# ---------------------------------------------------------------------------
# Identity-gate ansatz (gates frozen at identity)
# ---------------------------------------------------------------------------

def _identity_ansatz(n_qubits: int, n_layers: int) -> list[list[gck.GateLayer]]:
    """Ansatz with all gate parameters at zero (all gates = identity)."""
    layers = gck.build_hardware_efficient_ansatz(
        n_qubits, n_layers, full_generators=(n_qubits <= 6))
    for layer in layers:
        for gate in layer:
            gate.theta = 0.0
    return layers


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(n_qubits: int = 4):
    d = 2 ** n_qubits

    api_key = os.getenv("Q_ALCHEMY_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing Q_ALCHEMY_API_KEY. Export it and re-run:\n"
            "  export Q_ALCHEMY_API_KEY=\"...\""
        )

    fidelity_loss_levels = [0.05, 0.10, 0.20, 0.30, 0.40]

    if n_qubits <= 4:
        n_layers, n_iter, n_traj = 10, 100, 8
        lam_r = 0.1
        max_rate_step = 0.2
        n_teacher_cap = 10
    elif n_qubits <= 6:
        n_layers, n_iter, n_traj = 10, 100, 8
        lam_r = 0.1
        max_rate_step = 0.2
        n_teacher_cap = 20
    else:
        n_layers, n_iter, n_traj = 8, 80, 8
        lam_r = 0.1
        max_rate_step = 0.15
        n_teacher_cap = 20

    diss_dt = 0.3
    hw_init_rate = 0.01
    teacher_init_rate = 0.3

    print("=" * 72)
    print(f"  Q-Alchemy + Rate-Only Krotov ({n_qubits} qubits)")
    print("=" * 72)
    print(f"  System: d = {d}")
    print(f"  Dissipation layers: {n_layers}, Krotov iters: {n_iter}, "
          f"trajectories: {n_traj}")
    print(f"  lambda_rates={lam_r}, max_rate_step={max_rate_step}")
    print(f"  hw_init_rate={hw_init_rate}, teacher_init_rate={teacher_init_rate}")
    print(f"  diss_dt={diss_dt}")
    print(f"  Fidelity-loss levels: {fidelity_loss_levels}")

    psi_target = _hydrogen_target(n_qubits, n=2, l=1, m=0)

    basis_builders = {
        "hw_only": ("Hardware (AD+deph)",
                    lambda nl: _build_hw_basis(n_qubits, nl, hw_init_rate)),
        "teacher": ("Teacher cooling",
                    lambda nl: _build_teacher_basis(
                        psi_target, n_qubits, nl, teacher_init_rate,
                        max_ops=n_teacher_cap)),
        "mixed": ("AD + teacher (mixed)",
                  lambda nl: _build_mixed_basis(
                      psi_target, n_qubits, nl, teacher_rate=teacher_init_rate,
                      n_teacher=n_teacher_cap, hw_rate=hw_init_rate)),
    }

    results = {
        "n_qubits": n_qubits,
        "target": "H_2p",
        "n_layers": n_layers,
        "n_iter": n_iter,
        "n_traj": n_traj,
        "lambda_rates": lam_r,
        "dissipation_dt": diss_dt,
        "hw_init_rate": hw_init_rate,
        "teacher_init_rate": teacher_init_rate,
        "fidelity_loss_levels": fidelity_loss_levels,
    }

    for fl in fidelity_loss_levels:
        print(f"\n{'='*72}")
        print(f"  max_fidelity_loss = {fl}")
        print(f"{'='*72}")

        t_api = time.time()
        try:
            circuit = _get_qalchemy_circuit(psi_target, fl, api_key)
        except Exception as e:
            print(f"  [ERROR] Q-Alchemy API failed: {e}")
            results[f"fl_{fl}__error"] = str(e)
            continue
        t_api = time.time() - t_api

        psi_circuit = _circuit_output_state(circuit, n_qubits)
        F_circuit = float(abs(np.vdot(psi_target, psi_circuit)) ** 2)
        depth = circuit.depth()
        n_gates = sum(1 for inst in circuit.data
                      if inst.operation.name != "barrier")

        print(f"  Circuit: depth={depth}, gates={n_gates}, "
              f"API time={t_api:.1f}s")
        print(f"  Circuit fidelity: {F_circuit:.6f}  "
              f"(gap = {1 - F_circuit:.6f})")

        results[f"fl_{fl}__circuit_fidelity"] = F_circuit
        results[f"fl_{fl}__depth"] = depth
        results[f"fl_{fl}__n_gates"] = n_gates
        results[f"fl_{fl}__api_time"] = float(t_api)

        for basis_key, (basis_label, basis_fn) in basis_builders.items():
            print(f"\n  --- {basis_label} ---")

            db = basis_fn(n_layers)
            ansatz = _identity_ansatz(n_qubits, n_layers)

            print(f"  Operators: {db.n_ops}  "
                  f"({', '.join(db.names[:6])}{'...' if db.n_ops > 6 else ''})")

            t0 = time.time()
            res = rk.krotov_joint(
                psi_circuit, psi_target, ansatz, db,
                dissipation_dt=diss_dt,
                n_trajectories=n_traj,
                n_iterations=n_iter,
                lambda_gates=0.3,
                lambda_rates=lam_r,
                optimize_gates=False,
                optimize_rates=True,
                max_rate_step=max_rate_step,
                seed=42, verbose=True,
            )
            elapsed = time.time() - t0

            dm_eval = rk.evaluate_circuit_dm_rates(
                psi_circuit, psi_target, ansatz, db,
                dissipation_dt=diss_dt,
            )

            F_mcwf = res.fidelities[-1]
            F_dm = dm_eval["fidelity"]
            pur = dm_eval["purity"]
            delta_F = F_dm - F_circuit

            print(f"  Time: {elapsed:.1f}s")
            print(f"  F(circuit) = {F_circuit:.6f}")
            print(f"  F(MCWF)    = {F_mcwf:.6f}")
            print(f"  F(DM)      = {F_dm:.6f},  Purity = {pur:.4f}")
            print(f"  Gap closed = {delta_F:+.6f}  "
                  f"(from {1-F_circuit:.6f} to {1-F_dm:.6f})")

            final_rates = res.final_rates
            rate_zeros = int(np.sum(final_rates < 1e-10))
            print(f"  Rates: mean={final_rates.mean():.4f}, "
                  f"max={final_rates.max():.4f}, #zero={rate_zeros}")

            for op_idx, name in enumerate(db.names[:8]):
                r_vals = final_rates[:, op_idx]
                print(f"    {name}: [{', '.join(f'{v:.4f}' for v in r_vals)}]")
            if db.n_ops > 8:
                print(f"    ... ({db.n_ops - 8} more operators)")

            rkey = f"fl_{fl}__{basis_key}"
            results[f"{rkey}__fidelity_mcwf"] = float(F_mcwf)
            results[f"{rkey}__fidelity_dm"] = float(F_dm)
            results[f"{rkey}__purity"] = float(pur)
            results[f"{rkey}__delta_F"] = float(delta_F)
            results[f"{rkey}__elapsed"] = float(elapsed)
            results[f"{rkey}__fidelities"] = [float(f) for f in res.fidelities]
            results[f"{rkey}__rate_mean"] = float(final_rates.mean())
            results[f"{rkey}__rate_max"] = float(final_rates.max())
            results[f"{rkey}__rate_zeros"] = rate_zeros
            results[f"{rkey}__final_rates"] = final_rates.tolist()

    # --- Summary table ---
    print(f"\n{'='*72}")
    print("  Summary: Q-Alchemy circuit + rate-only Krotov")
    print(f"{'='*72}")
    header = f"  {'FL':>5s}  {'F_circ':>8s}"
    for bk in basis_builders:
        header += f"  {'F_'+bk:>12s}"
    print(header)
    print("  " + "-" * (5 + 8 + len(basis_builders) * 14 + 4))

    for fl in fidelity_loss_levels:
        Fc = results.get(f"fl_{fl}__circuit_fidelity", float("nan"))
        row = f"  {fl:5.2f}  {Fc:8.6f}"
        for bk in basis_builders:
            Fd = results.get(f"fl_{fl}__{bk}__fidelity_dm", float("nan"))
            row += f"  {Fd:12.6f}"
        print(row)
    print(f"{'='*72}")

    out_path = os.path.join(os.path.dirname(__file__),
                            f"qalchemy_rate_krotov_{n_qubits}q.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return results


if __name__ == "__main__":
    nq = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    run_experiment(nq)
