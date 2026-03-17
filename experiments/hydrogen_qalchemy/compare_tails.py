#!/usr/bin/env python
"""Compare the ideal (teacher) dissipative tail vs the optimised
lab-constrained tail after a state-preparation circuit.

Usage
-----
    export Q_ALCHEMY_API_KEY="..."
    python compare_tails.py                       # defaults: 10 qubits, n=2
    python compare_tails.py --num-qubits 6 --fit-iters 300 --gamma-max 3
    python compare_tails.py --fit-loss frobenius --no-2q

The script:
1. Builds the hydrogen target state and compiles the Q-Alchemy circuit.
2. Evolves the circuit (fast path) to get the output density matrix.
3. Runs the teacher (global target-cooling) tail.
4. Fits the constrained (lab-realistic) tail to the teacher.
5. Runs the constrained tail on the *same* circuit output.
6. Saves CSVs and produces a comparison plot.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# ---- repo / local imports ----
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "q-alchemy-sdk-py" / "src"))

from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix  # noqa: E402
from q_alchemy.initialize import OptParams  # noqa: E402

from run_hydrogen_qalchemy import (  # noqa: E402
    hydrogen_statevector,
    build_qalchemy_circuit,
)
from tail.target_cooling import run_target_cooling_trajectory  # noqa: E402
from tail.metrics import fidelity_to_pure  # noqa: E402
from tail.constraint_dissipation import ConstraintConfig  # noqa: E402
from tail.fit_constraint_to_target import (  # noqa: E402
    fit_to_teacher,
    generate_training_states,
    evaluate_and_report,
)
from tail.fit_constraint_direct_to_target import (  # noqa: E402
    fit_to_target,
    evaluate_and_report_direct,
)

OUT_DIR = REPO_ROOT / "experiments" / "hydrogen_qalchemy"


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare teacher vs constrained dissipative tail",
    )
    # State / circuit
    parser.add_argument("--num-qubits", type=int, default=10)
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--l", type=int, default=0)
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--extent-a-mu", type=float, default=30.0)
    parser.add_argument("--max-fidelity-loss", type=float, default=0.02)

    # Tail parameters (shared by both tails)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Teacher dissipation rate")
    parser.add_argument("--tmax", type=float, default=5.0)
    parser.add_argument("--steps", type=int, default=50)

    # Constrained-tail options
    parser.add_argument("--gamma-max", type=float, default=2.0)
    parser.add_argument("--connectivity", choices=["chain", "all_to_all"],
                        default="chain",
                        help="2-qubit connectivity for constrained tail")
    parser.add_argument("--no-2q", action="store_true",
                        help="Disable 2-qubit channels")
    parser.add_argument("--allow-ancilla-reset", action="store_true")
    parser.add_argument("--fit-loss", choices=["fidelity", "frobenius"],
                        default="fidelity")
    parser.add_argument("--direct-opt", action="store_true",
                        help="Optimize constrained tail directly to target "
                             "fidelity (no teacher in objective)")
    parser.add_argument("--direct-loss", choices=["curve", "terminal"],
                        default="curve")
    parser.add_argument("--direct-weight-mode",
                        choices=["uniform", "exp", "early"],
                        default="exp")
    parser.add_argument("--direct-weight-alpha", type=float, default=None)
    parser.add_argument("--fit-iters", type=int, default=200)
    parser.add_argument("--fit-steps", type=int, default=None,
                        help="Coarser time grid for fitting (default: same "
                             "as --steps).  Lower = faster optimisation.")
    parser.add_argument("--n-train-states", type=int, default=3)
    parser.add_argument("--n-valid-states", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    num_qubits = args.num_qubits
    d = 1 << num_qubits

    # ------------------------------------------------------------------
    # 1. Build target state + circuit
    # ------------------------------------------------------------------
    api_key = os.getenv("Q_ALCHEMY_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing Q_ALCHEMY_API_KEY.  Export it and re-run:\n"
            "  export Q_ALCHEMY_API_KEY=\"...\""
        )

    print("=" * 64)
    print("  Building target state and compiling circuit")
    print("=" * 64)

    psi_target = hydrogen_statevector(
        num_qubits=num_qubits,
        n=args.n, l=args.l, m=args.m,
        extent_a_mu=args.extent_a_mu,
    )
    opt_params = OptParams(
        api_key=api_key, max_fidelity_loss=args.max_fidelity_loss,
    )
    circuit = build_qalchemy_circuit(psi_target, opt_params=opt_params)
    print(f"  Qubits: {num_qubits},  d = {d}")
    print(f"  Circuit depth: {circuit.depth()}")

    # Fast-path: evolve the full circuit in one shot.
    sv_out = Statevector.from_label("0" * num_qubits).evolve(circuit)
    fid_circuit = state_fidelity(sv_out, Statevector(psi_target))
    rho0 = np.asarray(DensityMatrix(sv_out).data, dtype=complex)

    print(f"  Circuit fidelity: {fid_circuit:.10f}")
    print()

    # ------------------------------------------------------------------
    # 2. Teacher (global target-cooling) tail
    # ------------------------------------------------------------------
    print("=" * 64)
    print("  Teacher tail  (global target cooling)")
    print("=" * 64)

    times, teacher_fids, teacher_trs, teacher_purs = \
        run_target_cooling_trajectory(
            rho0, psi_target,
            gamma=args.gamma, tmax=args.tmax, steps=args.steps,
        )

    fids_analytic = 1.0 - (1.0 - teacher_fids[0]) * np.exp(
        -args.gamma * times
    )

    print(f"  F before tail: {teacher_fids[0]:.10f}")
    print(f"  F after tail : {teacher_fids[-1]:.10f}")
    print(f"  Fidelity gain: {teacher_fids[-1] - teacher_fids[0]:+.10f}")
    print(f"  Trace range  : [{teacher_trs.min():.12f}, "
          f"{teacher_trs.max():.12f}]")
    print()

    # ------------------------------------------------------------------
    # 3. Fit constrained tail
    # ------------------------------------------------------------------
    print("=" * 64)
    print("  Fitting constrained (lab-realistic) tail")
    print("=" * 64)

    cfg = ConstraintConfig(
        n_qubits=num_qubits,
        allow_2q=not args.no_2q,
        connectivity=args.connectivity,
        gamma_max=args.gamma_max,
        enable_amp_damp=True,
        enable_dephasing=True,
        enable_depolarizing=(num_qubits <= 6),
        allow_ancilla_reset=args.allow_ancilla_reset,
    )

    rho_train = generate_training_states(
        d,
        n_states=args.n_train_states,
        psi_target=psi_target,
        rho_circuit=rho0,
        seed=args.seed,
    )

    if args.direct_opt:
        fit_result = fit_to_target(
            psi_target=psi_target,
            rho_train=rho_train,
            cfg=cfg,
            tmax=args.tmax,
            steps=args.steps,
            loss=args.direct_loss,
            weight_mode=args.direct_weight_mode,
            weight_alpha=args.direct_weight_alpha,
            max_iter=args.fit_iters,
            seed=args.seed,
            verbose=True,
            fit_steps=args.fit_steps,
        )
    else:
        fit_result = fit_to_teacher(
            psi_target=psi_target,
            rho_train=rho_train,
            cfg=cfg,
            gamma_teacher=args.gamma,
            tmax=args.tmax,
            steps=args.steps,
            loss=args.fit_loss,
            max_iter=args.fit_iters,
            seed=args.seed,
            verbose=True,
            fit_steps=args.fit_steps,
        )
    model = fit_result["model"]
    print()

    # ------------------------------------------------------------------
    # 4. Run constrained tail on the circuit output
    # ------------------------------------------------------------------
    print("=" * 64)
    print("  Running constrained tail on circuit output")
    print("=" * 64)

    rhos_c = model.run_trajectory(rho0, args.tmax, args.steps)
    constr_fids = np.array([fidelity_to_pure(r, psi_target) for r in rhos_c])
    constr_trs = np.array([
        float(np.real(np.trace(r))) for r in rhos_c
    ])

    print(f"  F before tail: {constr_fids[0]:.10f}")
    print(f"  F after tail : {constr_fids[-1]:.10f}")
    print(f"  Fidelity gain: {constr_fids[-1] - constr_fids[0]:+.10f}")
    print(f"  Trace range  : [{constr_trs.min():.12f}, "
          f"{constr_trs.max():.12f}]")
    print()

    # ------------------------------------------------------------------
    # 5. Validation on held-out states
    # ------------------------------------------------------------------
    rho_valid = generate_training_states(
        d,
        n_states=args.n_valid_states,
        psi_target=psi_target,
        seed=args.seed + 999,
    )
    if args.direct_opt:
        evaluate_and_report_direct(
            model, psi_target, rho_valid, args.tmax, args.steps,
        )
    else:
        evaluate_and_report(
            model, psi_target, rho_valid,
            args.gamma, args.tmax, args.steps,
        )
    print()

    # ------------------------------------------------------------------
    # 6. Save CSVs
    # ------------------------------------------------------------------
    teacher_csv = OUT_DIR / "comparison_teacher.csv"
    with teacher_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "fidelity", "trace", "purity", "analytic"])
        w.writerows(zip(
            times, teacher_fids, teacher_trs, teacher_purs, fids_analytic,
        ))
    print(f"Saved teacher CSV  : {teacher_csv}")

    constr_csv = OUT_DIR / "comparison_constrained.csv"
    with constr_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "fidelity", "trace"])
        w.writerows(zip(times, constr_fids, constr_trs))
    print(f"Saved constrained CSV: {constr_csv}")

    # ------------------------------------------------------------------
    # 7. Comparison plot
    # ------------------------------------------------------------------
    C_TEACH = "#2c3e50"   # dark blue-gray
    C_ANALY = "#c0392b"   # red dashed
    C_CONST = "#27ae60"   # green

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax_idx, (label, fids, color) in enumerate([
        ("Teacher (global target cooling)", teacher_fids, C_TEACH),
        ("Constrained (lab-realistic)", constr_fids, C_CONST),
    ]):
        ax = axes[ax_idx]

        # Fidelity curves
        ax.plot(times, fids, "-", color=color, lw=1.5,
                label=label, zorder=3)
        ax.plot(times, fids_analytic, "--", color=C_ANALY, lw=1.0,
                alpha=0.7,
                label=r"Analytic $1\!-\!(1\!-\!F_0)\,e^{-\gamma t}$",
                zorder=2)

        # Also show the other tail as a faint reference
        if ax_idx == 0:
            ax.plot(times, constr_fids, ":", color=C_CONST, lw=0.9,
                    alpha=0.5, label="Constrained (ref)", zorder=1)
        else:
            ax.plot(times, teacher_fids, ":", color=C_TEACH, lw=0.9,
                    alpha=0.5, label="Teacher (ref)", zorder=1)

        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"Fidelity  $\langle\psi_*|\rho|\psi_*\rangle$")
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.25)

        # Annotate start / end fidelity
        ax.annotate(
            f"$F_0$ = {fids[0]:.4f}",
            xy=(times[0], fids[0]),
            xytext=(times[-1] * 0.15, fids[0] - 0.01),
            fontsize=7, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.6),
        )
        ax.annotate(
            f"$F_{{\\mathrm{{final}}}}$ = {fids[-1]:.4f}",
            xy=(times[-1], fids[-1]),
            xytext=(times[-1] * 0.55, fids[-1] - 0.02),
            fontsize=7, color=color,
            arrowprops=dict(arrowstyle="->", color=color, lw=0.6),
        )

    suptitle = (
        f"Tail comparison  |  n={args.n}, l={args.l}, m={args.m}"
        f"  |  {num_qubits} qubits"
        f"  |  " + r"$\gamma$" + f" = {args.gamma}"
        f"  |  " + r"$\gamma_{{\max}}$" + f" = {args.gamma_max}"
    )
    fig.suptitle(suptitle, fontsize=11)
    plt.tight_layout()

    plot_path = OUT_DIR / "tail_comparison.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()

    # ------------------------------------------------------------------
    # 8. Summary table
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print("  Summary")
    print("=" * 64)
    print(f"  {'':30s} {'Teacher':>12s}  {'Constrained':>12s}")
    print(f"  {'F before tail':30s} {teacher_fids[0]:12.6f}  "
          f"{constr_fids[0]:12.6f}")
    print(f"  {'F after tail':30s} {teacher_fids[-1]:12.6f}  "
          f"{constr_fids[-1]:12.6f}")
    print(f"  {'Fidelity gain':30s} "
          f"{teacher_fids[-1] - teacher_fids[0]:+12.6f}  "
          f"{constr_fids[-1] - constr_fids[0]:+12.6f}")
    gap = teacher_fids[-1] - constr_fids[-1]
    print(f"  {'Gap (teacher - constrained)':30s} {gap:+12.6f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
