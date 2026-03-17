from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity, DensityMatrix
import csv
import matplotlib.pyplot as plt

# Allow local imports without installing the SDK package.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "q-alchemy-sdk-py" / "src"))

from q_alchemy.initialize import OptParams  # noqa: E402
from q_alchemy.qiskit_integration import QAlchemyInitialize  # noqa: E402
from hydrogen_wavefunction import compute_psi_xz_slice  # noqa: E402
from tail.target_cooling import run_target_cooling_trajectory  # noqa: E402
from tail.metrics import fidelity_to_pure as tail_fidelity  # noqa: E402
from tail.constraint_dissipation import ConstraintConfig, ConstraintDissipation  # noqa: E402
from tail.fit_constraint_to_target import (  # noqa: E402
    fit_to_teacher,
    generate_training_states,
    evaluate_and_report,
)


def hydrogen_statevector(
    num_qubits: int,
    n: int = 4,
    l: int = 1,
    m: int = 0,
    extent_a_mu: float = 30.0,
    **kwargs,
):
    if num_qubits % 2 != 0:
        raise ValueError("num_qubits must be even to form a square grid")

    length = 2 ** (num_qubits // 2)
    _xg, _zg, psi, _a_mu = compute_psi_xz_slice(
        n,
        l,
        m,
        extent_a_mu=extent_a_mu,
        grid_points=length,
        **kwargs,
    )
    array = psi.astype(complex)
    array = array / np.linalg.norm(array)
    return array.flatten()


def build_qalchemy_circuit(statevector, opt_params=None):
    num_qubits = int(np.ceil(np.log2(len(statevector))))
    qc = QuantumCircuit(num_qubits)
    qc.append(QAlchemyInitialize(statevector, opt_params=opt_params), qc.qubits)

    # Trigger Q-Alchemy compilation and return a concrete circuit.
    compiled = qc.decompose()
    return compiled


def fidelity_to_target(statevector: np.ndarray, circuit: QuantumCircuit):
    target = Statevector(statevector)
    current = Statevector.from_label("0" * circuit.num_qubits)

    fidelities = []
    gate_indices = []
    gate_labels = []
    gate_qubit_counts = []

    for idx, instruction in enumerate(circuit.data):
        op = instruction.operation
        if op.name == "barrier":
            continue
        qargs = [circuit.find_bit(q).index for q in instruction.qubits]
        current = current.evolve(op, qargs=qargs)
        fidelities.append(state_fidelity(current, target))
        gate_indices.append(idx)
        gate_labels.append(op.name)
        gate_qubit_counts.append(op.num_qubits)

    final_fidelity = state_fidelity(current, target)
    final_density_matrix = DensityMatrix(current)
    return gate_indices, gate_labels, gate_qubit_counts, fidelities, final_fidelity, final_density_matrix


def monotonic_score(values: list[float], tol: float = 1e-9):
    """Return fraction of steps that are non-decreasing within tolerance."""
    if len(values) < 2:
        return 1.0
    good = 0
    for prev, cur in zip(values[:-1], values[1:]):
        if cur + tol >= prev:
            good += 1
    return good / (len(values) - 1)


def search_monotonic_targets(
    num_qubits: int,
    n_max: int,
    extent_a_mu: float,
    opt_params: OptParams,
    min_final_fidelity: float,
    min_monotonic_ratio: float,
    tol: float,
):
    results = []
    all_results = []
    for n in range(1, n_max + 1):
        for l in range(0, n):
            for m in range(-l, l + 1):
                statevector = hydrogen_statevector(
                    num_qubits=num_qubits,
                    n=n,
                    l=l,
                    m=m,
                    extent_a_mu=extent_a_mu,
                )
                circuit = build_qalchemy_circuit(statevector, opt_params=opt_params)
                _, _, _, fidelities, final_fidelity, _ = fidelity_to_target(
                    statevector, circuit
                )
                ratio = monotonic_score(fidelities, tol=tol)
                all_results.append((n, l, m, ratio, final_fidelity, circuit.depth()))
                if final_fidelity >= min_final_fidelity and ratio >= min_monotonic_ratio:
                    results.append((n, l, m, ratio, final_fidelity, circuit.depth()))
    return results, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Hydrogen Q-Alchemy state prep + fidelity tracing"
    )
    parser.add_argument("--num-qubits", type=int, default=10)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--extent-a-mu", type=float, default=30.0)
    parser.add_argument("--max-fidelity-loss", type=float, default=0.02)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--n-max", type=int, default=4)
    parser.add_argument("--min-final-fidelity", type=float, default=0.9)
    parser.add_argument("--min-monotonic-ratio", type=float, default=0.95)
    parser.add_argument("--monotonic-tol", type=float, default=1e-9)

    # Dissipative-tail flags
    parser.add_argument(
        "--tail", action="store_true",
        help="Enable dissipative cooling tail after the circuit",
    )
    parser.add_argument(
        "--tail-only", action="store_true",
        help="Skip gate-by-gate fidelity trace; run only the tail "
             "(implies --tail)",
    )
    parser.add_argument("--tail-gamma", type=float, default=1.0,
                        help="Dissipation rate gamma (default: 1.0)")
    parser.add_argument("--tail-tmax", type=float, default=5.0,
                        help="Total tail duration (default: 5.0)")
    parser.add_argument("--tail-steps", type=int, default=50,
                        help="Number of time steps in tail (default: 50)")

    # Constrained-tail fitting flags
    parser.add_argument(
        "--fit-constrained-tail", action="store_true",
        help="Fit a lab-constrained tail to the teacher and compare "
             "(implies --tail)",
    )
    parser.add_argument("--fit-loss", choices=["fidelity", "frobenius"],
                        default="fidelity",
                        help="Loss function for fitting (default: fidelity)")
    parser.add_argument("--gamma-max", type=float, default=2.0,
                        help="Max rate for constrained channels (default: 2.0)")
    parser.add_argument("--allow-2q", action="store_true", default=True,
                        help="Enable 2-qubit ZZ dephasing (default: on)")
    parser.add_argument("--no-2q", action="store_true",
                        help="Disable 2-qubit channels")
    parser.add_argument("--allow-ancilla-reset", action="store_true",
                        help="Enable ancilla-reset primitives")
    parser.add_argument("--n-train-states", type=int, default=5,
                        help="Number of training states (default: 5)")
    parser.add_argument("--n-valid-states", type=int, default=3,
                        help="Number of validation states (default: 3)")
    parser.add_argument("--fit-iters", type=int, default=200,
                        help="Max optimiser iterations (default: 200)")

    args = parser.parse_args()
    if args.tail_only:
        args.tail = True
    if args.fit_constrained_tail:
        args.tail = True
    if args.no_2q:
        args.allow_2q = False

    api_key = os.getenv("Q_ALCHEMY_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing Q_ALCHEMY_API_KEY. Export it and re-run:\n"
            "  export Q_ALCHEMY_API_KEY=\"...\""
        )

    opt_params = OptParams(api_key=api_key, max_fidelity_loss=args.max_fidelity_loss)
    num_qubits = args.num_qubits

    if args.search:
        results, all_results = search_monotonic_targets(
            num_qubits=num_qubits,
            n_max=args.n_max,
            extent_a_mu=args.extent_a_mu,
            opt_params=opt_params,
            min_final_fidelity=args.min_final_fidelity,
            min_monotonic_ratio=args.min_monotonic_ratio,
            tol=args.monotonic_tol,
        )
        all_results.sort(key=lambda x: (x[3], x[4]), reverse=True)
        print("Top candidates (n, l, m, monotonic_ratio, final_fidelity, depth):")
        for row in all_results[:10]:
            print(row)
        if not results:
            print("No targets found matching criteria.")
            return
        results.sort(key=lambda x: (x[3], x[4]), reverse=True)
        print("Candidates (n, l, m, monotonic_ratio, final_fidelity, depth):")
        for row in results:
            print(row)
        return

    statevector = hydrogen_statevector(
        num_qubits=num_qubits,
        n=args.n,
        l=args.l,
        m=args.m,
        extent_a_mu=args.extent_a_mu,
    )
    circuit = build_qalchemy_circuit(statevector, opt_params=opt_params)
    print("Depth:", circuit.depth())

    # -----------------------------------------------------------------
    # Gate-by-gate fidelity trace  (skipped when --tail-only)
    # -----------------------------------------------------------------
    if args.tail_only:
        # Fast path: evolve the full circuit in one shot.
        current = Statevector.from_label("0" * circuit.num_qubits).evolve(circuit)
        final_fidelity = state_fidelity(current, Statevector(statevector))
        final_density_matrix = DensityMatrix(current)
        print("Final fidelity (circuit):", final_fidelity)
    else:
        (
            gate_indices,
            gate_labels,
            gate_qubit_counts,
            fidelities,
            final_fidelity,
            final_density_matrix,
        ) = fidelity_to_target(statevector, circuit)
        if not fidelities:
            raise RuntimeError("No gates found to plot fidelity.")
        print("Final fidelity (circuit):", final_fidelity)

        csv_path = REPO_ROOT / "experiments" / "hydrogen_qalchemy" / "fidelities.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["gate_index", "gate_name", "num_qubits", "fidelity"])
            writer.writerows(
                zip(gate_indices, gate_labels, gate_qubit_counts, fidelities)
            )
        print("Saved fidelities to:", csv_path)

        # Prepare downsampled circuit data for the combined plot.
        plot_every = 5
        plot_indices = gate_indices[::plot_every]
        plot_fidelities = fidelities[::plot_every]
        if plot_indices and plot_indices[-1] != gate_indices[-1]:
            plot_indices.append(gate_indices[-1])
            plot_fidelities.append(fidelities[-1])

    # =================================================================
    # Dissipative tail  (enabled by --tail or --tail-only)
    # =================================================================
    if args.tail:
        print()
        print("=" * 64)
        print("  Dissipative tail  (target-cooling channel)")
        print("=" * 64)

        rho = np.asarray(final_density_matrix.data, dtype=complex)

        fid_before = tail_fidelity(rho, statevector)
        print(f"  Fidelity before tail : {fid_before:.10f}")
        print(f"  gamma = {args.tail_gamma},  "
              f"tmax = {args.tail_tmax},  steps = {args.tail_steps}")

        tail_times, tail_fids, tail_trs, tail_purs = run_target_cooling_trajectory(
            rho,
            statevector,
            gamma=args.tail_gamma,
            tmax=args.tail_tmax,
            steps=args.tail_steps,
        )

        fid_after = tail_fids[-1]
        print(f"  Fidelity after tail  : {fid_after:.10f}")
        print(f"  Fidelity gain        : {fid_after - fid_before:+.10f}")
        print(f"  Trace range          : "
              f"[{tail_trs.min():.12f}, {tail_trs.max():.12f}]")

        # Verify against the analytic formula.
        f0 = tail_fids[0]
        fids_analytic = 1.0 - (1.0 - f0) * np.exp(-args.tail_gamma * tail_times)
        max_formula_err = np.max(np.abs(tail_fids - fids_analytic))
        print(f"  Max deviation from F(t)=1-(1-F0)exp(-gt) : {max_formula_err:.2e}")
        print("=" * 64)

        # Save tail CSV.
        tail_csv = (
            REPO_ROOT / "experiments" / "hydrogen_qalchemy" / "tail_fidelities.csv"
        )
        with tail_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "fidelity", "trace", "purity"])
            writer.writerows(
                zip(tail_times, tail_fids, tail_trs, tail_purs)
            )
        print("Saved tail fidelities to:", tail_csv)

    # =================================================================
    # Constrained-tail fitting  (enabled by --fit-constrained-tail)
    # =================================================================
    constrained_fids = None
    if args.fit_constrained_tail:
        print()
        print("=" * 64)
        print("  Fitting lab-constrained tail to teacher")
        print("=" * 64)

        d = 1 << num_qubits
        rho_init = np.asarray(final_density_matrix.data, dtype=complex)

        cfg = ConstraintConfig(
            n_qubits=num_qubits,
            allow_2q=args.allow_2q,
            gamma_max=args.gamma_max,
            enable_amp_damp=True,
            enable_dephasing=True,
            enable_depolarizing=(num_qubits <= 6),
            allow_ancilla_reset=args.allow_ancilla_reset,
        )

        rho_train = generate_training_states(
            d,
            n_states=args.n_train_states,
            psi_target=statevector,
            rho_circuit=rho_init,
            seed=42,
        )

        result = fit_to_teacher(
            psi_target=statevector,
            rho_train=rho_train,
            cfg=cfg,
            gamma_teacher=args.tail_gamma,
            tmax=args.tail_tmax,
            steps=args.tail_steps,
            loss=args.fit_loss,
            max_iter=args.fit_iters,
            seed=0,
            verbose=True,
        )
        fitted_model = result["model"]

        # Run the constrained tail on the circuit output.
        rhos_c = fitted_model.run_trajectory(
            rho_init, args.tail_tmax, args.tail_steps,
        )
        constrained_fids = np.array([
            tail_fidelity(r, statevector) for r in rhos_c
        ])

        print(f"\n  Constrained tail fidelity: "
              f"{constrained_fids[0]:.6f} -> {constrained_fids[-1]:.6f}")
        print(f"  Teacher tail fidelity   : "
              f"{tail_fids[0]:.6f} -> {tail_fids[-1]:.6f}")

        # Validation.
        rho_valid = generate_training_states(
            d,
            n_states=args.n_valid_states,
            psi_target=statevector,
            seed=999,
        )
        evaluate_and_report(
            fitted_model, statevector, rho_valid,
            args.tail_gamma, args.tail_tmax, args.tail_steps,
        )

        # Save constrained-tail CSV.
        c_csv = (
            REPO_ROOT / "experiments" / "hydrogen_qalchemy"
            / "constrained_tail_fidelities.csv"
        )
        c_times = np.linspace(0, args.tail_tmax, args.tail_steps + 1)
        c_traces = np.array([float(np.real(np.trace(r))) for r in rhos_c])
        with c_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time", "fidelity", "trace"])
            writer.writerows(zip(c_times, constrained_fids, c_traces))
        print(f"  Saved constrained tail CSV to: {c_csv}")
        print("=" * 64)

    # =================================================================
    # Unified plot  (fidelity + infidelity, circuit and tail combined)
    # =================================================================
    has_circuit = not args.tail_only      # gate-by-gate data available
    has_tail = args.tail                  # tail data available

    fig, ax = plt.subplots(figsize=(10, 5))

    C_LINE = "#2c3e50"                    # dark blue-gray – the simulation
    C_ANLY = "#c0392b"                    # red – analytic formula
    C_BG_C = "#3498db"                    # blue  – circuit background
    C_BG_T = "#e74c3c"                    # red   – tail background

    # ---- Map both phases onto a shared x-axis ----
    if has_circuit and has_tail:
        circ_x = np.asarray(plot_indices, dtype=float)
        circ_fid = np.asarray(plot_fidelities)
        x_last = float(gate_indices[-1])
        gap = x_last * 0.04
        tail_span = x_last * 0.55
        tail_x = x_last + gap + (
            tail_times / max(args.tail_tmax, 1e-15) * tail_span
        )
        x_sep = x_last + gap / 2
    elif has_circuit:
        circ_x = np.asarray(plot_indices, dtype=float)
        circ_fid = np.asarray(plot_fidelities)
    else:                                 # tail-only
        tail_x = tail_times

    # ---- Plot data ----
    if has_circuit and has_tail:
        # --- Hybrid y-scale: log below F₀, linear above F₀ ---
        # This avoids a twin-axis misalignment: the circuit spans many
        # orders of magnitude (needs log), while the tail rises from
        # F₀→1 (needs linear).  A single custom scale gives a seamless
        # hand-off with no visual jump at the transition point.
        F0 = float(tail_fids[0])
        log_F0 = np.log10(max(F0, 1e-16))
        fid_min = float(np.clip(circ_fid.min(), 1e-16, None))
        log_min = np.log10(fid_min)
        log_range = log_F0 - log_min         # decades in the log part
        lin_span = log_range * 0.30          # visual height for linear tail

        def _fwd(y):
            y = np.asarray(y, dtype=float)
            return np.where(
                y <= F0,
                np.log10(np.clip(y, 1e-16, None)),
                log_F0 + (y - F0) / max(1.0 - F0, 1e-30) * lin_span,
            )

        def _inv(z):
            z = np.asarray(z, dtype=float)
            return np.where(
                z <= log_F0,
                10.0 ** z,
                F0 + (z - log_F0) / max(lin_span, 1e-30) * (1.0 - F0),
            )

        ax.set_yscale("function", functions=(_fwd, _inv))

        # Single continuous line: circuit → bridge → tail
        all_x = np.concatenate([circ_x, [x_sep], tail_x])
        all_fid = np.concatenate([circ_fid, [F0], tail_fids])
        ax.plot(all_x, np.clip(all_fid, 1e-16, None),
                "-", color=C_LINE, lw=1.3, label="Simulation", zorder=3)

        # Analytic overlay (tail region only)
        ax.plot(tail_x, fids_analytic, "--", color=C_ANLY, lw=1.0,
                alpha=0.7, label=r"$1\!-\!(1\!-\!F_0)\,e^{-\gamma t}$",
                zorder=2)

        # Constrained-tail overlay (if fitted)
        C_CONS = "#27ae60"                    # green – constrained tail
        if constrained_fids is not None:
            ax.plot(tail_x, constrained_fids, "-.", color=C_CONS, lw=1.2,
                    alpha=0.85, label="Constrained tail", zorder=2)

        # Hand-off dot
        ax.plot(x_sep, F0, "o", color=C_LINE, ms=4, zorder=4)

        # Background shading + separator
        ax.axvline(x_sep, color="#777777", ls="--", lw=0.8, zorder=1)
        ax.axvspan(circ_x[0], x_sep, alpha=0.06, color=C_BG_C,
                   label="Circuit phase", zorder=0)
        ax.axvspan(x_sep, tail_x[-1], alpha=0.06, color=C_BG_T,
                   label="Dissipative tail", zorder=0)

        ax.set_ylabel(r"Fidelity  $\langle\psi_*|\rho|\psi_*\rangle$")
        ax.legend(fontsize=8, loc="center right")
        ax.grid(True, alpha=0.25)

        # y-axis limits
        ax.set_ylim(fid_min * 0.5, 1.005)

        # Custom y-ticks: log powers below F₀, linear decimals above
        exp_lo = int(math.floor(np.log10(max(fid_min, 1e-16))))
        exp_hi = int(math.floor(log_F0))
        n_decades = exp_hi - exp_lo
        step = max(1, n_decades // 5)
        log_tv = [10.0 ** e for e in range(exp_lo, exp_hi + 1, step)
                  if 10.0 ** e <= F0]
        lin_tv = np.linspace(F0, 1.0, 4).tolist()
        all_tv = log_tv + lin_tv
        all_tl = ([f"$10^{{{int(np.log10(v))}}}$" for v in log_tv]
                  + [f"{v:.2f}" for v in lin_tv])
        ax.set_yticks(all_tv)
        ax.set_yticklabels(all_tl)

    elif has_circuit:
        ax.plot(circ_x, np.clip(circ_fid, 1e-16, None),
                "-", color=C_LINE, lw=1.3, label="Simulation", zorder=3)
        ax.set_yscale("log")
        ax.set_ylabel(r"Fidelity  $\langle\psi_*|\rho|\psi_*\rangle$")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25, which="both")

    elif has_tail:
        # Tail only – plain linear
        ax.plot(tail_x, tail_fids, "-", color=C_LINE, lw=1.3,
                label="Teacher", zorder=3)
        ax.plot(tail_x, fids_analytic, "--", color=C_ANLY, lw=1.0,
                alpha=0.7,
                label=r"$1\!-\!(1\!-\!F_0)\,e^{-\gamma t}$", zorder=2)
        if constrained_fids is not None:
            ax.plot(tail_x, constrained_fids, "-.", color="#27ae60", lw=1.2,
                    alpha=0.85, label="Constrained tail", zorder=2)
        ax.set_ylabel(r"Fidelity  $\langle\psi_*|\rho|\psi_*\rangle$")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.25)

    # ---- Custom coloured tick labels (gate indices + tail times) ----
    if has_circuit and has_tail:
        nc = min(6, len(circ_x))
        ci = np.round(np.linspace(0, len(circ_x) - 1, nc)).astype(int)
        nt = 5
        ti = np.round(np.linspace(0, len(tail_x) - 1, nt)).astype(int)
        positions = ([circ_x[i] for i in ci]
                     + [tail_x[j] for j in ti])
        labels = ([str(int(plot_indices[i])) for i in ci]
                  + [f"t={tail_times[j]:.1f}" for j in ti])
        colors = [C_BG_C] * nc + [C_BG_T] * nt
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        for lbl, c in zip(ax.get_xticklabels(), colors):
            lbl.set_color(c)
    elif has_circuit:
        ax.set_xlabel("Gate index")
    else:
        ax.set_xlabel(r"$t$")

    # ---- Suptitle ----
    title = f"n={args.n}, l={args.l}, m={args.m}  |  {num_qubits} qubits"
    if has_tail:
        title += r"  |  $\gamma$" + f" = {args.tail_gamma}"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plot_path = REPO_ROOT / "experiments" / "hydrogen_qalchemy" / "fidelity_plot.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
