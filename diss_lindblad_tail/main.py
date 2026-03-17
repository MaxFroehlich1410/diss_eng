#!/usr/bin/env python
"""CLI entry point: reproduce the dissipative-tail experiment.

Examples
--------
    # Default 3-qubit random-state experiment with cooling
    python main.py

    # GHZ target, 70 % of gates, amplitude damping, with plot
    python main.py --n-qubits 3 --target ghz --gate-fraction 0.7 \
                   --dissipation amplitude_damping --gamma 0.5 --plot

    # Run all sanity checks
    python main.py --check
"""

from __future__ import annotations

import argparse
import sys
import textwrap

import numpy as np

# Ensure the local package is importable without installation.
sys.path.insert(0, ".")

from diss_lindblad.experiment import ExperimentConfig, run
from tests.test_sanity import run_all as run_checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dissipative Lindblad tail after a shallow state-preparation circuit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            dissipation types
              cooling           - L_k = |psi*><psi_k^perp|  (drives toward target)
              amplitude_damping - sigma^- on each qubit      (T1 decay)
              dephasing         - Z on each qubit             (T2 dephasing)
        """),
    )
    p.add_argument("--n-qubits", type=int, default=3,
                   help="Number of qubits (default: 3)")
    p.add_argument("--target", choices=["random", "ghz", "w"], default="random",
                   help="Target state type (default: random)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--gate-fraction", type=float, default=0.5,
                   help="Fraction of exact-circuit gates to keep (0, 1] (default: 0.5)")
    p.add_argument("--dissipation",
                   choices=["cooling", "amplitude_damping", "dephasing"],
                   default="cooling",
                   help="Dissipation channel type (default: cooling)")
    p.add_argument("--gamma", type=float, default=1.0,
                   help="Dissipation rate (applied uniformly) (default: 1.0)")
    p.add_argument("--t-max", type=float, default=5.0,
                   help="Maximum evolution time (default: 5.0)")
    p.add_argument("--n-steps", type=int, default=50,
                   help="Number of time steps (default: 50)")
    p.add_argument("--plot", action="store_true",
                   help="Show matplotlib plots of fidelity, trace, purity vs. time")
    p.add_argument("--check", action="store_true",
                   help="Run sanity checks and exit")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------

def print_result(result) -> None:
    cfg = result.config
    print()
    print("=" * 66)
    print("  Dissipative Lindblad Tail  --  Experiment Results")
    print("=" * 66)
    print(f"  Qubits            : {cfg.n_qubits}  (d = {2**cfg.n_qubits})")
    print(f"  Target state      : {cfg.target_name}  (seed={cfg.seed})")
    print(f"  Circuit gates     : {result.n_gates_kept} / {result.n_gates_total}"
          f"  (fraction = {cfg.gate_fraction})")
    print(f"  Dissipation       : {cfg.dissipation_type}  (gamma = {cfg.gamma})")
    print(f"  Time range        : [0, {cfg.t_max}]  ({cfg.n_time_steps} steps)")
    print("-" * 66)
    print(f"  Fidelity (t=0)    : {result.fidelity_initial:.8f}")
    print(f"  Fidelity (t=end)  : {result.fidelity_final:.8f}")
    print(f"  Fidelity gain     : {result.fidelity_gain:+.8f}")
    print("-" * 66)

    # Compact table of selected time points.
    n_show = min(15, len(result.times))
    indices = np.linspace(0, len(result.times) - 1, n_show, dtype=int)
    print(f"  {'t':>8s}  {'Fidelity':>12s}  {'Tr(rho)':>10s}  {'Purity':>10s}")
    for i in indices:
        print(
            f"  {result.times[i]:8.4f}"
            f"  {result.fidelities[i]:12.8f}"
            f"  {result.traces[i]:10.8f}"
            f"  {result.purities[i]:10.8f}"
        )
    print("=" * 66)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_result(result) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # -- Fidelity --
    ax = axes[0]
    ax.plot(result.times, result.fidelities, "b-", linewidth=1.5)
    ax.axhline(result.fidelity_initial, color="gray", ls="--", lw=0.8,
               label=f"F_0 = {result.fidelity_initial:.4f}")
    ax.set_xlabel("t")
    ax.set_ylabel("Fidelity  <psi*|rho(t)|psi*>")
    ax.set_title("Fidelity vs. time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # -- Trace --
    ax = axes[1]
    ax.plot(result.times, result.traces, "r-", linewidth=1.5)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("Tr(rho)")
    ax.set_title("Trace preservation")
    ax.grid(True, alpha=0.3)
    pad = max(0.001, 3 * np.std(result.traces - 1.0))
    ax.set_ylim(1.0 - pad, 1.0 + pad)

    # -- Purity --
    ax = axes[2]
    ax.plot(result.times, result.purities, "g-", linewidth=1.5)
    ax.set_xlabel("t")
    ax.set_ylabel("Tr(rho^2)")
    ax.set_title("Purity vs. time")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{result.config.n_qubits}-qubit {result.config.target_name} state  |  "
        f"{result.config.dissipation_type} (gamma={result.config.gamma})",
        fontsize=12,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # -- sanity-check mode --
    if args.check:
        success = run_checks()
        sys.exit(0 if success else 1)

    cfg = ExperimentConfig(
        n_qubits=args.n_qubits,
        target_name=args.target,
        seed=args.seed,
        gate_fraction=args.gate_fraction,
        dissipation_type=args.dissipation,
        gamma=args.gamma,
        t_max=args.t_max,
        n_time_steps=args.n_steps,
    )

    # Warn about memory for large Liouvillian.
    d = 2 ** cfg.n_qubits
    liouv_elems = d ** 4          # d^2 x d^2 complex matrix
    mem_gb = liouv_elems * 16 / 1e9
    if mem_gb > 1.0:
        print(
            f"WARNING: Liouvillian is {d**2} x {d**2}  "
            f"(~{mem_gb:.1f} GB).  This may be slow or run out of memory."
        )

    print("Building circuit and evolving under Lindblad ...")
    result = run(cfg)
    print_result(result)

    if args.plot:
        plot_result(result)


if __name__ == "__main__":
    main()
