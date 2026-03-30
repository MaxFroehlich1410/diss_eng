r"""4-qubit experiment: Krotov dissipative steady-state engineering.

Demonstrates the generalised Krotov method by learning global dissipative
dynamics that have a GHZ state as unique steady state.

Three experiments are run:
    1. Target-cooling operators (known optimal -- baseline)
    2. Physical operators (single-qubit + nearest-neighbour)
    3. Random complex operators

Each experiment optimises time-dependent Lindblad amplitudes, then
analyses the resulting time-independent generator (from time-averaged
rates) for steady-state uniqueness and spectral gap.

Results are saved as .npz data files and matplotlib plots.
"""

from __future__ import annotations

import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .liouville import (
    pure_state_dm, fidelity_pure, dissipator_superop,
    build_liouvillian_from_amplitudes, vectorize, unvectorize,
    trace_dm, is_physical,
)
from .operators import (
    ghz_state, w_state, random_pure_state, maximally_mixed_state,
    physical_operator_basis, random_operators,
)
from .krotov import run_krotov, analyse_steady_state, KrotovConfig
from scipy.linalg import null_space, expm


def target_cooling_operators(psi_target: np.ndarray) -> list[np.ndarray]:
    """Lindblad operators L_k = |psi*><psi_k^perp| for target cooling."""
    psi = psi_target / np.linalg.norm(psi_target)
    orth = null_space(psi.conj().reshape(1, -1))
    return [np.outer(psi, orth[:, k].conj()) for k in range(orth.shape[1])]


def run_convergence_from_multiple_inits(
    psi_target: np.ndarray,
    lindblad_ops: list[np.ndarray],
    avg_rates: np.ndarray,
    n_init: int = 6,
    T: float = 20.0,
    N_steps: int = 200,
) -> dict:
    """Test convergence from multiple initial states with time-independent rates."""
    d = len(psi_target)
    S_ops = [dissipator_superop(L) for L in lindblad_ops]
    L_avg = build_liouvillian_from_amplitudes(S_ops, avg_rates)
    dt = T / N_steps
    propagator = expm(L_avg * dt)

    times = np.linspace(0, T, N_steps + 1)
    all_fidelities = []
    init_labels = []

    rng = np.random.default_rng(999)

    # 1. Computational basis |0...0>
    psi0 = np.zeros(d, dtype=complex); psi0[0] = 1.0
    init_labels.append("|0...0>")

    # 2. Maximally mixed
    init_labels.append("I/d")

    # 3-6. Random pure states
    for i in range(4):
        init_labels.append(f"random_{i}")

    inits = []
    inits.append(pure_state_dm(psi0))
    inits.append(maximally_mixed_state(d))
    for i in range(4):
        p = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        p /= np.linalg.norm(p)
        inits.append(pure_state_dm(p))

    for rho0 in inits:
        rho_vec = vectorize(rho0)
        fids = [fidelity_pure(rho0, psi_target)]
        for _ in range(N_steps):
            rho_vec = propagator @ rho_vec
            rho = unvectorize(rho_vec, d)
            fids.append(fidelity_pure(rho, psi_target))
        all_fidelities.append(np.array(fids))

    return {
        "times": times,
        "fidelities": all_fidelities,
        "labels": init_labels,
    }


def run_experiment(output_dir: str = "krotov_dissipative/results"):
    """Run the full 4-qubit experiment suite."""
    os.makedirs(output_dir, exist_ok=True)

    n_qubits = 4
    d = 2 ** n_qubits
    psi_target = ghz_state(n_qubits)
    rho0 = pure_state_dm(np.zeros(d, dtype=complex).__setitem__(0, 1.0) or np.eye(1))

    # Fix initial state properly
    psi0 = np.zeros(d, dtype=complex)
    psi0[0] = 1.0
    rho0 = pure_state_dm(psi0)

    print("=" * 70)
    print("  4-Qubit Krotov Dissipative Steady-State Engineering")
    print(f"  Target: GHZ state on {n_qubits} qubits (d = {d})")
    print(f"  Initial fidelity: {fidelity_pure(rho0, psi_target):.6f}")
    print("=" * 70)

    results = {}

    # ----------------------------------------------------------------
    # Experiment 1: Target-cooling operators (baseline)
    # ----------------------------------------------------------------
    print("\n--- Experiment 1: Target-cooling operators (known optimal) ---")
    ops_cooling = target_cooling_operators(psi_target)
    print(f"  Number of operators: {len(ops_cooling)}")

    config1 = KrotovConfig(
        n_qubits=n_qubits, T=5.0, N_t=50, max_iter=80,
        lambda_reg=0.5, tol=1e-7, u_init=0.3, verbose=True
    )
    t0 = time.time()
    res1 = run_krotov(rho0, psi_target, ops_cooling, config1)
    t1 = time.time()
    print(f"  Time: {t1-t0:.1f}s, Final F: {res1.fidelities[-1]:.8f}")

    analysis1 = analyse_steady_state(psi_target, ops_cooling, res1.final_controls)
    print(f"  Spectral gap: {analysis1['spectral_gap']:.6f}")
    print(f"  SS fidelity: {analysis1['steady_state_fidelity']:.8f}")
    print(f"  Unique SS: {analysis1['is_unique_steady_state']}")

    conv1 = run_convergence_from_multiple_inits(
        psi_target, ops_cooling, analysis1["avg_rates"]
    )

    results["cooling"] = {
        "fidelities": res1.fidelities,
        "infidelities": res1.infidelities,
        "analysis": analysis1,
        "convergence": conv1,
        "time": t1 - t0,
        "n_ops": len(ops_cooling),
    }

    # ----------------------------------------------------------------
    # Experiment 2: Physical operators (single-qubit + NN)
    # ----------------------------------------------------------------
    print("\n--- Experiment 2: Physical operators (1-body + 2-body NN) ---")
    ops_phys = physical_operator_basis(n_qubits)
    print(f"  Number of operators: {len(ops_phys)}")

    config2 = KrotovConfig(
        n_qubits=n_qubits, T=8.0, N_t=60, max_iter=120,
        lambda_reg=0.3, tol=1e-7, u_init=0.2, verbose=True
    )
    t0 = time.time()
    res2 = run_krotov(rho0, psi_target, ops_phys, config2)
    t2 = time.time()
    print(f"  Time: {t2-t0:.1f}s, Final F: {res2.fidelities[-1]:.8f}")

    analysis2 = analyse_steady_state(psi_target, ops_phys, res2.final_controls)
    print(f"  Spectral gap: {analysis2['spectral_gap']:.6f}")
    print(f"  SS fidelity: {analysis2['steady_state_fidelity']:.8f}")
    print(f"  Unique SS: {analysis2['is_unique_steady_state']}")

    conv2 = run_convergence_from_multiple_inits(
        psi_target, ops_phys, analysis2["avg_rates"]
    )

    results["physical"] = {
        "fidelities": res2.fidelities,
        "infidelities": res2.infidelities,
        "analysis": analysis2,
        "convergence": conv2,
        "time": t2 - t0,
        "n_ops": len(ops_phys),
    }

    # ----------------------------------------------------------------
    # Experiment 3: Random operators
    # ----------------------------------------------------------------
    print("\n--- Experiment 3: Random complex operators ---")
    ops_rand = random_operators(d, K=20, seed=42)
    print(f"  Number of operators: {len(ops_rand)}")

    config3 = KrotovConfig(
        n_qubits=n_qubits, T=8.0, N_t=60, max_iter=120,
        lambda_reg=0.3, tol=1e-7, u_init=0.2, verbose=True
    )
    t0 = time.time()
    res3 = run_krotov(rho0, psi_target, ops_rand, config3)
    t3 = time.time()
    print(f"  Time: {t3-t0:.1f}s, Final F: {res3.fidelities[-1]:.8f}")

    analysis3 = analyse_steady_state(psi_target, ops_rand, res3.final_controls)
    print(f"  Spectral gap: {analysis3['spectral_gap']:.6f}")
    print(f"  SS fidelity: {analysis3['steady_state_fidelity']:.8f}")
    print(f"  Unique SS: {analysis3['is_unique_steady_state']}")

    conv3 = run_convergence_from_multiple_inits(
        psi_target, ops_rand, analysis3["avg_rates"]
    )

    results["random"] = {
        "fidelities": res3.fidelities,
        "infidelities": res3.infidelities,
        "analysis": analysis3,
        "convergence": conv3,
        "time": t3 - t0,
        "n_ops": len(ops_rand),
    }

    # ----------------------------------------------------------------
    # Generate plots
    # ----------------------------------------------------------------
    print("\n--- Generating plots ---")
    _plot_optimisation_convergence(results, output_dir)
    _plot_steady_state_convergence(results, output_dir)
    _plot_spectral_analysis(results, output_dir)
    _plot_control_pulses(res1, res2, res3, config1, config2, config3, output_dir)

    # Save summary
    summary = {}
    for name, r in results.items():
        summary[name] = {
            "n_ops": r["n_ops"],
            "final_fidelity": r["fidelities"][-1],
            "n_iterations": len(r["fidelities"]) - 1,
            "spectral_gap": float(r["analysis"]["spectral_gap"]),
            "ss_fidelity": float(r["analysis"]["steady_state_fidelity"]),
            "is_unique_ss": bool(r["analysis"]["is_unique_steady_state"]),
            "time_s": r["time"],
        }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {output_dir}/summary.json")

    return results


def _plot_optimisation_convergence(results, output_dir):
    """Plot fidelity vs Krotov iteration for all three experiments."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    labels = {"cooling": "Target cooling", "physical": "Physical (1+2 body)",
              "random": "Random operators"}
    colors = {"cooling": "#2196F3", "physical": "#4CAF50", "random": "#FF9800"}

    for name, r in results.items():
        iters = range(len(r["fidelities"]))
        ax1.plot(iters, r["fidelities"], label=labels[name], color=colors[name], lw=2)
        ax2.semilogy(iters, r["infidelities"], label=labels[name], color=colors[name], lw=2)

    ax1.set_xlabel("Krotov iteration")
    ax1.set_ylabel("Fidelity F(ρ(T), ρ*)")
    ax1.set_title("Optimisation convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    ax2.set_xlabel("Krotov iteration")
    ax2.set_ylabel("Infidelity 1 - F")
    ax2.set_title("Infidelity convergence (log scale)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "optimisation_convergence.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_steady_state_convergence(results, output_dir):
    """Plot fidelity vs time for the time-independent generator from multiple inits."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = {"cooling": "Target cooling", "physical": "Physical (1+2 body)",
              "random": "Random operators"}

    for ax, (name, r) in zip(axes, results.items()):
        conv = r["convergence"]
        for fids, label in zip(conv["fidelities"], conv["labels"]):
            ax.plot(conv["times"], fids, label=label, lw=1.5)
        ax.set_xlabel("Time t")
        ax.set_ylabel("Fidelity F(ρ(t), ρ*)")
        ax.set_title(f"{labels[name]}\n(spectral gap = {r['analysis']['spectral_gap']:.4f})")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "steady_state_convergence.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_spectral_analysis(results, output_dir):
    """Plot eigenvalue spectrum of the time-averaged Liouvillian."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    labels = {"cooling": "Target cooling", "physical": "Physical (1+2 body)",
              "random": "Random operators"}
    colors = {"cooling": "#2196F3", "physical": "#4CAF50", "random": "#FF9800"}

    for ax, (name, r) in zip(axes, results.items()):
        # Recompute full eigenvalue spectrum
        from .liouville import dissipator_superop, build_liouvillian_from_amplitudes
        avg_rates = r["analysis"]["avg_rates"]

        # We only stored top-10 real parts, so let's just plot those
        eig_reals = r["analysis"]["eigenvalues_real"]
        ax.barh(range(len(eig_reals)), eig_reals, color=colors[name], alpha=0.7)
        ax.set_xlabel("Re(λ)")
        ax.set_ylabel("Eigenvalue index")
        ax.set_title(f"{labels[name]}")
        ax.axvline(x=0, color="k", ls="--", lw=0.8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "spectral_analysis.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _plot_control_pulses(res1, res2, res3, cfg1, cfg2, cfg3, output_dir):
    """Plot a sample of optimised control amplitudes vs time."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    data = [
        (res1, cfg1, "Target cooling"),
        (res2, cfg2, "Physical operators"),
        (res3, cfg3, "Random operators"),
    ]

    for ax, (res, cfg, title) in zip(axes, data):
        controls = res.final_controls
        N_t, K = controls.shape
        times = np.linspace(0, cfg.T, N_t)
        n_show = min(K, 8)
        for k in range(n_show):
            ax.plot(times, controls[:, k], lw=1.0, alpha=0.7, label=f"u_{k}")
        ax.set_xlabel("Time t")
        ax.set_ylabel("Amplitude u_k(t)")
        ax.set_title(title)
        if n_show <= 8:
            ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "control_pulses.pdf")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    run_experiment()
