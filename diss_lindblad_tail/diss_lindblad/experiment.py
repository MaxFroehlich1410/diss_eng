"""Experiment orchestrator: approximate circuit  +  dissipative Lindblad tail.

The :func:`run` function executes the full pipeline:

1. Construct a target state |psi*>.
2. Build the exact preparation circuit, truncate it -> U_approx.
3. Compute  rho_0 = U_approx |0><0| U_approx^dag.
4. Build the Liouvillian from the chosen Lindblad operators.
5. Evolve  rho(t) = exp(L t) rho_0  and record fidelity at each t.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import density_matrix as dm
from . import lindblad
from . import circuits


# ---------------------------------------------------------------------------
# Configuration & result containers
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """All tuneable knobs for a single run."""
    n_qubits: int = 3
    target_name: str = "random"
    seed: int = 42
    gate_fraction: float = 0.5
    dissipation_type: str = "cooling"       # cooling | amplitude_damping | dephasing
    gamma: float = 1.0
    t_max: float = 5.0
    n_time_steps: int = 50


@dataclass
class ExperimentResult:
    """Collected results of one experiment run."""
    config: ExperimentConfig
    psi_target: np.ndarray
    n_gates_total: int
    n_gates_kept: int
    fidelity_initial: float
    times: np.ndarray
    fidelities: np.ndarray
    traces: np.ndarray
    purities: np.ndarray

    @property
    def fidelity_final(self) -> float:
        return float(self.fidelities[-1])

    @property
    def fidelity_gain(self) -> float:
        return self.fidelity_final - self.fidelity_initial


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_target_state(name: str, n_qubits: int, seed: int = 42) -> np.ndarray:
    """Return the target state vector by name."""
    generators = {
        "random": lambda: circuits.random_statevector(n_qubits, seed=seed),
        "ghz": lambda: circuits.ghz_state(n_qubits),
        "w": lambda: circuits.w_state(n_qubits),
    }
    if name not in generators:
        raise ValueError(
            f"Unknown target: {name!r}.  Choose from: {list(generators)}."
        )
    return generators[name]()


def make_lindblad_ops(
    dissipation_type: str,
    psi_target: np.ndarray,
    n_qubits: int,
) -> list[np.ndarray]:
    """Return the Lindblad operator list for the requested channel."""
    if dissipation_type == "cooling":
        return lindblad.target_cooling_operators(psi_target)
    elif dissipation_type == "amplitude_damping":
        return lindblad.amplitude_damping_operators(n_qubits)
    elif dissipation_type == "dephasing":
        return lindblad.dephasing_operators(n_qubits)
    else:
        raise ValueError(
            f"Unknown dissipation type: {dissipation_type!r}.  "
            "Choose from: cooling, amplitude_damping, dephasing."
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: ExperimentConfig) -> ExperimentResult:
    """Execute the full pipeline and return an :class:`ExperimentResult`.

    Steps
    -----
    1. Construct target |psi*>.
    2. Build & truncate the exact preparation circuit  ->  U_approx.
    3. Compute  rho_0 = U_approx |0><0| U_approx^dag.
    4. Build the Liouvillian from the chosen Lindblad operators.
    5. Evolve  rho(t)  and record fidelity, trace, purity at each t.
    """
    # 1 ---- target state
    psi_target = make_target_state(cfg.target_name, cfg.n_qubits, cfg.seed)

    # 2 ---- circuit construction & truncation
    exact_circ = circuits.build_exact_circuit(psi_target)
    approx_circ, n_keep, n_total = circuits.truncate_circuit(
        exact_circ, cfg.gate_fraction,
    )
    U_approx = circuits.circuit_to_unitary(approx_circ)

    # 3 ---- initial density matrix
    d = 2 ** cfg.n_qubits
    zero_state = np.zeros(d, dtype=complex)
    zero_state[0] = 1.0
    rho_init = dm.pure_state_dm(U_approx @ zero_state)
    fidelity_init = dm.fidelity_to_pure(rho_init, psi_target)

    # 4 ---- Liouvillian
    ops = make_lindblad_ops(cfg.dissipation_type, psi_target, cfg.n_qubits)
    rates = np.full(len(ops), cfg.gamma)
    liouvillian = lindblad.build_liouvillian(ops, rates)

    # 5 ---- time evolution  (incremental propagation)
    times = np.linspace(0.0, cfg.t_max, cfg.n_time_steps + 1)
    snapshots = lindblad.evolve_trajectory(rho_init, liouvillian, times)

    fidelities = np.array([dm.fidelity_to_pure(rho, psi_target) for rho in snapshots])
    traces = np.array([float(np.real(dm.trace(rho))) for rho in snapshots])
    purities = np.array([dm.purity(rho) for rho in snapshots])

    return ExperimentResult(
        config=cfg,
        psi_target=psi_target,
        n_gates_total=n_total,
        n_gates_kept=n_keep,
        fidelity_initial=fidelity_init,
        times=times,
        fidelities=fidelities,
        traces=traces,
        purities=purities,
    )
