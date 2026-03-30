"""Plot the 8-qubit hydrogen state preparation circuit from Q-Alchemy."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, state_fidelity
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "q-alchemy-sdk-py" / "src"))

from q_alchemy.initialize import OptParams
from q_alchemy.qiskit_integration import QAlchemyInitialize
from hydrogen_wavefunction import compute_psi_xz_slice


NUM_QUBITS = 8
N, L, M = 4, 1, 0
EXTENT = 30.0
MAX_FIDELITY_LOSS = 0.02


def hydrogen_statevector(
    num_qubits: int, n: int, l: int, m: int, extent_a_mu: float,
) -> np.ndarray:
    grid_side = 2 ** (num_qubits // 2)
    _, _, psi, _ = compute_psi_xz_slice(
        n, l, m, extent_a_mu=extent_a_mu, grid_points=grid_side,
    )
    arr = psi.astype(complex).flatten()
    return arr / np.linalg.norm(arr)


def main():
    api_key = os.getenv("Q_ALCHEMY_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing Q_ALCHEMY_API_KEY. Export it and re-run:\n"
            "  export Q_ALCHEMY_API_KEY=\"...\""
        )

    print(f"Building hydrogen target: n={N}, l={L}, m={M}, "
          f"{NUM_QUBITS} qubits, extent={EXTENT}")
    sv = hydrogen_statevector(NUM_QUBITS, N, L, M, EXTENT)

    print(f"Calling Q-Alchemy (max_fidelity_loss={MAX_FIDELITY_LOSS})...")
    opt_params = OptParams(api_key=api_key, max_fidelity_loss=MAX_FIDELITY_LOSS)
    qc = QuantumCircuit(NUM_QUBITS)
    qc.append(QAlchemyInitialize(sv, opt_params=opt_params), qc.qubits)
    circuit = qc.decompose()

    final_sv = Statevector.from_label("0" * NUM_QUBITS).evolve(circuit)
    fidelity = state_fidelity(final_sv, Statevector(sv))
    print(f"Circuit depth: {circuit.depth()}")
    print(f"Gate count:    {sum(1 for i in circuit.data if i.operation.name != 'barrier')}")
    print(f"Fidelity:      {fidelity:.8f}")

    out_dir = Path(__file__).resolve().parent
    fig = circuit.draw(
        output="mpl",
        fold=-1,
        scale=0.5,
        style={
            "backgroundcolor": "#FFFFFF",
            "linecolor": "#2c3e50",
            "textcolor": "#2c3e50",
            "gatefacecolor": "#3498db",
            "gatetextcolor": "#FFFFFF",
            "barrierfacecolor": "#95a5a6",
            "fontsize": 8,
        },
    )
    fig.suptitle(
        f"Q-Alchemy: H atom ($n$={N}, $\\ell$={L}, $m$={M}), "
        f"{NUM_QUBITS} qubits, depth {circuit.depth()}, "
        f"$F$ = {fidelity:.4f}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()

    plot_path = out_dir / "hydrogen_8q_circuit.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    print(f"Circuit diagram saved to {plot_path}")

    folded_fig = circuit.draw(
        output="mpl",
        fold=40,
        scale=0.6,
        style={
            "backgroundcolor": "#FFFFFF",
            "linecolor": "#2c3e50",
            "textcolor": "#2c3e50",
            "gatefacecolor": "#3498db",
            "gatetextcolor": "#FFFFFF",
            "barrierfacecolor": "#95a5a6",
            "fontsize": 8,
        },
    )
    folded_fig.suptitle(
        f"Q-Alchemy: H atom ($n$={N}, $\\ell$={L}, $m$={M}), "
        f"{NUM_QUBITS} qubits, depth {circuit.depth()}, "
        f"$F$ = {fidelity:.4f}",
        fontsize=11,
        y=1.01,
    )
    folded_fig.tight_layout()

    folded_path = out_dir / "hydrogen_8q_circuit_folded.png"
    folded_fig.savefig(folded_path, dpi=200, bbox_inches="tight")
    print(f"Folded circuit diagram saved to {folded_path}")
    plt.show()


if __name__ == "__main__":
    main()
