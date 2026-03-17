"""Qiskit-based circuit construction for approximate state preparation.

Workflow
--------
1. Build the exact unitary U whose first column is |psi*> (so U|0> = |psi*>).
2. Wrap it in a Qiskit ``UnitaryGate`` and decompose to elementary gates.
3. Truncate the gate list to model a shallow / approximate circuit.
4. Extract the resulting unitary matrix via ``Operator``.

The module also provides small helper functions that generate well-known
target states (random, GHZ, W).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import null_space

from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator


# ---------------------------------------------------------------------------
# Unitary extension: |psi> -> d x d unitary with U[:,0] = psi
# ---------------------------------------------------------------------------

def extend_to_unitary(psi: np.ndarray) -> np.ndarray:
    """Build a d x d unitary U with  U |0> = |psi>.

    The remaining d-1 columns are an arbitrary orthonormal completion
    computed via the null-space of <psi|.
    """
    psi = np.asarray(psi, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    d = len(psi)

    if d == 1:
        return psi.reshape(1, 1)

    # Columns of N span the orthogonal complement of psi (shape d x d-1).
    N = null_space(psi.conj().reshape(1, -1))
    U = np.column_stack([psi, N])
    return U


# ---------------------------------------------------------------------------
# Circuit construction & truncation
# ---------------------------------------------------------------------------

def build_exact_circuit(psi_target: np.ndarray) -> QuantumCircuit:
    """Return a fully-decomposed Qiskit circuit that prepares |psi_target> from |0...0>.

    Uses ``UnitaryGate`` + recursive ``decompose`` to reach elementary gates
    (single-qubit rotations + CX).
    """
    psi = np.asarray(psi_target, dtype=complex).ravel()
    n_qubits = int(np.round(np.log2(len(psi))))

    U = extend_to_unitary(psi)
    qc = QuantumCircuit(n_qubits)
    qc.append(UnitaryGate(U), range(n_qubits))

    # Recursively decompose until only elementary gates remain.
    # For n <= 6 qubits, 25 rounds is more than enough.
    qc_dec = qc.decompose(reps=25)
    return qc_dec


def truncate_circuit(
    circuit: QuantumCircuit,
    gate_fraction: float,
) -> tuple[QuantumCircuit, int, int]:
    """Keep only the first *gate_fraction* of the gates in *circuit*.

    Barriers, resets, and measurements are skipped.

    Returns
    -------
    qc_trunc : QuantumCircuit
        Truncated circuit.
    n_kept : int
        Number of gates retained.
    n_total : int
        Total number of gates in the original circuit.
    """
    skip = {"barrier", "reset", "measure"}
    gates = [inst for inst in circuit.data if inst.operation.name not in skip]
    n_total = len(gates)
    n_keep = max(1, int(n_total * gate_fraction))

    qc = QuantumCircuit(circuit.num_qubits)
    for inst in gates[:n_keep]:
        qc.append(inst.operation, inst.qubits, inst.clbits)

    return qc, n_keep, n_total


def circuit_to_unitary(circuit: QuantumCircuit) -> np.ndarray:
    """Extract the unitary matrix implemented by a circuit."""
    return Operator(circuit).data


# ---------------------------------------------------------------------------
# Target-state generators
# ---------------------------------------------------------------------------

def random_statevector(n_qubits: int, seed: int = 42) -> np.ndarray:
    """Haar-random state vector on *n_qubits* qubits."""
    rng = np.random.default_rng(seed)
    d = 2 ** n_qubits
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return psi / np.linalg.norm(psi)


def ghz_state(n_qubits: int) -> np.ndarray:
    """GHZ state  (|00...0> + |11...1>) / sqrt(2)."""
    d = 2 ** n_qubits
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0
    psi[-1] = 1.0
    return psi / np.linalg.norm(psi)


def w_state(n_qubits: int) -> np.ndarray:
    """W state  (|10...0> + |01...0> + ... + |00...1>) / sqrt(n)."""
    d = 2 ** n_qubits
    psi = np.zeros(d, dtype=complex)
    for q in range(n_qubits):
        psi[1 << q] = 1.0
    return psi / np.linalg.norm(psi)
