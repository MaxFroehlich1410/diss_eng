r"""Lindblad operator basis construction for multi-qubit systems.

Provides several operator bases for parametrising the dissipative part
of the Lindblad master equation:

1. **Pauli basis** -- tensor products of single-qubit Paulis
   (Hermitian, traceless except identity).
2. **Raising/lowering basis** -- sigma^+, sigma^-, Z on each qubit
   plus nearest-neighbour two-body operators.
3. **Random basis** -- K random complex matrices (for expressivity tests).
4. **Gell-Mann basis** -- generalised Gell-Mann matrices for SU(d).

For steady-state engineering the operator basis must be rich enough
to span the space of generators that admit the target state as a
fixed point.
"""

from __future__ import annotations

import numpy as np
from itertools import product as iproduct


# -------------------------------------------------------------------
# Pauli matrices
# -------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]
PAULI_LABELS = ["I", "X", "Y", "Z"]


def pauli_basis(n_qubits: int, include_identity: bool = False) -> list[np.ndarray]:
    """All n-qubit Pauli strings (optionally excluding I^{\\otimes n}).

    Returns d^2 - 1 operators (or d^2 if include_identity).
    """
    ops = []
    for indices in iproduct(range(4), repeat=n_qubits):
        if not include_identity and all(i == 0 for i in indices):
            continue
        op = PAULIS[indices[0]]
        for idx in indices[1:]:
            op = np.kron(op, PAULIS[idx])
        ops.append(op)
    return ops


# -------------------------------------------------------------------
# Single-qubit embedded operators
# -------------------------------------------------------------------

def _embed_single_qubit(op_2x2: np.ndarray, qubit: int,
                        n_qubits: int) -> np.ndarray:
    """Embed a 2x2 operator on qubit `qubit` into the full 2^n space."""
    d = 2 ** n_qubits
    result = np.eye(1, dtype=complex)
    for q in range(n_qubits):
        if q == qubit:
            result = np.kron(result, op_2x2)
        else:
            result = np.kron(result, I2)
    return result


def _embed_two_qubit(op_4x4: np.ndarray, q1: int, q2: int,
                     n_qubits: int) -> np.ndarray:
    """Embed a 4x4 operator on qubits (q1, q2) into the full space.

    Assumes q1 < q2.  The 4x4 operator acts on the tensor product
    of qubit q1 (left factor) and qubit q2 (right factor).
    """
    # Build by inserting identities
    d = 2 ** n_qubits
    result = np.zeros((d, d), dtype=complex)
    # Use computational basis mapping
    for i in range(d):
        for j in range(d):
            # Extract bits for q1 and q2
            b1_i = (i >> q1) & 1
            b2_i = (i >> q2) & 1
            b1_j = (j >> q1) & 1
            b2_j = (j >> q2) & 1
            # Check that all OTHER bits match
            mask = ~((1 << q1) | (1 << q2)) & ((1 << n_qubits) - 1)
            if (i & mask) != (j & mask):
                continue
            # Index into 4x4: row = 2*b1_i + b2_i, col = 2*b1_j + b2_j
            r = 2 * b1_i + b2_i
            c = 2 * b1_j + b2_j
            result[i, j] = op_4x4[r, c]
    return result


# -------------------------------------------------------------------
# Physically motivated operator sets
# -------------------------------------------------------------------

def single_qubit_operators(n_qubits: int) -> list[np.ndarray]:
    """sigma_+, sigma_-, Z on each qubit (3 * n_qubits operators)."""
    sp = np.array([[0, 1], [0, 0]], dtype=complex)  # |0><1|
    sm = np.array([[0, 0], [1, 0]], dtype=complex)  # |1><0|
    z2 = np.array([[1, 0], [0, -1]], dtype=complex)
    ops = []
    for q in range(n_qubits):
        ops.append(_embed_single_qubit(sp, q, n_qubits))
        ops.append(_embed_single_qubit(sm, q, n_qubits))
        ops.append(_embed_single_qubit(z2, q, n_qubits))
    return ops


def nearest_neighbour_operators(n_qubits: int) -> list[np.ndarray]:
    """Two-body operators on nearest-neighbour pairs (chain topology).

    For each pair (q, q+1): sigma+sigma-, sigma-sigma+, ZZ.
    Returns 3 * (n_qubits - 1) operators.
    """
    sp = np.array([[0, 1], [0, 0]], dtype=complex)
    sm = np.array([[0, 0], [1, 0]], dtype=complex)
    z2 = np.array([[1, 0], [0, -1]], dtype=complex)

    sp_sm = np.kron(sp, sm)  # sigma+ otimes sigma-
    sm_sp = np.kron(sm, sp)  # sigma- otimes sigma+
    zz = np.kron(z2, z2)     # Z otimes Z

    ops = []
    for q in range(n_qubits - 1):
        ops.append(_embed_two_qubit(sp_sm, q, q + 1, n_qubits))
        ops.append(_embed_two_qubit(sm_sp, q, q + 1, n_qubits))
        ops.append(_embed_two_qubit(zz, q, q + 1, n_qubits))
    return ops


def physical_operator_basis(n_qubits: int) -> list[np.ndarray]:
    """Combined single-qubit + nearest-neighbour two-qubit operators."""
    return single_qubit_operators(n_qubits) + nearest_neighbour_operators(n_qubits)


# -------------------------------------------------------------------
# Random operator basis
# -------------------------------------------------------------------

def random_operators(d: int, K: int, seed: int = 42) -> list[np.ndarray]:
    """K random complex d x d matrices (Ginibre ensemble)."""
    rng = np.random.default_rng(seed)
    ops = []
    for _ in range(K):
        real = rng.standard_normal((d, d))
        imag = rng.standard_normal((d, d))
        ops.append((real + 1j * imag) / np.sqrt(2 * d))
    return ops


# -------------------------------------------------------------------
# Generalised Gell-Mann matrices
# -------------------------------------------------------------------

def gell_mann_basis(d: int) -> list[np.ndarray]:
    r"""Generalised Gell-Mann matrices for SU(d).

    Returns d^2 - 1 traceless Hermitian matrices forming an orthogonal
    basis of the Lie algebra su(d) with Tr(G_a G_b) = 2 delta_{ab}.

    Three types:
    1. Symmetric:  |j><k| + |k><j|   for j < k
    2. Antisymmetric: -i|j><k| + i|k><j|  for j < k
    3. Diagonal:  from the Cartan subalgebra
    """
    ops = []
    # Symmetric off-diagonal
    for j in range(d):
        for k in range(j + 1, d):
            G = np.zeros((d, d), dtype=complex)
            G[j, k] = 1.0
            G[k, j] = 1.0
            ops.append(G)
    # Anti-symmetric off-diagonal
    for j in range(d):
        for k in range(j + 1, d):
            G = np.zeros((d, d), dtype=complex)
            G[j, k] = -1j
            G[k, j] = 1j
            ops.append(G)
    # Diagonal
    for l in range(1, d):
        G = np.zeros((d, d), dtype=complex)
        coeff = np.sqrt(2.0 / (l * (l + 1)))
        for j in range(l):
            G[j, j] = coeff
        G[l, l] = -l * coeff
        ops.append(G)
    return ops


# -------------------------------------------------------------------
# Target states
# -------------------------------------------------------------------

def ghz_state(n_qubits: int) -> np.ndarray:
    """GHZ state (|00...0> + |11...1>) / sqrt(2)."""
    d = 2 ** n_qubits
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0
    psi[-1] = 1.0
    return psi / np.linalg.norm(psi)


def w_state(n_qubits: int) -> np.ndarray:
    """W state."""
    d = 2 ** n_qubits
    psi = np.zeros(d, dtype=complex)
    for q in range(n_qubits):
        psi[1 << q] = 1.0
    return psi / np.linalg.norm(psi)


def random_pure_state(d: int, seed: int = 42) -> np.ndarray:
    """Haar-random pure state."""
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return psi / np.linalg.norm(psi)


def maximally_mixed_state(d: int) -> np.ndarray:
    """Maximally mixed state rho = I/d."""
    return np.eye(d, dtype=complex) / d
