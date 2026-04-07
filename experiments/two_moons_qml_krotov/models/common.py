"""Shared linear-algebra helpers for alternative QML models."""

from __future__ import annotations

import numpy as np


EPS = 1e-7

I2 = np.eye(2, dtype=complex)
I4 = np.eye(4, dtype=complex)
X_PAULI = np.array([[0, 1], [1, 0]], dtype=complex)
Y_PAULI = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_PAULI = np.array([[1, 0], [0, -1]], dtype=complex)


def sigmoid(x):
    """Numerically stable logistic function."""
    x = np.asarray(x, dtype=float)
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def clip_probability(prob):
    """Clip probabilities away from the BCE singularities."""
    return np.clip(prob, EPS, 1.0 - EPS)


def ry(theta):
    """Single-qubit ``Ry(theta)`` rotation."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(phi):
    """Single-qubit ``Rz(phi)`` rotation."""
    return np.array(
        [[np.exp(-1j * phi / 2.0), 0.0], [0.0, np.exp(1j * phi / 2.0)]],
        dtype=complex,
    )


def one_parameter_unitary(angle, generator):
    """Return ``exp(-i angle generator / 2)`` for involutory generators."""
    return np.cos(angle / 2.0) * np.eye(generator.shape[0], dtype=complex) - 1j * np.sin(
        angle / 2.0
    ) * generator


def kron_n(matrices):
    """Kronecker product of a non-empty sequence of matrices."""
    result = matrices[0]
    for matrix in matrices[1:]:
        result = np.kron(result, matrix)
    return result


def single_qubit_gate(gate_2x2, qubit, n_qubits):
    """Embed a 2x2 gate on ``qubit`` into an ``n_qubits`` register."""
    ops = [I2] * n_qubits
    ops[qubit] = gate_2x2
    return kron_n(ops)


def embed_two_qubit_operator(local_op, qubit_a, qubit_b, n_qubits):
    """Embed a 4x4 operator acting on ``(qubit_a, qubit_b)``."""
    if qubit_a == qubit_b:
        raise ValueError("Two-qubit operators require distinct qubits.")

    qa, qb = sorted((qubit_a, qubit_b))
    dim = 2**n_qubits
    full = np.zeros((dim, dim), dtype=complex)

    for column in range(dim):
        bits = [(column >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        local_column = (bits[qa] << 1) | bits[qb]

        for local_row in range(4):
            amp = local_op[local_row, local_column]
            if amp == 0.0:
                continue
            out_bits = bits.copy()
            out_bits[qa] = (local_row >> 1) & 1
            out_bits[qb] = local_row & 1
            row = sum(bit << (n_qubits - 1 - idx) for idx, bit in enumerate(out_bits))
            full[row, column] += amp

    return full


def cnot(control, target, n_qubits):
    """Controlled-NOT embedded in the full register."""
    dim = 2**n_qubits
    gate = np.zeros((dim, dim), dtype=complex)

    for column in range(dim):
        bits = [(column >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[control] == 0:
            gate[column, column] = 1.0
            continue
        out_bits = bits.copy()
        out_bits[target] ^= 1
        row = sum(bit << (n_qubits - 1 - idx) for idx, bit in enumerate(out_bits))
        gate[row, column] = 1.0

    return gate


def zero_state(n_qubits):
    """Return ``|0...0>`` as a statevector."""
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    return state


def z_observable(qubit, n_qubits):
    """Full-register ``Z`` observable on one qubit."""
    return single_qubit_gate(Z_PAULI, qubit, n_qubits)


def expectation(state, observable):
    """Real expectation value of a Hermitian observable."""
    return float(np.real(state.conj() @ observable @ state))


class BaseQMLModel:
    """Small shared interface matching the existing two-moons model contract."""

    n_params: int

    def get_initial_params(self, seed=0):
        return self.init_params(seed=seed)

    def forward_batch(self, params, X):
        return np.array([self.forward(params, x) for x in X], dtype=float)

    def predict(self, params, X):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            return int(self.forward(params, X_arr) >= 0.5)
        return (self.forward_batch(params, X_arr) >= 0.5).astype(int)

    def loss(self, params, X, y):
        probs = self.forward_batch(params, X)
        y_arr = np.asarray(y, dtype=float)
        return float(-np.mean(y_arr * np.log(probs) + (1.0 - y_arr) * np.log(1.0 - probs)))

    def accuracy(self, params, X, y):
        preds = self.predict(params, X)
        return float(np.mean(preds == np.asarray(y, dtype=int)))

    def param_shift_gradient(self, params, X, y):
        return self.loss_gradient(params, X, y)
