"""4-qubit hardware-efficient ansatz VQC for binary classification.

Statevector simulation using numpy for exact, noiseless evaluation.

Circuit structure
-----------------
Encoding:
    Ry(x1) on q0, Ry(x2) on q1, Ry(x1) on q2, Ry(x2) on q3

Trainable layers (repeated n_layers times):
    Ry(theta) Rz(phi) on each qubit
    CNOT ring: (0,1), (1,2), (2,3), (3,0)

Observable:
    z = 0.5 * (<Z0> + <Z1>)
    p = clip((z + 1) / 2, eps, 1 - eps)

Performance note
----------------
The trainable layers are independent of input x, so we precompute the
full trainable unitary U_layers = prod(CNOT_ring @ Rz_all @ Ry_all)
and reuse it across all samples. Only the encoding varies per sample.
"""

import numpy as np

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X_PAULI = np.array([[0, 1], [1, 0]], dtype=complex)
Y_PAULI = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_PAULI = np.array([[1, 0], [0, -1]], dtype=complex)

EPS = 1e-7  # clipping epsilon


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz(phi):
    return np.array(
        [[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]], dtype=complex
    )


def _kron_n(matrices):
    result = matrices[0]
    for m in matrices[1:]:
        result = np.kron(result, m)
    return result


def _single_qubit_gate(gate_2x2, qubit, n_qubits=4):
    ops = [I2] * n_qubits
    ops[qubit] = gate_2x2
    return _kron_n(ops)


def _cnot(control, target, n_qubits=4):
    dim = 2**n_qubits
    gate = np.zeros((dim, dim), dtype=complex)
    for i in range(dim):
        bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[control] == 0:
            gate[i, i] = 1.0
        else:
            j_bits = bits.copy()
            j_bits[target] ^= 1
            j = sum(b << (n_qubits - 1 - q) for q, b in enumerate(j_bits))
            gate[j, i] = 1.0
    return gate


def _encode_state(x, n_qubits=4):
    """Encode 2D input into initial statevector.

    Ry(x1)|0> = [cos(x1/2), sin(x1/2)]^T, etc.
    Result is tensor product: q0(x1) x q1(x2) x q2(x1) x q3(x2).
    """
    vecs = []
    for q in range(n_qubits):
        angle = x[q % 2]
        vecs.append(np.array([np.cos(angle / 2), np.sin(angle / 2)], dtype=complex))
    state = vecs[0]
    for v in vecs[1:]:
        state = np.kron(state, v)
    return state


class VQCModel:
    """4-qubit variational quantum classifier with HEA."""

    def __init__(self, n_qubits=4, n_layers=3, entangler="ring"):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dim = 2**n_qubits
        self.n_params = n_layers * n_qubits * 2  # Ry + Rz per qubit per layer

        # Precompute CNOT entangler
        if entangler == "ring":
            pairs = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
        else:
            pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        self.entangler_unitary = np.eye(self.dim, dtype=complex)
        for c, t in pairs:
            self.entangler_unitary = _cnot(c, t, n_qubits) @ self.entangler_unitary

        # Precompute observable O = 0.5*(Z0 + Z1)
        self.obs = 0.5 * (
            _single_qubit_gate(Z_PAULI, 0, n_qubits)
            + _single_qubit_gate(Z_PAULI, 1, n_qubits)
        )

        # Cache for fast forward pass
        self._cached_params = None
        self._cached_U = None

    def init_params(self, seed=0):
        rng = np.random.RandomState(seed)
        return rng.uniform(-np.pi, np.pi, size=self.n_params)

    def _build_layer_unitary(self, params):
        """Build the full trainable unitary (all layers combined)."""
        nq = self.n_qubits
        U = np.eye(self.dim, dtype=complex)
        for layer in range(self.n_layers):
            base = layer * nq * 2
            # Ry on all qubits
            ry_all = _kron_n([_ry(params[base + q]) for q in range(nq)])
            # Rz on all qubits
            rz_all = _kron_n([_rz(params[base + nq + q]) for q in range(nq)])
            # Layer = entangler @ Rz_all @ Ry_all
            U = self.entangler_unitary @ rz_all @ ry_all @ U
        return U

    def _get_unitary(self, params):
        """Get trainable unitary, using cache if params unchanged."""
        # Check cache (exact float comparison is fine here for same array)
        if self._cached_params is not None and np.array_equal(params, self._cached_params):
            return self._cached_U
        U = self._build_layer_unitary(params)
        self._cached_params = params.copy()
        self._cached_U = U
        return U

    def forward(self, params, x):
        """Compute classification probability for a single input."""
        U = self._get_unitary(params)
        state = U @ _encode_state(x, self.n_qubits)
        z = np.real(state.conj() @ self.obs @ state)
        return np.clip((z + 1) / 2, EPS, 1 - EPS)

    def forward_batch(self, params, X):
        """Compute probabilities for a batch of inputs (vectorized)."""
        U = self._get_unitary(params)
        # Precompute O_eff = U^dag @ obs @ U
        O_eff = U.conj().T @ self.obs @ U
        probs = np.empty(len(X))
        for i, x in enumerate(X):
            enc = _encode_state(x, self.n_qubits)
            z = np.real(enc.conj() @ O_eff @ enc)
            probs[i] = np.clip((z + 1) / 2, EPS, 1 - EPS)
        return probs

    def loss(self, params, X, y):
        """Binary cross-entropy loss."""
        p = self.forward_batch(params, X)
        bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return np.mean(bce)

    def accuracy(self, params, X, y):
        p = self.forward_batch(params, X)
        return np.mean((p >= 0.5).astype(int) == y)

    def param_shift_gradient(self, params, X, y):
        """Exact gradient via parameter-shift rule.

        Returns (gradient, n_circuit_evaluations).
        Each shifted param requires rebuilding the unitary and evaluating
        all samples, so n_evals = 2 * n_params (in unitary-build units).
        """
        grad = np.zeros_like(params)
        shift = np.pi / 2
        for i in range(len(params)):
            p_plus = params.copy()
            p_plus[i] += shift
            p_minus = params.copy()
            p_minus[i] -= shift
            grad[i] = (self.loss(p_plus, X, y) - self.loss(p_minus, X, y)) / 2
        return grad, 2 * len(params)

    # ------------------------------------------------------------------
    # Gate-by-gate interface for Krotov
    # ------------------------------------------------------------------

    def _build_gate_sequence(self, params, x):
        """Return list of (gate_matrix, param_index_or_None) tuples."""
        gates = []
        nq = self.n_qubits

        # Encoding gates
        for q in range(nq):
            angle = x[q % 2]
            gates.append((_single_qubit_gate(_ry(angle), q, nq), None))

        # Trainable layers
        for layer in range(self.n_layers):
            base = layer * nq * 2
            for q in range(nq):
                pidx = base + q
                gates.append((_single_qubit_gate(_ry(params[pidx]), q, nq), pidx))
            for q in range(nq):
                pidx = base + nq + q
                gates.append((_single_qubit_gate(_rz(params[pidx]), q, nq), pidx))
            gates.append((self.entangler_unitary, None))

        return gates

    def get_gate_sequence_and_states(self, params, x):
        """Forward pass storing intermediate states for Krotov."""
        gates = self._build_gate_sequence(params, x)
        state = np.zeros(self.dim, dtype=complex)
        state[0] = 1.0

        states = [state.copy()]
        for gate, _ in gates:
            state = gate @ state
            states.append(state.copy())
        return gates, states

    def gate_derivative_generator(self, param_idx):
        """Return the generator -P/2 for the gate at param_idx.

        For Ry: generator = -iY/2 (as full-space operator)
        For Rz: generator = -iZ/2 (as full-space operator)

        d(R_P(theta))/dtheta = generator @ R_P(theta)
        """
        nq = self.n_qubits
        n_per_layer = nq * 2
        pos_in_layer = param_idx % n_per_layer

        if pos_in_layer < nq:
            qubit = pos_in_layer
            return -1j * 0.5 * _single_qubit_gate(Y_PAULI, qubit, nq)
        else:
            qubit = pos_in_layer - nq
            return -1j * 0.5 * _single_qubit_gate(Z_PAULI, qubit, nq)
