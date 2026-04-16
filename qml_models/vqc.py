"""Variational quantum classifiers for the two-moons benchmark.

Two architectures are supported:

``hea``
    The original 4-qubit hardware-efficient ansatz used in the benchmark:
    repeated ``Ry``/``Rz`` layers with a CNOT ring entangler.

``data_reuploading``
    A literature-aligned data-reuploading circuit for low-dimensional
    classification. Each re-upload block applies a single-qubit
    ``RZ-RY-RZ`` rotation whose angles depend affinely on the input
    features, following the ``U(theta + omega ∘ x)`` structure described in
    Pérez-Salinas et al. (2020) and later benchmark implementations.

``two_moons_dense_angle``
    A 2-qubit circuit used in a two-moons QML tutorial: a trainable dense
    angle-encoding front-end followed by repeated ``RY-CZ-RY`` blocks and a
    final ``RY`` layer, measured on ``Z0``.

The Krotov optimizer interacts with the model gate-by-gate, so the model also
provides helpers to rebuild individual trainable gates and the corresponding
gate generators.
"""

import numpy as np

from .losses import EPS

# Pauli matrices
I2 = np.eye(2, dtype=complex)
X_PAULI = np.array([[0, 1], [1, 0]], dtype=complex)
Y_PAULI = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_PAULI = np.array([[1, 0], [0, -1]], dtype=complex)


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


def _cz(control, target, n_qubits=2):
    dim = 2**n_qubits
    gate = np.eye(dim, dtype=complex)
    for i in range(dim):
        bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        if bits[control] == 1 and bits[target] == 1:
            gate[i, i] = -1.0
    return gate


def _rot(phi, theta, omega):
    """Single-qubit ``Rot`` gate in ``RZ(phi) RY(theta) RZ(omega)`` form."""
    return _rz(phi) @ _ry(theta) @ _rz(omega)


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
    """Variational quantum classifier with selectable circuit architecture."""

    def __init__(
        self,
        n_qubits=4,
        n_layers=3,
        entangler="ring",
        architecture="hea",
        observable="Z0Z1",
    ):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.architecture = architecture
        self.observable = observable
        self.dim = 2**n_qubits
        self.reupload_feature_dim = 3 * n_qubits

        if architecture == "hea":
            self.n_params = n_layers * n_qubits * 2
        elif architecture == "two_moons_dense_angle":
            self.n_params = 4 + n_layers * 4 + 2
        elif architecture == "data_reuploading":
            # Each re-upload block has trainable offset and feature-scale
            # parameters for the three Euler angles on every qubit.
            self.n_params = n_layers * n_qubits * 6
        else:
            raise ValueError(f"Unknown model architecture: {architecture}")

        # Precompute CNOT entangler
        if architecture == "two_moons_dense_angle":
            pairs = []
        elif n_qubits < 2 or entangler == "none":
            pairs = []
        elif entangler == "ring":
            pairs = [(i, (i + 1) % n_qubits) for i in range(n_qubits)]
        else:
            pairs = [(i, i + 1) for i in range(n_qubits - 1)]
        self.entangler_unitary = np.eye(self.dim, dtype=complex)
        for c, t in pairs:
            self.entangler_unitary = _cnot(c, t, n_qubits) @ self.entangler_unitary
        self.cz_unitary = (
            _cz(0, 1, n_qubits)
            if architecture == "two_moons_dense_angle" and n_qubits >= 2
            else np.eye(self.dim, dtype=complex)
        )

        if observable == "Z0" or n_qubits == 1:
            self.obs = _single_qubit_gate(Z_PAULI, 0, n_qubits)
        elif observable == "Z0Z1":
            self.obs = 0.5 * (
                _single_qubit_gate(Z_PAULI, 0, n_qubits)
                + _single_qubit_gate(Z_PAULI, 1, n_qubits)
            )
        else:
            raise ValueError(f"Unknown observable: {observable}")

        # Cache for fast forward pass
        self._cached_params = None
        self._cached_U = None

    def init_params(self, seed=0):
        rng = np.random.RandomState(seed)
        if self.architecture == "hea":
            return rng.uniform(-np.pi, np.pi, size=self.n_params)
        if self.architecture == "two_moons_dense_angle":
            params = np.empty(self.n_params, dtype=float)
            params[:4] = rng.normal(loc=1.0, scale=0.1, size=4)
            params[4:] = rng.uniform(-np.pi, np.pi, size=self.n_params - 4)
            return params

        params = np.empty(self.n_params, dtype=float)
        for idx in range(self.n_params):
            if idx % 6 in (0, 2, 4):
                params[idx] = rng.uniform(-np.pi, np.pi)
            else:
                params[idx] = rng.normal(loc=1.0, scale=0.1)
        return params

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
        if self.architecture != "hea":
            raise ValueError("Unitary caching is only available for the HEA model.")
        # Check cache (exact float comparison is fine here for same array)
        if self._cached_params is not None and np.array_equal(params, self._cached_params):
            return self._cached_U
        U = self._build_layer_unitary(params)
        self._cached_params = params.copy()
        self._cached_U = U
        return U

    def forward(self, params, x):
        """Compute classification probability for a single input."""
        if self.architecture == "hea":
            U = self._get_unitary(params)
            state = U @ _encode_state(x, self.n_qubits)
        else:
            _, states = self.get_gate_sequence_and_states(params, x)
            state = states[-1]
        z = np.real(state.conj() @ self.obs @ state)
        return np.clip((z + 1) / 2, EPS, 1 - EPS)

    def forward_batch(self, params, X):
        """Compute probabilities for a batch of inputs (vectorized)."""
        if self.architecture == "hea":
            U = self._get_unitary(params)
            O_eff = U.conj().T @ self.obs @ U
            probs = np.empty(len(X))
            for i, x in enumerate(X):
                enc = _encode_state(x, self.n_qubits)
                z = np.real(enc.conj() @ O_eff @ enc)
                probs[i] = np.clip((z + 1) / 2, EPS, 1 - EPS)
            return probs

        return np.array([self.forward(params, x) for x in X], dtype=float)

    def loss(self, params, X, y):
        """Binary cross-entropy loss."""
        p = self.forward_batch(params, X)
        bce = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        return np.mean(bce)

    def accuracy(self, params, X, y):
        p = self.forward_batch(params, X)
        return np.mean((p >= 0.5).astype(int) == y)

    def loss_gradient(self, params, X, y):
        """Exact gradient of the mean BCE loss via gate-state backpropagation.

        This routine differentiates the benchmark objective itself, not just the
        underlying expectation value. For each sample we

        1. build the gate sequence and store forward states,
        2. backpropagate the observable costate ``O|psi_T>``,
        3. obtain ``dz/dtheta`` gate by gate, and
        4. apply the BCE chain rule ``dL/dtheta = (dL/dp) * 0.5 * dz/dtheta``.

        This remains exact for parameters that scale the data-dependent gate
        angle, such as ``angle = w * x``, because the generator returned by
        ``gate_derivative_generator`` already includes the feature factor.
        """
        grad = np.zeros_like(params, dtype=float)

        for x_i, y_i in zip(X, y):
            gates, fwd_states = self.get_gate_sequence_and_states(params, x_i)
            final_state = fwd_states[-1]
            z = np.real(final_state.conj() @ self.obs @ final_state)
            p = np.clip((z + 1.0) / 2.0, EPS, 1.0 - EPS)
            dloss_dp = -y_i / p + (1.0 - y_i) / (1.0 - p)

            chi_states = [None] * len(fwd_states)
            chi_states[-1] = self.obs @ final_state
            for gate_idx in range(len(gates) - 1, -1, -1):
                gate_mat = gates[gate_idx][0]
                chi_states[gate_idx] = gate_mat.conj().T @ chi_states[gate_idx + 1]

            for gate_idx, (_, pidx) in enumerate(gates):
                if pidx is None:
                    continue
                gen = self.gate_derivative_generator(pidx, x_i)
                grad_vec = gen @ fwd_states[gate_idx + 1]
                dz_dtheta = 2.0 * np.real(chi_states[gate_idx + 1].conj() @ grad_vec)
                grad[pidx] += 0.5 * dloss_dp * dz_dtheta

        grad /= len(X)
        return grad, {
            "sample_forward_passes": len(X),
            "sample_backward_passes": len(X),
            "full_loss_evaluations": 0,
        }

    def param_shift_gradient(self, params, X, y):
        """Backward-compatible alias for the exact BCE gradient."""
        return self.loss_gradient(params, X, y)

    def parameter_metadata(self):
        """Return per-parameter metadata for generic optimizer consumption."""
        metadata = []
        nq = self.n_qubits
        if self.architecture == "hea":
            for pidx in range(self.n_params):
                layer = pidx // (nq * 2)
                pos = pidx % (nq * 2)
                if pos < nq:
                    qubit, axis = pos, "ry"
                else:
                    qubit, axis = pos - nq, "rz"
                metadata.append({
                    "index": pidx,
                    "name": f"hea_{axis}[{layer},{qubit}]",
                    "group": "hea_rotation",
                    "kind": "quantum",
                    "supports_gate_derivative": True,
                    "layer": layer,
                    "qubit": qubit,
                    "axis": axis,
                })
        elif self.architecture == "two_moons_dense_angle":
            for pidx in range(self.n_params):
                info = self._dense_angle_info(pidx)
                if pidx < 4:
                    layer, group = 0, "encoding_scale"
                elif pidx >= self.n_params - 2:
                    layer, group = self.n_layers + 1, "final_rotation"
                else:
                    layer = 1 + (pidx - 4) // 4
                    group = "trainable_rotation"
                metadata.append({
                    "index": pidx,
                    "name": f"dense_angle[{pidx}]",
                    "group": group,
                    "kind": "quantum",
                    "supports_gate_derivative": True,
                    "layer": layer,
                    "qubit": info["qubit"],
                    "axis": info["axis"],
                })
        else:
            for pidx in range(self.n_params):
                info = self._reupload_param_info(pidx)
                metadata.append({
                    "index": pidx,
                    "name": f"reupload[{info['block']},{info['qubit']},{pidx % 6}]",
                    "group": "scale" if info["is_scale"] else "offset",
                    "kind": "quantum",
                    "supports_gate_derivative": True,
                    "layer": info["block"],
                    "qubit": info["qubit"],
                    "axis": info["axis"],
                })
        return metadata

    def gate_parameter_indices(self):
        """Return indices of all gate-supported parameters."""
        return list(range(self.n_params))

    # ------------------------------------------------------------------
    # Gate-by-gate interface for Krotov
    # ------------------------------------------------------------------

    def _reupload_features(self, x):
        """Pad 2D inputs to the 3-angle-per-qubit data-reuploading layout."""
        features = np.zeros(self.reupload_feature_dim, dtype=float)
        limit = min(len(x), self.reupload_feature_dim)
        features[:limit] = np.asarray(x[:limit], dtype=float)
        return features

    def _dense_angle_info(self, param_idx):
        if param_idx < 4:
            return {
                "qubit": 0 if param_idx in (0, 2) else 1,
                "axis": "y" if param_idx < 2 else "z",
                "feature_idx": 0 if param_idx < 2 else 1,
                "is_scale": True,
            }

        final_base = self.n_params - 2
        if param_idx >= final_base:
            return {
                "qubit": param_idx - final_base,
                "axis": "y",
                "feature_idx": None,
                "is_scale": False,
            }

        offset = param_idx - 4
        pos_in_layer = offset % 4
        return {
            "qubit": 0 if pos_in_layer in (0, 2) else 1,
            "axis": "y",
            "feature_idx": None,
            "is_scale": False,
        }

    def _dense_angle_gate(self, param_idx, params, x):
        info = self._dense_angle_info(param_idx)
        angle = params[param_idx]
        if info["is_scale"]:
            angle *= float(x[info["feature_idx"]])
        gate_2x2 = _ry(angle) if info["axis"] == "y" else _rz(angle)
        return _single_qubit_gate(gate_2x2, info["qubit"], self.n_qubits)

    def _reupload_param_info(self, param_idx):
        block_size = self.n_qubits * 6
        block = param_idx // block_size
        offset = param_idx % block_size
        qubit = offset // 6
        slot = offset % 6
        axis = ("z", "z", "y", "y", "z", "z")[slot]
        feature_slot = qubit * 3 + slot // 2
        is_scale = slot % 2 == 1
        return {
            "block": block,
            "qubit": qubit,
            "axis": axis,
            "feature_slot": feature_slot,
            "is_scale": is_scale,
        }

    def _reupload_gate(self, param_idx, params, x):
        info = self._reupload_param_info(param_idx)
        features = self._reupload_features(x)
        angle = params[param_idx]
        if info["is_scale"]:
            angle *= features[info["feature_slot"]]
        gate_2x2 = _ry(angle) if info["axis"] == "y" else _rz(angle)
        return _single_qubit_gate(gate_2x2, info["qubit"], self.n_qubits)

    def rebuild_param_gate(self, param_idx, params, x):
        """Rebuild a single trainable gate after a parameter update."""
        if self.architecture == "hea":
            n_per_layer = self.n_qubits * 2
            pos_in_layer = param_idx % n_per_layer
            if pos_in_layer < self.n_qubits:
                return _single_qubit_gate(_ry(params[param_idx]), pos_in_layer, self.n_qubits)
            qubit = pos_in_layer - self.n_qubits
            return _single_qubit_gate(_rz(params[param_idx]), qubit, self.n_qubits)
        if self.architecture == "two_moons_dense_angle":
            return self._dense_angle_gate(param_idx, params, x)
        return self._reupload_gate(param_idx, params, x)

    def _build_gate_sequence(self, params, x):
        """Return list of ``(gate_matrix, param_index_or_None)`` tuples."""
        gates = []
        nq = self.n_qubits

        if self.architecture == "hea":
            for q in range(nq):
                angle = x[q % 2]
                gates.append((_single_qubit_gate(_ry(angle), q, nq), None))

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

        if self.architecture == "two_moons_dense_angle":
            for pidx in range(4):
                gates.append((self._dense_angle_gate(pidx, params, x), pidx))

            for layer in range(self.n_layers):
                base = 4 + layer * 4
                for local_idx in range(4):
                    gates.append((self._dense_angle_gate(base + local_idx, params, x), base + local_idx))
                    if local_idx == 1:
                        gates.append((self.cz_unitary, None))

            final_base = self.n_params - 2
            gates.append((self._dense_angle_gate(final_base, params, x), final_base))
            gates.append((self._dense_angle_gate(final_base + 1, params, x), final_base + 1))
            return gates

        for block in range(self.n_layers):
            block_base = block * nq * 6
            for q in range(nq):
                for local_idx in range(6):
                    pidx = block_base + q * 6 + local_idx
                    gates.append((self._reupload_gate(pidx, params, x), pidx))
            if block < self.n_layers - 1 and nq > 1:
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

    def gate_derivative_generator(self, param_idx, x=None):
        """Return the left-generator for the parameterized gate at ``param_idx``."""
        nq = self.n_qubits
        factor = 1.0

        if self.architecture == "hea":
            n_per_layer = nq * 2
            pos_in_layer = param_idx % n_per_layer
            if pos_in_layer < nq:
                qubit = pos_in_layer
                return -1j * 0.5 * _single_qubit_gate(Y_PAULI, qubit, nq)
            qubit = pos_in_layer - nq
            return -1j * 0.5 * _single_qubit_gate(Z_PAULI, qubit, nq)

        if self.architecture == "two_moons_dense_angle":
            info = self._dense_angle_info(param_idx)
            factor = float(x[info["feature_idx"]]) if info["is_scale"] else 1.0
            pauli = Y_PAULI if info["axis"] == "y" else Z_PAULI
            return factor * (-1j * 0.5 * _single_qubit_gate(pauli, info["qubit"], nq))

        info = self._reupload_param_info(param_idx)
        qubit = info["qubit"]
        if info["is_scale"]:
            if x is None:
                raise ValueError("Data-reuploading gate derivatives require the sample x.")
            factor = self._reupload_features(x)[info["feature_slot"]]

        pauli = Y_PAULI if info["axis"] == "y" else Z_PAULI
        return factor * (-1j * 0.5 * _single_qubit_gate(pauli, qubit, nq))
