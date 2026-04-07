"""Chen et al. (2025) SUN-VQC classifier reconstruction."""

from __future__ import annotations

import numpy as np

from .common import (
    BaseQMLModel,
    I2,
    X_PAULI,
    Y_PAULI,
    Z_PAULI,
    clip_probability,
    embed_two_qubit_operator,
    expectation,
    one_parameter_unitary,
    ry,
    rz,
    sigmoid,
    single_qubit_gate,
    z_observable,
    zero_state,
)


class ChenSUNVQCModel(BaseQMLModel):
    """4-qubit SUN-VQC with 15-parameter two-qubit blocks."""

    def __init__(
        self,
        n_macro_layers=2,
        encoding_axes=("y",),
        readout="simple_z0",
    ):
        if n_macro_layers <= 0:
            raise ValueError("SUN-VQC requires at least one macro-layer.")
        if readout not in {"simple_z0", "hybrid_linear"}:
            raise ValueError(f"Unknown SUN-VQC readout: {readout}")

        self.n_qubits = 4
        self.dim = 16
        self.n_macro_layers = n_macro_layers
        self.encoding_axes = tuple(encoding_axes)
        self.readout = readout
        self.block_pairs = ((0, 1), (2, 3), (1, 2))
        self.local_generators = self._build_su4_generators()
        self.generator_labels = [label for label, _ in self.local_generators]
        self.local_generator_matrices = [matrix for _, matrix in self.local_generators]
        self.z_observables = [z_observable(q, self.n_qubits) for q in range(self.n_qubits)]
        self.obs = self.z_observables[0] if self.readout == "simple_z0" else None
        self.n_quantum_params = (
            self.n_macro_layers * len(self.block_pairs) * len(self.local_generators)
        )
        self.n_output_params = 0 if self.readout == "simple_z0" else self.n_qubits + 1
        self.n_params = self.n_quantum_params + self.n_output_params
        self._metadata = self._build_parameter_metadata()

    def _build_su4_generators(self):
        single_generators = [
            ("XI", np.kron(X_PAULI, I2)),
            ("YI", np.kron(Y_PAULI, I2)),
            ("ZI", np.kron(Z_PAULI, I2)),
            ("IX", np.kron(I2, X_PAULI)),
            ("IY", np.kron(I2, Y_PAULI)),
            ("IZ", np.kron(I2, Z_PAULI)),
        ]
        entangling = []
        paulis = (("X", X_PAULI), ("Y", Y_PAULI), ("Z", Z_PAULI))
        for left_label, left_matrix in paulis:
            for right_label, right_matrix in paulis:
                entangling.append(
                    (f"{left_label}{right_label}", np.kron(left_matrix, right_matrix))
                )
        return single_generators + entangling

    def _rotation_encoding_gate(self, axis, angle, qubit):
        gate_2x2 = ry(angle) if axis == "y" else rz(angle)
        return single_qubit_gate(gate_2x2, qubit, self.n_qubits)

    def _quantum_param_info(self, param_idx):
        if not 0 <= param_idx < self.n_quantum_params:
            raise ValueError(f"Parameter {param_idx} is not a quantum SUN-VQC parameter.")

        per_block = len(self.local_generators)
        per_macro = len(self.block_pairs) * per_block
        macro_idx = param_idx // per_macro
        within_macro = param_idx % per_macro
        block_idx = within_macro // per_block
        generator_idx = within_macro % per_block
        return {
            "macro_layer": macro_idx,
            "block_idx": block_idx,
            "pair": self.block_pairs[block_idx],
            "generator_idx": generator_idx,
            "generator_label": self.generator_labels[generator_idx],
            "local_generator": self.local_generator_matrices[generator_idx],
        }

    def _output_weight_slice(self):
        return slice(self.n_quantum_params, self.n_quantum_params + self.n_qubits)

    def _output_bias_index(self):
        return self._output_weight_slice().stop

    def init_params(self, seed=0):
        rng = np.random.RandomState(seed)
        params = np.zeros(self.n_params, dtype=float)
        params[: self.n_quantum_params] = rng.uniform(
            -np.pi / 4.0, np.pi / 4.0, size=self.n_quantum_params
        )
        if self.readout == "hybrid_linear":
            params[self._output_weight_slice()] = rng.normal(scale=0.2, size=self.n_qubits)
            params[self._output_bias_index()] = rng.normal(scale=0.1)
        return params

    def parameter_metadata(self):
        return list(self._metadata)

    def gate_parameter_indices(self):
        return list(range(self.n_quantum_params))

    def nongate_parameter_indices(self):
        if self.readout == "simple_z0":
            return []
        return list(range(self.n_quantum_params, self.n_params))

    def _measurement_vector(self, state):
        return np.array([expectation(state, obs) for obs in self.z_observables], dtype=float)

    def _sample_forward_details(self, params, x):
        _, states = self.get_gate_sequence_and_states(params, x)
        final_state = states[-1]
        if self.readout == "simple_z0":
            prob = clip_probability(0.5 * (expectation(final_state, self.obs) + 1.0))
            return {
                "state": final_state,
                "probability": float(prob),
            }

        measurement_vector = self._measurement_vector(final_state)
        output_weights = params[self._output_weight_slice()]
        output_bias = params[self._output_bias_index()]
        logit = float(output_weights @ measurement_vector + output_bias)
        prob = float(clip_probability(sigmoid(logit)))
        return {
            "state": final_state,
            "measurement_vector": measurement_vector,
            "output_weights": output_weights,
            "probability": prob,
        }

    def forward(self, params, x):
        return self._sample_forward_details(np.asarray(params, dtype=float), x)["probability"]

    def terminal_costate(self, params, x, y, final_state):
        if self.readout == "simple_z0":
            z = expectation(final_state, self.obs)
            p = clip_probability(0.5 * (z + 1.0))
            dloss_dp = -float(y) / p + (1.0 - float(y)) / (1.0 - p)
            return 0.5 * dloss_dp * (self.obs @ final_state)

        sample = self._sample_forward_details(params, x)
        delta = sample["probability"] - float(y)
        operator = sum(
            weight * observable
            for weight, observable in zip(sample["output_weights"], self.z_observables)
        )
        return delta * (operator @ final_state)

    def loss_gradient(self, params, X, y):
        params_arr = np.asarray(params, dtype=float)
        grad = np.zeros_like(params_arr)

        for x_i, y_i in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=float)):
            gates, states = self.get_gate_sequence_and_states(params_arr, x_i)
            final_state = states[-1]

            if self.readout == "simple_z0":
                z = expectation(final_state, self.obs)
                p = clip_probability(0.5 * (z + 1.0))
                dloss_dp = -y_i / p + (1.0 - y_i) / (1.0 - p)
                chi_states = [None] * len(states)
                chi_states[-1] = self.obs @ final_state
                prefactor = 0.5 * dloss_dp
            else:
                sample = self._sample_forward_details(params_arr, x_i)
                delta = sample["probability"] - y_i
                grad[self._output_weight_slice()] += delta * sample["measurement_vector"]
                grad[self._output_bias_index()] += delta
                terminal_operator = sum(
                    weight * observable
                    for weight, observable in zip(
                        sample["output_weights"], self.z_observables
                    )
                )
                chi_states = [None] * len(states)
                chi_states[-1] = terminal_operator @ final_state
                prefactor = delta

            for gate_idx in range(len(gates) - 1, -1, -1):
                chi_states[gate_idx] = gates[gate_idx][0].conj().T @ chi_states[gate_idx + 1]

            for gate_idx, (_, pidx) in enumerate(gates):
                if pidx is None:
                    continue
                gen = self.gate_derivative_generator(pidx, x_i)
                grad_vec = gen @ states[gate_idx + 1]
                grad[pidx] += prefactor * 2.0 * np.real(
                    chi_states[gate_idx + 1].conj() @ grad_vec
                )

        grad /= len(X)
        return grad, {
            "sample_forward_passes": len(X),
            "sample_backward_passes": len(X),
            "full_loss_evaluations": 0,
        }

    def nongate_loss_gradient(self, params, X, y):
        """Return the exact gradient restricted to the hybrid linear readout."""
        params_arr = np.asarray(params, dtype=float)
        grad = np.zeros_like(params_arr)
        if self.readout == "simple_z0":
            return grad, {
                "sample_forward_passes": 0,
                "sample_backward_passes": 0,
                "full_loss_evaluations": 0,
            }

        output_slice = self._output_weight_slice()
        output_bias_idx = self._output_bias_index()
        for x_i, y_i in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=float)):
            sample = self._sample_forward_details(params_arr, x_i)
            delta = sample["probability"] - y_i
            grad[output_slice] += delta * sample["measurement_vector"]
            grad[output_bias_idx] += delta

        grad /= len(X)
        return grad, {
            "sample_forward_passes": len(X),
            "sample_backward_passes": 0,
            "full_loss_evaluations": 0,
        }

    def rebuild_param_gate(self, param_idx, params, x):
        info = self._quantum_param_info(param_idx)
        local_gate = one_parameter_unitary(
            float(params[param_idx]), info["local_generator"]
        )
        return embed_two_qubit_operator(local_gate, info["pair"][0], info["pair"][1], self.n_qubits)

    def gate_derivative_generator(self, param_idx, x=None):
        info = self._quantum_param_info(param_idx)
        local_generator = -1j * 0.5 * info["local_generator"]
        return embed_two_qubit_operator(
            local_generator, info["pair"][0], info["pair"][1], self.n_qubits
        )

    def get_gate_sequence_and_states(self, params, x):
        params_arr = np.asarray(params, dtype=float)
        x_arr = np.asarray(x, dtype=float)
        gates = []

        for qubit in range(self.n_qubits):
            feature_value = float(x_arr[qubit % 2])
            for axis in self.encoding_axes:
                gates.append((self._rotation_encoding_gate(axis, feature_value, qubit), None))

        for pidx in range(self.n_quantum_params):
            gates.append((self.rebuild_param_gate(pidx, params_arr, x_arr), pidx))

        state = zero_state(self.n_qubits)
        states = [state.copy()]
        for gate, _ in gates:
            state = gate @ state
            states.append(state.copy())
        return gates, states

    def _build_parameter_metadata(self):
        metadata = []
        for pidx in range(self.n_quantum_params):
            info = self._quantum_param_info(pidx)
            metadata.append(
                {
                    "index": pidx,
                    "name": (
                        f"su4[{info['macro_layer']},{info['block_idx']},"
                        f"{info['generator_idx']}]"
                    ),
                    "group": "quantum_block",
                    "kind": "quantum",
                    "supports_gate_derivative": True,
                    "macro_layer": info["macro_layer"],
                    "block_idx": info["block_idx"],
                    "pair": info["pair"],
                    "generator_idx": info["generator_idx"],
                    "generator_label": info["generator_label"],
                }
            )
        if self.readout == "hybrid_linear":
            for offset, pidx in enumerate(
                range(self._output_weight_slice().start, self._output_weight_slice().stop)
            ):
                metadata.append(
                    {
                        "index": pidx,
                        "name": f"output_weight[{offset}]",
                        "group": "classical_output",
                        "kind": "classical",
                        "supports_gate_derivative": False,
                        "macro_layer": None,
                        "block_idx": None,
                        "pair": None,
                        "generator_idx": None,
                        "generator_label": None,
                    }
                )
            metadata.append(
                {
                    "index": self._output_bias_index(),
                    "name": "output_bias",
                    "group": "classical_output",
                    "kind": "classical",
                    "supports_gate_derivative": False,
                    "macro_layer": None,
                    "block_idx": None,
                    "pair": None,
                    "generator_idx": None,
                    "generator_label": None,
                }
            )
        return metadata
