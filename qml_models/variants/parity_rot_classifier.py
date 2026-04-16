"""Parity classifier matching the PennyLane variational-classifier demo."""

from __future__ import annotations

import numpy as np

from ..common import BaseQMLModel, Y_PAULI, Z_PAULI, cnot, expectation, ry, rz, single_qubit_gate, z_observable


class ParityRotClassifierModel(BaseQMLModel):
    """4-qubit parity classifier with basis encoding and Rot/CNOT layers.

    The trainable circuit matches the first PennyLane variational-classifier
    demo architecture: basis-state preparation of a 4-bit input, followed by
    repeated layers of per-qubit ``Rot`` gates and the ring
    ``CNOT(0,1),(1,2),(2,3),(3,0)``, and finally an expectation value of
    ``Z`` on qubit 0 plus a single trainable scalar bias.
    """

    component_labels = ("phi", "theta", "omega")
    component_axes = ("rz", "ry", "rz")

    def __init__(self, n_layers=2):
        if n_layers <= 0:
            raise ValueError("ParityRotClassifierModel requires at least one layer.")

        self.n_qubits = 4
        self.dim = 2**self.n_qubits
        self.n_layers = int(n_layers)
        self.weights_shape = (self.n_layers, self.n_qubits, 3)
        self.n_quantum_params = int(np.prod(self.weights_shape))
        self.n_weight_params = 1
        self.n_params = self.n_quantum_params + self.n_weight_params
        self.obs = z_observable(0, self.n_qubits)
        self.entanglers = (
            cnot(0, 1, self.n_qubits),
            cnot(1, 2, self.n_qubits),
            cnot(2, 3, self.n_qubits),
            cnot(3, 0, self.n_qubits),
        )
        self._metadata = self._build_parameter_metadata()

    def _param_info(self, param_idx):
        if not 0 <= param_idx < self.n_quantum_params:
            raise ValueError(f"Parameter {param_idx} is not a quantum parity-classifier parameter.")
        layer, qubit, component = np.unravel_index(param_idx, self.weights_shape)
        return {
            "layer": int(layer),
            "qubit": int(qubit),
            "component": int(component),
            "component_label": self.component_labels[component],
            "axis": self.component_axes[component],
        }

    def _bias_index(self):
        return self.n_quantum_params

    def _prepare_basis_state(self, x):
        bits = np.asarray(x, dtype=int)
        if bits.shape != (self.n_qubits,):
            raise ValueError(f"Expected a 4-bit input vector, got shape {bits.shape}.")
        if np.any((bits != 0) & (bits != 1)):
            raise ValueError("Parity inputs must be bitstrings with entries in {0, 1}.")
        state = np.zeros(self.dim, dtype=complex)
        basis_index = 0
        for qubit, bit in enumerate(bits):
            basis_index |= int(bit) << (self.n_qubits - 1 - qubit)
        state[basis_index] = 1.0
        return state

    def _rotation_gate(self, axis, angle, qubit):
        gate_2x2 = rz(angle) if axis == "rz" else ry(angle)
        return single_qubit_gate(gate_2x2, qubit, self.n_qubits)

    def _sample_forward_details(self, params, x):
        _, states = self.get_gate_sequence_and_states(params, x)
        final_state = states[-1]
        quantum_output = expectation(final_state, self.obs)
        bias = float(np.asarray(params, dtype=float)[self._bias_index()])
        prediction = float(quantum_output + bias)
        return {
            "state": final_state,
            "quantum_output": float(quantum_output),
            "bias": bias,
            "prediction": prediction,
        }

    def init_params(self, seed=0):
        rng = np.random.RandomState(seed)
        params = np.zeros(self.n_params, dtype=float)
        params[: self.n_quantum_params] = rng.normal(scale=0.01, size=self.n_quantum_params)
        params[self._bias_index()] = 0.0
        return params

    def parameter_metadata(self):
        return list(self._metadata)

    def gate_parameter_indices(self):
        return list(range(self.n_quantum_params))

    def nongate_parameter_indices(self):
        return [self._bias_index()]

    def forward(self, params, x):
        return self._sample_forward_details(params, x)["prediction"]

    def predict(self, params, X):
        X_arr = np.asarray(X, dtype=int)
        if X_arr.ndim == 1:
            return int(1 if self.forward(params, X_arr) >= 0.0 else -1)
        scores = self.forward_batch(params, X_arr)
        return np.where(scores >= 0.0, 1, -1).astype(int)

    def loss(self, params, X, y):
        preds = self.forward_batch(params, np.asarray(X, dtype=int))
        y_arr = np.asarray(y, dtype=float)
        residual = y_arr - preds
        return float(np.mean(residual**2))

    def accuracy(self, params, X, y):
        preds = self.predict(params, X)
        return float(np.mean(preds == np.asarray(y, dtype=int)))

    def terminal_costate(self, params, x, y, final_state):
        sample = self._sample_forward_details(params, x)
        residual = sample["prediction"] - float(y)
        return 2.0 * residual * (self.obs @ final_state)

    def loss_gradient(self, params, X, y):
        params_arr = np.asarray(params, dtype=float)
        grad = np.zeros_like(params_arr)

        for x_i, y_i in zip(np.asarray(X, dtype=int), np.asarray(y, dtype=float)):
            gates, states = self.get_gate_sequence_and_states(params_arr, x_i)
            final_state = states[-1]
            sample = self._sample_forward_details(params_arr, x_i)
            residual = sample["prediction"] - y_i
            prefactor = 2.0 * residual
            grad[self._bias_index()] += prefactor

            chi_states = [None] * len(states)
            chi_states[-1] = self.obs @ final_state
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
        grad = np.zeros_like(np.asarray(params, dtype=float))
        preds = self.forward_batch(params, np.asarray(X, dtype=int))
        residual = preds - np.asarray(y, dtype=float)
        grad[self._bias_index()] = float(np.mean(2.0 * residual))
        return grad, {
            "sample_forward_passes": len(X),
            "sample_backward_passes": 0,
            "full_loss_evaluations": 0,
        }

    def rebuild_param_gate(self, param_idx, params, x):
        info = self._param_info(param_idx)
        angle = float(np.asarray(params, dtype=float)[param_idx])
        return self._rotation_gate(info["axis"], angle, info["qubit"])

    def gate_derivative_generator(self, param_idx, x=None):
        info = self._param_info(param_idx)
        pauli = Z_PAULI if info["axis"] == "rz" else Y_PAULI
        return -1j * 0.5 * single_qubit_gate(pauli, info["qubit"], self.n_qubits)

    def get_gate_sequence_and_states(self, params, x):
        params_arr = np.asarray(params, dtype=float)
        gates = []
        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                for component in range(3):
                    pidx = np.ravel_multi_index((layer, qubit, component), self.weights_shape)
                    gates.append((self.rebuild_param_gate(pidx, params_arr, x), pidx))
            for entangler in self.entanglers:
                gates.append((entangler, None))

        state = self._prepare_basis_state(x)
        states = [state.copy()]
        for gate, _ in gates:
            state = gate @ state
            states.append(state.copy())
        return gates, states

    def _build_parameter_metadata(self):
        metadata = []
        for pidx in range(self.n_quantum_params):
            info = self._param_info(pidx)
            metadata.append(
                {
                    "index": pidx,
                    "name": (
                        f"weights[{info['layer']},{info['qubit']},{info['component']}]"
                    ),
                    "group": "quantum_rotations",
                    "kind": "quantum",
                    "supports_gate_derivative": True,
                    "layer": info["layer"],
                    "qubit": info["qubit"],
                    "axis": info["axis"],
                    "component": info["component_label"],
                }
            )
        metadata.append(
            {
                "index": self._bias_index(),
                "name": "bias",
                "group": "classical_output",
                "kind": "classical",
                "supports_gate_derivative": False,
                "layer": None,
                "qubit": None,
                "axis": None,
                "component": None,
            }
        )
        return metadata
