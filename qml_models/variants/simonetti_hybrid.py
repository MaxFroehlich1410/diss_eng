"""Simonetti, Perri, Gervasi (2022) hybrid classifier reconstructions."""

from __future__ import annotations

import numpy as np

from ..common import (
    BaseQMLModel,
    X_PAULI,
    Y_PAULI,
    Z_PAULI,
    clip_probability,
    cnot,
    expectation,
    ry,
    rz,
    sigmoid,
    single_qubit_gate,
    z_observable,
    zero_state,
)


class SimonettiHybridModel(BaseQMLModel):
    """Hybrid classical-quantum-classical two-qubit classifier.

    Two modes are exposed:

    ``hybrid``
        Paper-faithful reconstruction with a dense 2 -> 16 classical map.

    ``explicit_angle``
        Ablation mode with trainable quantum angle offsets plus a fixed
        repeated raw-feature angle encoding.
    """

    def __init__(self, mode="hybrid", n_sublayers=4, entangler="cnot_01"):
        if mode not in {"hybrid", "explicit_angle"}:
            raise ValueError(f"Unknown Simonetti mode: {mode}")
        if n_sublayers <= 0:
            raise ValueError("Simonetti model requires at least one sublayer.")

        self.mode = mode
        self.n_qubits = 2
        self.dim = 4
        self.n_sublayers = n_sublayers
        self.n_angles = 4 * n_sublayers
        self.n_quantum_features = 2
        self.measurement_observables = [
            z_observable(0, self.n_qubits),
            z_observable(1, self.n_qubits),
        ]
        self.obs = None
        self.entangler = self._build_entangler(entangler)

        if self.mode == "hybrid":
            self.input_weight_size = self.n_angles * 2
            self.input_bias_size = self.n_angles
            self.output_weight_size = self.n_quantum_features
            self.output_bias_size = 1
            self.n_params = (
                self.input_weight_size
                + self.input_bias_size
                + self.output_weight_size
                + self.output_bias_size
            )
        else:
            self.explicit_angle_size = self.n_angles
            self.output_weight_size = self.n_quantum_features
            self.output_bias_size = 1
            self.n_params = (
                self.explicit_angle_size + self.output_weight_size + self.output_bias_size
            )

        self._metadata = self._build_parameter_metadata()

    def _build_entangler(self, entangler):
        if entangler == "cnot_01":
            return cnot(0, 1, self.n_qubits)
        if entangler == "cnot_10":
            return cnot(1, 0, self.n_qubits)
        if entangler == "bidirectional":
            return cnot(1, 0, self.n_qubits) @ cnot(0, 1, self.n_qubits)
        raise ValueError(f"Unknown Simonetti entangler: {entangler}")

    def _angle_slot_info(self, angle_idx):
        local_idx = angle_idx % 4
        return {
            "angle_idx": angle_idx,
            "sublayer": angle_idx // 4,
            "qubit": 0 if local_idx in (0, 2) else 1,
            "axis": "ry" if local_idx < 2 else "rz",
            "data_feature_idx": 0 if local_idx in (0, 2) else 1,
        }

    def _input_weight_index(self, angle_idx, feature_idx):
        return angle_idx * 2 + feature_idx

    def _input_bias_index(self, angle_idx):
        return self.input_weight_size + angle_idx

    def _output_weight_slice(self):
        if self.mode == "hybrid":
            start = self.input_weight_size + self.input_bias_size
        else:
            start = self.explicit_angle_size
        return slice(start, start + self.output_weight_size)

    def _output_bias_index(self):
        return self._output_weight_slice().stop

    def _rotation_gate(self, axis, angle, qubit):
        gate_2x2 = ry(angle) if axis == "ry" else rz(angle)
        return single_qubit_gate(gate_2x2, qubit, self.n_qubits)

    def _gate_param_info(self, param_idx):
        if self.mode == "hybrid":
            if param_idx < self.input_weight_size:
                angle_idx = param_idx // 2
                feature_idx = param_idx % 2
                info = self._angle_slot_info(angle_idx)
                info.update(
                    {
                        "parameter_role": "input_weight",
                        "feature_idx": feature_idx,
                        "factor_kind": "feature",
                    }
                )
                return info
            if param_idx < self.input_weight_size + self.input_bias_size:
                angle_idx = param_idx - self.input_weight_size
                info = self._angle_slot_info(angle_idx)
                info.update(
                    {
                        "parameter_role": "input_bias",
                        "feature_idx": None,
                        "factor_kind": "bias",
                    }
                )
                return info
        else:
            if param_idx < self.explicit_angle_size:
                info = self._angle_slot_info(param_idx)
                info.update(
                    {
                        "parameter_role": "explicit_angle",
                        "feature_idx": None,
                        "factor_kind": "offset",
                    }
                )
                return info
        raise ValueError(f"Parameter {param_idx} is not a gate-supported quantum parameter.")

    def _gate_factor(self, param_idx, x):
        info = self._gate_param_info(param_idx)
        if info["factor_kind"] == "feature":
            return float(x[info["feature_idx"]])
        return 1.0

    def _readout_params(self, params):
        output_weights = params[self._output_weight_slice()]
        output_bias = params[self._output_bias_index()]
        return output_weights, output_bias

    def _quantum_feature_vector(self, final_state):
        return np.array(
            [expectation(final_state, obs) for obs in self.measurement_observables],
            dtype=float,
        )

    def _sample_forward_details(self, params, x):
        _, states = self.get_gate_sequence_and_states(params, x)
        final_state = states[-1]
        quantum_features = self._quantum_feature_vector(final_state)
        output_weights, output_bias = self._readout_params(params)
        logit = float(output_weights @ quantum_features + output_bias)
        prob = float(clip_probability(sigmoid(logit)))
        return {
            "state": final_state,
            "quantum_features": quantum_features,
            "logit": logit,
            "probability": prob,
            "output_weights": output_weights,
        }

    def init_params(self, seed=0):
        rng = np.random.RandomState(seed)
        params = np.zeros(self.n_params, dtype=float)

        if self.mode == "hybrid":
            params[: self.input_weight_size] = rng.normal(scale=0.35, size=self.input_weight_size)
            params[
                self.input_weight_size : self.input_weight_size + self.input_bias_size
            ] = rng.normal(scale=0.2, size=self.input_bias_size)
        else:
            params[: self.explicit_angle_size] = rng.uniform(
                -np.pi / 3.0, np.pi / 3.0, size=self.explicit_angle_size
            )

        output_slice = self._output_weight_slice()
        params[output_slice] = rng.normal(scale=0.25, size=self.output_weight_size)
        params[self._output_bias_index()] = rng.normal(scale=0.1)
        return params

    def parameter_metadata(self):
        return list(self._metadata)

    def gate_parameter_indices(self):
        return [
            item["index"] for item in self._metadata if item["supports_gate_derivative"]
        ]

    def nongate_parameter_indices(self):
        return [
            item["index"] for item in self._metadata if not item["supports_gate_derivative"]
        ]

    def forward(self, params, x):
        return self._sample_forward_details(np.asarray(params, dtype=float), x)["probability"]

    def terminal_costate(self, params, x, y, final_state):
        sample = self._sample_forward_details(params, x)
        delta = sample["probability"] - float(y)
        operator = sum(
            weight * observable
            for weight, observable in zip(sample["output_weights"], self.measurement_observables)
        )
        return delta * (operator @ final_state)

    def loss_gradient(self, params, X, y):
        params_arr = np.asarray(params, dtype=float)
        grad = np.zeros_like(params_arr)

        for x_i, y_i in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=float)):
            gates, states = self.get_gate_sequence_and_states(params_arr, x_i)
            final_state = states[-1]
            sample = self._sample_forward_details(params_arr, x_i)
            delta = sample["probability"] - y_i

            output_slice = self._output_weight_slice()
            grad[output_slice] += delta * sample["quantum_features"]
            grad[self._output_bias_index()] += delta

            terminal_operator = sum(
                weight * observable
                for weight, observable in zip(
                    sample["output_weights"], self.measurement_observables
                )
            )
            chi_states = [None] * len(states)
            chi_states[-1] = terminal_operator @ final_state
            for gate_idx in range(len(gates) - 1, -1, -1):
                chi_states[gate_idx] = gates[gate_idx][0].conj().T @ chi_states[gate_idx + 1]

            for gate_idx, (_, pidx) in enumerate(gates):
                if pidx is None:
                    continue
                gen = self.gate_derivative_generator(pidx, x_i)
                grad_vec = gen @ states[gate_idx + 1]
                grad[pidx] += delta * 2.0 * np.real(
                    chi_states[gate_idx + 1].conj() @ grad_vec
                )

        grad /= len(X)
        return grad, {
            "sample_forward_passes": len(X),
            "sample_backward_passes": len(X),
            "full_loss_evaluations": 0,
        }

    def nongate_loss_gradient(self, params, X, y):
        """Return the exact gradient restricted to the classical readout head."""
        params_arr = np.asarray(params, dtype=float)
        grad = np.zeros_like(params_arr)
        output_slice = self._output_weight_slice()
        output_bias_idx = self._output_bias_index()

        for x_i, y_i in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=float)):
            sample = self._sample_forward_details(params_arr, x_i)
            delta = sample["probability"] - y_i
            grad[output_slice] += delta * sample["quantum_features"]
            grad[output_bias_idx] += delta

        grad /= len(X)
        return grad, {
            "sample_forward_passes": len(X),
            "sample_backward_passes": 0,
            "full_loss_evaluations": 0,
        }

    def rebuild_param_gate(self, param_idx, params, x):
        info = self._gate_param_info(param_idx)
        factor = self._gate_factor(param_idx, x)
        angle = factor * float(params[param_idx])
        return self._rotation_gate(info["axis"], angle, info["qubit"])

    def gate_derivative_generator(self, param_idx, x=None):
        if x is None:
            raise ValueError("Simonetti gate derivatives require the sample x.")
        info = self._gate_param_info(param_idx)
        factor = self._gate_factor(param_idx, x)
        pauli = Y_PAULI if info["axis"] == "ry" else Z_PAULI
        return factor * (-1j * 0.5 * single_qubit_gate(pauli, info["qubit"], self.n_qubits))

    def get_gate_sequence_and_states(self, params, x):
        params_arr = np.asarray(params, dtype=float)
        x_arr = np.asarray(x, dtype=float)
        gates = []

        if self.mode == "hybrid":
            for angle_idx in range(self.n_angles):
                info = self._angle_slot_info(angle_idx)
                for feature_idx in range(2):
                    pidx = self._input_weight_index(angle_idx, feature_idx)
                    gates.append((self.rebuild_param_gate(pidx, params_arr, x_arr), pidx))
                bidx = self._input_bias_index(angle_idx)
                gates.append((self.rebuild_param_gate(bidx, params_arr, x_arr), bidx))
                if angle_idx % 4 == 3:
                    gates.append((self.entangler, None))
        else:
            for angle_idx in range(self.n_angles):
                info = self._angle_slot_info(angle_idx)
                data_angle = float(x_arr[info["data_feature_idx"]])
                gates.append((self._rotation_gate(info["axis"], data_angle, info["qubit"]), None))
                gates.append((self.rebuild_param_gate(angle_idx, params_arr, x_arr), angle_idx))
                if angle_idx % 4 == 3:
                    gates.append((self.entangler, None))

        state = zero_state(self.n_qubits)
        states = [state.copy()]
        for gate, _ in gates:
            state = gate @ state
            states.append(state.copy())
        return gates, states

    def _build_parameter_metadata(self):
        metadata = []
        if self.mode == "hybrid":
            for angle_idx in range(self.n_angles):
                info = self._angle_slot_info(angle_idx)
                for feature_idx in range(2):
                    pidx = self._input_weight_index(angle_idx, feature_idx)
                    metadata.append(
                        {
                            "index": pidx,
                            "name": f"input_weight[{angle_idx},{feature_idx}]",
                            "group": "classical_input",
                            "kind": "classical",
                            "supports_gate_derivative": True,
                            "axis": info["axis"],
                            "qubit": info["qubit"],
                            "sublayer": info["sublayer"],
                            "feature_idx": feature_idx,
                        }
                    )
                bidx = self._input_bias_index(angle_idx)
                metadata.append(
                    {
                        "index": bidx,
                        "name": f"input_bias[{angle_idx}]",
                        "group": "classical_input",
                        "kind": "classical",
                        "supports_gate_derivative": True,
                        "axis": info["axis"],
                        "qubit": info["qubit"],
                        "sublayer": info["sublayer"],
                        "feature_idx": None,
                    }
                )
        else:
            for angle_idx in range(self.n_angles):
                info = self._angle_slot_info(angle_idx)
                metadata.append(
                    {
                        "index": angle_idx,
                        "name": f"angle_offset[{angle_idx}]",
                        "group": "explicit_angles",
                        "kind": "quantum",
                        "supports_gate_derivative": True,
                        "axis": info["axis"],
                        "qubit": info["qubit"],
                        "sublayer": info["sublayer"],
                        "feature_idx": info["data_feature_idx"],
                    }
                )

        output_slice = self._output_weight_slice()
        for offset, pidx in enumerate(range(output_slice.start, output_slice.stop)):
            metadata.append(
                {
                    "index": pidx,
                    "name": f"output_weight[{offset}]",
                    "group": "classical_output",
                    "kind": "classical",
                    "supports_gate_derivative": False,
                    "axis": None,
                    "qubit": None,
                    "sublayer": None,
                    "feature_idx": offset,
                }
            )
        metadata.append(
            {
                "index": self._output_bias_index(),
                "name": "output_bias",
                "group": "classical_output",
                "kind": "classical",
                "supports_gate_derivative": False,
                "axis": None,
                "qubit": None,
                "sublayer": None,
                "feature_idx": None,
            }
        )
        return metadata
