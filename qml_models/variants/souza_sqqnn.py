"""Souza et al. (2024) single-qubit quantum neural network models."""

from __future__ import annotations

import numpy as np

from ..common import (
    BaseQMLModel,
    I2,
    Y_PAULI,
    Z_PAULI,
    clip_probability,
    expectation,
    ry,
    rz,
    single_qubit_gate,
    zero_state,
)


class SouzaSQQNNModel(BaseQMLModel):
    """Single-qubit quantum neural network for two-moons classification."""

    def __init__(self, variant="reduced", n_neurons=4, basis_labels=None):
        if variant not in {"reduced", "full"}:
            raise ValueError(f"Unknown SQQNN variant: {variant}")
        if n_neurons <= 0:
            raise ValueError("SQQNN requires at least one neuron layer.")

        self.variant = variant
        self.n_neurons = n_neurons
        self.n_qubits = 1
        self.dim = 2
        self.obs = -single_qubit_gate(Z_PAULI, 0, self.n_qubits)
        self.basis_labels = basis_labels or ["1", "x1", "x2", "x1^2", "x1*x2", "x2^2"]
        self.basis_dim = len(self.basis_labels)
        self.angle_families = ("beta",) if self.variant == "reduced" else ("alpha", "beta", "gamma")
        self.axes = {"alpha": "rz", "beta": "ry", "gamma": "rz"}
        self.n_params = self.n_neurons * len(self.angle_families) * self.basis_dim
        self._metadata = self._build_parameter_metadata()

    def polynomial_features(self, x):
        x1, x2 = np.asarray(x, dtype=float)
        return np.array([1.0, x1, x2, x1**2, x1 * x2, x2**2], dtype=float)

    def init_params(self, seed=0):
        rng = np.random.RandomState(seed)
        return rng.normal(scale=0.3, size=self.n_params)

    def parameter_metadata(self):
        return list(self._metadata)

    def gate_parameter_indices(self):
        return list(range(self.n_params))

    def _param_info(self, param_idx):
        family_block = self.basis_dim
        neuron_block = len(self.angle_families) * family_block
        neuron_idx = param_idx // neuron_block
        within_neuron = param_idx % neuron_block
        family_idx = within_neuron // family_block
        basis_idx = within_neuron % family_block
        family = self.angle_families[family_idx]
        return {
            "neuron": neuron_idx,
            "family": family,
            "axis": self.axes[family],
            "basis_idx": basis_idx,
            "basis_label": self.basis_labels[basis_idx],
        }

    def _rotation_gate(self, axis, angle):
        gate_2x2 = ry(angle) if axis == "ry" else rz(angle)
        return single_qubit_gate(gate_2x2, 0, self.n_qubits)

    def forward(self, params, x):
        _, states = self.get_gate_sequence_and_states(params, x)
        state = states[-1]
        prob = 0.5 * (expectation(state, self.obs) + 1.0)
        return float(clip_probability(prob))

    def loss_gradient(self, params, X, y):
        params_arr = np.asarray(params, dtype=float)
        grad = np.zeros_like(params_arr)

        for x_i, y_i in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=float)):
            gates, states = self.get_gate_sequence_and_states(params_arr, x_i)
            final_state = states[-1]
            z = expectation(final_state, self.obs)
            p = clip_probability(0.5 * (z + 1.0))
            dloss_dp = -y_i / p + (1.0 - y_i) / (1.0 - p)

            chi_states = [None] * len(states)
            chi_states[-1] = self.obs @ final_state
            for gate_idx in range(len(gates) - 1, -1, -1):
                chi_states[gate_idx] = gates[gate_idx][0].conj().T @ chi_states[gate_idx + 1]

            for gate_idx, (_, pidx) in enumerate(gates):
                gen = self.gate_derivative_generator(pidx, x_i)
                grad_vec = gen @ states[gate_idx + 1]
                dz_dtheta = 2.0 * np.real(chi_states[gate_idx + 1].conj() @ grad_vec)
                grad[pidx] += 0.5 * dloss_dp * dz_dtheta

        grad /= len(X)
        return grad, {
            "sample_forward_passes": len(X),
            "sample_backward_passes": len(X),
            "full_loss_evaluations": 0,
        }

    def rebuild_param_gate(self, param_idx, params, x):
        info = self._param_info(param_idx)
        factor = self.polynomial_features(x)[info["basis_idx"]]
        angle = float(params[param_idx]) * factor
        return self._rotation_gate(info["axis"], angle)

    def gate_derivative_generator(self, param_idx, x=None):
        if x is None:
            raise ValueError("SQQNN gate derivatives require the sample x.")
        info = self._param_info(param_idx)
        factor = self.polynomial_features(x)[info["basis_idx"]]
        pauli = Y_PAULI if info["axis"] == "ry" else Z_PAULI
        return factor * (-1j * 0.5 * single_qubit_gate(pauli, 0, self.n_qubits))

    def get_gate_sequence_and_states(self, params, x):
        params_arr = np.asarray(params, dtype=float)
        gates = []
        for pidx in range(self.n_params):
            gates.append((self.rebuild_param_gate(pidx, params_arr, x), pidx))

        state = zero_state(self.n_qubits)
        states = [state.copy()]
        for gate, _ in gates:
            state = gate @ state
            states.append(state.copy())
        return gates, states

    def _build_parameter_metadata(self):
        metadata = []
        for pidx in range(self.n_params):
            info = self._param_info(pidx)
            metadata.append(
                {
                    "index": pidx,
                    "name": f"{info['family']}[{info['neuron']},{info['basis_idx']}]",
                    "group": info["family"],
                    "kind": "quantum",
                    "supports_gate_derivative": True,
                    "axis": info["axis"],
                    "neuron": info["neuron"],
                    "basis_idx": info["basis_idx"],
                    "basis_label": info["basis_label"],
                }
            )
        return metadata
