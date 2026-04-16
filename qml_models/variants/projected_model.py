"""Projected trainable-parameter views over larger QML models."""

from __future__ import annotations

import numpy as np

from ..common import EPS


class ProjectedTrainableModel:
    """Restrict optimization to a chosen parameter subset.

    This adapter exposes the same public training interface as the wrapped
    model, but the optimizer only sees the selected parameters. Internally the
    full parameter vector is reconstructed on every call, so existing loss and
    gate-derivative logic can be reused without modification.
    """

    def __init__(self, base_model, full_reference_params, trainable_indices, label=None):
        full_reference_params = np.asarray(full_reference_params, dtype=float)
        trainable_indices = np.asarray(trainable_indices, dtype=int)

        if trainable_indices.ndim != 1:
            raise ValueError("trainable_indices must be a 1D integer array.")
        if len(np.unique(trainable_indices)) != len(trainable_indices):
            raise ValueError("trainable_indices must be unique.")
        if np.any(trainable_indices < 0) or np.any(trainable_indices >= len(full_reference_params)):
            raise ValueError("trainable_indices contain out-of-range entries.")

        self.base_model = base_model
        self.full_reference_params = full_reference_params.copy()
        self.trainable_indices = trainable_indices.copy()
        self.label = label or getattr(base_model, "__class__", type(base_model)).__name__

        self.n_params = len(self.trainable_indices)
        self.n_qubits = getattr(base_model, "n_qubits", None)
        self.dim = getattr(base_model, "dim", None)
        self.obs = getattr(base_model, "obs", None)

        self._global_to_local = {int(global_idx): local_idx for local_idx, global_idx in enumerate(self.trainable_indices)}
        self._full_metadata = list(base_model.parameter_metadata()) if hasattr(base_model, "parameter_metadata") else []

    def expand_params(self, projected_params):
        projected_params = np.asarray(projected_params, dtype=float)
        if len(projected_params) != self.n_params:
            raise ValueError("Projected parameter vector has the wrong length.")
        full_params = self.full_reference_params.copy()
        full_params[self.trainable_indices] = projected_params
        return full_params

    def init_params(self, seed=0):
        base_init = np.asarray(self.base_model.init_params(seed=seed), dtype=float)
        self.full_reference_params = base_init.copy()
        return base_init[self.trainable_indices].copy()

    def get_initial_params(self, seed=0):
        return self.init_params(seed=seed)

    def parameter_metadata(self):
        if not self._full_metadata:
            return []
        return [self._full_metadata[idx] for idx in self.trainable_indices]

    def gate_parameter_indices(self):
        gate_indices = []
        for local_idx, global_idx in enumerate(self.trainable_indices):
            if self._full_metadata and self._full_metadata[global_idx].get("supports_gate_derivative", False):
                gate_indices.append(local_idx)
        return gate_indices

    def nongate_parameter_indices(self):
        gate_locals = set(self.gate_parameter_indices())
        return [idx for idx in range(self.n_params) if idx not in gate_locals]

    def forward(self, params, x):
        return self.base_model.forward(self.expand_params(params), x)

    def forward_batch(self, params, X):
        return self.base_model.forward_batch(self.expand_params(params), X)

    def predict(self, params, X):
        return self.base_model.predict(self.expand_params(params), X)

    def loss(self, params, X, y):
        return self.base_model.loss(self.expand_params(params), X, y)

    def accuracy(self, params, X, y):
        return self.base_model.accuracy(self.expand_params(params), X, y)

    def loss_gradient(self, params, X, y):
        full_params = self.expand_params(params)
        full_grad, stats = self.base_model.loss_gradient(full_params, X, y)
        return np.asarray(full_grad, dtype=float)[self.trainable_indices], stats

    def param_shift_gradient(self, params, X, y):
        return self.loss_gradient(params, X, y)

    def terminal_costate(self, params, x, y, final_state):
        if hasattr(self.base_model, "terminal_costate"):
            return self.base_model.terminal_costate(
                self.expand_params(params),
                x,
                y,
                final_state,
            )
        z = np.real(final_state.conj() @ self.base_model.obs @ final_state)
        p = np.clip((z + 1.0) / 2.0, EPS, 1.0 - EPS)
        dloss_dp = -y / p + (1.0 - y) / (1.0 - p)
        return 0.5 * dloss_dp * (self.base_model.obs @ final_state)

    def nongate_loss_gradient(self, params, X, y):
        if not hasattr(self.base_model, "nongate_loss_gradient"):
            full_grad, stats = self.base_model.loss_gradient(self.expand_params(params), X, y)
            return np.asarray(full_grad, dtype=float)[self.trainable_indices], stats
        full_grad, stats = self.base_model.nongate_loss_gradient(self.expand_params(params), X, y)
        return np.asarray(full_grad, dtype=float)[self.trainable_indices], stats

    def _to_global_index(self, local_idx):
        if not 0 <= local_idx < self.n_params:
            raise ValueError(f"Local parameter index {local_idx} is out of range.")
        return int(self.trainable_indices[local_idx])

    def rebuild_param_gate(self, param_idx, params, x):
        full_params = self.expand_params(params)
        return self.base_model.rebuild_param_gate(self._to_global_index(param_idx), full_params, x)

    def gate_derivative_generator(self, param_idx, x=None):
        return self.base_model.gate_derivative_generator(self._to_global_index(param_idx), x)

    def get_gate_sequence_and_states(self, params, x):
        full_params = self.expand_params(params)
        gates, states = self.base_model.get_gate_sequence_and_states(full_params, x)
        projected_gates = []
        for gate, global_idx in gates:
            if global_idx is None:
                projected_gates.append((gate, None))
                continue
            local_idx = self._global_to_local.get(int(global_idx))
            projected_gates.append((gate, local_idx))
        return projected_gates, states
