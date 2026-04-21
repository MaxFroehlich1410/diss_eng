"""PennyLane implementation of the Perez-Salinas data re-uploading classifier."""

from __future__ import annotations

import os
import sys

import numpy as np

VENDOR_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "pennylane")
)
if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

import pennylane as qml
from pennylane import numpy as pnp

from .perez_salinas_reuploading import PerezSalinasReuploadingModel


class PennyLanePerezSalinasReuploadingModel:
    """PennyLane mirror of :class:`PerezSalinasReuploadingModel`.

    The circuit, parameter layout, local-fidelity readout, and weighted-fidelity
    loss are matched to the native implementation so optimizer comparisons can
    isolate the training algorithm rather than a model mismatch.
    """

    def __init__(
        self,
        n_qubits=4,
        n_layers=8,
        n_classes=2,
        input_dim=2,
        use_entanglement=True,
        use_classical_head=True,
        loss_mode="weighted_fidelity",
        interface="autograd",
    ):
        self.native_reference = PerezSalinasReuploadingModel(
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_classes=n_classes,
            input_dim=input_dim,
            use_entanglement=use_entanglement,
            use_classical_head=use_classical_head,
            loss_mode=loss_mode,
        )
        self.n_qubits = self.native_reference.n_qubits
        self.n_layers = self.native_reference.n_layers
        self.n_classes = self.native_reference.n_classes
        self.input_dim = self.native_reference.input_dim
        self.use_entanglement = self.native_reference.use_entanglement
        self.use_classical_head = self.native_reference.use_classical_head
        self.loss_mode = self.native_reference.loss_mode
        self.n_quantum_params = self.native_reference.n_quantum_params
        self.n_weight_params = self.native_reference.n_weight_params
        self.n_params = self.native_reference.n_params
        self.params_per_block = self.native_reference.params_per_block
        self.label_states = tuple(
            pnp.array(state, dtype=complex, requires_grad=False)
            for state in self.native_reference.label_states
        )
        self.projector_ops = tuple(
            tuple(pnp.array(op, dtype=complex, requires_grad=False) for op in class_ops)
            for class_ops in self.native_reference.projector_ops
        )
        self.weights_shape = (self.n_layers, self.n_qubits, self.params_per_block)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface=interface)
        def circuit(x, quantum_params):
            x_arr = qml.math.asarray(x)
            reshaped = qml.math.reshape(quantum_params, self.weights_shape)
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    theta0, theta1, theta2, alpha0, alpha1 = reshaped[layer, qubit]
                    qml.RZ(-theta1, wires=qubit)
                    qml.RZ(-(alpha1 * x_arr[1]), wires=qubit)
                    qml.RY(theta0, wires=qubit)
                    qml.RY(alpha0 * x_arr[0], wires=qubit)
                    qml.RZ(-theta2, wires=qubit)
                if self.use_entanglement and layer < self.n_layers - 1:
                    if self.n_qubits == 2:
                        qml.CZ(wires=[0, 1])
                    elif self.n_qubits == 4:
                        if layer % 2 == 0:
                            qml.CZ(wires=[0, 1])
                            qml.CZ(wires=[2, 3])
                        else:
                            qml.CZ(wires=[1, 2])
                            qml.CZ(wires=[0, 3])
            return qml.state()

        self.circuit = circuit
        self.metric_tensor_fn = qml.metric_tensor(self.circuit, approx=None)

    def init_params(self, seed=0):
        return self.native_reference.init_params(seed=seed)

    def _split_params(self, params):
        params_arr = pnp.array(params, dtype=float, requires_grad=getattr(params, "requires_grad", False))
        quantum = qml.math.reshape(params_arr[: self.n_quantum_params], self.weights_shape)
        classical = params_arr[self.n_quantum_params :]
        return params_arr, quantum, classical

    def _weights(self, params):
        if not self.use_classical_head:
            return pnp.ones((self.n_classes, self.n_qubits), dtype=float) / float(self.n_qubits)
        params_arr, _, classical = self._split_params(params)
        del params_arr
        return qml.math.reshape(classical, (self.n_classes, self.n_qubits))

    def state(self, params, x):
        _, quantum, _ = self._split_params(params)
        return self.circuit(pnp.array(x, dtype=float, requires_grad=False), quantum)

    def sample_fidelities(self, params, x):
        state = self.state(params, x)
        fidelities = []
        for class_ops in self.projector_ops:
            class_vals = []
            for op in class_ops:
                class_vals.append(qml.math.real(qml.math.conj(state) @ (op @ state)))
            fidelities.append(class_vals)
        return qml.math.stack([qml.math.stack(class_vals) for class_vals in fidelities])

    def class_scores(self, params, x):
        fidelities = self.sample_fidelities(params, x)
        return qml.math.sum(self._weights(params) * fidelities, axis=1)

    def forward(self, params, x):
        if self.n_classes != 2:
            raise ValueError("forward() is scalar only for binary tasks; use class_scores() instead.")
        return float(self.class_scores(params, x)[1])

    def forward_batch(self, params, X):
        if self.n_classes != 2:
            raise ValueError("forward_batch() is scalar only for binary tasks; use predict() instead.")
        return np.array([self.forward(params, x) for x in np.asarray(X, dtype=float)], dtype=float)

    def predict(self, params, X):
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            return int(np.argmax(np.asarray(self.class_scores(params, X_arr), dtype=float)))
        return np.array(
            [int(np.argmax(np.asarray(self.class_scores(params, x), dtype=float))) for x in X_arr],
            dtype=int,
        )

    def accuracy(self, params, X, y):
        preds = self.predict(params, X)
        return float(np.mean(preds == np.asarray(y, dtype=int)))

    def _target_vector(self, y):
        return pnp.array(self.native_reference._target_vector(y), dtype=float, requires_grad=False)

    def loss(self, params, X, y):
        losses = []
        for x_i, y_i in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=int)):
            residual = self.class_scores(params, x_i) - self._target_vector(y_i)
            losses.append(0.5 * qml.math.dot(residual, residual))
        return qml.math.mean(qml.math.stack(losses))

    def metric_tensor(self, params, X, approx="block-diag"):
        _, quantum, _ = self._split_params(params)
        approx_value = None if approx == "full" else approx
        metric_fn = qml.metric_tensor(self.circuit, approx=approx_value)
        metrics = [metric_fn(pnp.array(x, dtype=float, requires_grad=False), quantum) for x in X]
        mean_metric = sum(metrics) / len(metrics)
        mean_metric = qml.math.reshape(mean_metric, (self.n_quantum_params, self.n_quantum_params))
        if not self.use_classical_head:
            return mean_metric
        classical_block = pnp.eye(self.n_weight_params, dtype=float)
        top = qml.math.concatenate(
            [mean_metric, qml.math.zeros((mean_metric.shape[0], self.n_weight_params), dtype=float)],
            axis=1,
        )
        bottom = qml.math.concatenate(
            [qml.math.zeros((self.n_weight_params, mean_metric.shape[1]), dtype=float), classical_block],
            axis=1,
        )
        return qml.math.concatenate([top, bottom], axis=0)
