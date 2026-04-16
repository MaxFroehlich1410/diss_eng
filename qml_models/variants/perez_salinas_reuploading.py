"""Perez-Salinas et al. (2020) multi-qubit data re-uploading classifier.

This reconstruction mirrors the executable reference implementation published
alongside "Data re-uploading for a universal quantum classifier" while exposing
the gate-by-gate hooks required by the repository's Krotov optimizers.

The paper's 2D multi-qubit model applies, on every qubit and layer, a
three-angle single-qubit block

    U3(theta, phi, lam) = Rz(-phi) @ Ry(theta) @ Rz(-lam)

with the first two angles depending affinely on the input coordinates.
Following the reference code, for a 2D task each layer/qubit block carries
five trainable parameters:

    theta0, theta1, theta2, alpha0, alpha1

such that

    theta = theta0 + alpha0 * x0
    phi   = theta1 + alpha1 * x1
    lam   = theta2

To preserve exact gatewise derivatives, the affine same-axis contributions are
represented as separate commuting rotations in the gate sequence.
"""

from __future__ import annotations

import numpy as np

from ..common import Y_PAULI, Z_PAULI, embed_two_qubit_operator, ry, rz, single_qubit_gate, zero_state


_CZ_LOCAL = np.diag([1.0, 1.0, 1.0, -1.0]).astype(complex)


class PerezSalinasReuploadingModel:
    """Paper-faithful multi-qubit data re-uploading classifier.

    Parameters
    ----------
    n_qubits:
        Number of classifier qubits. The original paper studies ``1``, ``2``,
        and ``4`` qubits. The requested benchmark preset is ``4``.
    n_layers:
        Number of re-uploading layers. The requested benchmark preset is ``6``.
    n_classes:
        Number of classes. The label-state constructions from the paper support
        ``2``, ``3``, and ``4`` classes.
    input_dim:
        Input dimensionality. This reconstruction currently targets the paper's
        2D classification problems.
    use_entanglement:
        Whether to insert the paper's alternating ``CZ`` entanglers between
        layers. The final layer never receives an entangler.
    use_classical_head:
        Whether to learn per-class, per-qubit readout weights over the local
        fidelities. When disabled, the model uses a fixed mean-fidelity readout
        with no extra classical parameters.
    loss_mode:
        Optimization objective. Only ``"weighted_fidelity"`` is implemented
        here because it is the paper's most useful multi-qubit benchmark loss
        and matches the released reference code.
    """

    _PARAM_SPECS = (
        {
            "slot_name": "theta0",
            "axis": "ry",
            "group": "theta",
            "feature_idx": None,
            "factor_sign": 1.0,
        },
        {
            "slot_name": "theta1",
            "axis": "rz",
            "group": "theta",
            "feature_idx": None,
            "factor_sign": -1.0,
        },
        {
            "slot_name": "theta2",
            "axis": "rz",
            "group": "theta",
            "feature_idx": None,
            "factor_sign": -1.0,
        },
        {
            "slot_name": "alpha0",
            "axis": "ry",
            "group": "alpha",
            "feature_idx": 0,
            "factor_sign": 1.0,
        },
        {
            "slot_name": "alpha1",
            "axis": "rz",
            "group": "alpha",
            "feature_idx": 1,
            "factor_sign": -1.0,
        },
    )

    # One U3 block factorised into commuting one-parameter gates.
    _GATE_SLOT_ORDER = (1, 4, 0, 3, 2)

    def __init__(
        self,
        n_qubits=4,
        n_layers=6,
        n_classes=2,
        input_dim=2,
        use_entanglement=True,
        use_classical_head=True,
        loss_mode="weighted_fidelity",
    ):
        if n_qubits not in {1, 2, 4}:
            raise ValueError("Perez-Salinas model currently supports 1, 2, or 4 qubits.")
        if n_layers <= 0:
            raise ValueError("Perez-Salinas model requires at least one layer.")
        if n_classes not in {2, 3, 4}:
            raise ValueError("Perez-Salinas model supports 2, 3, or 4 classes.")
        if input_dim != 2:
            raise ValueError("This reconstruction currently targets the paper's 2D tasks only.")
        if loss_mode != "weighted_fidelity":
            raise ValueError("Only weighted_fidelity is implemented for this model.")

        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.n_classes = int(n_classes)
        self.input_dim = int(input_dim)
        self.use_entanglement = bool(use_entanglement)
        self.use_classical_head = bool(use_classical_head)
        self.loss_mode = str(loss_mode)
        self.dim = 2**self.n_qubits

        self.params_per_block = len(self._PARAM_SPECS)
        self.n_quantum_params = self.n_qubits * self.n_layers * self.params_per_block
        self.n_weight_params = self.n_classes * self.n_qubits if self.use_classical_head else 0
        self.n_params = self.n_quantum_params + self.n_weight_params

        self.label_states = self._build_label_states(self.n_classes)
        self.local_projectors = [
            np.outer(label_state, np.conjugate(label_state)) for label_state in self.label_states
        ]
        self.projector_ops = [
            [single_qubit_gate(projector, qubit, self.n_qubits) for qubit in range(self.n_qubits)]
            for projector in self.local_projectors
        ]
        self._metadata = self._build_parameter_metadata()

        if self.n_qubits == 2:
            self._two_qubit_entangler = embed_two_qubit_operator(_CZ_LOCAL, 0, 1, self.n_qubits)
        elif self.n_qubits == 4:
            self._four_qubit_entanglers = (
                embed_two_qubit_operator(_CZ_LOCAL, 0, 1, self.n_qubits)
                @ embed_two_qubit_operator(_CZ_LOCAL, 2, 3, self.n_qubits),
                embed_two_qubit_operator(_CZ_LOCAL, 1, 2, self.n_qubits)
                @ embed_two_qubit_operator(_CZ_LOCAL, 0, 3, self.n_qubits),
            )

    def _build_label_states(self, n_classes):
        if n_classes == 2:
            return np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=complex,
            )
        if n_classes == 3:
            return np.array(
                [
                    [1.0, 0.0],
                    [0.5, np.sqrt(3.0) / 2.0],
                    [0.5, -np.sqrt(3.0) / 2.0],
                ],
                dtype=complex,
            )
        return np.array(
            [
                [1.0, 0.0],
                [1.0 / np.sqrt(3.0), np.sqrt(2.0 / 3.0)],
                [1.0 / np.sqrt(3.0), np.exp(1j * 2.0 * np.pi / 3.0) * np.sqrt(2.0 / 3.0)],
                [1.0 / np.sqrt(3.0), np.exp(-1j * 2.0 * np.pi / 3.0) * np.sqrt(2.0 / 3.0)],
            ],
            dtype=complex,
        )

    def _target_vector(self, y):
        if self.n_classes == 2:
            target = np.zeros(self.n_classes, dtype=float)
        elif self.n_classes == 3:
            target = 0.25 * np.ones(self.n_classes, dtype=float)
        else:
            target = (1.0 / 3.0) * np.ones(self.n_classes, dtype=float)
        target[int(y)] = 1.0
        return target

    def _block_base(self, layer, qubit):
        return (layer * self.n_qubits + qubit) * self.params_per_block

    def _quantum_param_info(self, param_idx):
        if not 0 <= param_idx < self.n_quantum_params:
            raise ValueError(f"Parameter {param_idx} is not a gate-supported quantum parameter.")
        block = param_idx // self.params_per_block
        slot = param_idx % self.params_per_block
        layer = block // self.n_qubits
        qubit = block % self.n_qubits
        spec = dict(self._PARAM_SPECS[slot])
        spec.update(
            {
                "index": int(param_idx),
                "layer": int(layer),
                "qubit": int(qubit),
                "slot": int(slot),
            }
        )
        return spec

    def _weight_slice(self):
        return slice(self.n_quantum_params, self.n_params)

    def _rotation_gate(self, axis, angle, qubit):
        gate_2x2 = ry(angle) if axis == "ry" else rz(angle)
        return single_qubit_gate(gate_2x2, qubit, self.n_qubits)

    def _gate_factor(self, info, x):
        factor = float(info["factor_sign"])
        if info["feature_idx"] is not None:
            x_arr = np.asarray(x, dtype=float)
            factor *= float(x_arr[info["feature_idx"]])
        return factor

    def _weights(self, params):
        if not self.use_classical_head:
            return np.full((self.n_classes, self.n_qubits), 1.0 / float(self.n_qubits), dtype=float)
        params_arr = np.asarray(params, dtype=float)
        return params_arr[self._weight_slice()].reshape(self.n_classes, self.n_qubits)

    def _layer_entangler(self, layer):
        if not self.use_entanglement or self.n_qubits == 1 or layer >= self.n_layers - 1:
            return None
        if self.n_qubits == 2:
            return self._two_qubit_entangler
        return self._four_qubit_entanglers[layer % 2]

    def _sample_fidelities(self, final_state):
        fidelities = np.empty((self.n_classes, self.n_qubits), dtype=float)
        for class_idx in range(self.n_classes):
            for qubit in range(self.n_qubits):
                value = np.real(final_state.conj() @ self.projector_ops[class_idx][qubit] @ final_state)
                fidelities[class_idx, qubit] = float(np.clip(value, 0.0, 1.0))
        return fidelities

    def _sample_scores(self, params, fidelities):
        return np.sum(self._weights(params) * fidelities, axis=1)

    def _sample_forward_details(self, params, x):
        _, states = self.get_gate_sequence_and_states(params, x)
        final_state = states[-1]
        fidelities = self._sample_fidelities(final_state)
        scores = self._sample_scores(params, fidelities)
        return {
            "state": final_state,
            "fidelities": fidelities,
            "scores": scores,
        }

    def get_initial_params(self, seed=0):
        return self.init_params(seed=seed)

    def init_params(self, seed=0):
        rng = np.random.RandomState(seed)
        return rng.rand(self.n_params).astype(float)

    def parameter_metadata(self):
        return list(self._metadata)

    def gate_parameter_indices(self):
        return list(range(self.n_quantum_params))

    def nongate_parameter_indices(self):
        if not self.use_classical_head:
            return []
        return list(range(self.n_quantum_params, self.n_params))

    def class_scores(self, params, x):
        return self._sample_forward_details(np.asarray(params, dtype=float), x)["scores"]

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
            return int(np.argmax(self.class_scores(params, X_arr)))
        return np.array([int(np.argmax(self.class_scores(params, x))) for x in X_arr], dtype=int)

    def accuracy(self, params, X, y):
        preds = self.predict(params, X)
        return float(np.mean(preds == np.asarray(y, dtype=int)))

    def loss(self, params, X, y):
        params_arr = np.asarray(params, dtype=float)
        losses = []
        for x_i, y_i in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=int)):
            scores = self.class_scores(params_arr, x_i)
            residual = scores - self._target_vector(y_i)
            losses.append(0.5 * float(residual @ residual))
        return float(np.mean(losses))

    def _terminal_operator(self, params, residual):
        operator = np.zeros((self.dim, self.dim), dtype=complex)
        weights = self._weights(params)
        for class_idx in range(self.n_classes):
            for qubit in range(self.n_qubits):
                operator += residual[class_idx] * weights[class_idx, qubit] * self.projector_ops[class_idx][qubit]
        return operator

    def terminal_costate(self, params, _x, y, final_state):
        fidelities = self._sample_fidelities(final_state)
        scores = self._sample_scores(params, fidelities)
        residual = scores - self._target_vector(y)
        return self._terminal_operator(params, residual) @ final_state

    def loss_gradient(self, params, X, y):
        params_arr = np.asarray(params, dtype=float)
        grad = np.zeros_like(params_arr)

        for x_i, y_i in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=int)):
            gates, states = self.get_gate_sequence_and_states(params_arr, x_i)
            final_state = states[-1]
            fidelities = self._sample_fidelities(final_state)
            scores = self._sample_scores(params_arr, fidelities)
            residual = scores - self._target_vector(y_i)

            if self.use_classical_head:
                grad[self._weight_slice()] += (residual[:, None] * fidelities).reshape(-1)

            chi_states = [None] * len(states)
            chi_states[-1] = self._terminal_operator(params_arr, residual) @ final_state
            for gate_idx in range(len(gates) - 1, -1, -1):
                chi_states[gate_idx] = gates[gate_idx][0].conj().T @ chi_states[gate_idx + 1]

            for gate_idx, (_, pidx) in enumerate(gates):
                if pidx is None:
                    continue
                gen = self.gate_derivative_generator(pidx, x_i)
                grad_vec = gen @ states[gate_idx + 1]
                grad[pidx] += 2.0 * np.real(chi_states[gate_idx + 1].conj() @ grad_vec)

        grad /= len(X)
        return grad, {
            "sample_forward_passes": len(X),
            "sample_backward_passes": len(X),
            "full_loss_evaluations": 0,
        }

    def nongate_loss_gradient(self, params, X, y):
        params_arr = np.asarray(params, dtype=float)
        grad = np.zeros_like(params_arr)
        if not self.use_classical_head:
            return grad, {
                "sample_forward_passes": 0,
                "sample_backward_passes": 0,
                "full_loss_evaluations": 0,
            }

        for x_i, y_i in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=int)):
            details = self._sample_forward_details(params_arr, x_i)
            residual = details["scores"] - self._target_vector(y_i)
            grad[self._weight_slice()] += (residual[:, None] * details["fidelities"]).reshape(-1)

        grad /= len(X)
        return grad, {
            "sample_forward_passes": len(X),
            "sample_backward_passes": 0,
            "full_loss_evaluations": 0,
        }

    def rebuild_param_gate(self, param_idx, params, x):
        info = self._quantum_param_info(param_idx)
        angle = self._gate_factor(info, x) * float(params[param_idx])
        return self._rotation_gate(info["axis"], angle, info["qubit"])

    def gate_derivative_generator(self, param_idx, x=None):
        if x is None:
            raise ValueError("Perez-Salinas gate derivatives require the sample x.")
        info = self._quantum_param_info(param_idx)
        factor = self._gate_factor(info, x)
        pauli = Y_PAULI if info["axis"] == "ry" else Z_PAULI
        return factor * (-1j * 0.5 * single_qubit_gate(pauli, info["qubit"], self.n_qubits))

    def get_gate_sequence_and_states(self, params, x):
        params_arr = np.asarray(params, dtype=float)
        x_arr = np.asarray(x, dtype=float)
        gates = []

        for layer in range(self.n_layers):
            for qubit in range(self.n_qubits):
                block_base = self._block_base(layer, qubit)
                for slot in self._GATE_SLOT_ORDER:
                    pidx = block_base + slot
                    gates.append((self.rebuild_param_gate(pidx, params_arr, x_arr), pidx))
            entangler = self._layer_entangler(layer)
            if entangler is not None:
                gates.append((entangler, None))

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
                    "name": f"{info['slot_name']}[{info['layer']},{info['qubit']}]",
                    "group": info["group"],
                    "kind": "quantum",
                    "supports_gate_derivative": True,
                    "layer": info["layer"],
                    "qubit": info["qubit"],
                    "axis": info["axis"],
                    "feature_idx": info["feature_idx"],
                }
            )

        if self.use_classical_head:
            for class_idx in range(self.n_classes):
                for qubit in range(self.n_qubits):
                    pidx = self.n_quantum_params + class_idx * self.n_qubits + qubit
                    metadata.append(
                        {
                            "index": pidx,
                            "name": f"weight[{class_idx},{qubit}]",
                            "group": "classical_weight",
                            "kind": "classical",
                            "supports_gate_derivative": False,
                            "layer": None,
                            "qubit": qubit,
                            "axis": None,
                            "feature_idx": None,
                        }
                    )
        return metadata
