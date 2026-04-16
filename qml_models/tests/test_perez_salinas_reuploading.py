"""Tests for the Perez-Salinas data re-uploading benchmark model."""

from __future__ import annotations

import unittest

import numpy as np

from qml_models.variants import PerezSalinasReuploadingModel
from datasets import (
    generate_perez_salinas_dataset,
    perez_salinas_benchmark_preset,
    perez_salinas_4q8l_preset,
    perez_salinas_problem_num_classes,
)


def finite_difference_gradient(model, params, X, y, eps=1e-6, indices=None):
    params_arr = np.asarray(params, dtype=float)
    test_indices = np.arange(len(params_arr)) if indices is None else np.asarray(indices, dtype=int)
    approx = np.zeros(len(test_indices), dtype=float)

    for out_idx, param_idx in enumerate(test_indices):
        plus = params_arr.copy()
        minus = params_arr.copy()
        plus[param_idx] += eps
        minus[param_idx] -= eps
        approx[out_idx] = (model.loss(plus, X, y) - model.loss(minus, X, y)) / (2.0 * eps)

    return test_indices, approx


class PerezSalinasDatasetTests(unittest.TestCase):
    def test_non_convex_generation_and_preset(self):
        X_train, X_test, y_train, y_test = generate_perez_salinas_dataset(
            problem="non_convex",
            n_samples=60,
            test_fraction=0.25,
            seed=7,
        )

        self.assertEqual(X_train.shape, (45, 2))
        self.assertEqual(X_test.shape, (15, 2))
        self.assertTrue(set(np.unique(y_train)).issubset({0, 1}))
        self.assertTrue(set(np.unique(y_test)).issubset({0, 1}))

        preset = perez_salinas_4q8l_preset("non_convex")
        self.assertEqual(preset["n_qubits"], 4)
        self.assertEqual(preset["n_layers"], 8)
        self.assertEqual(preset["loss_mode"], "weighted_fidelity")
        self.assertEqual(preset["n_classes"], 2)

    def test_multiclass_problem_metadata(self):
        self.assertEqual(perez_salinas_problem_num_classes("3_circles"), 4)
        X_train, X_test, y_train, y_test = generate_perez_salinas_dataset(
            problem="3_circles",
            n_samples=80,
            test_fraction=0.25,
            seed=5,
        )

        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(X_test.shape[1], 2)
        self.assertTrue(set(np.unique(y_train)).issubset({0, 1, 2, 3}))
        self.assertTrue(set(np.unique(y_test)).issubset({0, 1, 2, 3}))

    def test_generic_preset_supports_8_layers(self):
        preset = perez_salinas_benchmark_preset(
            problem="crown",
            n_qubits=4,
            n_layers=8,
            use_entanglement=True,
        )

        self.assertEqual(preset["problem"], "crown")
        self.assertEqual(preset["n_qubits"], 4)
        self.assertEqual(preset["n_layers"], 8)
        self.assertEqual(preset["n_classes"], 2)
        self.assertEqual(preset["loss_mode"], "weighted_fidelity")


class PerezSalinasModelTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.15, -0.25], [0.40, 0.10]], dtype=float)
        self.y = np.array([1, 0], dtype=int)

    def test_public_api_and_gate_sequence(self):
        model = PerezSalinasReuploadingModel(n_qubits=4, n_layers=6, n_classes=2, use_entanglement=True)
        params = model.init_params(seed=3)

        self.assertEqual(model.n_quantum_params, 4 * 6 * 5)
        self.assertEqual(model.n_params, model.n_quantum_params + 2 * 4)
        self.assertEqual(len(model.parameter_metadata()), model.n_params)
        self.assertEqual(len(model.gate_parameter_indices()), model.n_quantum_params)
        self.assertEqual(len(model.nongate_parameter_indices()), 8)

        gates, states = model.get_gate_sequence_and_states(params, self.X[0])
        self.assertEqual(len(gates), 4 * 6 * 5 + 5)
        self.assertEqual(len(states), len(gates) + 1)
        self.assertTrue(np.allclose(np.linalg.norm(states[-1]), 1.0, atol=1e-8))

        scores = model.class_scores(params, self.X[0])
        self.assertEqual(scores.shape, (2,))
        self.assertIn(model.predict(params, self.X[0]), (0, 1))
        self.assertGreaterEqual(model.loss(params, self.X, self.y), 0.0)

    def test_binary_gradient_matches_finite_difference(self):
        model = PerezSalinasReuploadingModel(n_qubits=2, n_layers=2, n_classes=2, use_entanglement=True)
        params = model.init_params(seed=11)

        analytic, stats = model.loss_gradient(params, self.X, self.y)
        _, numeric = finite_difference_gradient(model, params, self.X, self.y)

        self.assertEqual(stats["sample_forward_passes"], len(self.X))
        self.assertEqual(stats["sample_backward_passes"], len(self.X))
        self.assertTrue(np.allclose(analytic, numeric, atol=4e-5, rtol=4e-4))

    def test_nongate_gradient_matches_full_gradient_slice(self):
        model = PerezSalinasReuploadingModel(n_qubits=2, n_layers=2, n_classes=2, use_entanglement=True)
        params = model.init_params(seed=13)

        full_grad, _ = model.loss_gradient(params, self.X, self.y)
        nongate_grad, stats = model.nongate_loss_gradient(params, self.X, self.y)
        nongate_idx = np.asarray(model.nongate_parameter_indices(), dtype=int)
        gate_idx = np.asarray(model.gate_parameter_indices(), dtype=int)

        self.assertEqual(stats["sample_forward_passes"], len(self.X))
        self.assertEqual(stats["sample_backward_passes"], 0)
        self.assertTrue(np.allclose(nongate_grad[nongate_idx], full_grad[nongate_idx]))
        self.assertTrue(np.allclose(nongate_grad[gate_idx], 0.0))

    def test_multiclass_scores_and_prediction(self):
        model = PerezSalinasReuploadingModel(n_qubits=4, n_layers=1, n_classes=4, use_entanglement=True)
        params = model.init_params(seed=17)
        x = np.array([0.2, -0.1], dtype=float)

        scores = model.class_scores(params, x)
        self.assertEqual(scores.shape, (4,))
        self.assertIn(model.predict(params, x), {0, 1, 2, 3})

    def test_no_classical_head_uses_fixed_mean_fidelity_readout(self):
        model = PerezSalinasReuploadingModel(
            n_qubits=2,
            n_layers=2,
            n_classes=2,
            use_entanglement=True,
            use_classical_head=False,
        )
        params = model.init_params(seed=19)
        x = self.X[0]

        self.assertEqual(model.n_weight_params, 0)
        self.assertEqual(model.n_params, model.n_quantum_params)
        self.assertEqual(len(model.parameter_metadata()), model.n_quantum_params)
        self.assertEqual(model.nongate_parameter_indices(), [])

        details = model._sample_forward_details(params, x)
        expected_scores = np.mean(details["fidelities"], axis=1)
        self.assertTrue(np.allclose(details["scores"], expected_scores))

        nongate_grad, stats = model.nongate_loss_gradient(params, self.X, self.y)
        self.assertTrue(np.allclose(nongate_grad, 0.0))
        self.assertEqual(stats["sample_forward_passes"], 0)
        self.assertEqual(stats["sample_backward_passes"], 0)


if __name__ == "__main__":
    unittest.main()
