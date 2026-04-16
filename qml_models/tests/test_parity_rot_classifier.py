"""Tests for the PennyLane-style 4-bit parity classifier."""

from __future__ import annotations

import unittest

import numpy as np

from datasets import generate_parity_4bit
from qml_models.variants import ParityRotClassifierModel


def finite_difference_gradient(model, params, X, y, eps=1e-6):
    params = np.asarray(params, dtype=float)
    approx = np.zeros_like(params)
    for param_idx in range(len(params)):
        plus = params.copy()
        minus = params.copy()
        plus[param_idx] += eps
        minus[param_idx] -= eps
        approx[param_idx] = (model.loss(plus, X, y) - model.loss(minus, X, y)) / (2.0 * eps)
    return approx


class ParityRotClassifierModelTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 1, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=int,
        )
        self.y = np.array([-1, 1, -1, -1], dtype=float)

    def test_public_api_and_metadata(self):
        model = ParityRotClassifierModel(n_layers=2)
        params_a = model.init_params(seed=7)
        params_b = model.get_initial_params(seed=7)

        self.assertEqual(model.weights_shape, (2, 4, 3))
        self.assertEqual(model.n_quantum_params, 24)
        self.assertEqual(model.n_params, 25)
        self.assertTrue(np.allclose(params_a, params_b))

        metadata = model.parameter_metadata()
        self.assertEqual(len(metadata), model.n_params)
        self.assertEqual(sum(item["supports_gate_derivative"] for item in metadata), 24)
        self.assertEqual(model.nongate_parameter_indices(), [24])

    def test_gate_sequence_matches_rot_ring_structure(self):
        model = ParityRotClassifierModel(n_layers=2)
        params = model.init_params(seed=3)
        gates, states = model.get_gate_sequence_and_states(params, self.X[1])

        self.assertEqual(len(gates), 2 * (4 * 3 + 4))
        self.assertEqual(len(states), len(gates) + 1)
        self.assertTrue(np.allclose(np.linalg.norm(states[0]), 1.0, atol=1e-8))
        self.assertTrue(np.allclose(np.linalg.norm(states[-1]), 1.0, atol=1e-8))

        gate = model.rebuild_param_gate(0, params, self.X[1])
        eps = 1e-7
        plus = params.copy()
        minus = params.copy()
        plus[0] += eps
        minus[0] -= eps
        fd = (model.rebuild_param_gate(0, plus, self.X[1]) - model.rebuild_param_gate(0, minus, self.X[1])) / (
            2.0 * eps
        )
        analytic = model.gate_derivative_generator(0, self.X[1]) @ gate
        self.assertTrue(np.allclose(fd, analytic, atol=1e-6))

        with self.assertRaises(ValueError):
            model.rebuild_param_gate(model.n_params - 1, params, self.X[1])

    def test_square_loss_gradient_matches_finite_difference(self):
        model = ParityRotClassifierModel(n_layers=1)
        params = model.init_params(seed=11)
        analytic, stats = model.loss_gradient(params, self.X, self.y)
        numeric = finite_difference_gradient(model, params, self.X, self.y)

        self.assertEqual(stats["sample_forward_passes"], len(self.X))
        self.assertEqual(stats["sample_backward_passes"], len(self.X))
        self.assertTrue(np.allclose(analytic, numeric, atol=4e-5, rtol=3e-4))

    def test_nongate_gradient_matches_bias_slice(self):
        model = ParityRotClassifierModel(n_layers=1)
        params = model.init_params(seed=5)
        full_grad, _ = model.loss_gradient(params, self.X, self.y)
        nongate_grad, stats = model.nongate_loss_gradient(params, self.X, self.y)

        self.assertEqual(stats["sample_forward_passes"], len(self.X))
        self.assertEqual(stats["sample_backward_passes"], 0)
        self.assertEqual(model.nongate_parameter_indices(), [model.n_quantum_params])
        self.assertTrue(np.allclose(nongate_grad[-1], full_grad[-1]))
        self.assertTrue(np.allclose(nongate_grad[:-1], 0.0))

    def test_prediction_rule_uses_output_sign(self):
        model = ParityRotClassifierModel(n_layers=1)
        params = np.zeros(model.n_params, dtype=float)
        scores = model.forward_batch(params, self.X[:2])
        preds = model.predict(params, self.X[:2])
        self.assertTrue(np.array_equal(preds, np.where(scores >= 0.0, 1, -1)))

        params[-1] = -1.5
        shifted_scores = model.forward_batch(params, self.X[:2])
        preds = model.predict(params, self.X[:2])
        self.assertTrue(np.array_equal(preds, np.where(shifted_scores >= 0.0, 1, -1)))


class ParityDatasetTests(unittest.TestCase):
    def test_dataset_labels_follow_parity(self):
        X_train, X_test, y_train, y_test = generate_parity_4bit(test_fraction=0.25, seed=0, repeats=2)
        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)

        expected = np.where(np.sum(X_all, axis=1) % 2 == 1, 1, -1)
        self.assertTrue(np.array_equal(y_all, expected))


if __name__ == "__main__":
    unittest.main()
