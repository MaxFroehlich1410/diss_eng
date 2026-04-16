"""Unit tests for the alternative two-moons QML models."""

from __future__ import annotations

import unittest

import numpy as np

from qml_models.variants import (
    ChenSUNVQCModel,
    SimonettiHybridModel,
    SouzaSQQNNModel,
)


def finite_difference_gradient(model, params, X, y, eps=1e-6, indices=None):
    params = np.asarray(params, dtype=float)
    test_indices = np.arange(len(params)) if indices is None else np.asarray(indices, dtype=int)
    approx = np.zeros(len(test_indices), dtype=float)
    for out_idx, param_idx in enumerate(test_indices):
        plus = params.copy()
        minus = params.copy()
        plus[param_idx] += eps
        minus[param_idx] -= eps
        approx[out_idx] = (model.loss(plus, X, y) - model.loss(minus, X, y)) / (2.0 * eps)
    return test_indices, approx


def assert_unitary(testcase, matrix, atol=1e-8):
    ident = np.eye(matrix.shape[0], dtype=complex)
    testcase.assertTrue(np.allclose(matrix.conj().T @ matrix, ident, atol=atol))


class SimonettiHybridModelTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.2, -0.4], [-0.1, 0.3]], dtype=float)
        self.y = np.array([1.0, 0.0], dtype=float)

    def test_hybrid_public_api_and_metadata(self):
        model = SimonettiHybridModel(mode="hybrid")
        params_a = model.init_params(seed=7)
        params_b = model.get_initial_params(seed=7)

        self.assertEqual(model.n_params, 51)
        self.assertTrue(np.allclose(params_a, params_b))

        metadata = model.parameter_metadata()
        self.assertEqual(len(metadata), model.n_params)
        self.assertEqual(sum(item["supports_gate_derivative"] for item in metadata), 48)

        prob = model.forward(params_a, self.X[0])
        self.assertGreater(prob, 0.0)
        self.assertLess(prob, 1.0)
        self.assertEqual(model.predict(params_a, self.X[0]) in (0, 1), True)
        self.assertEqual(model.forward_batch(params_a, self.X).shape, (2,))
        self.assertGreaterEqual(model.accuracy(params_a, self.X, self.y), 0.0)
        self.assertGreaterEqual(model.loss(params_a, self.X, self.y), 0.0)

    def test_hybrid_gate_sequence_and_gate_derivative(self):
        model = SimonettiHybridModel(mode="hybrid")
        params = model.init_params(seed=3)
        gates, states = model.get_gate_sequence_and_states(params, self.X[0])

        self.assertEqual(len(gates), 52)
        self.assertEqual(len(states), 53)
        self.assertTrue(np.allclose(np.linalg.norm(states[-1]), 1.0, atol=1e-8))

        gate = model.rebuild_param_gate(0, params, self.X[0])
        assert_unitary(self, gate)

        eps = 1e-7
        plus = params.copy()
        minus = params.copy()
        plus[0] += eps
        minus[0] -= eps
        fd = (model.rebuild_param_gate(0, plus, self.X[0]) - model.rebuild_param_gate(0, minus, self.X[0])) / (
            2.0 * eps
        )
        analytic = model.gate_derivative_generator(0, self.X[0]) @ gate
        self.assertTrue(np.allclose(fd, analytic, atol=1e-6))

        with self.assertRaises(ValueError):
            model.rebuild_param_gate(model.n_params - 1, params, self.X[0])

    def test_hybrid_loss_gradient_matches_finite_difference(self):
        model = SimonettiHybridModel(mode="hybrid")
        params = model.init_params(seed=1)
        analytic, stats = model.loss_gradient(params, self.X, self.y)
        _, numeric = finite_difference_gradient(model, params, self.X, self.y)

        self.assertEqual(stats["sample_forward_passes"], len(self.X))
        self.assertEqual(stats["sample_backward_passes"], len(self.X))
        self.assertTrue(np.allclose(analytic, numeric, atol=3e-5, rtol=2e-4))
        alias, _ = model.param_shift_gradient(params, self.X, self.y)
        self.assertTrue(np.allclose(alias, analytic))

    def test_hybrid_nongate_gradient_matches_full_gradient_slice(self):
        model = SimonettiHybridModel(mode="hybrid")
        params = model.init_params(seed=12)
        full_grad, _ = model.loss_gradient(params, self.X, self.y)
        nongate_grad, stats = model.nongate_loss_gradient(params, self.X, self.y)
        nongate_idx = model.nongate_parameter_indices()

        self.assertEqual(stats["sample_forward_passes"], len(self.X))
        self.assertEqual(stats["sample_backward_passes"], 0)
        self.assertTrue(np.allclose(nongate_grad[nongate_idx], full_grad[nongate_idx]))
        gate_idx = model.gate_parameter_indices()
        self.assertTrue(np.allclose(nongate_grad[gate_idx], 0.0))

    def test_explicit_angle_mode_behaves_and_has_gate_support(self):
        model = SimonettiHybridModel(mode="explicit_angle")
        params = model.init_params(seed=11)
        metadata = model.parameter_metadata()

        self.assertEqual(model.n_params, 19)
        self.assertEqual(sum(item["supports_gate_derivative"] for item in metadata), 16)

        gates, states = model.get_gate_sequence_and_states(params, self.X[0])
        self.assertEqual(len(gates), 36)
        self.assertEqual(len(states), 37)
        self.assertGreater(model.forward(params, self.X[0]), 0.0)

        analytic, _ = model.loss_gradient(params, self.X, self.y)
        _, numeric = finite_difference_gradient(model, params, self.X, self.y)
        self.assertTrue(np.allclose(analytic, numeric, atol=3e-5, rtol=2e-4))


class SouzaSQQNNModelTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.15, -0.25], [0.4, 0.1]], dtype=float)
        self.y = np.array([1.0, 0.0], dtype=float)

    def test_reduced_variant_public_api(self):
        model = SouzaSQQNNModel(variant="reduced", n_neurons=4)
        params = model.init_params(seed=5)

        self.assertEqual(model.n_params, 24)
        self.assertEqual(model.parameter_metadata()[0]["group"], "beta")
        self.assertTrue(np.allclose(model.polynomial_features(self.X[0]), [1.0, 0.15, -0.25, 0.0225, -0.0375, 0.0625]))
        self.assertEqual(model.predict(params, self.X).shape, (2,))
        self.assertGreater(model.forward(params, self.X[0]), 0.0)
        self.assertGreaterEqual(model.loss(params, self.X, self.y), 0.0)

    def test_reduced_gate_sequence_and_derivative(self):
        model = SouzaSQQNNModel(variant="reduced", n_neurons=3)
        params = model.init_params(seed=2)
        gates, states = model.get_gate_sequence_and_states(params, self.X[0])

        self.assertEqual(len(gates), model.n_params)
        self.assertEqual(len(states), model.n_params + 1)
        self.assertTrue(np.allclose(np.linalg.norm(states[-1]), 1.0, atol=1e-8))

        gate = model.rebuild_param_gate(0, params, self.X[0])
        assert_unitary(self, gate)

        eps = 1e-7
        plus = params.copy()
        minus = params.copy()
        plus[0] += eps
        minus[0] -= eps
        fd = (model.rebuild_param_gate(0, plus, self.X[0]) - model.rebuild_param_gate(0, minus, self.X[0])) / (
            2.0 * eps
        )
        analytic = model.gate_derivative_generator(0, self.X[0]) @ gate
        self.assertTrue(np.allclose(fd, analytic, atol=1e-6))

    def test_reduced_gradient_matches_finite_difference(self):
        model = SouzaSQQNNModel(variant="reduced", n_neurons=4)
        params = model.init_params(seed=9)
        analytic, stats = model.loss_gradient(params, self.X, self.y)
        _, numeric = finite_difference_gradient(model, params, self.X, self.y)

        self.assertEqual(stats["sample_forward_passes"], len(self.X))
        self.assertTrue(np.allclose(analytic, numeric, atol=2e-5, rtol=2e-4))

    def test_full_variant_public_api_and_gradient(self):
        model = SouzaSQQNNModel(variant="full", n_neurons=2)
        params = model.init_params(seed=4)

        self.assertEqual(model.n_params, 36)
        metadata = model.parameter_metadata()
        self.assertEqual({item["group"] for item in metadata}, {"alpha", "beta", "gamma"})

        analytic, _ = model.loss_gradient(params, self.X, self.y)
        _, numeric = finite_difference_gradient(model, params, self.X, self.y)
        self.assertTrue(np.allclose(analytic, numeric, atol=2e-5, rtol=2e-4))


class ChenSUNVQCModelTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.1, -0.2], [0.35, 0.25]], dtype=float)
        self.y = np.array([1.0, 0.0], dtype=float)

    def test_simple_readout_public_api_and_gate_access(self):
        model = ChenSUNVQCModel(n_macro_layers=1, readout="simple_z0")
        params = model.init_params(seed=6)
        metadata = model.parameter_metadata()

        self.assertEqual(model.n_params, 45)
        self.assertEqual(len(metadata), 45)
        self.assertTrue(all(item["supports_gate_derivative"] for item in metadata))

        gates, states = model.get_gate_sequence_and_states(params, self.X[0])
        self.assertEqual(len(gates), 49)
        self.assertEqual(len(states), 50)
        self.assertTrue(np.allclose(np.linalg.norm(states[-1]), 1.0, atol=1e-8))

        gate = model.rebuild_param_gate(0, params, self.X[0])
        assert_unitary(self, gate, atol=1e-7)

        eps = 1e-7
        plus = params.copy()
        minus = params.copy()
        plus[0] += eps
        minus[0] -= eps
        fd = (model.rebuild_param_gate(0, plus, self.X[0]) - model.rebuild_param_gate(0, minus, self.X[0])) / (
            2.0 * eps
        )
        analytic = model.gate_derivative_generator(0, self.X[0]) @ gate
        self.assertTrue(np.allclose(fd, analytic, atol=2e-6))

    def test_simple_readout_gradient_matches_finite_difference(self):
        model = ChenSUNVQCModel(n_macro_layers=1, readout="simple_z0")
        params = model.init_params(seed=8)
        analytic, _ = model.loss_gradient(params, self.X, self.y)
        _, numeric = finite_difference_gradient(model, params, self.X, self.y)
        self.assertTrue(np.allclose(analytic, numeric, atol=4e-5, rtol=3e-4))

    def test_hybrid_readout_gradient_matches_finite_difference_and_flags_classical_params(self):
        model = ChenSUNVQCModel(n_macro_layers=1, readout="hybrid_linear")
        params = model.init_params(seed=10)
        metadata = model.parameter_metadata()

        self.assertEqual(model.n_params, 50)
        self.assertEqual(sum(item["supports_gate_derivative"] for item in metadata), 45)

        analytic, _ = model.loss_gradient(params, self.X, self.y)
        _, numeric = finite_difference_gradient(model, params, self.X, self.y)
        self.assertTrue(np.allclose(analytic, numeric, atol=5e-5, rtol=4e-4))

        with self.assertRaises(ValueError):
            model.rebuild_param_gate(model.n_params - 1, params, self.X[0])

    def test_hybrid_readout_nongate_gradient_matches_full_gradient_slice(self):
        model = ChenSUNVQCModel(n_macro_layers=1, readout="hybrid_linear")
        params = model.init_params(seed=13)
        full_grad, _ = model.loss_gradient(params, self.X, self.y)
        nongate_grad, stats = model.nongate_loss_gradient(params, self.X, self.y)
        nongate_idx = model.nongate_parameter_indices()

        self.assertEqual(stats["sample_forward_passes"], len(self.X))
        self.assertEqual(stats["sample_backward_passes"], 0)
        self.assertTrue(np.allclose(nongate_grad[nongate_idx], full_grad[nongate_idx]))
        gate_idx = model.gate_parameter_indices()
        self.assertTrue(np.allclose(nongate_grad[gate_idx], 0.0))


if __name__ == "__main__":
    unittest.main()
