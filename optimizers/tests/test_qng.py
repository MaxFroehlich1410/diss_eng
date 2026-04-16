"""Tests for the Quantum Natural Gradient (QNG) optimizer.

Covers:
- Correctness of state-derivative computation (finite-difference check)
- Fubini-Study metric tensor properties (symmetry, positive semi-definiteness)
- Metric tensor agreement with finite-difference approximation
- Regularisation and diagonal approximation behaviour
- QNG training loop integration for VQCModel and alternative models
- Dispatch through ``run_optimizer``
- Cost-accounting counters
"""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

import numpy as np


from experiments.two_moons_common.config import DEFAULT_CONFIG
from datasets import generate_two_moons
from qml_models import VQCModel
from qml_models.variants import (
    ChenSUNVQCModel,
    ProjectedTrainableModel,
    SimonettiHybridModel,
    SouzaSQQNNModel,
)
from optimizers import (
    _compute_metric_tensor,
    _compute_state_derivatives,
    run_optimizer,
    train_qng,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fd_state_derivative(model, params, x, pidx, eps=1e-7):
    """Finite-difference approximation of d|psi>/d(theta_pidx)."""
    p_plus = params.copy()
    p_minus = params.copy()
    p_plus[pidx] += eps
    p_minus[pidx] -= eps
    _, s_plus = model.get_gate_sequence_and_states(p_plus, x)
    _, s_minus = model.get_gate_sequence_and_states(p_minus, x)
    return (s_plus[-1] - s_minus[-1]) / (2.0 * eps)


def _fd_metric_tensor(model, params, X, gate_indices, eps=1e-5):
    """Finite-difference Fubini-Study metric tensor for validation."""
    n = len(gate_indices)
    MT = np.zeros((n, n), dtype=float)

    for x in X:
        _, states_ref = model.get_gate_sequence_and_states(params, x)
        psi = states_ref[-1]

        dpsi_fd = []
        for pidx in gate_indices:
            dpsi_fd.append(_fd_state_derivative(model, params, x, pidx, eps))

        for i in range(n):
            for j in range(i, n):
                inner = np.real(dpsi_fd[i].conj() @ dpsi_fd[j])
                ov_i = psi.conj() @ dpsi_fd[i]
                ov_j = psi.conj() @ dpsi_fd[j]
                val = (inner - np.real(ov_i.conj() * ov_j)) / len(X)
                MT[i, j] += val
                if i != j:
                    MT[j, i] += val

    return MT


# ---------------------------------------------------------------------------
# State derivative tests
# ---------------------------------------------------------------------------

class TestComputeStateDerivatives(unittest.TestCase):
    """Verify analytical state derivatives against finite differences."""

    def _assert_state_derivatives(self, model, params, x, atol=1e-5):
        dpsi, final_state = _compute_state_derivatives(model, params, x)

        _, states_ref = model.get_gate_sequence_and_states(params, x)
        np.testing.assert_allclose(final_state, states_ref[-1], atol=1e-12)

        for pidx, analytic in dpsi.items():
            fd = _fd_state_derivative(model, params, x, pidx)
            np.testing.assert_allclose(
                analytic, fd, atol=atol,
                err_msg=f"State derivative mismatch for param {pidx}",
            )

    def test_vqc_hea_2q1l(self):
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=42)
        self._assert_state_derivatives(model, params, np.array([0.5, -0.3]))

    def test_vqc_hea_4q2l(self):
        model = VQCModel(n_qubits=4, n_layers=2, entangler="ring", observable="Z0Z1")
        params = model.init_params(seed=7)
        self._assert_state_derivatives(model, params, np.array([1.1, -0.6]))

    def test_simonetti_explicit_angle(self):
        model = SimonettiHybridModel(mode="explicit_angle")
        params = model.init_params(seed=7)
        self._assert_state_derivatives(model, params, np.array([0.2, -0.4]))

    def test_simonetti_hybrid(self):
        model = SimonettiHybridModel(mode="hybrid")
        params = model.init_params(seed=3)
        self._assert_state_derivatives(model, params, np.array([0.15, 0.35]))

    def test_souza_reduced(self):
        model = SouzaSQQNNModel(variant="reduced", n_neurons=3)
        params = model.init_params(seed=11)
        self._assert_state_derivatives(model, params, np.array([0.3, -0.2]))

    def test_chen_simple_z0(self):
        model = ChenSUNVQCModel(n_macro_layers=1, readout="simple_z0")
        params = model.init_params(seed=6)
        self._assert_state_derivatives(model, params, np.array([0.1, -0.2]))


# ---------------------------------------------------------------------------
# Metric tensor tests
# ---------------------------------------------------------------------------

class TestMetricTensor(unittest.TestCase):
    """Properties and numerical accuracy of the Fubini-Study metric tensor."""

    def setUp(self):
        self.model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        self.params = self.model.init_params(seed=10)
        self.X = np.array([[0.5, -0.3], [-0.2, 0.7]])
        self.gate_indices = np.arange(self.model.n_params, dtype=int)

    def test_symmetric(self):
        MT, _ = _compute_metric_tensor(
            self.model, self.params, self.X, self.gate_indices,
        )
        np.testing.assert_allclose(MT, MT.T, atol=1e-12)

    def test_positive_semidefinite(self):
        MT, _ = _compute_metric_tensor(
            self.model, self.params, self.X, self.gate_indices, lam=0.0,
        )
        eigenvalues = np.linalg.eigvalsh(MT)
        self.assertTrue(np.all(eigenvalues >= -1e-10),
                        f"Negative eigenvalues: {eigenvalues[eigenvalues < -1e-10]}")

    def test_regularisation_adds_lambda_identity(self):
        lam = 0.05
        MT_unreg, _ = _compute_metric_tensor(
            self.model, self.params, self.X, self.gate_indices, lam=0.0,
        )
        MT_reg, _ = _compute_metric_tensor(
            self.model, self.params, self.X, self.gate_indices, lam=lam,
        )
        expected = MT_unreg + lam * np.eye(len(self.gate_indices))
        np.testing.assert_allclose(MT_reg, expected, atol=1e-12)

    def test_diagonal_approx_matches_full_diagonal(self):
        MT_full, _ = _compute_metric_tensor(
            self.model, self.params, self.X, self.gate_indices, approx=None,
        )
        MT_diag, _ = _compute_metric_tensor(
            self.model, self.params, self.X, self.gate_indices, approx="diag",
        )
        np.testing.assert_allclose(
            np.diag(MT_diag), np.diag(MT_full), atol=1e-12,
        )
        offdiag = MT_diag.copy()
        np.fill_diagonal(offdiag, 0.0)
        self.assertTrue(np.allclose(offdiag, 0.0))

    def test_matches_finite_difference(self):
        MT_analytic, _ = _compute_metric_tensor(
            self.model, self.params, self.X, self.gate_indices, lam=0.0,
        )
        MT_numeric = _fd_metric_tensor(
            self.model, self.params, self.X, self.gate_indices,
        )
        np.testing.assert_allclose(MT_analytic, MT_numeric, atol=1e-5)

    def test_matches_finite_difference_4q2l(self):
        model = VQCModel(n_qubits=4, n_layers=2, entangler="ring", observable="Z0Z1")
        params = model.init_params(seed=20)
        X = np.array([[0.3, -0.1], [0.6, 0.2]])
        gi = np.arange(model.n_params, dtype=int)

        MT_a, _ = _compute_metric_tensor(model, params, X, gi, lam=0.0)
        MT_n = _fd_metric_tensor(model, params, X, gi)
        np.testing.assert_allclose(MT_a, MT_n, atol=1e-4)

    def test_matches_finite_difference_simonetti(self):
        model = SimonettiHybridModel(mode="explicit_angle")
        params = model.init_params(seed=5)
        X = np.array([[0.2, -0.4], [-0.1, 0.3]])
        gi = np.asarray(model.gate_parameter_indices(), dtype=int)

        MT_a, _ = _compute_metric_tensor(model, params, X, gi, lam=0.0)
        MT_n = _fd_metric_tensor(model, params, X, gi)
        np.testing.assert_allclose(MT_a, MT_n, atol=1e-4)

    def test_cost_accounting(self):
        _, stats = _compute_metric_tensor(
            self.model, self.params, self.X, self.gate_indices,
        )
        self.assertEqual(stats["sample_forward_passes"], len(self.X))
        self.assertEqual(stats["sample_backward_passes"], len(self.X))
        self.assertEqual(stats["full_loss_evaluations"], 0)
        self.assertEqual(stats["gradient_evaluations"], 0)

    def test_empty_gate_indices(self):
        MT, stats = _compute_metric_tensor(
            self.model, self.params, self.X, np.array([], dtype=int),
        )
        self.assertEqual(MT.shape, (0, 0))
        self.assertEqual(stats["sample_forward_passes"], 0)


# ---------------------------------------------------------------------------
# QNG training loop tests
# ---------------------------------------------------------------------------

class TestTrainQNG(unittest.TestCase):
    """Smoke and integration tests for the full QNG training loop."""

    def setUp(self):
        self.dataset = generate_two_moons(
            n_samples=60, noise=0.15, test_fraction=0.3, seed=0,
            encoding="tanh_0_pi",
        )
        self.config = replace(
            DEFAULT_CONFIG,
            max_iterations=3,
            qng_lr=0.05,
            qng_lam=0.01,
            early_stopping_enabled=False,
            run_krotov_batch_sweep=False,
            run_krotov_hybrid_sweep=False,
        )

    def _assert_valid_trace(self, trace, n_steps):
        self.assertEqual(len(trace["loss"]), n_steps + 1)
        self.assertTrue(np.isfinite(trace["loss"]).all())
        self.assertTrue(np.isfinite(trace["train_acc"]).all())
        self.assertTrue(np.isfinite(trace["test_acc"]).all())
        self.assertEqual(trace["step"][0], 0)
        self.assertEqual(trace["step"][-1], n_steps)
        self.assertGreater(trace["cost_units"][-1], 0)

    def test_vqc_model_full_metric(self):
        X_train, X_test, y_train, y_test = self.dataset
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=1)
        final, trace = train_qng(
            model, params, X_train, y_train, X_test, y_test,
            max_iterations=3, lr=0.05, lam=0.01,
        )
        self.assertTrue(np.isfinite(final).all())
        self._assert_valid_trace(trace, 3)

    def test_vqc_model_diagonal_approx(self):
        X_train, X_test, y_train, y_test = self.dataset
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=2)
        final, trace = train_qng(
            model, params, X_train, y_train, X_test, y_test,
            max_iterations=3, lr=0.05, lam=0.01, approx="diag",
        )
        self.assertTrue(np.isfinite(final).all())
        self._assert_valid_trace(trace, 3)

    def test_run_optimizer_dispatch(self):
        X_train, X_test, y_train, y_test = self.dataset
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=3)
        final, trace = run_optimizer(
            "qng", model, params, X_train, y_train, X_test, y_test,
            self.config,
        )
        self.assertTrue(np.isfinite(final).all())
        self._assert_valid_trace(trace, 3)

    def test_projected_simonetti(self):
        X_train, X_test, y_train, y_test = self.dataset
        base = SimonettiHybridModel(mode="hybrid")
        full_init = base.init_params(seed=4)
        model = ProjectedTrainableModel(
            base, full_reference_params=full_init,
            trainable_indices=np.asarray(base.gate_parameter_indices(), dtype=int),
        )
        params = full_init[model.trainable_indices].copy()
        final, trace = train_qng(
            model, params, X_train, y_train, X_test, y_test,
            max_iterations=3, lr=0.02, lam=0.01,
        )
        self.assertTrue(np.isfinite(final).all())
        self._assert_valid_trace(trace, 3)

    def test_full_hybrid_model_updates_both_param_types(self):
        X_train, X_test, y_train, y_test = self.dataset
        model = SimonettiHybridModel(mode="hybrid")
        params = model.init_params(seed=5)
        final, trace = train_qng(
            model, params, X_train, y_train, X_test, y_test,
            max_iterations=3, lr=0.02, lam=0.01,
        )
        gate_idx = np.asarray(model.gate_parameter_indices(), dtype=int)
        nongate_idx = np.asarray(model.nongate_parameter_indices(), dtype=int)
        self.assertFalse(np.allclose(final[gate_idx], params[gate_idx]))
        self.assertFalse(np.allclose(final[nongate_idx], params[nongate_idx]))
        self._assert_valid_trace(trace, 3)

    def test_chen_sun_vqc(self):
        X_train, X_test, y_train, y_test = self.dataset
        base = ChenSUNVQCModel(n_macro_layers=1, readout="simple_z0")
        full_init = base.init_params(seed=6)
        model = ProjectedTrainableModel(
            base, full_reference_params=full_init,
            trainable_indices=np.asarray(base.gate_parameter_indices(), dtype=int),
        )
        params = full_init[model.trainable_indices].copy()
        final, trace = train_qng(
            model, params, X_train, y_train, X_test, y_test,
            max_iterations=3, lr=0.01, lam=0.01,
        )
        self.assertTrue(np.isfinite(final).all())
        self._assert_valid_trace(trace, 3)

    def test_souza_sqqnn(self):
        X_train, X_test, y_train, y_test = self.dataset
        model = SouzaSQQNNModel(variant="reduced", n_neurons=3)
        params = model.init_params(seed=8)
        final, trace = train_qng(
            model, params, X_train, y_train, X_test, y_test,
            max_iterations=3, lr=0.03, lam=0.01,
        )
        self.assertTrue(np.isfinite(final).all())
        self._assert_valid_trace(trace, 3)

    def test_large_regularisation_approaches_vanilla_gd(self):
        """With overwhelming regularisation the natural gradient reduces to
        a rescaled vanilla gradient step."""
        X_train, X_test, y_train, y_test = self.dataset
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=6)

        grad, _ = model.loss_gradient(params, X_train, y_train)
        lr = 0.05
        lam = 1e4

        final_qng, _ = train_qng(
            model, params.copy(), X_train, y_train, X_test, y_test,
            max_iterations=1, lr=lr, lam=lam,
        )
        expected = params - (lr / lam) * grad
        np.testing.assert_allclose(final_qng, expected, atol=1e-5)

    def test_cost_counters_increase(self):
        X_train, X_test, y_train, y_test = self.dataset
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=9)
        _, trace = train_qng(
            model, params, X_train, y_train, X_test, y_test,
            max_iterations=2, lr=0.05, lam=0.01,
        )
        costs = trace["cost_units"]
        self.assertGreater(costs[1], costs[0])
        self.assertGreater(costs[2], costs[1])
        self.assertGreater(trace["gradient_evaluations"][-1], 0)


if __name__ == "__main__":
    unittest.main()
