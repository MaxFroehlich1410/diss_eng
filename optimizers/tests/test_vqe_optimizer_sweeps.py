"""Smoke and utility tests for the exact-statevector VQE optimizer sweeps."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


from qml_models.vqe import Hubbard1x2HVVQEProblem
from optimizers._vqe_impl import (
    compute_metric_tensor,
    compute_state_derivatives,
    exact_energy_and_gradient,
    train_adam,
    train_bfgs,
    train_krotov_hybrid,
    train_qng,
)


def _fd_state_derivative(problem: Hubbard1x2HVVQEProblem, theta: np.ndarray, param_idx: int, eps: float = 1e-7) -> np.ndarray:
    theta_plus = theta.copy()
    theta_minus = theta.copy()
    theta_plus[param_idx] += eps
    theta_minus[param_idx] -= eps
    psi_plus = problem.apply_ansatz(theta_plus)
    psi_minus = problem.apply_ansatz(theta_minus)
    return (psi_plus - psi_minus) / (2.0 * eps)


class TestVQEOptimizerUtilities(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = Hubbard1x2HVVQEProblem(U=4.0)
        rng = np.random.default_rng(5)
        self.theta = rng.uniform(-0.4, 0.4, size=self.problem.n_params)

    def test_state_derivatives_match_finite_differences(self) -> None:
        dpsi, final_state = compute_state_derivatives(self.problem, self.theta)
        np.testing.assert_allclose(final_state, self.problem.apply_ansatz(self.theta), atol=1e-12)
        for param_idx, analytic in dpsi.items():
            numeric = _fd_state_derivative(self.problem, self.theta, param_idx)
            np.testing.assert_allclose(analytic, numeric, atol=1e-5)

    def test_metric_tensor_is_symmetric_and_positive_semidefinite(self) -> None:
        metric, _ = compute_metric_tensor(self.problem, self.theta, lam=0.0)
        np.testing.assert_allclose(metric, metric.T, atol=1e-12)
        eigenvalues = np.linalg.eigvalsh(metric)
        self.assertTrue(np.all(eigenvalues >= -1e-10))

    def test_energy_gradient_is_finite(self) -> None:
        energy, grad, stats = exact_energy_and_gradient(self.problem, self.theta)
        self.assertTrue(np.isfinite(energy))
        self.assertTrue(np.all(np.isfinite(grad)))
        self.assertEqual(stats["state_forward_passes"], 1)
        self.assertEqual(stats["state_backward_passes"], 1)


class TestVQEOptimizerSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.problem = Hubbard1x2HVVQEProblem(U=4.0)
        rng = np.random.default_rng(11)
        self.theta0 = rng.uniform(-0.2, 0.2, size=self.problem.n_params)
        self.exact_ground_energy = self.problem.exact_ground_energy()

    def _assert_valid_trace(self, trace: dict[str, list[float | int | str]]) -> None:
        self.assertGreaterEqual(len(trace["energy"]), 2)
        self.assertEqual(trace["step"][0], 0)
        self.assertTrue(np.isfinite(np.asarray(trace["energy"], dtype=float)).all())
        self.assertTrue(np.isfinite(np.asarray(trace["energy_error"], dtype=float)).all())
        self.assertGreater(trace["cost_units"][-1], 0)

    def _assert_variational_bound(self, theta: np.ndarray) -> None:
        energy = self.problem.energy(theta)
        self.assertGreaterEqual(energy, self.exact_ground_energy - 1e-8)

    def test_adam_smoke(self) -> None:
        theta, trace = train_adam(self.problem, self.theta0, max_iterations=2, lr=0.05)
        self._assert_valid_trace(trace)
        self._assert_variational_bound(theta)

    def test_bfgs_smoke(self) -> None:
        theta, trace = train_bfgs(self.problem, self.theta0, max_iterations=2, gtol=1e-7)
        self._assert_valid_trace(trace)
        self._assert_variational_bound(theta)

    def test_qng_smoke(self) -> None:
        theta, trace = train_qng(self.problem, self.theta0, max_iterations=2, lr=0.05, lam=1e-3)
        self._assert_valid_trace(trace)
        self._assert_variational_bound(theta)

    def test_hybrid_krotov_smoke(self) -> None:
        theta, trace = train_krotov_hybrid(
            self.problem,
            self.theta0,
            max_iterations=3,
            switch_iteration=1,
            online_step_size=0.05,
            batch_step_size=0.1,
        )
        self._assert_valid_trace(trace)
        self._assert_variational_bound(theta)


if __name__ == "__main__":
    unittest.main()
