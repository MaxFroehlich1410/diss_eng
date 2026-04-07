"""Optimizer smoke tests for the alternative QML models."""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

import numpy as np

SCRIPT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from experiments.two_moons_qml_krotov.config import DEFAULT_CONFIG
from experiments.two_moons_qml_krotov.dataset import generate_two_moons
from experiments.two_moons_qml_krotov.models import (
    ChenSUNVQCModel,
    ProjectedTrainableModel,
    SimonettiHybridModel,
    SouzaSQQNNModel,
)
from experiments.two_moons_qml_krotov.optimizers import run_optimizer


def make_projected_model(base_model, seed):
    full_init = np.asarray(base_model.init_params(seed=seed), dtype=float)
    projected = ProjectedTrainableModel(
        base_model,
        full_reference_params=full_init,
        trainable_indices=np.asarray(base_model.gate_parameter_indices(), dtype=int),
    )
    return projected, full_init[projected.trainable_indices].copy()


class AlternativeQMLTrainingTests(unittest.TestCase):
    def setUp(self):
        self.dataset = generate_two_moons(
            n_samples=80,
            noise=0.15,
            test_fraction=0.3,
            seed=0,
            encoding="linear_pm_pi",
        )
        self.optimizers = ["krotov_hybrid", "adam", "lbfgs"]
        self.config = replace(
            DEFAULT_CONFIG,
            max_iterations=3,
            lbfgs_maxiter=3,
            adam_lr=0.03,
            hybrid_switch_iteration=1,
            hybrid_online_step_size=0.02,
            hybrid_batch_step_size=0.04,
            hybrid_online_schedule="constant",
            hybrid_batch_schedule="constant",
            early_stopping_enabled=False,
            run_krotov_batch_sweep=False,
            run_krotov_hybrid_sweep=False,
        )

    def _assert_training_run(self, model, init_params):
        X_train, X_test, y_train, y_test = self.dataset
        for optimizer_name in self.optimizers:
            final_params, trace = run_optimizer(
                optimizer_name,
                model,
                init_params.copy(),
                X_train,
                y_train,
                X_test,
                y_test,
                self.config,
            )
            self.assertTrue(np.isfinite(final_params).all())
            self.assertGreaterEqual(len(trace["loss"]), 2)
            self.assertTrue(np.isfinite(trace["loss"]).all())
            self.assertTrue(np.isfinite(trace["train_acc"]).all())
            self.assertTrue(np.isfinite(trace["test_acc"]).all())
            self.assertEqual(trace["step"][0], 0)
            self.assertGreater(trace["step"][-1], 0)
            self.assertGreaterEqual(trace["cost_units"][-1], 0)

    def test_simonetti_projected_training(self):
        model, init_params = make_projected_model(SimonettiHybridModel(mode="hybrid"), seed=1)
        self._assert_training_run(model, init_params)

    def test_souza_training(self):
        model, init_params = make_projected_model(SouzaSQQNNModel(variant="reduced", n_neurons=4), seed=2)
        self._assert_training_run(model, init_params)

    def test_chen_training(self):
        model, init_params = make_projected_model(ChenSUNVQCModel(n_macro_layers=2, readout="simple_z0"), seed=3)
        self._assert_training_run(model, init_params)

    def test_full_simonetti_hybrid_krotov_updates_classical_head(self):
        X_train, X_test, y_train, y_test = self.dataset
        model = SimonettiHybridModel(mode="hybrid")
        init_params = model.init_params(seed=4)
        final_params, trace = run_optimizer(
            "krotov_hybrid",
            model,
            init_params.copy(),
            X_train,
            y_train,
            X_test,
            y_test,
            self.config,
        )
        nongate_idx = np.asarray(model.nongate_parameter_indices(), dtype=int)
        self.assertGreater(len(nongate_idx), 0)
        self.assertFalse(np.allclose(final_params[nongate_idx], init_params[nongate_idx]))
        self.assertTrue(np.isfinite(trace["loss"]).all())

    def test_full_chen_hybrid_readout_krotov_updates_classical_head(self):
        X_train, X_test, y_train, y_test = self.dataset
        model = ChenSUNVQCModel(n_macro_layers=2, readout="hybrid_linear")
        init_params = model.init_params(seed=5)
        final_params, trace = run_optimizer(
            "krotov_hybrid",
            model,
            init_params.copy(),
            X_train,
            y_train,
            X_test,
            y_test,
            self.config,
        )
        nongate_idx = np.asarray(model.nongate_parameter_indices(), dtype=int)
        self.assertGreater(len(nongate_idx), 0)
        self.assertFalse(np.allclose(final_params[nongate_idx], init_params[nongate_idx]))
        self.assertTrue(np.isfinite(trace["loss"]).all())


if __name__ == "__main__":
    unittest.main()
