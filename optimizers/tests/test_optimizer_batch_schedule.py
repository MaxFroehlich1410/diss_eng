"""Tests for mini-batch to full-batch schedules in Adam and QNG."""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

import numpy as np


from experiments.two_moons_common.config import DEFAULT_CONFIG
from datasets import generate_two_moons
from qml_models import VQCModel
from optimizers.runner import run_optimizer


class OptimizerBatchScheduleTests(unittest.TestCase):
    def setUp(self):
        X_train, X_test, y_train, y_test = generate_two_moons(
            n_samples=60,
            noise=0.15,
            test_fraction=0.3,
            seed=0,
            encoding="tanh_0_pi",
        )
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = VQCModel(
            n_qubits=2,
            n_layers=1,
            entangler="ring",
            observable="Z0",
        )
        self.params = self.model.init_params(seed=7)

    def _make_config(self, **kwargs):
        return replace(
            DEFAULT_CONFIG,
            max_iterations=2,
            adam_lr=0.03,
            qng_lr=0.05,
            qng_lam=0.01,
            early_stopping_enabled=False,
            run_krotov_batch_sweep=False,
            run_krotov_hybrid_sweep=False,
            **kwargs,
        )

    def _run(self, optimizer_name, config):
        return run_optimizer(
            optimizer_name,
            self.model,
            self.params.copy(),
            self.X_train,
            self.y_train,
            self.X_test,
            self.y_test,
            config,
        )

    def test_adam_can_switch_from_minibatch_to_full_batch(self):
        full_config = self._make_config()
        scheduled_config = self._make_config(adam_batch_size=8, adam_switch_iteration=1)

        final_full, trace_full = self._run("adam", full_config)
        final_scheduled, trace_scheduled = self._run("adam", scheduled_config)

        self.assertTrue(np.isfinite(final_full).all())
        self.assertTrue(np.isfinite(final_scheduled).all())
        self.assertEqual(trace_scheduled["phase"][1], "adam_minibatch")
        self.assertEqual(trace_scheduled["phase"][2], "adam")

        full_step_cost = trace_full["cost_units"][1] - trace_full["cost_units"][0]
        scheduled_step_cost = trace_scheduled["cost_units"][1] - trace_scheduled["cost_units"][0]
        self.assertLess(scheduled_step_cost, full_step_cost)

    def test_qng_can_switch_from_minibatch_to_full_batch(self):
        full_config = self._make_config()
        scheduled_config = self._make_config(qng_batch_size=8, qng_switch_iteration=1)

        final_full, trace_full = self._run("qng", full_config)
        final_scheduled, trace_scheduled = self._run("qng", scheduled_config)

        self.assertTrue(np.isfinite(final_full).all())
        self.assertTrue(np.isfinite(final_scheduled).all())
        self.assertEqual(trace_scheduled["phase"][1], "qng_minibatch")
        self.assertEqual(trace_scheduled["phase"][2], "qng")

        full_step_cost = trace_full["cost_units"][1] - trace_full["cost_units"][0]
        scheduled_step_cost = trace_scheduled["cost_units"][1] - trace_scheduled["cost_units"][0]
        self.assertLess(scheduled_step_cost, full_step_cost)


if __name__ == "__main__":
    unittest.main()
