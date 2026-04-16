"""Tests for the generic Krotov scaling infrastructure."""

from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace

import numpy as np


from optimizers.krotov import (
    AdaptiveClipScaling,
    AdaptiveSmoothScaling,
    GroupwiseAdaptiveScaling,
    GroupwiseScaling,
    LayerwiseScaling,
    NoScaling,
    build_scaling_strategy,
    get_gate_metadata,
)
from experiments.two_moons_common.config import DEFAULT_CONFIG
from datasets import generate_two_moons
from qml_models import VQCModel
from qml_models.variants import (
    ChenSUNVQCModel,
    SimonettiHybridModel,
    SouzaSQQNNModel,
)
from optimizers.runner import run_optimizer


# -----------------------------------------------------------------------
# Unit tests: individual scaling strategies
# -----------------------------------------------------------------------

class TestNoScaling(unittest.TestCase):
    def test_returns_ones(self):
        s = NoScaling()
        raw = np.array([0.1, -0.5, 0.0, 2.3])
        factors = s.compute_scaling_factors(raw, np.arange(4))
        np.testing.assert_array_equal(factors, np.ones(4))

    def test_empty_input(self):
        s = NoScaling()
        factors = s.compute_scaling_factors(np.array([]), np.array([], dtype=int))
        self.assertEqual(len(factors), 0)


class TestAdaptiveClipScaling(unittest.TestCase):
    def test_known_values(self):
        s = AdaptiveClipScaling(tau=0.5, eps=0.0)
        raw = np.array([0.1, 1.0, 2.0, 0.0])
        factors = s.compute_scaling_factors(raw, np.arange(4))
        # |0.1| -> min(1, 0.5/0.1) = 1.0
        # |1.0| -> min(1, 0.5/1.0) = 0.5
        # |2.0| -> min(1, 0.5/2.0) = 0.25
        # |0.0| -> min(1, inf)     = 1.0
        expected = np.array([1.0, 0.5, 0.25, 1.0])
        np.testing.assert_allclose(factors, expected)

    def test_all_small_updates_give_ones(self):
        s = AdaptiveClipScaling(tau=1.0, eps=1e-10)
        raw = np.array([0.01, 0.001, 0.0001])
        factors = s.compute_scaling_factors(raw, np.arange(3))
        np.testing.assert_allclose(factors, np.ones(3), atol=1e-6)

    def test_zero_updates_safe(self):
        s = AdaptiveClipScaling(tau=0.1, eps=1e-8)
        raw = np.zeros(5)
        factors = s.compute_scaling_factors(raw, np.arange(5))
        self.assertTrue(np.all(np.isfinite(factors)))
        # tau/(0+eps) is huge, so min(1, huge) = 1.0
        np.testing.assert_allclose(factors, np.ones(5))

    def test_negative_updates(self):
        s = AdaptiveClipScaling(tau=0.5, eps=0.0)
        raw = np.array([-1.0, -2.0])
        factors = s.compute_scaling_factors(raw, np.arange(2))
        expected = np.array([0.5, 0.25])
        np.testing.assert_allclose(factors, expected)

    def test_factors_at_most_one(self):
        s = AdaptiveClipScaling(tau=0.3, eps=1e-8)
        rng = np.random.RandomState(42)
        raw = rng.randn(50)
        factors = s.compute_scaling_factors(raw, np.arange(50))
        self.assertTrue(np.all(factors <= 1.0 + 1e-12))
        self.assertTrue(np.all(factors > 0.0))

    def test_invalid_tau(self):
        with self.assertRaises(ValueError):
            AdaptiveClipScaling(tau=-1.0)
        with self.assertRaises(ValueError):
            AdaptiveClipScaling(tau=0.0)

    def test_invalid_eps(self):
        with self.assertRaises(ValueError):
            AdaptiveClipScaling(eps=-1.0)


class TestAdaptiveSmoothScaling(unittest.TestCase):
    def test_known_values(self):
        s = AdaptiveSmoothScaling(beta=2.0)
        raw = np.array([0.0, 0.5, 1.0, 2.0])
        factors = s.compute_scaling_factors(raw, np.arange(4))
        expected = 1.0 / (1.0 + 2.0 * np.abs(raw))
        np.testing.assert_allclose(factors, expected)

    def test_zero_beta_gives_ones(self):
        s = AdaptiveSmoothScaling(beta=0.0)
        raw = np.array([1.0, 100.0, -50.0])
        factors = s.compute_scaling_factors(raw, np.arange(3))
        np.testing.assert_allclose(factors, np.ones(3))

    def test_zero_updates_give_ones(self):
        s = AdaptiveSmoothScaling(beta=5.0)
        raw = np.zeros(4)
        factors = s.compute_scaling_factors(raw, np.arange(4))
        np.testing.assert_allclose(factors, np.ones(4))

    def test_factors_positive_and_at_most_one(self):
        s = AdaptiveSmoothScaling(beta=1.0)
        rng = np.random.RandomState(7)
        raw = rng.randn(50)
        factors = s.compute_scaling_factors(raw, np.arange(50))
        self.assertTrue(np.all(factors > 0.0))
        self.assertTrue(np.all(factors <= 1.0 + 1e-12))

    def test_invalid_beta(self):
        with self.assertRaises(ValueError):
            AdaptiveSmoothScaling(beta=-1.0)


class TestLayerwiseScaling(unittest.TestCase):
    def _make_metadata(self, layers):
        return [{"index": i, "layer": l, "supports_gate_derivative": True}
                for i, l in enumerate(layers)]

    def test_exponential_decay(self):
        s = LayerwiseScaling(gamma=0.5)
        meta = self._make_metadata([0, 0, 1, 1, 2])
        factors = s.compute_scaling_factors(
            np.ones(5), np.arange(5), metadata=meta
        )
        expected = np.array([1.0, 1.0, 0.5, 0.5, 0.25])
        np.testing.assert_allclose(factors, expected)

    def test_explicit_mapping(self):
        s = LayerwiseScaling(layer_scales={0: 1.0, 1: 0.8, 2: 0.5})
        meta = self._make_metadata([0, 1, 2, 0])
        factors = s.compute_scaling_factors(
            np.ones(4), np.arange(4), metadata=meta
        )
        expected = np.array([1.0, 0.8, 0.5, 1.0])
        np.testing.assert_allclose(factors, expected)

    def test_missing_layer_in_explicit_mapping_fails(self):
        s = LayerwiseScaling(layer_scales={0: 1.0})
        meta = self._make_metadata([0, 1])
        with self.assertRaises(ValueError):
            s.compute_scaling_factors(np.ones(2), np.arange(2), metadata=meta)

    def test_no_metadata_fails_on_validate(self):
        s = LayerwiseScaling(gamma=0.9)
        with self.assertRaises(ValueError):
            s.validate(np.arange(3), metadata=None)

    def test_no_layer_field_in_metadata_fails(self):
        s = LayerwiseScaling(gamma=0.9)
        meta = [{"index": 0, "group": "a"}]
        with self.assertRaises(ValueError):
            s.validate(np.arange(1), metadata=meta)

    def test_custom_layer_field(self):
        s = LayerwiseScaling(gamma=0.8, layer_field="macro_layer")
        meta = [
            {"index": 0, "macro_layer": 0},
            {"index": 1, "macro_layer": 1},
        ]
        factors = s.compute_scaling_factors(
            np.ones(2), np.arange(2), metadata=meta
        )
        expected = np.array([1.0, 0.8])
        np.testing.assert_allclose(factors, expected)

    def test_custom_field_not_found_fails(self):
        s = LayerwiseScaling(gamma=0.9, layer_field="nonexistent")
        meta = [{"index": 0, "layer": 0}]
        with self.assertRaises(ValueError):
            s.validate(np.arange(1), metadata=meta)

    def test_both_gamma_and_layer_scales_fails(self):
        with self.assertRaises(ValueError):
            LayerwiseScaling(gamma=0.5, layer_scales={0: 1.0})

    def test_neither_gamma_nor_layer_scales_fails(self):
        with self.assertRaises(ValueError):
            LayerwiseScaling()

    def test_invalid_gamma(self):
        with self.assertRaises(ValueError):
            LayerwiseScaling(gamma=0.0)
        with self.assertRaises(ValueError):
            LayerwiseScaling(gamma=1.5)


class TestGroupwiseScaling(unittest.TestCase):
    def test_known_values(self):
        s = GroupwiseScaling(
            group_field="group",
            group_scales={"local": 0.7, "entangling": 1.0},
        )
        meta = [
            {"index": 0, "group": "local"},
            {"index": 1, "group": "entangling"},
            {"index": 2, "group": "local"},
        ]
        factors = s.compute_scaling_factors(
            np.ones(3), np.arange(3), metadata=meta
        )
        expected = np.array([0.7, 1.0, 0.7])
        np.testing.assert_allclose(factors, expected)

    def test_axis_based_scaling(self):
        s = GroupwiseScaling(
            group_field="axis",
            group_scales={"ry": 1.0, "rz": 0.8},
        )
        meta = [
            {"index": 0, "axis": "ry"},
            {"index": 1, "axis": "rz"},
            {"index": 2, "axis": "ry"},
            {"index": 3, "axis": "rz"},
        ]
        factors = s.compute_scaling_factors(
            np.ones(4), np.arange(4), metadata=meta
        )
        expected = np.array([1.0, 0.8, 1.0, 0.8])
        np.testing.assert_allclose(factors, expected)

    def test_missing_field_in_metadata_fails(self):
        s = GroupwiseScaling(group_field="role", group_scales={"a": 1.0})
        meta = [{"index": 0, "group": "a"}]
        with self.assertRaises(ValueError):
            s.validate(np.arange(1), metadata=meta)

    def test_missing_label_without_default_fails(self):
        s = GroupwiseScaling(group_field="group", group_scales={"a": 1.0})
        meta = [{"index": 0, "group": "b"}]
        with self.assertRaises(ValueError):
            s.compute_scaling_factors(np.ones(1), np.arange(1), metadata=meta)

    def test_missing_label_uses_default(self):
        s = GroupwiseScaling(
            group_field="group",
            group_scales={"a": 0.5},
            default_group_scale=1.0,
        )
        meta = [
            {"index": 0, "group": "a"},
            {"index": 1, "group": "unknown_group"},
        ]
        factors = s.compute_scaling_factors(
            np.ones(2), np.arange(2), metadata=meta
        )
        expected = np.array([0.5, 1.0])
        np.testing.assert_allclose(factors, expected)

    def test_no_metadata_fails(self):
        s = GroupwiseScaling(group_field="group", group_scales={"a": 1.0})
        with self.assertRaises(ValueError):
            s.validate(np.arange(1), metadata=None)

    def test_empty_group_scales_fails(self):
        with self.assertRaises(ValueError):
            GroupwiseScaling(group_field="group", group_scales={})

    def test_empty_group_field_fails(self):
        with self.assertRaises(ValueError):
            GroupwiseScaling(group_field="", group_scales={"a": 1.0})


class TestGroupwiseAdaptiveScaling(unittest.TestCase):
    def test_product_of_components_clip(self):
        groupwise = GroupwiseScaling(
            group_field="group",
            group_scales={"a": 0.5, "b": 1.0},
        )
        adaptive = AdaptiveClipScaling(tau=1.0, eps=0.0)
        combined = GroupwiseAdaptiveScaling(groupwise, adaptive)

        raw = np.array([0.5, 2.0, 4.0])
        meta = [
            {"index": 0, "group": "a"},
            {"index": 1, "group": "b"},
            {"index": 2, "group": "a"},
        ]
        gate_indices = np.arange(3)

        total = combined.compute_scaling_factors(raw, gate_indices, metadata=meta)
        group_only = groupwise.compute_scaling_factors(raw, gate_indices, metadata=meta)
        adaptive_only = adaptive.compute_scaling_factors(raw, gate_indices, metadata=meta)
        np.testing.assert_allclose(total, group_only * adaptive_only)

    def test_product_of_components_smooth(self):
        groupwise = GroupwiseScaling(
            group_field="group",
            group_scales={"x": 0.8},
            default_group_scale=1.0,
        )
        adaptive = AdaptiveSmoothScaling(beta=2.0)
        combined = GroupwiseAdaptiveScaling(groupwise, adaptive)

        raw = np.array([1.0, 0.5])
        meta = [
            {"index": 0, "group": "x"},
            {"index": 1, "group": "y"},
        ]
        total = combined.compute_scaling_factors(raw, np.arange(2), metadata=meta)
        expected_group = np.array([0.8, 1.0])
        expected_adaptive = 1.0 / (1.0 + 2.0 * np.abs(raw))
        np.testing.assert_allclose(total, expected_group * expected_adaptive)

    def test_invalid_groupwise_type_fails(self):
        with self.assertRaises(TypeError):
            GroupwiseAdaptiveScaling(NoScaling(), AdaptiveClipScaling())

    def test_invalid_adaptive_type_fails(self):
        g = GroupwiseScaling(group_field="g", group_scales={"a": 1.0})
        with self.assertRaises(TypeError):
            GroupwiseAdaptiveScaling(g, NoScaling())


# -----------------------------------------------------------------------
# Factory tests
# -----------------------------------------------------------------------

class TestBuildScalingStrategy(unittest.TestCase):
    def test_none_mode(self):
        s = build_scaling_strategy("none")
        self.assertIsInstance(s, NoScaling)

    def test_adaptive_clip_mode(self):
        s = build_scaling_strategy("adaptive_clip", {"tau": 0.2, "eps": 1e-6})
        self.assertIsInstance(s, AdaptiveClipScaling)
        self.assertAlmostEqual(s.tau, 0.2)
        self.assertAlmostEqual(s.eps, 1e-6)

    def test_adaptive_clip_defaults(self):
        s = build_scaling_strategy("adaptive_clip")
        self.assertIsInstance(s, AdaptiveClipScaling)
        self.assertAlmostEqual(s.tau, 0.1)

    def test_adaptive_smooth_mode(self):
        s = build_scaling_strategy("adaptive_smooth", {"beta": 3.0})
        self.assertIsInstance(s, AdaptiveSmoothScaling)
        self.assertAlmostEqual(s.beta, 3.0)

    def test_layerwise_mode(self):
        s = build_scaling_strategy("layerwise", {"gamma": 0.9})
        self.assertIsInstance(s, LayerwiseScaling)

    def test_groupwise_mode(self):
        s = build_scaling_strategy("groupwise", {
            "group_field": "axis",
            "group_scales": {"ry": 1.0, "rz": 0.5},
        })
        self.assertIsInstance(s, GroupwiseScaling)

    def test_groupwise_adaptive_mode(self):
        s = build_scaling_strategy("groupwise_adaptive", {
            "group_field": "group",
            "group_scales": {"a": 1.0},
            "adaptive_mode": "adaptive_smooth",
            "adaptive_config": {"beta": 2.0},
        })
        self.assertIsInstance(s, GroupwiseAdaptiveScaling)

    def test_groupwise_adaptive_defaults_to_clip(self):
        s = build_scaling_strategy("groupwise_adaptive", {
            "group_field": "g",
            "group_scales": {"a": 1.0},
        })
        self.assertIsInstance(s, GroupwiseAdaptiveScaling)
        self.assertIsInstance(s.adaptive, AdaptiveClipScaling)

    def test_invalid_mode_fails(self):
        with self.assertRaises(ValueError):
            build_scaling_strategy("invalid_mode")

    def test_groupwise_adaptive_invalid_adaptive_mode_fails(self):
        with self.assertRaises(ValueError):
            build_scaling_strategy("groupwise_adaptive", {
                "group_field": "g",
                "group_scales": {"a": 1.0},
                "adaptive_mode": "nonexistent",
            })


# -----------------------------------------------------------------------
# Model metadata tests
# -----------------------------------------------------------------------

class TestVQCModelMetadata(unittest.TestCase):
    def test_hea_metadata_count_and_fields(self):
        model = VQCModel(n_qubits=2, n_layers=2, architecture="hea")
        meta = model.parameter_metadata()
        self.assertEqual(len(meta), model.n_params)
        for m in meta:
            self.assertIn("index", m)
            self.assertIn("layer", m)
            self.assertIn("axis", m)
            self.assertIn("supports_gate_derivative", m)
            self.assertTrue(m["supports_gate_derivative"])

    def test_hea_layer_assignment(self):
        nq, nl = 2, 2
        model = VQCModel(n_qubits=nq, n_layers=nl, architecture="hea")
        meta = model.parameter_metadata()
        per_layer = nq * 2
        for pidx, m in enumerate(meta):
            expected_layer = pidx // per_layer
            self.assertEqual(m["layer"], expected_layer)

    def test_hea_axis_assignment(self):
        nq = 3
        model = VQCModel(n_qubits=nq, n_layers=1, architecture="hea")
        meta = model.parameter_metadata()
        for m in meta:
            pos = m["index"] % (nq * 2)
            if pos < nq:
                self.assertEqual(m["axis"], "ry")
            else:
                self.assertEqual(m["axis"], "rz")

    def test_gate_parameter_indices_hea(self):
        model = VQCModel(n_qubits=3, n_layers=2, architecture="hea")
        gate_idx = model.gate_parameter_indices()
        self.assertEqual(gate_idx, list(range(model.n_params)))

    def test_data_reuploading_metadata(self):
        model = VQCModel(n_qubits=2, n_layers=1, architecture="data_reuploading")
        meta = model.parameter_metadata()
        self.assertEqual(len(meta), model.n_params)
        for m in meta:
            self.assertIn("layer", m)
            self.assertIn("axis", m)
            self.assertIn("group", m)
            self.assertIn(m["group"], {"scale", "offset"})

    def test_dense_angle_metadata(self):
        model = VQCModel(n_qubits=2, n_layers=2, architecture="two_moons_dense_angle")
        meta = model.parameter_metadata()
        self.assertEqual(len(meta), model.n_params)
        for m in meta:
            self.assertIn("layer", m)
            self.assertIsNotNone(m["layer"])

    def test_alternative_model_metadata_basics(self):
        cases = [
            (ChenSUNVQCModel, {"n_macro_layers": 1}),
            (SimonettiHybridModel, {"mode": "hybrid"}),
            (SouzaSQQNNModel, {"variant": "reduced", "n_neurons": 2}),
        ]
        for ModelCls, kwargs in cases:
            with self.subTest(model=ModelCls.__name__):
                model = ModelCls(**kwargs)
                meta = model.parameter_metadata()
                self.assertEqual(len(meta), model.n_params)
                gate_idx = model.gate_parameter_indices()
                self.assertGreater(len(gate_idx), 0)


class TestGetGateMetadata(unittest.TestCase):
    def test_with_vqc_model(self):
        model = VQCModel(n_qubits=2, n_layers=1, architecture="hea")
        gate_indices = np.arange(model.n_params, dtype=int)
        meta = get_gate_metadata(model, gate_indices)
        self.assertIsNotNone(meta)
        self.assertEqual(len(meta), model.n_params)
        for m in meta:
            self.assertIsNotNone(m)

    def test_partial_indices(self):
        model = VQCModel(n_qubits=2, n_layers=2, architecture="hea")
        gate_indices = np.array([0, 2, 4], dtype=int)
        meta = get_gate_metadata(model, gate_indices)
        self.assertEqual(len(meta), 3)
        for m, idx in zip(meta, gate_indices):
            self.assertEqual(m["index"], idx)

    def test_without_metadata_returns_none(self):
        class DummyModel:
            pass
        meta = get_gate_metadata(DummyModel(), np.array([0, 1]))
        self.assertIsNone(meta)


# -----------------------------------------------------------------------
# Optimizer integration tests
# -----------------------------------------------------------------------

class TestOptimizerScalingIntegration(unittest.TestCase):
    """Integration tests for scaling in the hybrid Krotov optimizer."""

    def setUp(self):
        self.dataset = generate_two_moons(
            n_samples=40,
            noise=0.15,
            test_fraction=0.3,
            seed=42,
            encoding="tanh_0_pi",
        )
        self.base_config = replace(
            DEFAULT_CONFIG,
            max_iterations=3,
            hybrid_switch_iteration=1,
            hybrid_online_step_size=0.05,
            hybrid_batch_step_size=0.1,
            hybrid_online_schedule="constant",
            hybrid_batch_schedule="constant",
            early_stopping_enabled=False,
        )

    def _run(self, model, params, config):
        X_train, X_test, y_train, y_test = self.dataset
        return run_optimizer(
            "krotov_hybrid", model, params.copy(),
            X_train, y_train, X_test, y_test, config,
        )

    def test_baseline_preservation_no_scaling(self):
        """scaling_mode='none' produces identical results to default (unscaled)."""
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=0)

        config_default = self.base_config
        config_none = replace(self.base_config, hybrid_scaling_mode="none")

        p1, t1 = self._run(model, params, config_default)
        p2, t2 = self._run(model, params, config_none)

        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(t1["loss"], t2["loss"])

    def test_adaptive_clip_runs_and_differs_from_baseline(self):
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=1)

        config_scaled = replace(
            self.base_config,
            hybrid_scaling_mode="adaptive_clip",
            hybrid_scaling_config={"tau": 0.05, "eps": 1e-8},
        )
        p_scaled, t_scaled = self._run(model, params, config_scaled)
        p_base, _ = self._run(model, params, self.base_config)

        self.assertTrue(np.isfinite(p_scaled).all())
        self.assertTrue(np.isfinite(t_scaled["loss"]).all())
        self.assertFalse(np.allclose(p_scaled, p_base, atol=1e-12))

    def test_adaptive_smooth_runs(self):
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=2)

        config_scaled = replace(
            self.base_config,
            hybrid_scaling_mode="adaptive_smooth",
            hybrid_scaling_config={"beta": 2.0},
        )
        p, t = self._run(model, params, config_scaled)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.isfinite(t["loss"]).all())

    def test_online_only_scaling(self):
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=2)

        config = replace(
            self.base_config,
            hybrid_scaling_mode="adaptive_smooth",
            hybrid_scaling_apply_phase="online",
            hybrid_scaling_config={"beta": 2.0},
        )
        p, t = self._run(model, params, config)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.isfinite(t["loss"]).all())

    def test_batch_only_scaling(self):
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=3)

        config = replace(
            self.base_config,
            hybrid_scaling_mode="adaptive_clip",
            hybrid_scaling_apply_phase="batch",
            hybrid_scaling_config={"tau": 0.1},
        )
        p, t = self._run(model, params, config)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.isfinite(t["loss"]).all())

    def test_both_phase_scaling(self):
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=4)

        config = replace(
            self.base_config,
            hybrid_scaling_mode="adaptive_smooth",
            hybrid_scaling_apply_phase="both",
            hybrid_scaling_config={"beta": 1.0},
        )
        p, t = self._run(model, params, config)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.isfinite(t["loss"]).all())

    def test_online_vs_batch_scaling_differ(self):
        """Scaling only online vs only batch should give different results."""
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=5)

        config_online = replace(
            self.base_config,
            hybrid_scaling_mode="adaptive_clip",
            hybrid_scaling_apply_phase="online",
            hybrid_scaling_config={"tau": 0.05},
        )
        config_batch = replace(
            self.base_config,
            hybrid_scaling_mode="adaptive_clip",
            hybrid_scaling_apply_phase="batch",
            hybrid_scaling_config={"tau": 0.05},
        )
        p_online, _ = self._run(model, params, config_online)
        p_batch, _ = self._run(model, params, config_batch)
        self.assertFalse(np.allclose(p_online, p_batch, atol=1e-12))

    def test_nongate_params_updated_with_scaling(self):
        """Nongate (classical head) params should still be updated."""
        model = SimonettiHybridModel(mode="hybrid")
        params = model.init_params(seed=5)
        nongate_idx = np.asarray(model.nongate_parameter_indices(), dtype=int)
        self.assertGreater(len(nongate_idx), 0)

        config = replace(
            self.base_config,
            hybrid_scaling_mode="adaptive_clip",
            hybrid_scaling_config={"tau": 0.01},
        )
        p_scaled, t = self._run(model, params, config)
        self.assertTrue(np.isfinite(p_scaled).all())
        self.assertTrue(np.isfinite(t["loss"]).all())
        self.assertFalse(np.allclose(p_scaled[nongate_idx], params[nongate_idx]))

    def test_layerwise_scaling_with_hea(self):
        model = VQCModel(n_qubits=2, n_layers=2, entangler="ring", observable="Z0")
        params = model.init_params(seed=6)

        config = replace(
            self.base_config,
            hybrid_scaling_mode="layerwise",
            hybrid_scaling_config={"gamma": 0.8},
        )
        p, t = self._run(model, params, config)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.isfinite(t["loss"]).all())

    def test_layerwise_explicit_map_with_hea(self):
        model = VQCModel(n_qubits=2, n_layers=2, entangler="ring", observable="Z0")
        params = model.init_params(seed=6)

        config = replace(
            self.base_config,
            hybrid_scaling_mode="layerwise",
            hybrid_scaling_config={"layer_scales": {0: 1.0, 1: 0.5}},
        )
        p, t = self._run(model, params, config)
        self.assertTrue(np.isfinite(p).all())

    def test_groupwise_scaling_with_hea(self):
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=7)

        config = replace(
            self.base_config,
            hybrid_scaling_mode="groupwise",
            hybrid_scaling_config={
                "group_field": "axis",
                "group_scales": {"ry": 1.0, "rz": 0.5},
            },
        )
        p, t = self._run(model, params, config)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.isfinite(t["loss"]).all())

    def test_groupwise_adaptive_scaling(self):
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=8)

        config = replace(
            self.base_config,
            hybrid_scaling_mode="groupwise_adaptive",
            hybrid_scaling_config={
                "group_field": "axis",
                "group_scales": {"ry": 1.0, "rz": 0.5},
                "adaptive_mode": "adaptive_clip",
                "adaptive_config": {"tau": 0.1},
            },
        )
        p, t = self._run(model, params, config)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.isfinite(t["loss"]).all())

    def test_chen_model_with_scaling(self):
        model = ChenSUNVQCModel(n_macro_layers=1, readout="simple_z0")
        params = model.init_params(seed=9)

        config = replace(
            self.base_config,
            hybrid_scaling_mode="adaptive_smooth",
            hybrid_scaling_config={"beta": 1.0},
        )
        p, t = self._run(model, params, config)
        self.assertTrue(np.isfinite(p).all())
        self.assertTrue(np.isfinite(t["loss"]).all())

    def test_chen_hybrid_readout_with_layerwise_scaling(self):
        model = ChenSUNVQCModel(n_macro_layers=1, readout="hybrid_linear")
        params = model.init_params(seed=10)

        config = replace(
            self.base_config,
            hybrid_scaling_mode="layerwise",
            hybrid_scaling_config={"layer_field": "macro_layer", "gamma": 0.9},
        )
        p, t = self._run(model, params, config)
        self.assertTrue(np.isfinite(p).all())

    def test_souza_model_with_groupwise_scaling(self):
        model = SouzaSQQNNModel(variant="reduced", n_neurons=2)
        params = model.init_params(seed=11)

        config = replace(
            self.base_config,
            hybrid_scaling_mode="groupwise",
            hybrid_scaling_config={
                "group_field": "group",
                "group_scales": {"beta": 0.8},
                "default_group_scale": 1.0,
            },
        )
        p, t = self._run(model, params, config)
        self.assertTrue(np.isfinite(p).all())

    def test_layerwise_without_metadata_fails(self):
        class BareModel:
            n_params = 4
            obs = np.eye(2)
            dim = 2
            n_qubits = 1

            def init_params(self, seed=0):
                return np.zeros(4)

            def get_gate_sequence_and_states(self, params, x):
                return [], [np.array([1, 0], dtype=complex)]

            def gate_derivative_generator(self, pidx, x=None):
                return np.eye(2)

            def rebuild_param_gate(self, pidx, params, x):
                return np.eye(2)

            def forward(self, params, x):
                return 0.5

            def forward_batch(self, params, X):
                return np.full(len(X), 0.5)

            def loss(self, params, X, y):
                return 0.5

            def accuracy(self, params, X, y):
                return 0.5

        config = replace(
            self.base_config,
            hybrid_scaling_mode="layerwise",
            hybrid_scaling_config={"gamma": 0.9},
        )
        with self.assertRaises(ValueError):
            self._run(BareModel(), np.zeros(4), config)

    def test_invalid_scaling_mode_fails(self):
        config = replace(self.base_config, hybrid_scaling_mode="invalid")
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        with self.assertRaises(ValueError):
            self._run(model, model.init_params(seed=0), config)

    def test_invalid_apply_phase_fails(self):
        config = replace(
            self.base_config,
            hybrid_scaling_mode="adaptive_clip",
            hybrid_scaling_apply_phase="invalid_phase",
        )
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        with self.assertRaises(ValueError):
            self._run(model, model.init_params(seed=0), config)

    def test_scaling_preserves_trace_structure(self):
        """Trace should have the same keys and structure regardless of scaling."""
        model = VQCModel(n_qubits=2, n_layers=1, entangler="ring", observable="Z0")
        params = model.init_params(seed=12)

        config_none = self.base_config
        config_scaled = replace(
            self.base_config,
            hybrid_scaling_mode="adaptive_clip",
            hybrid_scaling_config={"tau": 0.1},
        )

        _, t_none = self._run(model, params, config_none)
        _, t_scaled = self._run(model, params, config_scaled)

        self.assertEqual(set(t_none.keys()), set(t_scaled.keys()))
        self.assertEqual(len(t_none["loss"]), len(t_scaled["loss"]))


if __name__ == "__main__":
    unittest.main()
