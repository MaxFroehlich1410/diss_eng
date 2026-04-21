"""Equivalence tests for the PennyLane Perez-Salinas mirror model."""

from __future__ import annotations

import unittest

import numpy as np

from qml_models.variants import (
    PennyLanePerezSalinasReuploadingModel,
    PerezSalinasReuploadingModel,
)


class PennyLanePerezSalinasEquivalenceTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[0.15, -0.25], [0.40, 0.10]], dtype=float)
        self.y = np.array([1, 0], dtype=int)

    def test_state_scores_and_loss_match_native_model(self):
        native = PerezSalinasReuploadingModel(
            n_qubits=2,
            n_layers=2,
            n_classes=2,
            use_entanglement=True,
            use_classical_head=True,
        )
        pennylane_model = PennyLanePerezSalinasReuploadingModel(
            n_qubits=2,
            n_layers=2,
            n_classes=2,
            use_entanglement=True,
            use_classical_head=True,
        )
        params = native.init_params(seed=7)

        native_state = native._sample_forward_details(params, self.X[0])["state"]
        pennylane_state = np.asarray(pennylane_model.state(params, self.X[0]), dtype=complex)
        self.assertTrue(np.allclose(native_state, pennylane_state, atol=1e-10, rtol=1e-10))

        native_scores = native.class_scores(params, self.X[0])
        pennylane_scores = np.asarray(pennylane_model.class_scores(params, self.X[0]), dtype=float)
        self.assertTrue(np.allclose(native_scores, pennylane_scores, atol=1e-10, rtol=1e-10))

        native_loss = native.loss(params, self.X, self.y)
        pennylane_loss = float(pennylane_model.loss(params, self.X, self.y))
        self.assertAlmostEqual(native_loss, pennylane_loss, places=10)

    def test_fixed_mean_fidelity_no_head_matches_native_model(self):
        native = PerezSalinasReuploadingModel(
            n_qubits=2,
            n_layers=2,
            n_classes=2,
            use_entanglement=True,
            use_classical_head=False,
        )
        pennylane_model = PennyLanePerezSalinasReuploadingModel(
            n_qubits=2,
            n_layers=2,
            n_classes=2,
            use_entanglement=True,
            use_classical_head=False,
        )
        params = native.init_params(seed=11)

        native_scores = native.class_scores(params, self.X[0])
        pennylane_scores = np.asarray(pennylane_model.class_scores(params, self.X[0]), dtype=float)
        self.assertTrue(np.allclose(native_scores, pennylane_scores, atol=1e-10, rtol=1e-10))
        self.assertAlmostEqual(native.loss(params, self.X, self.y), float(pennylane_model.loss(params, self.X, self.y)))

    def test_full_loss_gradient_matches_native_model(self):
        native = PerezSalinasReuploadingModel(
            n_qubits=2,
            n_layers=2,
            n_classes=2,
            use_entanglement=True,
            use_classical_head=True,
        )
        pennylane_model = PennyLanePerezSalinasReuploadingModel(
            n_qubits=2,
            n_layers=2,
            n_classes=2,
            use_entanglement=True,
            use_classical_head=True,
        )
        params = native.init_params(seed=13)

        native_grad, _ = native.loss_gradient(params, self.X, self.y)
        import pennylane as qml
        from pennylane import numpy as pnp

        params_pl = pnp.array(params, requires_grad=True)
        grad_fn = qml.grad(lambda p: pennylane_model.loss(p, self.X, self.y))
        pennylane_grad = np.asarray(grad_fn(params_pl), dtype=float)

        self.assertTrue(np.allclose(native_grad, pennylane_grad, atol=5e-6, rtol=5e-5))

    def test_4q_8l_matches_across_multiple_seeds_and_inputs(self):
        native = PerezSalinasReuploadingModel(
            n_qubits=4,
            n_layers=8,
            n_classes=2,
            use_entanglement=True,
            use_classical_head=True,
        )
        pennylane_model = PennyLanePerezSalinasReuploadingModel(
            n_qubits=4,
            n_layers=8,
            n_classes=2,
            use_entanglement=True,
            use_classical_head=True,
        )

        probe_inputs = np.array(
            [
                [0.15, -0.25],
                [0.40, 0.10],
                [-0.62, 0.37],
                [0.81, -0.44],
                [-0.05, -0.91],
            ],
            dtype=float,
        )
        batch_y = np.array([1, 0, 1, 0, 1], dtype=int)

        for seed in (0, 1, 3):
            params = native.init_params(seed=seed)
            for x in probe_inputs:
                native_state = native._sample_forward_details(params, x)["state"]
                pennylane_state = np.asarray(pennylane_model.state(params, x), dtype=complex)
                self.assertTrue(np.allclose(native_state, pennylane_state, atol=1e-9, rtol=1e-9))

                native_scores = native.class_scores(params, x)
                pennylane_scores = np.asarray(pennylane_model.class_scores(params, x), dtype=float)
                self.assertTrue(np.allclose(native_scores, pennylane_scores, atol=1e-9, rtol=1e-9))

            native_loss = native.loss(params, probe_inputs, batch_y)
            pennylane_loss = float(pennylane_model.loss(params, probe_inputs, batch_y))
            self.assertAlmostEqual(native_loss, pennylane_loss, places=9)


if __name__ == "__main__":
    unittest.main()
