"""Validation tests for the exact 1x2 Fermi-Hubbard HV-VQE benchmark."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize


from qml_models.vqe import (
    HALF_FILLING_PARTICLES,
    Hubbard1x2HVVQEProblem,
    apply_hv_ansatz,
    basis_indices_with_particle_number,
    build_double_occupancy_projector,
    build_hop_block_unitary,
    build_hopping_generator_term,
    build_hubbard_1x2_hamiltonian,
    build_hv_unit_generators,
    build_onsite_block_unitary,
    build_reference_state_half_filling_noninteracting,
    exact_ground_state_energy,
    forward_states_hv_ansatz,
    get_generators_hv_1x2_5layer,
    parameter_count_hv,
    vqe_energy,
    vqe_energy_gradient,
)


class TestHubbard1x2HVBenchmark(unittest.TestCase):
    def setUp(self) -> None:
        self.psi_ref = build_reference_state_half_filling_noninteracting()

    def test_hamiltonian_is_hermitian(self) -> None:
        for U in (2.0, 4.0, 8.0):
            hamiltonian = build_hubbard_1x2_hamiltonian(U=U)
            np.testing.assert_allclose(hamiltonian, hamiltonian.conj().T, atol=1e-12)

    def test_generators_are_hermitian(self) -> None:
        onsite_unit, hop_unit = build_hv_unit_generators()
        np.testing.assert_allclose(onsite_unit, onsite_unit.conj().T, atol=1e-12)
        np.testing.assert_allclose(hop_unit, hop_unit.conj().T, atol=1e-12)

    def test_parametrized_blocks_are_unitary(self) -> None:
        onsite = build_onsite_block_unitary(0.37)
        hop = build_hop_block_unitary(-0.19)
        identity = np.eye(onsite.shape[0], dtype=complex)
        np.testing.assert_allclose(onsite.conj().T @ onsite, identity, atol=1e-12)
        np.testing.assert_allclose(hop.conj().T @ hop, identity, atol=1e-12)

    def test_five_layer_ansatz_uses_exactly_ten_parameters(self) -> None:
        self.assertEqual(parameter_count_hv(5), 10)
        self.assertEqual(len(get_generators_hv_1x2_5layer()), 10)

    def test_parameter_sharing_matches_products_of_local_terms(self) -> None:
        phi = 0.41
        tau = -0.23

        onsite_02 = build_double_occupancy_projector(0, 2)
        onsite_13 = build_double_occupancy_projector(1, 3)
        hop_01 = build_hopping_generator_term(0, 1)
        hop_23 = build_hopping_generator_term(2, 3)

        onsite_from_shared_angle = build_onsite_block_unitary(phi)
        onsite_from_local_factors = expm(1j * phi * onsite_13) @ expm(1j * phi * onsite_02)
        np.testing.assert_allclose(
            onsite_from_shared_angle,
            onsite_from_local_factors,
            atol=1e-12,
        )

        hop_from_shared_angle = build_hop_block_unitary(tau)
        hop_from_local_factors = expm(1j * tau * hop_23) @ expm(1j * tau * hop_01)
        np.testing.assert_allclose(hop_from_shared_angle, hop_from_local_factors, atol=1e-12)

    def test_reference_state_lies_in_half_filling_sector(self) -> None:
        allowed = set(basis_indices_with_particle_number(HALF_FILLING_PARTICLES))
        support = {index for index, amp in enumerate(self.psi_ref) if abs(amp) > 1e-12}
        self.assertTrue(support.issubset(allowed))
        np.testing.assert_allclose(np.linalg.norm(self.psi_ref), 1.0, atol=1e-12)

    def test_ansatz_preserves_state_norm(self) -> None:
        theta = np.linspace(-0.4, 0.5, 10)
        final_state = apply_hv_ansatz(theta, self.psi_ref)
        np.testing.assert_allclose(np.linalg.norm(final_state), 1.0, atol=1e-12)

    def test_vqe_energy_returns_real_scalar(self) -> None:
        theta = np.linspace(-0.2, 0.25, 10)
        hamiltonian = build_hubbard_1x2_hamiltonian(U=4.0)
        energy = vqe_energy(theta, hamiltonian, self.psi_ref)
        self.assertIsInstance(energy, float)
        self.assertTrue(np.isfinite(energy))

    def test_forward_states_bookkeeping(self) -> None:
        theta = np.linspace(-0.3, 0.3, 10)
        states = forward_states_hv_ansatz(theta, self.psi_ref)
        self.assertEqual(len(states), 11)

    def test_forward_states_match_apply_hv_ansatz(self) -> None:
        theta = np.linspace(-0.3, 0.3, 10)
        states = forward_states_hv_ansatz(theta, self.psi_ref)
        final_state = apply_hv_ansatz(theta, self.psi_ref)
        np.testing.assert_allclose(states[-1], final_state, atol=1e-12)

    def test_exact_energy_gradient_matches_problem_wrapper(self) -> None:
        theta = np.linspace(-0.25, 0.35, 10)
        problem = Hubbard1x2HVVQEProblem(U=4.0)
        direct = vqe_energy_gradient(theta, problem.H, problem.psi_ref)
        wrapped = problem.energy_gradient(theta)
        np.testing.assert_allclose(direct, wrapped, atol=1e-12)

    def test_small_exact_benchmark_respects_variational_bound(self) -> None:
        problem = Hubbard1x2HVVQEProblem(U=4.0)
        exact_ground = problem.exact_ground_energy(n_particles=HALF_FILLING_PARTICLES)

        rng = np.random.default_rng(7)
        initial_thetas = [rng.uniform(-0.5, 0.5, size=problem.n_params) for _ in range(3)]
        best_initial_energy = min(problem.energy(theta0) for theta0 in initial_thetas)

        best_energy = np.inf
        for theta0 in initial_thetas:
            result = minimize(
                problem.energy,
                theta0,
                jac=problem.energy_gradient,
                method="L-BFGS-B",
                options={"maxiter": 60},
            )
            self.assertTrue(np.isfinite(result.fun))
            best_energy = min(best_energy, float(result.fun))

        self.assertLessEqual(best_energy, best_initial_energy + 1e-9)
        self.assertGreaterEqual(best_energy, exact_ground - 1e-8)

    def test_exact_ground_energy_matches_function_helper(self) -> None:
        hamiltonian = build_hubbard_1x2_hamiltonian(U=4.0)
        energy_from_function = exact_ground_state_energy(
            hamiltonian,
            n_particles=HALF_FILLING_PARTICLES,
        )
        energy_from_problem = Hubbard1x2HVVQEProblem(U=4.0).exact_ground_energy()
        self.assertAlmostEqual(energy_from_function, energy_from_problem, places=12)


if __name__ == "__main__":
    unittest.main()
