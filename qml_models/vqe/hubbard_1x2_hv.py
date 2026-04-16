"""Exact-statevector HV-VQE benchmark for the 1x2 Fermi-Hubbard model.

This module implements the 4-qubit, 5-layer Hamiltonian-variational (HV)
benchmark instance used for optimizer comparisons in the Fermi-Hubbard VQE
literature, specialized to the 1x2 chain with spin.

Qubit ordering
--------------
The computational basis is ordered as ``|q0 q1 q2 q3>`` with

- ``q0 = (site 1, spin up)``
- ``q1 = (site 2, spin up)``
- ``q2 = (site 1, spin down)``
- ``q3 = (site 2, spin down)``

Hamiltonian
-----------
The physical Hamiltonian is

``H(U, t) = t * H_hop_unit + U * H_onsite_unit``

with

``H_hop_unit = 0.5 * (X0 X1 + Y0 Y1) + 0.5 * (X2 X3 + Y2 Y3)``

and

``H_onsite_unit = |11><11|_(0,2) + |11><11|_(1,3)``.

Ansatz
------
The HV ansatz uses a fixed-generator block sequence with 2 trainable scalars
per layer:

``theta = [phi_1, tau_1, ..., phi_5, tau_5]``

and

``U_layer(phi_l, tau_l) = exp(+i * tau_l * H_hop_unit) @ exp(+i * phi_l * H_onsite_unit)``.

The sign convention is therefore ``U_k(theta_k) = exp(+i * theta_k * G_k)`` for
Hermitian block generator ``G_k``. The corresponding left-derivative operator is
``dU_k/dtheta_k = (i G_k) U_k``.

Simulation model
----------------
Everything here is exact dense linear algebra:

- exact statevectors only
- exact expectation values only
- no shot noise
- no device noise
- no finite-difference gradients as the primary gradient method

The helper class :class:`Hubbard1x2HVVQEProblem` exposes both a simple VQE API
and a gate-by-gate interface that matches the needs of the hybrid Krotov code:
forward states after each parametrized block and fixed block generators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.linalg import expm

N_QUBITS = 4
HILBERT_DIM = 2 ** N_QUBITS
DEFAULT_N_LAYERS = 5
HALF_FILLING_PARTICLES = 2

I2 = np.eye(2, dtype=complex)
X_PAULI = np.array([[0, 1], [1, 0]], dtype=complex)
Y_PAULI = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z_PAULI = np.array([[1, 0], [0, -1]], dtype=complex)


def _kron_n(operators: tuple[np.ndarray, ...]) -> np.ndarray:
    result = operators[0]
    for op in operators[1:]:
        result = np.kron(result, op)
    return result


def _single_qubit_operator(
    operator: np.ndarray,
    qubit: int,
    n_qubits: int = N_QUBITS,
) -> np.ndarray:
    if qubit < 0 or qubit >= n_qubits:
        raise ValueError(f"Qubit index {qubit} out of range for {n_qubits} qubits.")
    ops = [I2] * n_qubits
    ops[qubit] = np.asarray(operator, dtype=complex)
    return _kron_n(tuple(ops))


def _validate_state(state: np.ndarray, n_qubits: int = N_QUBITS) -> np.ndarray:
    array = np.asarray(state, dtype=complex)
    expected_shape = (2 ** n_qubits,)
    if array.shape != expected_shape:
        raise ValueError(f"State must have shape {expected_shape}, got {array.shape}.")
    return array


def _validate_hamiltonian(hamiltonian: np.ndarray) -> np.ndarray:
    array = np.asarray(hamiltonian, dtype=complex)
    if array.shape != (HILBERT_DIM, HILBERT_DIM):
        raise ValueError(
            f"Hamiltonian must have shape {(HILBERT_DIM, HILBERT_DIM)}, got {array.shape}."
        )
    return array


def _validate_theta(theta: np.ndarray, n_layers: int | None = None) -> np.ndarray:
    array = np.asarray(theta, dtype=float)
    if array.ndim != 1:
        raise ValueError("Theta must be a one-dimensional array.")
    if n_layers is None:
        if array.size % 2 != 0:
            raise ValueError("Theta must contain an even number of entries.")
    elif array.size != parameter_count_hv(n_layers):
        raise ValueError(
            f"Theta must contain exactly {parameter_count_hv(n_layers)} parameters "
            f"for {n_layers} layers, got {array.size}."
        )
    return array


def _canonicalize_global_phase(state: np.ndarray) -> np.ndarray:
    state = np.asarray(state, dtype=complex).copy()
    pivot = int(np.argmax(np.abs(state)))
    amplitude = state[pivot]
    if np.abs(amplitude) > 0.0:
        state *= np.exp(-1j * np.angle(amplitude))
    return state


def parameter_count_hv(n_layers: int = DEFAULT_N_LAYERS) -> int:
    """Return the number of trainable HV parameters for ``n_layers``."""
    if n_layers <= 0:
        raise ValueError("The number of layers must be positive.")
    return 2 * n_layers


def basis_indices_with_particle_number(
    n_particles: int,
    n_qubits: int = N_QUBITS,
) -> np.ndarray:
    """Return computational-basis indices with fixed particle number."""
    if n_particles < 0 or n_particles > n_qubits:
        raise ValueError(
            f"Particle number must lie in [0, {n_qubits}], got {n_particles}."
        )
    return np.array(
        [index for index in range(2 ** n_qubits) if index.bit_count() == n_particles],
        dtype=int,
    )


def build_double_occupancy_projector(
    qubit_i: int,
    qubit_j: int,
    n_qubits: int = N_QUBITS,
) -> np.ndarray:
    """Build ``|11><11|_(i,j) = ((I-Z_i)/2) * ((I-Z_j)/2)``."""
    if qubit_i == qubit_j:
        raise ValueError("Double-occupancy projector requires two distinct qubits.")
    identity = np.eye(2 ** n_qubits, dtype=complex)
    z_i = _single_qubit_operator(Z_PAULI, qubit_i, n_qubits)
    z_j = _single_qubit_operator(Z_PAULI, qubit_j, n_qubits)
    return 0.25 * (identity - z_i - z_j + z_i @ z_j)


def build_hopping_generator_term(
    qubit_i: int,
    qubit_j: int,
    n_qubits: int = N_QUBITS,
) -> np.ndarray:
    """Build ``0.5 * (X_i X_j + Y_i Y_j)`` on the full Hilbert space."""
    if qubit_i == qubit_j:
        raise ValueError("Hopping term requires two distinct qubits.")
    x_i = _single_qubit_operator(X_PAULI, qubit_i, n_qubits)
    x_j = _single_qubit_operator(X_PAULI, qubit_j, n_qubits)
    y_i = _single_qubit_operator(Y_PAULI, qubit_i, n_qubits)
    y_j = _single_qubit_operator(Y_PAULI, qubit_j, n_qubits)
    return 0.5 * (x_i @ x_j + y_i @ y_j)


@lru_cache(maxsize=1)
def _cached_hv_unit_generators() -> tuple[np.ndarray, np.ndarray]:
    onsite_unit = (
        build_double_occupancy_projector(0, 2)
        + build_double_occupancy_projector(1, 3)
    )
    hop_unit = (
        build_hopping_generator_term(0, 1)
        + build_hopping_generator_term(2, 3)
    )
    return onsite_unit, hop_unit


def build_hv_unit_generators() -> tuple[np.ndarray, np.ndarray]:
    """Return ``(H_onsite_unit, H_hop_unit)`` for the 1x2 HV ansatz."""
    onsite_unit, hop_unit = _cached_hv_unit_generators()
    return onsite_unit.copy(), hop_unit.copy()


def get_generators_hv_1x2(n_layers: int = DEFAULT_N_LAYERS) -> list[np.ndarray]:
    """Return the Hermitian block generators ``[onsite_1, hop_1, ..., hop_L]``."""
    onsite_unit, hop_unit = _cached_hv_unit_generators()
    generators: list[np.ndarray] = []
    for _ in range(n_layers):
        generators.append(onsite_unit.copy())
        generators.append(hop_unit.copy())
    return generators


def get_generators_hv_1x2_5layer() -> list[np.ndarray]:
    """Return the 10 Hermitian generators for the 5-layer benchmark."""
    return get_generators_hv_1x2(DEFAULT_N_LAYERS)


def parameter_metadata_hv_1x2(n_layers: int = DEFAULT_N_LAYERS) -> list[dict[str, object]]:
    """Return optimizer-facing metadata for the shared HV parameters."""
    metadata: list[dict[str, object]] = []
    for layer in range(n_layers):
        metadata.append(
            {
                "index": 2 * layer,
                "name": f"phi_{layer + 1}",
                "group": "onsite",
                "block_type": "onsite",
                "layer": layer,
                "supports_gate_derivative": True,
            }
        )
        metadata.append(
            {
                "index": 2 * layer + 1,
                "name": f"tau_{layer + 1}",
                "group": "hop",
                "block_type": "hop",
                "layer": layer,
                "supports_gate_derivative": True,
            }
        )
    return metadata


def build_hubbard_1x2_hamiltonian(U: float, t: float = -1.0) -> np.ndarray:
    """Build the exact 4-qubit Fermi-Hubbard Hamiltonian for the 1x2 chain."""
    onsite_unit, hop_unit = _cached_hv_unit_generators()
    return (float(t) * hop_unit + float(U) * onsite_unit).copy()


def exact_ground_state(
    hamiltonian: np.ndarray,
    n_particles: int | None = HALF_FILLING_PARTICLES,
) -> tuple[float, np.ndarray]:
    """Return the exact ground energy and state, optionally in a fixed sector."""
    hamiltonian = _validate_hamiltonian(hamiltonian)

    if n_particles is None:
        evals, evecs = np.linalg.eigh(hamiltonian)
        state = evecs[:, 0]
        return float(np.real(evals[0])), _canonicalize_global_phase(state)

    sector = basis_indices_with_particle_number(n_particles, N_QUBITS)
    restricted = hamiltonian[np.ix_(sector, sector)]
    evals, evecs = np.linalg.eigh(restricted)

    state = np.zeros(HILBERT_DIM, dtype=complex)
    state[sector] = evecs[:, 0]
    state /= np.linalg.norm(state)
    state = _canonicalize_global_phase(state)
    return float(np.real(evals[0])), state


def exact_ground_state_energy(
    hamiltonian: np.ndarray,
    n_particles: int | None = HALF_FILLING_PARTICLES,
) -> float:
    """Return the exact ground energy, defaulting to the half-filling sector."""
    energy, _ = exact_ground_state(hamiltonian, n_particles=n_particles)
    return energy


def build_reference_state_half_filling_noninteracting(t: float = -1.0) -> np.ndarray:
    """Build the exact ``U=0`` half-filling reference state as a statevector."""
    hamiltonian = build_hubbard_1x2_hamiltonian(U=0.0, t=t)
    _, state = exact_ground_state(hamiltonian, n_particles=HALF_FILLING_PARTICLES)
    return state


def build_parametrized_block_unitary(theta_k: float, generator: np.ndarray) -> np.ndarray:
    """Build ``exp(+i * theta_k * G_k)`` for a fixed Hermitian generator ``G_k``."""
    generator = _validate_hamiltonian(generator)
    return expm(1j * float(theta_k) * generator)


def build_onsite_block_unitary(phi: float) -> np.ndarray:
    """Build ``U_onsite(phi) = exp(+i * phi * H_onsite_unit)``."""
    onsite_unit, _ = _cached_hv_unit_generators()
    return build_parametrized_block_unitary(phi, onsite_unit)


def build_hop_block_unitary(tau: float) -> np.ndarray:
    """Build ``U_hop(tau) = exp(+i * tau * H_hop_unit)``."""
    _, hop_unit = _cached_hv_unit_generators()
    return build_parametrized_block_unitary(tau, hop_unit)


def build_hv_layer_unitary(phi: float, tau: float) -> np.ndarray:
    """Build one HV layer in the documented order ``U_hop @ U_onsite``."""
    return build_hop_block_unitary(tau) @ build_onsite_block_unitary(phi)


def apply_hv_layer(state: np.ndarray, phi: float, tau: float) -> np.ndarray:
    """Apply one HV layer to ``state`` using ``U_hop(tau) @ U_onsite(phi)``."""
    state = _validate_state(state)
    return build_hv_layer_unitary(phi, tau) @ state


def build_hv_block_sequence(theta: np.ndarray) -> list[tuple[np.ndarray, int]]:
    """Return the sequential trainable HV blocks as ``(unitary, param_idx)``."""
    theta = _validate_theta(theta)
    onsite_unit, hop_unit = _cached_hv_unit_generators()
    blocks: list[tuple[np.ndarray, int]] = []
    for param_idx, angle in enumerate(theta):
        generator = onsite_unit if param_idx % 2 == 0 else hop_unit
        blocks.append((build_parametrized_block_unitary(angle, generator), param_idx))
    return blocks


def apply_hv_ansatz(theta: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
    """Apply the full HV ansatz to the reference state."""
    return forward_states_hv_ansatz(theta, psi_ref)[-1]


def forward_states_hv_ansatz(theta: np.ndarray, psi_ref: np.ndarray) -> list[np.ndarray]:
    """Return ``[psi_0, psi_1, ..., psi_K]`` at trainable-block resolution."""
    theta = _validate_theta(theta)
    state = _validate_state(psi_ref)
    states = [state.copy()]
    for gate, _ in build_hv_block_sequence(theta):
        state = gate @ state
        states.append(state.copy())
    return states


def vqe_energy(theta: np.ndarray, H: np.ndarray, psi_ref: np.ndarray) -> float:
    """Return the exact VQE loss ``<psi(theta)|H|psi(theta)>`` as a real scalar."""
    H = _validate_hamiltonian(H)
    final_state = apply_hv_ansatz(theta, psi_ref)
    energy = np.vdot(final_state, H @ final_state)
    return float(np.real_if_close(energy).real)


def vqe_energy_gradient(theta: np.ndarray, H: np.ndarray, psi_ref: np.ndarray) -> np.ndarray:
    """Return the exact energy gradient via forward/backward block propagation."""
    theta = _validate_theta(theta)
    H = _validate_hamiltonian(H)
    states = forward_states_hv_ansatz(theta, psi_ref)
    final_state = states[-1]
    generators = get_generators_hv_1x2(len(theta) // 2)
    block_sequence = build_hv_block_sequence(theta)

    costates: list[np.ndarray | None] = [None] * len(states)
    costates[-1] = H @ final_state
    for block_idx in range(len(block_sequence) - 1, -1, -1):
        gate = block_sequence[block_idx][0]
        costates[block_idx] = gate.conj().T @ costates[block_idx + 1]

    grad = np.zeros_like(theta, dtype=float)
    for param_idx, generator in enumerate(generators):
        left_generator = 1j * generator
        grad[param_idx] = 2.0 * np.real(
            np.vdot(costates[param_idx + 1], left_generator @ states[param_idx + 1])
        )
    return grad


@dataclass
class Hubbard1x2HVVQEProblem:
    """Convenience wrapper for the exact 1x2 HV-VQE benchmark instance."""

    U: float
    t: float = -1.0
    n_layers: int = DEFAULT_N_LAYERS
    psi_ref: np.ndarray | None = None
    H: np.ndarray = field(init=False, repr=False)
    _generators: tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_layers <= 0:
            raise ValueError("The number of layers must be positive.")
        self.H = build_hubbard_1x2_hamiltonian(self.U, self.t)
        if self.psi_ref is None:
            self.psi_ref = build_reference_state_half_filling_noninteracting(t=self.t)
        else:
            self.psi_ref = _validate_state(self.psi_ref)
        self._generators = tuple(get_generators_hv_1x2(self.n_layers))

    @property
    def dim(self) -> int:
        return HILBERT_DIM

    @property
    def n_qubits(self) -> int:
        return N_QUBITS

    @property
    def n_params(self) -> int:
        return parameter_count_hv(self.n_layers)

    def parameter_metadata(self) -> list[dict[str, object]]:
        return parameter_metadata_hv_1x2(self.n_layers)

    def gate_parameter_indices(self) -> list[int]:
        return list(range(self.n_params))

    def get_generators(self) -> list[np.ndarray]:
        return [generator.copy() for generator in self._generators]

    def block_generator(self, param_idx: int) -> np.ndarray:
        if param_idx < 0 or param_idx >= self.n_params:
            raise IndexError(f"Parameter index {param_idx} out of range.")
        return self._generators[param_idx].copy()

    def gate_derivative_generator(self, param_idx: int, x: None = None) -> np.ndarray:
        """Return the left-generator ``i G_k`` used in exact gate derivatives."""
        del x
        return 1j * self.block_generator(param_idx)

    def terminal_costate(self, final_state: np.ndarray) -> np.ndarray:
        return self.H @ _validate_state(final_state)

    def get_gate_sequence_and_states(
        self,
        theta: np.ndarray,
        psi_ref: np.ndarray | None = None,
    ) -> tuple[list[tuple[np.ndarray, int]], list[np.ndarray]]:
        theta = _validate_theta(theta, self.n_layers)
        state = self.psi_ref if psi_ref is None else _validate_state(psi_ref)
        states = [state.copy()]
        gates: list[tuple[np.ndarray, int]] = []
        for param_idx, (angle, generator) in enumerate(zip(theta, self._generators)):
            gate = build_parametrized_block_unitary(angle, generator)
            gates.append((gate, param_idx))
            state = gate @ state
            states.append(state.copy())
        return gates, states

    def apply_ansatz(self, theta: np.ndarray, psi_ref: np.ndarray | None = None) -> np.ndarray:
        return self.get_gate_sequence_and_states(theta, psi_ref=psi_ref)[1][-1]

    def forward_states(self, theta: np.ndarray, psi_ref: np.ndarray | None = None) -> list[np.ndarray]:
        return self.get_gate_sequence_and_states(theta, psi_ref=psi_ref)[1]

    def energy(self, theta: np.ndarray, psi_ref: np.ndarray | None = None) -> float:
        state = self.psi_ref if psi_ref is None else _validate_state(psi_ref)
        return vqe_energy(theta, self.H, state)

    def energy_gradient(self, theta: np.ndarray, psi_ref: np.ndarray | None = None) -> np.ndarray:
        state = self.psi_ref if psi_ref is None else _validate_state(psi_ref)
        theta = _validate_theta(theta, self.n_layers)
        return vqe_energy_gradient(theta, self.H, state)

    def exact_ground_state(
        self,
        n_particles: int | None = HALF_FILLING_PARTICLES,
    ) -> tuple[float, np.ndarray]:
        return exact_ground_state(self.H, n_particles=n_particles)

    def exact_ground_energy(self, n_particles: int | None = HALF_FILLING_PARTICLES) -> float:
        return exact_ground_state_energy(self.H, n_particles=n_particles)
