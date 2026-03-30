r"""Gate-based circuit Krotov optimisation with interleaved dissipation.

Extends the MCWF-Krotov method to parameterised quantum circuits where
each layer consists of:

  1. A set of parameterised unitary gates  U_j(theta_j)
  2. A dissipative channel  D(gamma, dt)  modelled via MCWF / Lindblad

The Krotov update for gate parameter theta_j at layer j is:

    Delta theta_j = (S / (M lambda))  sum_k  Im< chi_k(t_j) | G_j | psi_k(t_j) >

where G_j is the Hermitian generator of the gate  U_j = exp(-i theta_j G_j).

This module supports two dissipation models:
  a) Global target-cooling Lindblad operators (ideal teacher)
  b) Hardware-constrained local channels (amp-damping, dephasing, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.linalg import expm

from .mcwf import build_effective_hamiltonian


# ---------------------------------------------------------------------------
# Gate layer definitions
# ---------------------------------------------------------------------------

@dataclass
class GateLayer:
    """One parameterised gate in the circuit.

    generator : ndarray (d, d) or None
        Full Hermitian generator G so that U = exp(-i theta G).
        Can be None for large systems when local_generator is provided.
    theta : float
        Current parameter value.
    name : str
        Human-readable label.
    local_generator : ndarray (2, 2) or (4, 4) or None
        Local generator acting on 1 or 2 qubits.
    qubits : tuple of int
        Qubit indices the gate acts on.
    """
    generator: Optional[np.ndarray] = None
    theta: float = 0.0
    name: str = ""
    local_generator: Optional[np.ndarray] = None
    qubits: tuple = ()


def ry_generator(qubit: int, n_qubits: int) -> np.ndarray:
    """Generator for Ry(theta) = exp(-i theta Y/2) on a specific qubit."""
    d = 2 ** n_qubits
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    G = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            diff = i ^ j
            if diff == (1 << qubit):
                bit_i = (i >> qubit) & 1
                bit_j = (j >> qubit) & 1
                G[i, j] = Y[bit_i, bit_j] / 2
    for i in range(d):
        bit = (i >> qubit) & 1
        G[i, i] = Y[bit, bit] / 2
    return G


def rz_generator(qubit: int, n_qubits: int) -> np.ndarray:
    """Generator for Rz(theta) = exp(-i theta Z/2) on a specific qubit."""
    d = 2 ** n_qubits
    G = np.zeros((d, d), dtype=complex)
    for i in range(d):
        bit = (i >> qubit) & 1
        G[i, i] = (1.0 - 2.0 * bit) / 2
    return G


def rx_generator(qubit: int, n_qubits: int) -> np.ndarray:
    """Generator for Rx(theta) = exp(-i theta X/2) on a specific qubit."""
    d = 2 ** n_qubits
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    G = np.zeros((d, d), dtype=complex)
    for i in range(d):
        for j in range(d):
            diff = i ^ j
            if diff == (1 << qubit) or i == j:
                bit_i = (i >> qubit) & 1
                bit_j = (j >> qubit) & 1
                G[i, j] = X[bit_i, bit_j] / 2
    return G


def cnot_zx_generator(control: int, target: int, n_qubits: int) -> np.ndarray:
    r"""Generator for a CNOT-like entangling gate.

    CNOT = exp(-i pi/4 (I - Z_c)(I - X_t) / 2) up to global phase.
    We use G = (I - Z_c) x (I - X_t) / 4 as the generator.
    """
    d = 2 ** n_qubits
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    ops_c = [I] * n_qubits
    ops_c[control] = I - Z

    ops_t = [I] * n_qubits
    ops_t[target] = I - X

    Gc = ops_c[0]
    for op in ops_c[1:]:
        Gc = np.kron(Gc, op)

    Gt = ops_t[0]
    for op in ops_t[1:]:
        Gt = np.kron(Gt, op)

    return (Gc @ Gt) / 4


_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_LOCAL_RY = _Y / 2
_LOCAL_RZ = _Z / 2
_LOCAL_CNOT = np.kron(_I2 - _Z, _I2 - _X) / 4


def build_hardware_efficient_ansatz(
    n_qubits: int,
    n_layers: int,
    full_generators: bool = True,
) -> list[list[GateLayer]]:
    """Build a hardware-efficient variational ansatz.

    Each layer: Ry on all qubits, Rz on all qubits, then CNOT chain.
    Set full_generators=False for n_qubits > 6 to save memory.
    """
    circuit_layers = []
    for layer in range(n_layers):
        gates = []
        for q in range(n_qubits):
            gen = ry_generator(q, n_qubits) if full_generators else None
            gates.append(GateLayer(
                generator=gen, theta=0.1, name=f"Ry_{q}_L{layer}",
                local_generator=_LOCAL_RY, qubits=(q,),
            ))
        for q in range(n_qubits):
            gen = rz_generator(q, n_qubits) if full_generators else None
            gates.append(GateLayer(
                generator=gen, theta=0.1, name=f"Rz_{q}_L{layer}",
                local_generator=_LOCAL_RZ, qubits=(q,),
            ))
        for q in range(n_qubits - 1):
            gen = cnot_zx_generator(q, q + 1, n_qubits) if full_generators else None
            gates.append(GateLayer(
                generator=gen, theta=np.pi / 4, name=f"CNOT_{q}_{q+1}_L{layer}",
                local_generator=_LOCAL_CNOT, qubits=(q, q + 1),
            ))
        circuit_layers.append(gates)
    return circuit_layers


# ---------------------------------------------------------------------------
# Efficient tensor-contraction gate application (O(2^k * d) instead of O(d^3))
# ---------------------------------------------------------------------------

def _apply_1q_to_sv(psi: np.ndarray, M2: np.ndarray,
                     qubit: int, n_qubits: int) -> np.ndarray:
    """Apply a 2x2 matrix to one qubit of a statevector via tensor contraction."""
    psi_t = psi.reshape((2,) * n_qubits)
    psi_t = np.tensordot(M2, psi_t, axes=([1], [qubit]))
    psi_t = np.moveaxis(psi_t, 0, qubit)
    return psi_t.reshape(-1)


def _apply_2q_to_sv(psi: np.ndarray, M4: np.ndarray,
                     qi: int, qj: int, n_qubits: int) -> np.ndarray:
    """Apply a 4x4 matrix to two qubits of a statevector."""
    d = len(psi)
    psi_t = psi.reshape((2,) * n_qubits)
    axes = list(range(n_qubits))
    rest = [a for a in axes if a not in (qi, qj)]
    perm = [qi, qj] + rest
    psi_t = psi_t.transpose(perm)
    d_rest = d // 4
    psi_t = (M4 @ psi_t.reshape(4, d_rest)).reshape(
        [2, 2] + [2] * (n_qubits - 2))
    inv_perm = [0] * n_qubits
    for new_pos, old_pos in enumerate(perm):
        inv_perm[old_pos] = new_pos
    return psi_t.transpose(inv_perm).reshape(d)


def _apply_local_to_sv(psi, mat, qubits, n_qubits):
    """Dispatch to 1q or 2q tensor-contraction helper."""
    if len(qubits) == 1:
        return _apply_1q_to_sv(psi, mat, qubits[0], n_qubits)
    return _apply_2q_to_sv(psi, mat, qubits[0], qubits[1], n_qubits)


def _apply_1q_unitary_to_dm(rho, U2, qubit, n_qubits):
    """Apply rho -> U rho U† for a 1q unitary via tensor contraction."""
    d = 2 ** n_qubits
    shape = (2,) * (2 * n_qubits)
    rho_t = rho.reshape(shape)
    ax_ket, ax_bra = qubit, n_qubits + qubit
    tmp = np.tensordot(U2, rho_t, axes=([1], [ax_ket]))
    tmp = np.moveaxis(tmp, 0, ax_ket)
    tmp = np.tensordot(tmp, U2.conj(), axes=([ax_bra], [1]))
    tmp = np.moveaxis(tmp, -1, ax_bra)
    return tmp.reshape(d, d)


def _apply_2q_unitary_to_dm(rho, U4, qi, qj, n_qubits):
    """Apply rho -> U rho U† for a 2q unitary via tensor contraction."""
    d = 2 ** n_qubits
    shape = (2,) * (2 * n_qubits)
    rho_t = rho.reshape(shape)
    ax_bra_i, ax_bra_j = n_qubits + qi, n_qubits + qj
    axes_ket = list(range(n_qubits))
    axes_bra = list(range(n_qubits, 2 * n_qubits))
    ket_rest = [k for k in axes_ket if k not in (qi, qj)]
    bra_rest = [k for k in axes_bra if k not in (ax_bra_i, ax_bra_j)]
    perm = [qi, qj] + ket_rest + [ax_bra_i, ax_bra_j] + bra_rest
    rho_p = rho_t.transpose(perm)
    n_rest = n_qubits - 2
    d_rest = 2 ** n_rest if n_rest > 0 else 1
    rho_p = rho_p.reshape(4, d_rest, 4, d_rest)
    out_p = np.einsum("ab,brcs,dc->ards", U4, rho_p, U4.conj(), optimize=True)
    out_p = out_p.reshape([2, 2] + [2] * n_rest + [2, 2] + [2] * n_rest)
    inv_perm = [0] * len(perm)
    for new_pos, old_pos in enumerate(perm):
        inv_perm[old_pos] = new_pos
    return out_p.transpose(inv_perm).reshape(d, d)


# ---------------------------------------------------------------------------
# Gate application
# ---------------------------------------------------------------------------

def apply_gate(psi: np.ndarray, gate: GateLayer) -> np.ndarray:
    """Apply U = exp(-i theta G) to psi."""
    if gate.local_generator is not None and gate.qubits:
        n_qubits = int(np.log2(len(psi)))
        U_local = expm(-1j * gate.theta * gate.local_generator)
        return _apply_local_to_sv(psi, U_local, gate.qubits, n_qubits)
    U = expm(-1j * gate.theta * gate.generator)
    return U @ psi


def apply_gate_adjoint(chi: np.ndarray, gate: GateLayer) -> np.ndarray:
    """Apply U† = exp(+i theta G) to chi."""
    if gate.local_generator is not None and gate.qubits:
        n_qubits = int(np.log2(len(chi)))
        Ud_local = expm(1j * gate.theta * gate.local_generator)
        return _apply_local_to_sv(chi, Ud_local, gate.qubits, n_qubits)
    Ud = expm(1j * gate.theta * gate.generator)
    return Ud @ chi


def apply_generator(psi: np.ndarray, gate: GateLayer) -> np.ndarray:
    """Compute G @ psi efficiently."""
    if gate.local_generator is not None and gate.qubits:
        n_qubits = int(np.log2(len(psi)))
        return _apply_local_to_sv(psi, gate.local_generator, gate.qubits, n_qubits)
    return gate.generator @ psi


def apply_gate_layer(psi: np.ndarray, gates: list[GateLayer]) -> np.ndarray:
    """Apply a sequence of gates in a layer."""
    for gate in gates:
        psi = apply_gate(psi, gate)
    return psi


# ---------------------------------------------------------------------------
# MCWF dissipative step
# ---------------------------------------------------------------------------

def mcwf_dissipation_step(
    psi: np.ndarray,
    lindblad_ops: list,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool, int]:
    """Apply one MCWF dissipation step (possible quantum jump).

    Supports both dense numpy arrays and scipy sparse operators.
    Returns (new_psi, jumped, jump_operator_index).
    """
    LdL_list = [L.conj().T @ L for L in lindblad_ops]
    dp_list = np.array([
        np.real(np.vdot(psi, np.asarray(LdL @ psi).ravel()))
        for LdL in LdL_list
    ])
    dp_total = dp_list.sum() * dt

    r = rng.uniform()

    if dp_total > r and dp_total > 1e-15:
        probs = dp_list / dp_list.sum()
        l_idx = rng.choice(len(lindblad_ops), p=probs)
        psi_new = np.asarray(lindblad_ops[l_idx] @ psi).ravel()
        norm = np.linalg.norm(psi_new)
        if norm > 1e-15:
            return psi_new / norm, True, l_idx
        return psi / np.linalg.norm(psi), False, -1

    H_eff_part = np.zeros_like(psi)
    for LdL in LdL_list:
        H_eff_part -= 0.5 * dt * np.asarray(LdL @ psi).ravel()
    psi_new = psi + H_eff_part
    norm = np.linalg.norm(psi_new)
    return psi_new / max(norm, 1e-30), False, -1


def mcwf_kraus_step(
    psi: np.ndarray,
    kraus_ops: list,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Apply a discrete Kraus channel stochastically (MCWF unravelling).

    Given Kraus operators {K_m} with sum_m K_m^dag K_m = I, sample
    outcome m with probability p_m = ||K_m|psi>||^2 and return the
    post-measurement state K_m|psi>/||K_m|psi>||.

    Supports both dense numpy arrays and scipy sparse operators.
    Returns (new_psi, selected_outcome_index).
    """
    probs = np.array([
        np.real(np.vdot(psi, np.asarray(K.conj().T @ K @ psi).ravel()))
        for K in kraus_ops
    ])
    total = probs.sum()
    if total < 1e-30:
        return psi.copy(), 0
    probs /= total
    m = rng.choice(len(kraus_ops), p=probs)
    psi_new = np.asarray(kraus_ops[m] @ psi).ravel()
    norm = np.linalg.norm(psi_new)
    return psi_new / max(norm, 1e-30), int(m)


# ---------------------------------------------------------------------------
# Krotov result container
# ---------------------------------------------------------------------------

@dataclass
class GateKrotovResult:
    """Result of a gate-circuit Krotov optimisation."""
    gate_params: list[list[float]]
    errors: list[float]
    errors_functional: list[float]
    fidelities: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core Krotov optimiser for gate circuits
# ---------------------------------------------------------------------------

def krotov_gate_circuit(
    psi0: np.ndarray,
    psi_target: np.ndarray,
    circuit_layers: list[list[GateLayer]],
    lindblad_ops: list[np.ndarray],
    dissipation_dt: float,
    n_trajectories: int = 2,
    n_iterations: int = 100,
    lambda_a: float = 1.0,
    seed: int = 42,
    verbose: bool = True,
    kraus_ops_sets: Optional[list[list[np.ndarray]]] = None,
    max_gate_step: Optional[float] = None,
) -> GateKrotovResult:
    r"""Krotov optimisation for an interleaved gate + dissipation circuit.

    Architecture per iteration:
        for each layer L:
            1. Apply gates: psi -> U_1 ... U_k psi
            2. Apply MCWF dissipation step for duration dissipation_dt
            3. (Optional) Apply discrete Kraus channels from kraus_ops_sets
        Then compute fidelity to target.

    Parameters
    ----------
    psi0 : initial state
    psi_target : target state
    circuit_layers : list of gate layers
    lindblad_ops : Lindblad jump operators
    dissipation_dt : duration of dissipative step per layer
    n_trajectories : M
    n_iterations : number of Krotov iterations
    lambda_a : regularisation
    seed : random seed
    verbose : print progress
    kraus_ops_sets : optional list of Kraus operator sets for discrete
        channels (e.g. ancilla-reset gadgets).  Each set is a list of
        d×d matrices [K_0, K_1, ...] satisfying sum K_m† K_m = I.
    """
    rng = np.random.default_rng(seed)
    d = len(psi0)
    n_layers = len(circuit_layers)
    psi_tgt = psi_target / np.linalg.norm(psi_target)
    M = n_trajectories
    _kraus = kraus_ops_sets or []

    errors = []
    errors_func = []
    fidelities_list = []

    if lindblad_ops:
        _LdL_sum = lindblad_ops[0].conj().T @ lindblad_ops[0]
        for _L in lindblad_ops[1:]:
            _LdL_sum = _LdL_sum + _L.conj().T @ _L
        if hasattr(_LdL_sum, 'toarray'):
            _LdL_sum = _LdL_sum.toarray()
    else:
        _LdL_sum = np.zeros((d, d), dtype=complex)

    for iteration in range(n_iterations):
        # ====== Forward propagation of M trajectories (guess) ======
        fwd_trajs = []
        jump_records = []
        kraus_records = []

        for k in range(M):
            psi = psi0.copy()
            states = [psi.copy()]
            jumps_k = []
            kraus_k = []

            for layer_idx, gates in enumerate(circuit_layers):
                for gate in gates:
                    psi = apply_gate(psi, gate)

                psi, jumped, j_idx = mcwf_dissipation_step(
                    psi, lindblad_ops, dissipation_dt, rng,
                )
                jumps_k.append((jumped, j_idx))

                layer_kraus_outcomes = []
                for kset in _kraus:
                    psi, m_idx = mcwf_kraus_step(psi, kset, rng)
                    layer_kraus_outcomes.append(m_idx)
                kraus_k.append(layer_kraus_outcomes)

                states.append(psi.copy())

            fwd_trajs.append(states)
            jump_records.append(jumps_k)
            kraus_records.append(kraus_k)

        # ====== Evaluate JT ======
        tau_list = [np.vdot(psi_tgt, fwd_trajs[k][-1]) for k in range(M)]
        JT_traj = 1.0 - np.mean([abs(tau) ** 2 for tau in tau_list])
        errors_func.append(float(np.real(JT_traj)))

        psi_finals = np.column_stack([fwd_trajs[k][-1] for k in range(M)])
        rho_approx = (psi_finals @ psi_finals.conj().T) / M
        F = float(np.real(psi_tgt.conj() @ rho_approx @ psi_tgt))
        JT = 1.0 - F
        errors.append(JT)
        fidelities_list.append(F)

        if verbose and iteration % max(1, n_iterations // 20) == 0:
            print(f"  iter {iteration:4d}  JT = {JT:.6e}  F = {F:.6f}")

        # ====== Backward propagation ======
        bwd_trajs = []
        for k in range(M):
            tau_k = tau_list[k]
            chi = tau_k * psi_tgt
            chi_states = [None] * (n_layers + 1)
            chi_states[n_layers] = chi.copy()

            for layer_idx in range(n_layers - 1, -1, -1):
                # Undo Kraus steps (reverse order)
                for s_idx in range(len(_kraus) - 1, -1, -1):
                    m_idx = kraus_records[k][layer_idx][s_idx]
                    K_m = _kraus[s_idx][m_idx]
                    chi = np.asarray(K_m.conj().T @ chi).ravel()
                    norm = np.linalg.norm(chi)
                    if norm > 1e-15:
                        chi = chi / norm

                # Undo Lindblad MCWF step
                jumped, j_idx = jump_records[k][layer_idx]
                if jumped and 0 <= j_idx < len(lindblad_ops):
                    L = lindblad_ops[j_idx]
                    chi = np.asarray(L.conj().T @ chi).ravel()
                    norm = np.linalg.norm(chi)
                    if norm > 1e-15:
                        chi = chi / norm
                else:
                    chi = np.asarray(chi + 0.5 * dissipation_dt * (_LdL_sum @ chi)).ravel()
                    norm = np.linalg.norm(chi)
                    if norm > 1e-15:
                        chi = chi / norm

                for gate in reversed(circuit_layers[layer_idx]):
                    chi = apply_gate_adjoint(chi, gate)

                chi_states[layer_idx] = chi.copy()

            bwd_trajs.append(chi_states)

        # ====== Sequential forward update (Krotov) ======
        fwd_new = [[psi0.copy()] for _ in range(M)]

        for layer_idx, gates in enumerate(circuit_layers):
            for gate in gates:
                delta_theta = 0.0
                for k in range(M):
                    psi_k = fwd_new[k][-1]
                    chi_k = bwd_trajs[k][layer_idx]
                    delta_theta += np.imag(np.vdot(chi_k, apply_generator(psi_k, gate)))
                delta_theta /= (M * lambda_a)
                if max_gate_step is not None:
                    delta_theta = np.clip(delta_theta,
                                          -max_gate_step, max_gate_step)
                gate.theta += delta_theta

            for k in range(M):
                psi_k = fwd_new[k][-1]
                for gate in gates:
                    psi_k = apply_gate(psi_k, gate)

                jumped, j_idx = jump_records[k][layer_idx]
                if jumped and 0 <= j_idx < len(lindblad_ops):
                    psi_jumped = np.asarray(lindblad_ops[j_idx] @ psi_k).ravel()
                    norm = np.linalg.norm(psi_jumped)
                    if norm > 1e-15:
                        psi_k = psi_jumped / norm
                else:
                    psi_k = np.asarray(psi_k - 0.5 * dissipation_dt * (_LdL_sum @ psi_k)).ravel()
                    norm = np.linalg.norm(psi_k)
                    psi_k = psi_k / max(norm, 1e-30)

                for s_idx, kset in enumerate(_kraus):
                    m_idx = kraus_records[k][layer_idx][s_idx]
                    psi_k = np.asarray(kset[m_idx] @ psi_k).ravel()
                    norm = np.linalg.norm(psi_k)
                    psi_k = psi_k / max(norm, 1e-30)

                fwd_new[k].append(psi_k.copy())

    gate_params = [[g.theta for g in layer] for layer in circuit_layers]

    return GateKrotovResult(
        gate_params=gate_params,
        errors=errors,
        errors_functional=errors_func,
        fidelities=fidelities_list,
    )


# ---------------------------------------------------------------------------
# Density-matrix evaluation (exact, for validation)
# ---------------------------------------------------------------------------

def evaluate_circuit_dm(
    psi0: np.ndarray,
    psi_target: np.ndarray,
    circuit_layers: list[list[GateLayer]],
    lindblad_ops: list[np.ndarray],
    dissipation_dt: float,
    kraus_ops_sets: Optional[list[list[np.ndarray]]] = None,
) -> dict:
    """Evaluate the circuit with exact density-matrix propagation.

    Uses Lindblad master equation for the dissipative steps and
    exact Kraus-map application for any discrete channels.
    """
    d = len(psi0)
    rho = np.outer(psi0, psi0.conj())
    psi_tgt = psi_target / np.linalg.norm(psi_target)
    _kraus = kraus_ops_sets or []

    n_qubits = int(np.log2(d))
    for layer_idx, gates in enumerate(circuit_layers):
        for gate in gates:
            if gate.local_generator is not None and gate.qubits:
                U_local = expm(-1j * gate.theta * gate.local_generator)
                if len(gate.qubits) == 1:
                    rho = _apply_1q_unitary_to_dm(rho, U_local,
                                                  gate.qubits[0], n_qubits)
                else:
                    rho = _apply_2q_unitary_to_dm(rho, U_local,
                                                  gate.qubits[0], gate.qubits[1],
                                                  n_qubits)
            else:
                U_g = expm(-1j * gate.theta * gate.generator)
                rho = U_g @ rho @ U_g.conj().T

        from .utils import Rank1Op
        rank1_ops = [L for L in lindblad_ops if isinstance(L, Rank1Op)]
        other_ops = [L for L in lindblad_ops if not isinstance(L, Rank1Op)]

        if rank1_ops:
            V = np.column_stack([L.row for L in rank1_ops])          # (d, k)
            gammas = np.array([abs(L._scale)**2 for L in rank1_ops]) # (k,)
            tgt_col = rank1_ops[0].col
            VtR = V.conj().T @ rho                                   # (k, d)
            overlaps = np.sum(VtR * V.T, axis=1).real                 # (k,)
            s = float(np.dot(gammas, overlaps))
            VgR = VtR * gammas[:, None]                               # (k, d)
            P_rho = V @ VgR                                           # (d, d)
            rho_P = P_rho.conj().T
            rho = rho + dissipation_dt * (
                s * np.outer(tgt_col, tgt_col.conj())
                - 0.5 * P_rho - 0.5 * rho_P
            )
            del VtR, VgR, P_rho, rho_P

        for L in other_ops:
            LdL = L.conj().T @ L
            L_rho = np.asarray(L @ rho) if hasattr(L, 'toarray') else L @ rho
            LdL_rho = np.asarray(LdL @ rho) if hasattr(LdL, 'toarray') else LdL @ rho
            rho_LdL = np.asarray(rho @ LdL) if hasattr(LdL, 'toarray') else rho @ LdL
            L_dag = L.conj().T
            drho = (
                L_rho @ (L_dag.toarray() if hasattr(L_dag, 'toarray') else L_dag)
                - 0.5 * LdL_rho
                - 0.5 * rho_LdL
            )
            rho = rho + dissipation_dt * drho

        for kset in _kraus:
            rho_new = np.zeros_like(rho)
            for K in kset:
                K_rho = np.asarray(K @ rho) if hasattr(K, 'toarray') else K @ rho
                K_dag = K.conj().T
                rho_new += K_rho @ (K_dag.toarray() if hasattr(K_dag, 'toarray') else K_dag)
            rho = rho_new

        rho = 0.5 * (rho + rho.conj().T)
        tr = np.trace(rho)
        if abs(tr) > 1e-15:
            rho /= tr

    F = float(np.real(psi_tgt.conj() @ rho @ psi_tgt))
    return {
        "fidelity": F,
        "error": 1.0 - F,
        "trace": float(np.real(np.trace(rho))),
        "purity": float(np.real(np.trace(rho @ rho))),
        "rho": rho,
    }
