r"""Extended Krotov optimiser: joint gate + Lindblad-rate optimisation.

Extends the gate-circuit Krotov method so that the **per-layer Lindblad
dissipation rates** gamma_k^{(l)} are treated as additional Krotov
control variables alongside the existing gate parameters theta_j.

Mathematical basis
------------------
At each circuit layer *l* the dynamics are:

  1. Unitary gates:   |psi'> = U_gates |psi>
  2. Dissipation:     MCWF step with operators {sqrt(gamma_k^{(l)}) L_k}

The no-jump propagation (first order in dt) reads

  |psi_out> ~ |psi'> - (dt/2) sum_k gamma_k L_k†L_k |psi'>

Differentiating w.r.t. the rate gamma_k:

  d|psi_out>/d(gamma_k) = -(dt/2) L_k†L_k |psi'>

For jump trajectories the post-jump state L_j|psi'>/||...|| is
independent of gamma_k (the rate cancels in normalisation), giving
zero contribution.

The resulting Krotov update rule for rates is:

  Delta gamma_k^{(l)} =
    -(dt / 2 M lambda_gamma)
    * sum_{m : no-jump at l}  Re< chi'_m | L_k†L_k | psi'_m >

where psi'_m is the *updated* forward state after gates and chi'_m is
the backward co-state at the intermediate point (after gates, before
dissipation), both at layer l.

Physical constraint:  gamma_k >= 0  enforced by projection after update.

Supported modes
---------------
  optimize_gates=True,  optimize_rates=False  → Hamiltonian-only Krotov
  optimize_gates=False, optimize_rates=True   → rate-only Krotov
  optimize_gates=True,  optimize_rates=True   → joint Krotov
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
from scipy.linalg import expm

from . import utils
from .gate_circuit_krotov import (
    GateLayer,
    apply_gate,
    apply_gate_adjoint,
    apply_generator,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DissipationBasis:
    r"""Fixed jump-operator basis with optimisable per-layer rates.

    bare_ops : list of K  d x d  matrices  [L_1, ..., L_K]
        Unit-scale jump operators (rate-free).
    rates : ndarray of shape (n_layers, K)
        gamma_k^{(l)} >= 0 for each layer and operator.
    names : list of K human-readable labels
    """
    bare_ops: list[np.ndarray]
    rates: np.ndarray
    names: list[str] = field(default_factory=list)

    @property
    def n_ops(self) -> int:
        return len(self.bare_ops)

    @property
    def n_layers(self) -> int:
        return self.rates.shape[0]

    def effective_ops(self, layer: int) -> list[np.ndarray]:
        """Return sqrt(gamma_k^{(l)}) * L_k for each k at the given layer."""
        return [
            np.sqrt(max(g, 0.0)) * L
            for g, L in zip(self.rates[layer], self.bare_ops)
        ]


@dataclass
class JointKrotovResult:
    """Result container for the extended Krotov optimisation."""
    gate_params: list[list[float]]
    final_rates: np.ndarray
    rate_history: list[np.ndarray]
    errors: list[float]
    fidelities: list[float]
    errors_functional: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dissipation basis builders
# ---------------------------------------------------------------------------

def build_dissipation_basis(
    n_qubits: int,
    n_layers: int,
    include_amp_damp: bool = True,
    include_dephasing: bool = True,
    include_zz: bool = False,
    zz_edges: list[tuple[int, int]] | None = None,
    init_rate: float = 0.1,
) -> DissipationBasis:
    """Build a fixed basis of jump operators with uniform initial rates.

    Parameters
    ----------
    n_qubits : number of system qubits
    n_layers : number of circuit layers (determines rate array shape)
    include_amp_damp : add single-qubit sigma^- on each qubit
    include_dephasing : add single-qubit Z on each qubit
    include_zz : add two-qubit Z_i Z_j
    zz_edges : qubit pairs for ZZ; defaults to all-to-all
    init_rate : initial value for all gamma_k^{(l)}
    """
    bare_ops: list[np.ndarray] = []
    names: list[str] = []

    if include_amp_damp:
        for q, op in enumerate(utils.amplitude_damping_operators(n_qubits)):
            bare_ops.append(op)
            names.append(f"AD_q{q}")

    if include_dephasing:
        for q, op in enumerate(utils.dephasing_operators(n_qubits)):
            bare_ops.append(op)
            names.append(f"Deph_q{q}")

    if include_zz:
        if zz_edges is None:
            zz_edges = list(combinations(range(n_qubits), 2))
        for (qi, qj), op in zip(
            zz_edges,
            utils.zz_dephasing_operators(n_qubits, zz_edges),
        ):
            bare_ops.append(op)
            names.append(f"ZZ_{qi}_{qj}")

    K = len(bare_ops)
    rates = np.full((n_layers, K), init_rate, dtype=float)
    return DissipationBasis(bare_ops=bare_ops, rates=rates, names=names)


# ---------------------------------------------------------------------------
# MCWF step with explicit (bare ops, rates) separation
# ---------------------------------------------------------------------------

def _mcwf_step_rates(
    psi: np.ndarray,
    bare_ops: list[np.ndarray],
    LdL_bare: list[np.ndarray],
    rates: np.ndarray,
    dt: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, bool, int]:
    """MCWF dissipation step with separated bare operators and rates.

    Jump probability for channel k:  dp_k = gamma_k <psi|L_k†L_k|psi> dt
    Post-jump state: L_k|psi> / ||L_k|psi>||  (rate cancels)
    No-jump: |psi> - (dt/2) sum_k gamma_k L_k†L_k |psi>
    """
    dp_list = np.array([
        g * np.real(np.vdot(psi, LdL @ psi))
        for g, LdL in zip(rates, LdL_bare)
    ])
    dp_list = np.maximum(dp_list, 0.0)
    dp_total = dp_list.sum() * dt

    r = rng.uniform()

    if dp_total > r and dp_total > 1e-15:
        psum = dp_list.sum()
        if psum < 1e-30:
            return psi / np.linalg.norm(psi), False, -1
        probs = dp_list / psum
        l_idx = rng.choice(len(bare_ops), p=probs)
        psi_new = bare_ops[l_idx] @ psi
        norm = np.linalg.norm(psi_new)
        if norm > 1e-15:
            return psi_new / norm, True, l_idx
        return psi / np.linalg.norm(psi), False, -1

    H_eff_part = np.zeros_like(psi)
    for g, LdL in zip(rates, LdL_bare):
        H_eff_part -= 0.5 * dt * g * (LdL @ psi)
    psi_new = psi + H_eff_part
    norm = np.linalg.norm(psi_new)
    return psi_new / max(norm, 1e-30), False, -1


def _weighted_LdL_sum(rates: np.ndarray,
                      LdL_bare: list[np.ndarray]) -> np.ndarray:
    """Compute sum_k gamma_k L_k†L_k for a given rate vector."""
    result = rates[0] * LdL_bare[0]
    for g, M in zip(rates[1:], LdL_bare[1:]):
        result = result + g * M
    return result


# ---------------------------------------------------------------------------
# Core: joint Krotov optimiser
# ---------------------------------------------------------------------------

def krotov_joint(
    psi0: np.ndarray,
    psi_target: np.ndarray,
    circuit_layers: list[list[GateLayer]],
    diss_basis: DissipationBasis,
    dissipation_dt: float,
    *,
    n_trajectories: int = 4,
    n_iterations: int = 100,
    lambda_gates: float = 0.3,
    lambda_rates: float = 1.0,
    optimize_gates: bool = True,
    optimize_rates: bool = True,
    max_gate_step: float | None = None,
    max_rate_step: float | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> JointKrotovResult:
    r"""Joint Krotov optimisation of gate parameters AND Lindblad rates.

    Architecture per layer l:
        1. Apply parameterised gates  U_j(theta_j)
        2. Apply MCWF dissipation with rates gamma_k^{(l)} and bare ops L_k

    Parameters
    ----------
    psi0, psi_target : initial / target states
    circuit_layers : list of gate layers (as in krotov_gate_circuit)
    diss_basis : DissipationBasis with bare operators and initial rates
    dissipation_dt : duration of dissipative step per layer
    n_trajectories : M  (number of MCWF trajectories)
    n_iterations : total Krotov iterations
    lambda_gates : regularisation for gate parameters
    lambda_rates : regularisation for rate controls
    optimize_gates : if True, update gate thetas (set False for rate-only)
    optimize_rates : if True, update gamma_k (set False for gates-only)
    max_gate_step : optional per-gate update clip
    max_rate_step : optional per-rate update clip
    seed : random seed
    verbose : print progress
    """
    rng = np.random.default_rng(seed)
    d = len(psi0)
    n_layers = len(circuit_layers)
    psi_tgt = psi_target / np.linalg.norm(psi_target)
    M = n_trajectories
    K = diss_basis.n_ops

    assert diss_basis.n_layers == n_layers, (
        f"DissipationBasis has {diss_basis.n_layers} layers but circuit "
        f"has {n_layers}")

    # Pre-compute L_k†L_k for each bare operator (fixed across iterations)
    LdL_bare = [L.conj().T @ L for L in diss_basis.bare_ops]

    errors = []
    errors_func = []
    fidelities_list = []
    rate_history = [diss_basis.rates.copy()]

    for iteration in range(n_iterations):
        # Current rates snapshot (used for forward+backward with old controls)
        rates_old = diss_basis.rates.copy()

        # ====== Forward propagation (M trajectories, OLD controls) ======
        fwd_trajs = []          # fwd_trajs[k][l] = state AFTER layer l
        jump_records = []       # jump_records[k][l] = (jumped, j_idx)

        for k in range(M):
            psi = psi0.copy()
            states = [psi.copy()]
            jumps_k = []

            for li, gates in enumerate(circuit_layers):
                for gate in gates:
                    psi = apply_gate(psi, gate)

                psi, jumped, j_idx = _mcwf_step_rates(
                    psi, diss_basis.bare_ops, LdL_bare,
                    rates_old[li], dissipation_dt, rng,
                )
                jumps_k.append((jumped, j_idx))
                states.append(psi.copy())

            fwd_trajs.append(states)
            jump_records.append(jumps_k)

        # ====== Evaluate cost ======
        tau_list = [np.vdot(psi_tgt, fwd_trajs[k][-1]) for k in range(M)]
        JT_traj = 1.0 - np.mean([abs(tau) ** 2 for tau in tau_list])
        errors_func.append(float(np.real(JT_traj)))

        psi_finals = np.column_stack([fwd_trajs[k][-1] for k in range(M)])
        rho_approx = (psi_finals @ psi_finals.conj().T) / M
        F = float(np.real(psi_tgt.conj() @ rho_approx @ psi_tgt))
        errors.append(1.0 - F)
        fidelities_list.append(F)

        if verbose and iteration % max(1, n_iterations // 20) == 0:
            print(f"  iter {iteration:4d}  JT = {1-F:.6e}  F = {F:.6f}")

        # ====== Backward propagation (OLD controls) ======
        # Store both chi at layer start and chi at intermediate point
        bwd_start = []          # bwd_start[k][l]  = chi at START of layer l
        bwd_intermediate = []   # bwd_intermediate[k][l] = chi AFTER gates,
        #                         BEFORE dissipation at layer l

        for k in range(M):
            tau_k = tau_list[k]
            chi = tau_k * psi_tgt
            chi_at_start = [None] * (n_layers + 1)
            chi_at_mid = [None] * n_layers
            chi_at_start[n_layers] = chi.copy()

            for li in range(n_layers - 1, -1, -1):
                # Undo MCWF step (backward through dissipation)
                jumped, j_idx = jump_records[k][li]
                if jumped and 0 <= j_idx < K:
                    L = diss_basis.bare_ops[j_idx]
                    chi = L.conj().T @ chi
                    norm = np.linalg.norm(chi)
                    if norm > 1e-15:
                        chi = chi / norm
                else:
                    LdL_sum = _weighted_LdL_sum(rates_old[li], LdL_bare)
                    chi = chi + 0.5 * dissipation_dt * (LdL_sum @ chi)
                    norm = np.linalg.norm(chi)
                    if norm > 1e-15:
                        chi = chi / norm

                # chi is now at the intermediate point
                chi_at_mid[li] = chi.copy()

                # Undo gates
                for gate in reversed(circuit_layers[li]):
                    chi = apply_gate_adjoint(chi, gate)

                chi_at_start[li] = chi.copy()

            bwd_start.append(chi_at_start)
            bwd_intermediate.append(chi_at_mid)

        # ====== Sequential forward update (Krotov) ======
        fwd_new = [[psi0.copy()] for _ in range(M)]

        for li, gates in enumerate(circuit_layers):
            # --- Step A: update gate parameters (using OLD chi, NEW psi) ---
            if optimize_gates:
                for gate in gates:
                    delta_theta = 0.0
                    for k in range(M):
                        psi_k = fwd_new[k][-1]
                        chi_k = bwd_start[k][li]
                        delta_theta += np.imag(
                            np.vdot(chi_k, apply_generator(psi_k, gate)))
                    delta_theta /= (M * lambda_gates)
                    if max_gate_step is not None:
                        delta_theta = np.clip(
                            delta_theta, -max_gate_step, max_gate_step)
                    gate.theta += delta_theta

            # --- Propagate forward through UPDATED gates ---
            psi_post_gates = []
            for k in range(M):
                psi_k = fwd_new[k][-1]
                for gate in gates:
                    psi_k = apply_gate(psi_k, gate)
                psi_post_gates.append(psi_k)

            # --- Step B: update rates at this layer ---
            if optimize_rates:
                for op_idx in range(K):
                    delta_gamma = 0.0
                    n_nojump = 0
                    for k in range(M):
                        jumped, _ = jump_records[k][li]
                        if jumped:
                            continue
                        n_nojump += 1
                        chi_mid = bwd_intermediate[k][li]
                        psi_mid = psi_post_gates[k]
                        LdL_psi = LdL_bare[op_idx] @ psi_mid
                        delta_gamma += np.real(np.vdot(chi_mid, LdL_psi))

                    delta_gamma *= -dissipation_dt / (2.0 * M * lambda_rates)
                    if max_rate_step is not None:
                        delta_gamma = np.clip(
                            delta_gamma, -max_rate_step, max_rate_step)
                    diss_basis.rates[li, op_idx] = max(
                        0.0, diss_basis.rates[li, op_idx] + delta_gamma)

            # --- Propagate forward through dissipation (UPDATED rates) ---
            for k in range(M):
                psi_k = psi_post_gates[k]
                jumped, j_idx = jump_records[k][li]

                if jumped and 0 <= j_idx < K:
                    psi_jumped = diss_basis.bare_ops[j_idx] @ psi_k
                    norm = np.linalg.norm(psi_jumped)
                    if norm > 1e-15:
                        psi_k = psi_jumped / norm
                else:
                    LdL_sum = _weighted_LdL_sum(
                        diss_basis.rates[li], LdL_bare)
                    psi_k = psi_k - 0.5 * dissipation_dt * (LdL_sum @ psi_k)
                    norm = np.linalg.norm(psi_k)
                    psi_k = psi_k / max(norm, 1e-30)

                fwd_new[k].append(psi_k.copy())

        rate_history.append(diss_basis.rates.copy())

    gate_params = [[g.theta for g in layer] for layer in circuit_layers]

    return JointKrotovResult(
        gate_params=gate_params,
        final_rates=diss_basis.rates.copy(),
        rate_history=rate_history,
        errors=errors,
        fidelities=fidelities_list,
        errors_functional=errors_func,
    )


# ---------------------------------------------------------------------------
# Density-matrix evaluation with per-layer rates
# ---------------------------------------------------------------------------

def evaluate_circuit_dm_rates(
    psi0: np.ndarray,
    psi_target: np.ndarray,
    circuit_layers: list[list[GateLayer]],
    diss_basis: DissipationBasis,
    dissipation_dt: float,
) -> dict:
    """Exact density-matrix propagation with per-layer Lindblad rates.

    Uses Euler-step Lindblad master equation:
        drho = dt * sum_k gamma_k^{(l)} (L_k rho L_k† - 0.5{L_k†L_k, rho})
    """
    rho = np.outer(psi0, psi0.conj())
    psi_tgt = psi_target / np.linalg.norm(psi_target)

    n_qubits = int(np.log2(len(psi0)))
    for li, gates in enumerate(circuit_layers):
        for gate in gates:
            if gate.local_generator is not None and gate.qubits:
                from .gate_circuit_krotov import (
                    _apply_1q_unitary_to_dm,
                    _apply_2q_unitary_to_dm,
                )
                U_local = expm(-1j * gate.theta * gate.local_generator)
                if len(gate.qubits) == 1:
                    rho = _apply_1q_unitary_to_dm(
                        rho, U_local, gate.qubits[0], n_qubits)
                else:
                    rho = _apply_2q_unitary_to_dm(
                        rho, U_local, gate.qubits[0], gate.qubits[1],
                        n_qubits)
            else:
                U_g = expm(-1j * gate.theta * gate.generator)
                rho = U_g @ rho @ U_g.conj().T

        for op_idx, L in enumerate(diss_basis.bare_ops):
            g = diss_basis.rates[li, op_idx]
            if g < 1e-30:
                continue
            LdL = L.conj().T @ L
            L_rho = L @ rho
            drho = g * (
                L_rho @ L.conj().T
                - 0.5 * (LdL @ rho)
                - 0.5 * (rho @ LdL)
            )
            rho = rho + dissipation_dt * drho

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
