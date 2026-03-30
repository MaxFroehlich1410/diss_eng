r"""Cascade quantum network model from Goerz & Jacobs, arXiv:1801.04382v2.

Section 3.1: N cavities each containing a 3-level atom (Lambda config),
connected via a unidirectional traveling-wave field.

After adiabatic elimination of the excited atomic level |r>, each node
becomes an effective 2-level system coupled to its cavity mode.

Since the Hamiltonian preserves total excitation, we restrict to the
single-excitation subspace plus the ground state.  For N nodes this gives
dimension d = 2N + 1 instead of 4^N.

Encoding
--------
We use a compact Hilbert space of dimension 2N+1:
  - Index 0           : global ground state  |gg...g, 00...0>
  - Index 2i-1  (odd) : qubit i excited       |g...e_i...g, 00...0>
  - Index 2i    (even): cavity i excited       |gg...g, 0...1_i...0>
for i = 1, ..., N.

Parameters (from paper, dimensionless units with energies in g)
---------------------------------------------------------------
  Delta = 100 g   (detuning, large for adiabatic elimination)
  kappa = g       (cavity decay rate)
  g = 1           (atom-cavity coupling, sets the energy scale)
  T = 5 hbar/g    (for 2 nodes), T = 50 hbar/g (for 20 nodes)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Default parameters (dimensionless, energy in units of g)
# ---------------------------------------------------------------------------

DEFAULT_PARAMS = {
    "g_coupling": 1.0,
    "Delta": 100.0,
    "kappa": 1.0,
}


def hilbert_dim(n_nodes: int) -> int:
    """Dimension of the single-excitation-plus-ground subspace: 2N+1."""
    return 2 * n_nodes + 1


def qubit_index(node: int) -> int:
    """Index of |qubit_i excited> in the compact basis (1-indexed node)."""
    return 2 * node - 1


def cavity_index(node: int) -> int:
    """Index of |cavity_i excited> in the compact basis (1-indexed node)."""
    return 2 * node


# ---------------------------------------------------------------------------
# Operators in the compact single-excitation basis
# ---------------------------------------------------------------------------

def _sigma_eg(node: int, d: int) -> np.ndarray:
    r"""sigma_{e,g}^{(i)} raising operator: |qubit_i> <- |ground>.

    In the single-excitation subspace this couples the global ground
    state (index 0) to the qubit-excited state.
    """
    op = np.zeros((d, d), dtype=complex)
    qi = qubit_index(node)
    op[qi, 0] = 1.0
    return op


def _a_lower(node: int, d: int) -> np.ndarray:
    r"""Cavity lowering operator a_i: |ground> <- |cavity_i excited>.

    In the compact basis this takes |cavity_i> to |ground>.
    """
    op = np.zeros((d, d), dtype=complex)
    ci = cavity_index(node)
    op[0, ci] = 1.0
    return op


def _a_raise(node: int, d: int) -> np.ndarray:
    """Cavity raising operator a_i^dag."""
    return _a_lower(node, d).conj().T


def _proj_g_qubit(node: int, d: int) -> np.ndarray:
    r"""Projector onto qubit-ground for node i.

    In the compact basis, the qubit at node i is in ground state for
    every basis vector *except* the one with qubit i excited.
    """
    op = np.eye(d, dtype=complex)
    qi = qubit_index(node)
    op[qi, qi] = 0.0
    return op


def _n_cavity(node: int, d: int) -> np.ndarray:
    """Cavity number operator a_i^dag a_i (0 or 1 in single-exc subspace)."""
    ci = cavity_index(node)
    op = np.zeros((d, d), dtype=complex)
    op[ci, ci] = 1.0
    return op


# ---------------------------------------------------------------------------
# Hamiltonian construction  (Eqs. 17-22)
# ---------------------------------------------------------------------------

def build_drift_hamiltonian(
    n_nodes: int,
    g_coupling: float = 1.0,
    Delta: float = 100.0,
    kappa: float = 1.0,
) -> np.ndarray:
    r"""Time-independent part of the network Hamiltonian.

    H_0 = sum_i H_0^{(i)} + sum_{i<j} H^{(i,j)}

    H_0^{(i)} = -(g^2/Delta) a_i^dag a_i
                + (g^2/Delta) Pi_g^{(i)} a_i^dag a_i        (Eq. 18)

    H^{(i,j)} = i kappa (a_i^dag a_j - a_j^dag a_i)         (Eq. 22)
    """
    d = hilbert_dim(n_nodes)
    H = np.zeros((d, d), dtype=complex)
    g2_Delta = g_coupling ** 2 / Delta

    for i in range(1, n_nodes + 1):
        n_cav = _n_cavity(i, d)
        proj_g = _proj_g_qubit(i, d)
        H += -g2_Delta * n_cav + g2_Delta * (proj_g @ n_cav)

    for i in range(1, n_nodes + 1):
        for j in range(i + 1, n_nodes + 1):
            ai_dag = _a_raise(i, d)
            aj = _a_lower(j, d)
            H += 1j * kappa * (ai_dag @ aj) + (-1j * kappa) * (aj.conj().T @ ai_dag.conj().T)

    return H


def build_control_hamiltonians(
    n_nodes: int,
    g_coupling: float = 1.0,
    Delta: float = 100.0,
) -> list[np.ndarray]:
    r"""Control Hamiltonians H_d^{(i)} for each node (Eq. 19).

    H_d^{(i)} = -i Omega_i(t) g / (2 Delta)  (sigma_eg a_i - h.c.)

    We factor out Omega_i(t) and return H_i such that
        H_total = H_drift + sum_i  Omega_i(t) * H_i

    where H_i = -i g/(2*Delta) (sigma_eg a_i - a_i^dag sigma_ge).
    """
    d = hilbert_dim(n_nodes)
    g_over_2Delta = g_coupling / (2 * Delta)
    H_ctrl = []

    for i in range(1, n_nodes + 1):
        seg = _sigma_eg(i, d)
        ai = _a_lower(i, d)
        Hi = -1j * g_over_2Delta * (seg @ ai - ai.conj().T @ seg.conj().T)
        H_ctrl.append(Hi)

    return H_ctrl


def build_total_hamiltonian(
    n_nodes: int,
    controls: np.ndarray | list[float],
    g_coupling: float = 1.0,
    Delta: float = 100.0,
    kappa: float = 1.0,
) -> np.ndarray:
    """Build H(t) = H_drift + sum_i Omega_i * H_ctrl_i for given control values."""
    H = build_drift_hamiltonian(n_nodes, g_coupling, Delta, kappa)
    H_ctrls = build_control_hamiltonians(n_nodes, g_coupling, Delta)
    for omega_i, Hi in zip(controls, H_ctrls):
        H += omega_i * Hi
    return H


def build_network_lindblad_op(
    n_nodes: int,
    kappa: float = 1.0,
) -> np.ndarray:
    r"""Total Lindblad operator L = sum_i sqrt(2 kappa) a_i  (Eq. 20, 23)."""
    d = hilbert_dim(n_nodes)
    L = np.zeros((d, d), dtype=complex)
    for i in range(1, n_nodes + 1):
        L += np.sqrt(2 * kappa) * _a_lower(i, d)
    return L


# ---------------------------------------------------------------------------
# Initial and target states  (Section 3.2)
# ---------------------------------------------------------------------------

def initial_state(n_nodes: int) -> np.ndarray:
    r"""|Psi(0)> = |e g...g, 0...0>: qubit 1 excited, rest ground."""
    d = hilbert_dim(n_nodes)
    psi = np.zeros(d, dtype=complex)
    psi[qubit_index(1)] = 1.0
    return psi


def dark_state_target(n_nodes: int) -> np.ndarray:
    r"""Target: (1/sqrt(N)) sum_i |g...e_i...g, 0...0>.

    Equal superposition of single-qubit excitations (W-type state).
    """
    d = hilbert_dim(n_nodes)
    psi = np.zeros(d, dtype=complex)
    for i in range(1, n_nodes + 1):
        psi[qubit_index(i)] = 1.0
    return psi / np.linalg.norm(psi)


def dark_state_condition(psi: np.ndarray, L: np.ndarray) -> float:
    r"""Evaluate <psi|L^dag L|psi> -- should be zero in a dark state."""
    LdL = L.conj().T @ L
    return float(np.real(np.vdot(psi, LdL @ psi)))


# ---------------------------------------------------------------------------
# Convenience: time-dependent Hamiltonian factory
# ---------------------------------------------------------------------------

class NetworkHamiltonian:
    """Callable that returns H(t) given time-dependent control pulses.

    Parameters
    ----------
    n_nodes : int
    pulse_funcs : list of callable(t) -> float
        One function per node returning Omega_i(t).
    """

    def __init__(
        self,
        n_nodes: int,
        pulse_funcs: list,
        g_coupling: float = 1.0,
        Delta: float = 100.0,
        kappa: float = 1.0,
    ):
        self.n_nodes = n_nodes
        self.pulse_funcs = pulse_funcs
        self.H_drift = build_drift_hamiltonian(n_nodes, g_coupling, Delta, kappa)
        self.H_ctrls = build_control_hamiltonians(n_nodes, g_coupling, Delta)

    def __call__(self, t: float) -> np.ndarray:
        H = self.H_drift.copy()
        for omega_func, Hi in zip(self.pulse_funcs, self.H_ctrls):
            H += omega_func(t) * Hi
        return H


class PiecewiseConstantHamiltonian:
    """H(t) with piecewise-constant control amplitudes on a time grid.

    Parameters
    ----------
    n_nodes : int
    controls : ndarray (n_controls, nt)
        Control amplitudes Omega_i(t_j) for each node and time step.
    times : ndarray (nt,)
        Time grid.
    """

    def __init__(
        self,
        n_nodes: int,
        controls: np.ndarray,
        times: np.ndarray,
        g_coupling: float = 1.0,
        Delta: float = 100.0,
        kappa: float = 1.0,
    ):
        self.n_nodes = n_nodes
        self.controls = np.asarray(controls, dtype=float)
        self.times = np.asarray(times, dtype=float)
        self.H_drift = build_drift_hamiltonian(n_nodes, g_coupling, Delta, kappa)
        self.H_ctrls = build_control_hamiltonians(n_nodes, g_coupling, Delta)

    def __call__(self, t: float) -> np.ndarray:
        idx = np.searchsorted(self.times, t, side="right") - 1
        idx = np.clip(idx, 0, len(self.times) - 1)
        H = self.H_drift.copy()
        for c in range(self.controls.shape[0]):
            H += self.controls[c, idx] * self.H_ctrls[c]
        return H

    def get_controls_at(self, time_idx: int) -> np.ndarray:
        """Return control vector at a specific time index."""
        return self.controls[:, time_idx]
