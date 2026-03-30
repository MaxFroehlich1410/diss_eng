r"""Utility functions for MCWF-Krotov: states, fidelities, operators."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, null_space


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def basis_state(d: int, index: int) -> np.ndarray:
    """Computational basis state |index> in dimension d."""
    psi = np.zeros(d, dtype=complex)
    psi[index] = 1.0
    return psi


def random_statevector(n_qubits: int, seed: int | None = None) -> np.ndarray:
    """Haar-random pure state on n_qubits qubits."""
    rng = np.random.default_rng(seed)
    d = 2 ** n_qubits
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return psi / np.linalg.norm(psi)


def w_state(n_qubits: int) -> np.ndarray:
    r"""W state: (1/sqrt(N)) sum_i |0...1_i...0>.

    The single-excitation symmetric state.
    """
    d = 2 ** n_qubits
    psi = np.zeros(d, dtype=complex)
    for i in range(n_qubits):
        idx = 1 << i
        psi[idx] = 1.0
    return psi / np.linalg.norm(psi)


def ghz_state(n_qubits: int) -> np.ndarray:
    r"""GHZ state: (1/sqrt(2)) (|00...0> + |11...1>)."""
    d = 2 ** n_qubits
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0
    psi[d - 1] = 1.0
    return psi / np.linalg.norm(psi)


# ---------------------------------------------------------------------------
# Density matrix operations
# ---------------------------------------------------------------------------

def pure_state_dm(psi: np.ndarray) -> np.ndarray:
    r"""Convert |psi> to |psi><psi|."""
    psi = np.asarray(psi, dtype=complex).ravel()
    return np.outer(psi, psi.conj())


def fidelity_to_pure(rho: np.ndarray, psi: np.ndarray) -> float:
    r"""F = <psi|rho|psi>."""
    psi = np.asarray(psi, dtype=complex).ravel()
    return float(np.real(psi.conj() @ rho @ psi))


def overlap(psi1: np.ndarray, psi2: np.ndarray) -> complex:
    """<psi1|psi2>."""
    return np.vdot(psi1, psi2)


def trace(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho)))


def purity(rho: np.ndarray) -> float:
    return float(np.real(np.trace(rho @ rho)))


def is_physical(rho: np.ndarray, atol: float = 1e-8) -> dict:
    """Check whether rho is a valid density matrix."""
    tr = trace(rho)
    hermitian = bool(np.allclose(rho, rho.conj().T, atol=atol))
    eigs = np.linalg.eigvalsh(rho)
    min_eig = float(eigs[0])
    psd = bool(min_eig >= -atol)
    valid = abs(tr - 1.0) < atol and hermitian and psd
    return {
        "trace": tr,
        "hermitian": hermitian,
        "positive_semidefinite": psd,
        "min_eigenvalue": min_eig,
        "is_valid": valid,
    }


# ---------------------------------------------------------------------------
# Operator construction helpers
# ---------------------------------------------------------------------------

def tensor(*ops: np.ndarray) -> np.ndarray:
    """Kronecker product of an arbitrary number of operators."""
    result = ops[0]
    for op in ops[1:]:
        result = np.kron(result, op)
    return result


def embed_operator(op: np.ndarray, qubit: int, n_qubits: int) -> np.ndarray:
    """Embed a single-qubit operator into the full Hilbert space."""
    ops = []
    for q in range(n_qubits):
        ops.append(op if q == qubit else np.eye(op.shape[0], dtype=complex))
    return tensor(*ops)


# ---------------------------------------------------------------------------
# Standard single-qubit operators
# ---------------------------------------------------------------------------

SIGMA_X = np.array([[0, 1], [1, 0]], dtype=complex)
SIGMA_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
SIGMA_Z = np.array([[1, 0], [0, -1]], dtype=complex)
SIGMA_PLUS = np.array([[0, 1], [0, 0]], dtype=complex)
SIGMA_MINUS = np.array([[0, 0], [1, 0]], dtype=complex)
PROJ_E = np.array([[1, 0], [0, 0]], dtype=complex)  # |e><e| = |0><0|
PROJ_G = np.array([[0, 0], [0, 1]], dtype=complex)  # |g><g| = |1><1|


# ---------------------------------------------------------------------------
# Lindblad operator sets
# ---------------------------------------------------------------------------

def target_cooling_operators(psi_target: np.ndarray) -> list[np.ndarray]:
    r"""Operators L_k = |psi*><psi_k^perp| that cool toward |psi*>."""
    psi = np.asarray(psi_target, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    orth = null_space(psi.conj().reshape(1, -1))
    ops = []
    for k in range(orth.shape[1]):
        L_k = np.outer(psi, orth[:, k].conj())
        ops.append(L_k)
    return ops


# ---------------------------------------------------------------------------
# Memory-efficient rank-1 operators for large Hilbert spaces
# ---------------------------------------------------------------------------

class Rank1Op:
    r"""Memory-efficient rank-1 operator  L = scale * |col><row|.

    Stores only two vectors (O(d) memory) and performs mat-vec in O(d).
    Supports the full interface needed by mcwf_dissipation_step and
    krotov_gate_circuit:  L@psi,  L.conj().T (adjoint),
    L†@L (returns Rank1Op projector), L@rho@L† (density matrix).
    """
    __slots__ = ('col', 'row', '_scale')
    __array_ufunc__ = None
    __array_priority__ = 20.0

    def __init__(self, col, row, scale=1.0):
        self.col = np.asarray(col, dtype=complex).ravel()
        self.row = np.asarray(row, dtype=complex).ravel()
        self._scale = complex(scale)

    def __matmul__(self, other):
        if isinstance(other, Rank1Op):
            overlap = np.vdot(self.row, other.col)
            return Rank1Op(self.col, other.row,
                           self._scale * other._scale * overlap)
        other = np.asarray(other)
        if other.ndim == 1:
            return self._scale * np.vdot(self.row, other) * self.col
        if other.ndim == 2:
            return self._scale * np.outer(self.col, self.row.conj() @ other)
        return NotImplemented

    def __rmatmul__(self, other):
        other = np.asarray(other)
        if other.ndim == 2:
            return self._scale * np.outer(other @ self.col, self.row.conj())
        return NotImplemented

    def _to_dense(self):
        return self._scale * np.outer(self.col, self.row.conj())

    def __add__(self, other):
        if isinstance(other, Rank1Op):
            return self._to_dense() + other._to_dense()
        return self._to_dense() + np.asarray(other)

    def __radd__(self, other):
        if isinstance(other, (int, float, complex)) and other == 0:
            return self._to_dense()
        return np.asarray(other) + self._to_dense()

    def conj(self):
        return _Rank1Conj(self)

    @property
    def T(self):
        return Rank1Op(self.row.conj(), self.col.conj(), self._scale)


class _Rank1Conj:
    """Intermediate for the `.conj().T` → adjoint chain."""
    __slots__ = ('_p',)
    def __init__(self, parent):
        self._p = parent
    @property
    def T(self):
        return Rank1Op(self._p.row, self._p.col,
                       self._p._scale.conjugate())


def target_cooling_operators_rank1(
    psi_target: np.ndarray,
    gamma: float = 1.0,
    max_ops: int | None = None,
    ref_state: np.ndarray | None = None,
) -> list[Rank1Op]:
    r"""Memory-efficient teacher operators as Rank1Op objects.

    Returns L_k = sqrt(gamma) |target><perp_k|.
    If max_ops is set, keeps only the top-k operators sorted by
    overlap |<perp_k|ref_state>|^2 (defaults to uniform if ref_state
    is None).
    """
    psi = np.asarray(psi_target, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    orth = null_space(psi.conj().reshape(1, -1))

    n_perp = orth.shape[1]
    indices = list(range(n_perp))

    if max_ops is not None and max_ops < n_perp and ref_state is not None:
        ref = np.asarray(ref_state, dtype=complex).ravel()
        overlaps = np.array([abs(np.vdot(orth[:, k], ref))**2
                             for k in range(n_perp)])
        indices = np.argsort(overlaps)[::-1][:max_ops].tolist()

    sg = np.sqrt(gamma)
    return [Rank1Op(psi, orth[:, k], sg) for k in indices]


def amplitude_damping_operators(n_qubits: int) -> list[np.ndarray]:
    r"""Single-qubit sigma^- = |0><1| on each qubit."""
    d = 2 ** n_qubits
    ops = []
    for q in range(n_qubits):
        sigma_m = np.zeros((d, d), dtype=complex)
        for i in range(d):
            if (i >> q) & 1:
                j = i ^ (1 << q)
                sigma_m[j, i] = 1.0
        ops.append(sigma_m)
    return ops


def dephasing_operators(n_qubits: int) -> list[np.ndarray]:
    r"""Single-qubit Z_i on each qubit (pure dephasing jump operators)."""
    d = 2 ** n_qubits
    ops = []
    for q in range(n_qubits):
        Z_q = np.zeros((d, d), dtype=complex)
        for i in range(d):
            bit = (i >> q) & 1
            Z_q[i, i] = 1.0 - 2.0 * bit
        ops.append(Z_q)
    return ops


def zz_dephasing_operators(
    n_qubits: int,
    edges: list[tuple[int, int]],
) -> list[np.ndarray]:
    r"""Two-qubit Z_i Z_j on each edge (correlated dephasing jump operators)."""
    d = 2 ** n_qubits
    ops = []
    for qi, qj in edges:
        ZZ = np.zeros((d, d), dtype=complex)
        for i in range(d):
            val = (1.0 - 2.0 * ((i >> qi) & 1)) * (1.0 - 2.0 * ((i >> qj) & 1))
            ZZ[i, i] = val
        ops.append(ZZ)
    return ops


def embed_2q_operator(
    op: np.ndarray,
    qi: int,
    qj: int,
    n_qubits: int,
) -> np.ndarray:
    r"""Embed a 4×4 two-qubit operator into the full 2^n Hilbert space.

    The 4×4 operator acts on basis {|00⟩,|01⟩,|10⟩,|11⟩} where the
    first index is qubit qi and the second is qubit qj.
    """
    d = 2 ** n_qubits
    result = np.zeros((d, d), dtype=complex)
    mask = (1 << qi) | (1 << qj)
    for i in range(d):
        for j in range(d):
            if (i ^ j) & ~mask != 0:
                continue
            row_2q = ((i >> qi) & 1) * 2 + ((i >> qj) & 1)
            col_2q = ((j >> qi) & 1) * 2 + ((j >> qj) & 1)
            result[i, j] = op[row_2q, col_2q]
    return result


def ancilla_reset_kraus_1q(
    n_qubits: int,
    seed: int = 0,
    angle_scale: float = 0.5,
) -> list[list[np.ndarray]]:
    r"""Per-qubit ancilla-reset Kraus operators in the full Hilbert space.

    For each system qubit, appends a fresh |0⟩ ancilla, applies a random
    parameterised 4×4 unitary U on (system, ancilla), then resets the
    ancilla to |0⟩.  The resulting single-qubit CPTP map has Kraus ops
    K_m = ⟨m|_anc U |0⟩_anc (m = 0, 1), each embedded as d×d.

    Returns a list of n_qubits Kraus-operator sets, each [K0, K1].
    """
    rng = np.random.default_rng(seed)

    I2 = np.eye(2, dtype=complex)
    paulis = [I2, SIGMA_X, SIGMA_Y, SIGMA_Z]
    basis_4 = []
    for a in paulis:
        for b in paulis:
            mat = np.kron(a, b)
            if not np.allclose(mat, np.eye(4)):
                basis_4.append(mat)

    all_sets: list[list[np.ndarray]] = []
    for q in range(n_qubits):
        params = rng.uniform(-angle_scale, angle_scale, size=min(15, len(basis_4)))
        H = np.zeros((4, 4), dtype=complex)
        for k in range(len(params)):
            H += params[k] * basis_4[k]
        H = 0.5 * (H + H.conj().T)
        U = expm(-1j * H)

        U_r = U.reshape(2, 2, 2, 2)  # [sys_out, anc_out, sys_in, anc_in]
        K0_local = U_r[:, 0, :, 0]
        K1_local = U_r[:, 1, :, 0]

        K0_full = embed_operator(K0_local, q, n_qubits)
        K1_full = embed_operator(K1_local, q, n_qubits)
        all_sets.append([K0_full, K1_full])
    return all_sets


def ancilla_reset_kraus_2q(
    n_qubits: int,
    edges: list[tuple[int, int]],
    seed: int = 100,
    angle_scale: float = 0.5,
    n_params: int = 30,
) -> list[list[np.ndarray]]:
    r"""Per-edge 2q ancilla-reset Kraus operators in the full Hilbert space.

    For each edge (qi, qj), appends a |0⟩ ancilla to the 2-qubit
    subsystem, applies a random 8×8 unitary, then resets the ancilla.
    Resulting Kraus ops K_m = ⟨m|_anc U |0⟩_anc are 4×4, embedded as d×d.

    Returns a list of len(edges) Kraus-operator sets, each [K0, K1].
    """
    rng = np.random.default_rng(seed)

    I2 = np.eye(2, dtype=complex)
    paulis = [I2, SIGMA_X, SIGMA_Y, SIGMA_Z]
    basis_8: list[np.ndarray] = []
    for a in paulis:
        for b in paulis:
            for c in paulis:
                mat = np.kron(np.kron(a, b), c)
                if not np.allclose(mat, np.eye(8)):
                    basis_8.append(mat)

    all_sets: list[list[np.ndarray]] = []
    for qi, qj in edges:
        params = rng.uniform(-angle_scale, angle_scale,
                             size=min(n_params, len(basis_8)))
        H = np.zeros((8, 8), dtype=complex)
        for k in range(len(params)):
            H += params[k] * basis_8[k]
        H = 0.5 * (H + H.conj().T)
        U = expm(-1j * H)

        U_r = U.reshape(4, 2, 4, 2)  # [sys_pair_out, anc_out, sys_pair_in, anc_in]
        K0_local = U_r[:, 0, :, 0]
        K1_local = U_r[:, 1, :, 0]

        K0_full = embed_2q_operator(K0_local, qi, qj, n_qubits)
        K1_full = embed_2q_operator(K1_local, qi, qj, n_qubits)
        all_sets.append([K0_full, K1_full])
    return all_sets


# ---------------------------------------------------------------------------
# Blackman pulse shape (used as guess in the paper)
# ---------------------------------------------------------------------------

def blackman_pulse(t: float, T: float, amplitude: float = 1.0) -> float:
    """Blackman window shape, zero at boundaries."""
    if t <= 0 or t >= T:
        return 0.0
    a0, a1, a2 = 0.42, 0.5, 0.08
    return amplitude * (a0 - a1 * np.cos(2 * np.pi * t / T)
                        + a2 * np.cos(4 * np.pi * t / T))


def shape_function(t: float, T: float, t_rise: float = 0.0) -> float:
    r"""S(t) in [0, 1] that enforces pulse boundary conditions.

    Smooth switch-on/off over t_rise at each end.  If t_rise=0 returns 1
    everywhere (no shaping).
    """
    if t_rise <= 0:
        return 1.0
    if t < t_rise:
        return np.sin(np.pi * t / (2 * t_rise)) ** 2
    if t > T - t_rise:
        return np.sin(np.pi * (T - t) / (2 * t_rise)) ** 2
    return 1.0


# ---------------------------------------------------------------------------
# Liouvillian (for reference / validation against MCWF)
# ---------------------------------------------------------------------------

def build_liouvillian(
    lindblad_ops: list[np.ndarray],
    rates: np.ndarray | list[float] | None = None,
    hamiltonian: np.ndarray | None = None,
) -> np.ndarray:
    """Build the full d^2 x d^2 Liouvillian superoperator."""
    d = lindblad_ops[0].shape[0]
    d2 = d * d
    eye = np.eye(d, dtype=complex)
    if rates is None:
        rates = np.ones(len(lindblad_ops))
    rates = np.asarray(rates, dtype=float)

    L = np.zeros((d2, d2), dtype=complex)
    if hamiltonian is not None:
        H = np.asarray(hamiltonian, dtype=complex)
        L += -1j * (np.kron(eye, H) - np.kron(H.T, eye))

    for gamma_k, L_k in zip(rates, lindblad_ops):
        L_k = np.asarray(L_k, dtype=complex)
        LdL = L_k.conj().T @ L_k
        L += gamma_k * (
            np.kron(L_k.conj(), L_k)
            - 0.5 * np.kron(eye, LdL)
            - 0.5 * np.kron(LdL.T, eye)
        )
    return L


def evolve_density_matrix(
    rho: np.ndarray,
    liouvillian: np.ndarray,
    t: float,
) -> np.ndarray:
    """Evolve rho under the Liouvillian for time t."""
    d = rho.shape[0]
    rho_vec = rho.flatten(order="F")
    prop = expm(liouvillian * t)
    return (prop @ rho_vec).reshape((d, d), order="F")


# ---------------------------------------------------------------------------
# Sparse Lindblad operator builders (memory-efficient for large n_qubits)
# ---------------------------------------------------------------------------

def amplitude_damping_operators_sparse(n_qubits: int):
    r"""Single-qubit sigma^- on each qubit as scipy.sparse CSR matrices."""
    from scipy import sparse
    d = 2 ** n_qubits
    ops = []
    for q in range(n_qubits):
        rows, cols = [], []
        for i in range(d):
            if (i >> q) & 1:
                rows.append(i ^ (1 << q))
                cols.append(i)
        data = np.ones(len(rows), dtype=complex)
        ops.append(sparse.csr_matrix((data, (rows, cols)), shape=(d, d)))
    return ops


def dephasing_operators_sparse(n_qubits: int):
    r"""Single-qubit Z_i on each qubit as sparse diagonal matrices."""
    from scipy import sparse
    d = 2 ** n_qubits
    ops = []
    arange = np.arange(d)
    for q in range(n_qubits):
        diag = 1.0 - 2.0 * ((arange >> q) & 1).astype(complex)
        ops.append(sparse.diags(diag, format='csr'))
    return ops


def zz_dephasing_operators_sparse(n_qubits: int, edges: list[tuple[int, int]]):
    r"""Two-qubit Z_i Z_j on each edge as sparse diagonal matrices."""
    from scipy import sparse
    d = 2 ** n_qubits
    ops = []
    arange = np.arange(d)
    for qi, qj in edges:
        diag = ((1.0 - 2.0 * ((arange >> qi) & 1))
                * (1.0 - 2.0 * ((arange >> qj) & 1))).astype(complex)
        ops.append(sparse.diags(diag, format='csr'))
    return ops
