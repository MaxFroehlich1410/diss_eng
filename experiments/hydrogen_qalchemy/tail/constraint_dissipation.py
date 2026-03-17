r"""Laboratory-constrained dissipative tail.

Why constraints?
----------------
The *teacher* tail (``target_cooling.py``) uses global jump operators
L_k = sqrt(gamma)|psi_*><psi_k^perp| -- one per orthogonal direction.
While mathematically ideal (monotone fidelity, unique steady state), these
operators are non-local: each L_k couples *every* qubit.  No current
hardware can implement them natively.

This module provides a **lab-realistic** approximation built from:

1. **Local 1-qubit channels** (amplitude damping, dephasing, depolarising)
   that can be implemented via spontaneous emission, T2 decay, or
   randomised Pauli gates on individual qubits.

2. **2-qubit channels** (correlated ZZ dephasing, engineered two-qubit
   pumping) corresponding to nearest-neighbour couplings -- turned on
   by ``allow_2q``.

3. **(Optional) ancilla-reset steps** where a short system--ancilla unitary
   followed by ancilla |0>-reset implements a programmable dissipative
   map -- behind ``allow_ancilla_reset``.

All rates are bounded in [0, gamma_max].  3-body terms are gated behind
``allow_3q`` (default off) and left as an extension point.

How step() works
----------------
Each ``step(rho, dt)`` applies a **first-order Trotter decomposition**:
the total channel for one time step is approximated as a sequential
composition of independent local CPTP maps.  Each local map is exact
(Kraus-form, analytic for the Lindbladian at that rate and dt).

Efficiency: each 1-qubit channel is applied via a *reshape + tensor
contraction* trick that avoids constructing the full d×d Kraus operator.
For n qubits with d = 2^n, the cost per 1-qubit channel is O(d²) with
a small constant.  2-qubit channels use 4×4 Kraus sets applied via
reshaping on qubit pairs: O(d²) per edge.  For 2q ancilla-reset, the
3-qubit Pauli basis used in the unitary parameterization is cached once
at module import time.

With strictly local noise the teacher (global target cooling) *cannot* be
matched exactly in general.  The optimizer in ``fit_constraint_to_target``
finds the best approximation under these hardware constraints.

Oracle connectivity mode
------------------------
For benchmarking upper bounds, ``ConstraintConfig(connectivity="all_to_all")``
sets the allowed 2-qubit pairs to the complete graph on n qubits.  This
removes geometric locality constraints while still restricting primitives to
2-qubit CPTP maps (plus optional ancilla reset gadgets).

Optimization space (fit_constraint_to_target)
----------------------------------------------
The optimizer (e.g. L-BFGS-B) sees a flat vector of length ``num_params()``.
- **Rate parameters** (amp_damp, dephasing, depolar, zz_deph): stored in
  *sigmoid space*: the optimizer gets unbounded reals x, and we map to
  physical rates via γ = γ_max * sigmoid(x) ∈ (0, γ_max).  So physical
  rates are always in [0, γ_max].
- **Ancilla-reset parameters** (if enabled): unbounded real angles passed
  through directly (no sigmoid).

Lindblad operators (per channel)
---------------------------------
The dynamics are implemented as Trotterized Kraus maps; each map is the
exact solution of a Lindblad equation for one channel over dt.  The
corresponding jump operators (acting on the full Hilbert space, with
identity on other qubits implied) are:

**1-qubit, per qubit i:**
- Amplitude damping:  L_i = √γ_i  σ^-_i   (σ^- = |0⟩⟨1|)
- Dephasing:         L_i = √γ_i  Z_i
- Depolarizing:      L_{i,X} = √(γ_i/3) X_i,  L_{i,Y} = √(γ_i/3) Y_i,
                     L_{i,Z} = √(γ_i/3) Z_i

**2-qubit, per edge (i,j):**
- ZZ dephasing:      L_{ij} = √γ_{ij}  Z_i Z_j

**Ancilla-reset:** Not in Lindblad form; it is a discrete CPTP map
(Kraus operators from unitary + ancilla reset).  No jump operator L.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np


def _complete_graph_edges(n: int) -> list[tuple[int, int]]:
    """All unordered pairs (i, j), 0 <= i < j < n, in lexicographic order."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def _canonicalize_edges(
    edges: list[tuple[int, int]],
    n_qubits: int,
) -> list[tuple[int, int]]:
    """Validate and canonicalize user-provided edge list."""
    out: set[tuple[int, int]] = set()
    for a, b in edges:
        i = int(a)
        j = int(b)
        if i == j:
            raise ValueError(f"Invalid edge ({i},{j}): self-loops are not allowed")
        if not (0 <= i < n_qubits and 0 <= j < n_qubits):
            raise ValueError(
                f"Invalid edge ({i},{j}): indices must be in [0, {n_qubits - 1}]"
            )
        if i > j:
            i, j = j, i
        out.add((i, j))
    return sorted(out)

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------

@dataclass
class ConstraintConfig:
    """Specifies which lab-realistic dissipation channels are available."""

    n_qubits: int
    allow_2q: bool = True
    allow_3q: bool = False
    allow_ancilla_reset: bool = False
    allow_2q_ancilla_reset: bool = False
    gamma_max: float = 2.0

    # Topology for 2-qubit jumps: list of edges (i, j).
    # Connectivity policy:
    #   chain      : linear chain (if edges not provided)
    #   custom     : use edges exactly (must be provided)
    #   all_to_all : complete graph edges (ignores provided edges)
    connectivity: str = "chain"  # {"chain", "custom", "all_to_all"}
    # Custom topology for 2-qubit jumps. If provided with connectivity=chain,
    # we treat it as a custom topology for backward compatibility.
    edges: Optional[List[Tuple[int, int]]] = None

    # Which 1-qubit channel families are enabled.
    enable_amp_damp: bool = True
    enable_dephasing: bool = True
    enable_depolarizing: bool = False

    # New expressive edge channels.
    enable_edge_pumping: bool = False
    edge_pump_target: str = "phi_plus"       # {"phi_plus", "psi_minus"}
    edge_parity_target: str = "even"         # {"even", "odd"}
    edge_parity_basis: str = "ZZ"            # {"ZZ", "XX"}

    # Optional collective decay on each edge.
    enable_edge_collective_decay: bool = False
    edge_collective_sign: str = "plus"       # {"plus", "minus"}

    # Number of parameters per edge for the 2q ancilla-reset unitary.
    # 63 = all non-identity 3-qubit Pauli products.
    anc2q_n_params: int = 30

    def __post_init__(self) -> None:
        if self.connectivity not in {"chain", "custom", "all_to_all"}:
            raise ValueError(
                "connectivity must be one of {'chain','custom','all_to_all'}"
            )

        # Backward compatibility:
        # if edges are explicitly provided while connectivity remains default
        # ('chain'), treat this as custom connectivity.
        mode = self.connectivity
        if self.edges is not None and mode == "chain":
            mode = "custom"

        if self.allow_2q:
            if mode == "all_to_all":
                self.edges = _complete_graph_edges(self.n_qubits)
            elif mode == "chain":
                self.edges = [(i, i + 1) for i in range(self.n_qubits - 1)]
            else:  # custom
                if self.edges is None:
                    raise ValueError(
                        "connectivity='custom' requires an explicit edges list"
                    )
                self.edges = _canonicalize_edges(self.edges, self.n_qubits)
        else:
            self.edges = []

        # Quick self-check for the oracle connectivity mode.
        if self.allow_2q and mode == "all_to_all" and self.n_qubits <= 6:
            expected = self.n_qubits * (self.n_qubits - 1) // 2
            if len(self.edges) != expected:
                raise ValueError(
                    "all_to_all connectivity edge count mismatch: "
                    f"got {len(self.edges)}, expected {expected}"
                )

        if self.edge_pump_target not in {"phi_plus", "psi_minus"}:
            raise ValueError(
                "edge_pump_target must be 'phi_plus' or 'psi_minus'"
            )
        if self.edge_parity_target not in {"even", "odd"}:
            raise ValueError("edge_parity_target must be 'even' or 'odd'")
        if self.edge_parity_basis not in {"ZZ", "XX"}:
            raise ValueError("edge_parity_basis must be 'ZZ' or 'XX'")
        if self.edge_collective_sign not in {"plus", "minus"}:
            raise ValueError("edge_collective_sign must be 'plus' or 'minus'")
        if self.anc2q_n_params < 1:
            raise ValueError("anc2q_n_params must be >= 1")


# -----------------------------------------------------------------------
# Helper: efficient local-channel application via reshape + einsum
# -----------------------------------------------------------------------

def _apply_1q_kraus(
    rho: np.ndarray,
    kraus_ops: list[np.ndarray],
    qubit: int,
    n_qubits: int,
) -> np.ndarray:
    r"""Apply a 1-qubit CPTP map to *qubit* using Kraus operators.

    Instead of lifting each 2×2 Kraus operator to d×d, we reshape rho
    into a tensor of shape (2,)*2n, contract on the target qubit indices,
    and reshape back.  Cost: O(K · d²) where K = |Kraus set|.
    """
    d = 1 << n_qubits
    shape = (2,) * (2 * n_qubits)
    rho_t = rho.reshape(shape)

    # Axes of the target qubit in the ket (row) and bra (col) side.
    ax_ket = qubit
    ax_bra = n_qubits + qubit

    out = np.zeros_like(rho_t)
    for K in kraus_ops:
        # K is 2×2.  We need:  K_{a,b} * rho_{..b.., ..c..} * K^*_{d,c}
        # = sum_{b,c} K_{a,b} conj(K_{d,c}) rho_{..b.., ..c..}
        # Use tensordot / einsum for clarity.
        # Step 1:  contract ket side  tmp_{..a.., ..c..} = K_{a,b} rho_{..b..}
        tmp = np.tensordot(K, rho_t, axes=([1], [ax_ket]))
        # tensordot puts the new 'a' axis at position 0; we need to
        # move it to position ax_ket.
        tmp = np.moveaxis(tmp, 0, ax_ket)

        # Step 2: contract bra side  res_{..a.., ..d..} = tmp * K^*_{d,c}
        Kd = K.conj()
        tmp2 = np.tensordot(tmp, Kd, axes=([ax_bra], [1]))
        # The new 'd' axis lands at the end; move it to ax_bra.
        tmp2 = np.moveaxis(tmp2, -1, ax_bra)

        out += tmp2

    return out.reshape(d, d)


def _apply_2q_kraus(
    rho: np.ndarray,
    kraus_ops: list[np.ndarray],
    qubit_i: int,
    qubit_j: int,
    n_qubits: int,
) -> np.ndarray:
    r"""Apply a 2-qubit CPTP map on qubits (i, j) via Kraus operators.

    Each Kraus operator is 4×4.  We permute the tensor axes to group the
    target qubit pair, apply the Kraus map via einsum on the 4×d_rest
    sub-blocks, then undo the permutation.  Cost: O(K · d²).
    """
    d = 1 << n_qubits
    shape = (2,) * (2 * n_qubits)
    rho_t = rho.reshape(shape)

    ax_bra_i = n_qubits + qubit_i
    ax_bra_j = n_qubits + qubit_j

    # Permute axes: target qubit pair to front.
    axes_ket = list(range(n_qubits))
    axes_bra = list(range(n_qubits, 2 * n_qubits))
    ket_rest = [k for k in axes_ket if k not in (qubit_i, qubit_j)]
    bra_rest = [k for k in axes_bra if k not in (ax_bra_i, ax_bra_j)]
    perm = [qubit_i, qubit_j] + ket_rest + [ax_bra_i, ax_bra_j] + bra_rest
    rho_p = rho_t.transpose(perm)

    n_rest = n_qubits - 2
    d_rest = 1 << n_rest if n_rest > 0 else 1
    rho_p = rho_p.reshape(4, d_rest, 4, d_rest)

    out_p = np.zeros_like(rho_p)
    for K4 in kraus_ops:
        # K4 is (4, 4).  Apply: K rho K^dag on the pair subspace.
        # out_p[a, r, d, s] += K[a,b] * rho_p[b, r, c, s] * conj(K[d,c])
        tmp = np.einsum("ab,brcs,dc->ards", K4, rho_p, K4.conj(),
                        optimize=True)
        out_p += tmp

    # Undo the permutation.
    out_p = out_p.reshape([2, 2] + [2] * n_rest + [2, 2] + [2] * n_rest)
    inv_perm = [0] * len(perm)
    for new_pos, old_pos in enumerate(perm):
        inv_perm[old_pos] = new_pos
    out_t = out_p.transpose(inv_perm)
    return out_t.reshape(d, d)


# -----------------------------------------------------------------------
# Kraus sets for elementary channels
# -----------------------------------------------------------------------

def _kraus_amplitude_damping(gamma: float, dt: float) -> list[np.ndarray]:
    r"""Kraus operators for amplitude damping on 1 qubit.

    L = sqrt(gamma) sigma^-.  For time step dt, the damping probability
    is  p = 1 - exp(-gamma dt).
    """
    p = 1.0 - np.exp(-gamma * dt)
    p = np.clip(p, 0.0, 1.0)
    sp = np.sqrt(p)
    K0 = np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - p)]], dtype=complex)
    K1 = np.array([[0.0, sp], [0.0, 0.0]], dtype=complex)
    return [K0, K1]


def _kraus_dephasing(gamma: float, dt: float) -> list[np.ndarray]:
    r"""Kraus operators for pure dephasing (L = sqrt(gamma) Z).

    Off-diagonal elements decay as exp(-2 gamma dt).
    Equivalent Kraus:
        K0 = sqrt((1+exp(-2gdt))/2) I
        K1 = sqrt((1-exp(-2gdt))/2) Z
    """
    e = np.exp(-2.0 * gamma * dt)
    a = np.sqrt(0.5 * (1.0 + e))
    b = np.sqrt(0.5 * (1.0 - e))
    I2 = np.eye(2, dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    return [a * I2, b * Z]


def _kraus_depolarizing(gamma: float, dt: float) -> list[np.ndarray]:
    r"""Kraus operators for single-qubit depolarising channel.

    The channel mixes the state toward I/2 at rate gamma:
        rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z)
    with  p = (1 - exp(-gamma dt)) * 3/4.
    """
    p = (1.0 - np.exp(-gamma * dt)) * 0.75
    p = np.clip(p, 0.0, 0.75)
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [
        np.sqrt(1.0 - p) * I2,
        np.sqrt(p / 3.0) * X,
        np.sqrt(p / 3.0) * Y,
        np.sqrt(p / 3.0) * Z,
    ]


def _kraus_zz_dephasing(gamma: float, dt: float) -> list[np.ndarray]:
    r"""Kraus operators for ZZ correlated dephasing on a qubit pair.

    L = sqrt(gamma) Z_i Z_j.  The 4×4 Kraus set is:
        K0 = sqrt((1+e)/2) I4,   K1 = sqrt((1-e)/2) (Z⊗Z)
    with e = exp(-2 gamma dt).
    """
    e = np.exp(-2.0 * gamma * dt)
    a = np.sqrt(0.5 * (1.0 + e))
    b = np.sqrt(0.5 * (1.0 - e))
    I4 = np.eye(4, dtype=complex)
    Z = np.diag([1.0, -1.0])
    ZZ = np.kron(Z, Z).astype(complex)
    return [a * I4, b * ZZ]


# -----------------------------------------------------------------------
# Expressive 2-qubit edge channels
# -----------------------------------------------------------------------

def _bell_target_state(name: str) -> np.ndarray:
    """Return a Bell target statevector in computational basis."""
    if name == "phi_plus":
        return np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)
    if name == "psi_minus":
        return np.array([0.0, 1.0, -1.0, 0.0], dtype=complex) / np.sqrt(2.0)
    raise ValueError(f"Unsupported Bell target: {name}")


def _kraus_edge_bell_pump(gamma: float, dt: float, target: str) -> list[np.ndarray]:
    r"""Bell pumping map on one edge.

    Implements:
        E(rho) = (1-p) rho + p |beta><beta|
    with p = 1 - exp(-gamma dt), and |beta> chosen by ``target``.

    Kraus representation:
        K0 = sqrt(1-p) I4
        Kj = sqrt(p) |beta><j|,  j=0..3
    """
    p = float(np.clip(1.0 - np.exp(-gamma * dt), 0.0, 1.0))
    beta = _bell_target_state(target)
    I4 = np.eye(4, dtype=complex)
    kraus = [np.sqrt(1.0 - p) * I4]
    for j in range(4):
        ej = np.zeros(4, dtype=complex)
        ej[j] = 1.0
        kraus.append(np.sqrt(p) * np.outer(beta, ej.conj()))
    return kraus


def _kraus_edge_parity_pump(
    gamma: float,
    dt: float,
    target_parity: str,
    basis: str,
) -> list[np.ndarray]:
    r"""Parity pumping map with measurement+feedback semantics.

    For ZZ basis:
      - even sector: span{|00>, |11>}
      - odd  sector: span{|01>, |10>}

    For XX basis we conjugate the ZZ map by H⊗H.
    """
    p = float(np.clip(1.0 - np.exp(-gamma * dt), 0.0, 1.0))

    I2 = np.eye(2, dtype=complex)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    H = (1.0 / np.sqrt(2.0)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
    I4 = np.eye(4, dtype=complex)

    P_even = np.diag([1.0, 0.0, 0.0, 1.0]).astype(complex)
    P_odd = np.diag([0.0, 1.0, 1.0, 0.0]).astype(complex)
    X2 = np.kron(I2, X).astype(complex)  # conditional correction on qubit-j

    if target_parity == "even":
        K0 = P_even
        K1 = np.sqrt(1.0 - p) * P_odd
        K2 = np.sqrt(p) * (X2 @ P_odd)
    else:  # pump toward odd
        K0 = P_odd
        K1 = np.sqrt(1.0 - p) * P_even
        K2 = np.sqrt(p) * (X2 @ P_even)

    kraus = [K0, K1, K2]
    if basis == "XX":
        U = np.kron(H, H).astype(complex)
        kraus = [U.conj().T @ K @ U for K in kraus]
    return kraus


def _pauli_product_basis(n_qubits: int, *, include_identity: bool = False) -> list[np.ndarray]:
    """Dense Pauli-product basis for n_qubits (cached per call site)."""
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    Y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    paulis = [I2, X, Y, Z]

    basis = []
    cur = [np.array([[1.0 + 0.0j]])]
    for _ in range(n_qubits):
        nxt = []
        for a in cur:
            for p in paulis:
                nxt.append(np.kron(a, p))
        cur = nxt
    for mat in cur:
        if not include_identity and np.allclose(mat, np.eye(1 << n_qubits)):
            continue
        basis.append(mat.astype(complex))
    return basis


# Cached once for 2q ancilla-reset (3-qubit unitary basis).
_PAULI_BASIS_3Q: list[np.ndarray] = _pauli_product_basis(
    3, include_identity=False
)


def _superop_to_kraus(superop: np.ndarray, d: int, *, tol: float = 1e-12) -> list[np.ndarray]:
    """Convert a column-stacked superoperator to Kraus operators."""
    choi = np.zeros((d * d, d * d), dtype=complex)
    for m in range(d):
        for n in range(d):
            basis = np.zeros((d, d), dtype=complex)
            basis[m, n] = 1.0
            v = basis.reshape(d * d, order="F")
            out = (superop @ v).reshape((d, d), order="F")
            choi += np.kron(out, basis)

    choi = 0.5 * (choi + choi.conj().T)
    evals, evecs = np.linalg.eigh(choi)
    kraus = []
    for ev, vec in zip(evals, evecs.T):
        if ev > tol:
            K = np.sqrt(ev) * vec.reshape((d, d), order="F")
            kraus.append(K)
    return kraus


def _kraus_edge_collective_decay(
    gamma: float,
    dt: float,
    sign: str,
) -> list[np.ndarray]:
    r"""Collective edge decay with jump L = sqrt(gamma)(sigma^-_i ± sigma^-_j).

    Uses exact local Lindbladian exponentiation on the 4x4 edge subsystem,
    then converts the resulting CPTP map to Kraus operators.
    """
    from scipy.linalg import expm

    sm = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    I2 = np.eye(2, dtype=complex)
    s = 1.0 if sign == "plus" else -1.0
    L = np.sqrt(max(gamma, 0.0)) * (np.kron(sm, I2) + s * np.kron(I2, sm))
    A = L.conj().T @ L
    I4 = np.eye(4, dtype=complex)

    # Column-stacked vec convention: vec(AXB) = (B^T ⊗ A) vec(X)
    D = (
        np.kron(L, L.conj())
        - 0.5 * np.kron(I4, A.T)
        - 0.5 * np.kron(A, I4)
    )
    E = expm(dt * D)
    return _superop_to_kraus(E, d=4)


# -----------------------------------------------------------------------
# Ancilla-reset primitives (optional)
# -----------------------------------------------------------------------

def _ancilla_reset_kraus(
    U_params: np.ndarray,
    target_qubit: int,  # noqa: ARG001 – reserved for future topology
    n_qubits: int,      # noqa: ARG001 – reserved for multi-qubit ancilla
) -> list[np.ndarray]:
    r"""Build Kraus operators for one ancilla-reset step.

    Model: append a fresh |0> ancilla, apply a parameterised 4×4 unitary
    U(theta) on (system_qubit, ancilla), measure/reset ancilla to |0>.

    The resulting single-qubit CPTP map has Kraus operators:
        K_m = <m|_ancilla U |0>_ancilla   for m in {0, 1}
    where each K_m is 2×2 acting on the system qubit.

    U is parameterised as exp(-i H) with H a real 4×4 Hermitian built
    from 6 real parameters (upper-triangular real part + imaginary part).
    For simplicity we use a general SU(4) parameterisation based on the
    first 15 Gell-Mann-like generators scaled by the parameter vector.
    Here we use a simpler approach: U = expm(-i * sum_k theta_k G_k).
    """
    from scipy.linalg import expm

    # Simple parameterisation: 6-parameter Hermitian 4×4
    # Pack into a Hermitian matrix via:
    #   H = sum_{k} theta_k * basis_k
    # where basis_k are a chosen set of Hermitian generators.
    # For a 4×4 system we use the standard Pauli-product basis:
    # {IX, IY, IZ, XI, XX, XY, XZ, YI, YX, YY, YZ, ZI, ZX, ZY, ZZ}
    # We use the first len(U_params) of them.
    I2 = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    paulis = [I2, X, Y, Z]
    basis = []
    for a in paulis:
        for b in paulis:
            if np.allclose(a, I2) and np.allclose(b, I2):
                continue  # skip identity
            basis.append(np.kron(a, b))
    # Truncate to parameter count
    n_p = len(U_params)
    H = np.zeros((4, 4), dtype=complex)
    for k in range(min(n_p, len(basis))):
        H += U_params[k] * basis[k]
    H = 0.5 * (H + H.conj().T)  # ensure Hermitian
    U = expm(-1j * H)

    # Extract Kraus ops: K_m = <m|_anc U |0>_anc, m=0,1
    # U is 4×4 acting on (system, ancilla).
    # |0>_anc selects columns 0,2 (ancilla=0).  <m|_anc selects rows.
    U = U.reshape(2, 2, 2, 2)  # [sys_out, anc_out, sys_in, anc_in]
    K0 = U[:, 0, :, 0]  # anc_in=0, anc_out=0 → 2×2
    K1 = U[:, 1, :, 0]  # anc_in=0, anc_out=1 → 2×2
    return [K0, K1]


def _ancilla_reset_kraus_2q(
    U_params: np.ndarray,
    anc2q_n_params: int,
) -> list[np.ndarray]:
    r"""Kraus operators for edge ancilla-reset on two system qubits.

    Build a 3-qubit unitary U on (qi, qj, anc) as:
        U = exp(-i H(theta))
    where H is Hermitian from a truncated 3-qubit Pauli-product basis.
    Then trace/reset ancilla:
        K_m = <m|_anc U |0>_anc,  m in {0,1}
    giving two 4x4 Kraus operators acting on the edge subsystem.
    """
    from scipy.linalg import expm

    basis = _PAULI_BASIS_3Q
    n_use = min(int(anc2q_n_params), len(basis), len(U_params))
    H = np.zeros((8, 8), dtype=complex)
    for k in range(n_use):
        H += U_params[k] * basis[k]
    H = 0.5 * (H + H.conj().T)
    U = expm(-1j * H)

    # Reshape with ordering (sys2_out, anc_out, sys2_in, anc_in)
    U = U.reshape(4, 2, 4, 2)
    K0 = U[:, 0, :, 0]
    K1 = U[:, 1, :, 0]
    return [K0, K1]


# -----------------------------------------------------------------------
# Main class
# -----------------------------------------------------------------------

class ConstraintDissipation:
    """Family of lab-realistic dissipative tails.

    Parameters are optimisable rates (and optionally ancilla-reset
    unitary parameters) that define a CPTP map per time step.
    """

    def __init__(self, cfg: ConstraintConfig) -> None:
        self.cfg = cfg
        self.n = cfg.n_qubits

        # ---- Build parameter layout ----
        # Track (name, index_start, count) for each block.
        self._layout: list[tuple[str, int, int]] = []
        idx = 0

        # 1-qubit amplitude damping: one rate per qubit
        if cfg.enable_amp_damp:
            self._layout.append(("amp_damp", idx, self.n))
            idx += self.n

        # 1-qubit dephasing: one rate per qubit
        if cfg.enable_dephasing:
            self._layout.append(("dephasing", idx, self.n))
            idx += self.n

        # 1-qubit depolarising: one rate per qubit
        if cfg.enable_depolarizing:
            self._layout.append(("depolar", idx, self.n))
            idx += self.n

        # 2-qubit ZZ dephasing: one rate per edge
        if cfg.allow_2q and cfg.edges:
            # Note: for all_to_all connectivity, len(edges)=O(n^2), so
            # parameter count grows quadratically for all edge-based blocks.
            self._layout.append(("zz_deph", idx, len(cfg.edges)))
            idx += len(cfg.edges)

        # Structured edge pumping baselines (rate-only)
        if cfg.allow_2q and cfg.edges and cfg.enable_edge_pumping:
            self._layout.append(("edge_bell_pump", idx, len(cfg.edges)))
            idx += len(cfg.edges)
            self._layout.append(("edge_parity_pump", idx, len(cfg.edges)))
            idx += len(cfg.edges)

        # Optional collective edge decay (rate-only)
        if cfg.allow_2q and cfg.edges and cfg.enable_edge_collective_decay:
            self._layout.append(("edge_collective_decay", idx, len(cfg.edges)))
            idx += len(cfg.edges)

        # Ancilla-reset (1q): unitary params per qubit
        self._ancilla_n_params = 15  # per qubit
        if cfg.allow_ancilla_reset:
            self._layout.append(
                ("ancilla_reset", idx, self.n * self._ancilla_n_params)
            )
            idx += self.n * self._ancilla_n_params

        # Ancilla-reset (2q on edges): unitary params per edge
        self._anc2q_n_params = int(cfg.anc2q_n_params)
        if cfg.allow_2q and cfg.edges and cfg.allow_2q_ancilla_reset:
            self._layout.append(
                ("ancilla_reset_2q", idx, len(cfg.edges) * self._anc2q_n_params)
            )
            idx += len(cfg.edges) * self._anc2q_n_params

        # 3-body placeholder (not implemented, reserved)
        if cfg.allow_3q:
            pass  # extension point

        self._n_params = idx
        self._params = np.zeros(idx, dtype=float)

        # Initialise rates to small positive values.
        self._init_rates(0.1)

    # ---- Parameter access ----

    def num_params(self) -> int:
        """Return number of optimisable parameters."""
        return self._n_params

    def pack_params(self) -> np.ndarray:
        """Return current parameters as a flat vector."""
        return self._params.copy()

    def _is_angle_block(self, name: str) -> bool:
        """Return True if a block contains unbounded angle parameters."""
        return name.startswith("ancilla_reset")

    def rate_mask(self) -> np.ndarray:
        """Boolean mask over flat params: True=rate, False=angle."""
        mask = np.ones(self._n_params, dtype=bool)
        for name, start, count in self._layout:
            if self._is_angle_block(name):
                mask[start:start + count] = False
        if mask.size != self._n_params:
            raise RuntimeError("rate_mask size mismatch")
        return mask

    def unpack_params(self, x: np.ndarray) -> None:
        """Set internal parameters from a flat vector, enforcing bounds.

        Rate parameters are clipped to [0, gamma_max].
        Ancilla-reset unitary parameters are unbounded (angles).
        """
        x = np.asarray(x, dtype=float).ravel()
        if x.size != self._n_params:
            raise ValueError(
                f"Expected {self._n_params} params, got {x.size}"
            )
        for name, start, count in self._layout:
            if self._is_angle_block(name):
                self._params[start:start + count] = x[start:start + count]
            else:
                self._params[start:start + count] = np.clip(
                    x[start:start + count], 0.0, self.cfg.gamma_max
                )

    def _init_rates(self, val: float) -> None:
        """Set all rate parameters to *val*."""
        self.set_all_rate_params(val)

    def _get_block(self, name: str) -> np.ndarray:
        """Return the sub-array of parameters for a named block."""
        for n, s, c in self._layout:
            if n == name:
                return self._params[s:s + c]
        return np.array([], dtype=float)

    def set_all_rate_params(self, value: float) -> None:
        """Set every rate parameter block to a single clipped value."""
        v = float(np.clip(value, 0.0, self.cfg.gamma_max))
        for name, start, count in self._layout:
            if not self._is_angle_block(name):
                self._params[start:start + count] = v

    def set_rate_block(self, name: str, value: float | np.ndarray) -> None:
        """Set a single *rate* block by scalar or elementwise array.

        Parameters
        ----------
        name : str
            Block name in ``self._layout``.
        value : float or array-like
            Scalar value for full block, or array of exact block length.
        """
        start = count = None
        for n, s, c in self._layout:
            if n == name:
                if self._is_angle_block(n):
                    raise ValueError(f"Block '{name}' is an angle block")
                start, count = s, c
                break
        if start is None or count is None:
            raise ValueError(f"Unknown parameter block: '{name}'")

        if np.isscalar(value):
            v = float(np.clip(value, 0.0, self.cfg.gamma_max))
            self._params[start:start + count] = v
            return

        arr = np.asarray(value, dtype=float).ravel()
        if arr.size != count:
            raise ValueError(
                f"Block '{name}' expects {count} values, got {arr.size}"
            )
        self._params[start:start + count] = np.clip(
            arr, 0.0, self.cfg.gamma_max
        )

    def set_all_angle_params(
        self,
        scale: float = 0.0,
        seed: int | None = None,
    ) -> None:
        """Set all angle blocks to zero or small random values.

        If ``scale == 0``, angle params are set to exactly zero.
        Otherwise, values are sampled uniformly from [-scale, scale].
        """
        sc = float(scale)
        if sc < 0.0:
            raise ValueError("scale must be >= 0")
        rng = np.random.default_rng(seed)
        for name, start, count in self._layout:
            if self._is_angle_block(name):
                if sc == 0.0:
                    self._params[start:start + count] = 0.0
                else:
                    self._params[start:start + count] = rng.uniform(
                        -sc, sc, size=count
                    )

    # ---- Kraus cache ----

    def prepare_step(self, dt: float) -> None:
        """Pre-build all Kraus operators for a given dt.

        Call this once before a trajectory loop.  Then use
        ``cached_step(rho)`` instead of ``step(rho, dt)`` to avoid
        rebuilding the operators every call.  This gives a large
        speed-up when the same dt is used many times (fitting loops).
        """
        n = self.n
        self._cached_ops: list[tuple[str, int, int, list[np.ndarray]]] = []
        # (kind, qubit_or_edge_i, qubit_or_edge_j, kraus_list)
        # kind: "1q" or "2q"

        if self.cfg.enable_amp_damp:
            rates = self._get_block("amp_damp")
            for q in range(n):
                if rates[q] > 1e-14:
                    self._cached_ops.append(
                        ("1q", q, -1,
                         _kraus_amplitude_damping(rates[q], dt)))

        if self.cfg.enable_dephasing:
            rates = self._get_block("dephasing")
            for q in range(n):
                if rates[q] > 1e-14:
                    self._cached_ops.append(
                        ("1q", q, -1,
                         _kraus_dephasing(rates[q], dt)))

        if self.cfg.enable_depolarizing:
            rates = self._get_block("depolar")
            for q in range(n):
                if rates[q] > 1e-14:
                    self._cached_ops.append(
                        ("1q", q, -1,
                         _kraus_depolarizing(rates[q], dt)))

        if self.cfg.allow_2q and self.cfg.edges:
            rates = self._get_block("zz_deph")
            for k, (qi, qj) in enumerate(self.cfg.edges):
                if rates[k] > 1e-14:
                    self._cached_ops.append(
                        ("2q", qi, qj,
                         _kraus_zz_dephasing(rates[k], dt)))

        if self.cfg.allow_2q and self.cfg.edges and self.cfg.enable_edge_pumping:
            rates_bell = self._get_block("edge_bell_pump")
            rates_par = self._get_block("edge_parity_pump")
            for k, (qi, qj) in enumerate(self.cfg.edges):
                if rates_bell[k] > 1e-14:
                    self._cached_ops.append(
                        ("2q", qi, qj,
                         _kraus_edge_bell_pump(
                             rates_bell[k], dt, self.cfg.edge_pump_target
                         ))
                    )
                if rates_par[k] > 1e-14:
                    self._cached_ops.append(
                        ("2q", qi, qj,
                         _kraus_edge_parity_pump(
                             rates_par[k],
                             dt,
                             self.cfg.edge_parity_target,
                             self.cfg.edge_parity_basis,
                         ))
                    )

        if (
            self.cfg.allow_2q
            and self.cfg.edges
            and self.cfg.enable_edge_collective_decay
        ):
            rates_cd = self._get_block("edge_collective_decay")
            for k, (qi, qj) in enumerate(self.cfg.edges):
                if rates_cd[k] > 1e-14:
                    self._cached_ops.append(
                        ("2q", qi, qj,
                         _kraus_edge_collective_decay(
                             rates_cd[k], dt, self.cfg.edge_collective_sign
                         ))
                    )

        if self.cfg.allow_ancilla_reset:
            all_p = self._get_block("ancilla_reset")
            for q in range(n):
                p = all_p[
                    q * self._ancilla_n_params:(q + 1) * self._ancilla_n_params
                ]
                self._cached_ops.append(
                    ("1q", q, -1, _ancilla_reset_kraus(p, q, n)))

        if self.cfg.allow_2q and self.cfg.edges and self.cfg.allow_2q_ancilla_reset:
            all_p2 = self._get_block("ancilla_reset_2q")
            for k, (qi, qj) in enumerate(self.cfg.edges):
                p = all_p2[
                    k * self._anc2q_n_params:(k + 1) * self._anc2q_n_params
                ]
                self._cached_ops.append(
                    ("2q", qi, qj,
                     _ancilla_reset_kraus_2q(p, self._anc2q_n_params))
                )

    def cached_step(self, rho: np.ndarray) -> np.ndarray:
        """Apply one step using pre-built Kraus operators (fast path).

        Must call ``prepare_step(dt)`` first.
        """
        n = self.n
        for kind, qi, qj, kraus in self._cached_ops:
            if kind == "1q":
                rho = _apply_1q_kraus(rho, kraus, qi, n)
            else:
                rho = _apply_2q_kraus(rho, kraus, qi, qj, n)
        return rho

    # ---- Core CPTP step (uncached, convenience) ----

    def step(self, rho: np.ndarray, dt: float) -> np.ndarray:
        """Apply one time step of the constrained dissipation (Trotter).

        For repeated calls with the same dt, prefer
        ``prepare_step(dt)`` + ``cached_step(rho)`` instead.
        """
        self.prepare_step(dt)
        return self.cached_step(rho)

    # ---- Full trajectory ----

    def run_trajectory(
        self,
        rho0: np.ndarray,
        tmax: float,
        steps: int,
    ) -> list[np.ndarray]:
        """Return density matrices at each time-grid point.

        Parameters
        ----------
        rho0 : ndarray (d, d)
        tmax : float > 0
        steps : int > 0

        Returns
        -------
        rhos : list of ndarrays, length steps + 1
        """
        dt = tmax / steps if steps > 0 else 0.0
        self.prepare_step(dt)
        rho = rho0.copy()
        rhos = [rho.copy()]
        for _ in range(steps):
            rho = self.cached_step(rho)
            rhos.append(rho.copy())
        return rhos

    # ---- Small-n debug: full Liouvillian validation (n <= 4 only) ----

    def _build_liouvillian_debug(self, dt: float) -> np.ndarray:
        """Build the d²×d² Liouvillian superoperator for validation.

        Only usable for n_qubits ≤ 4 (d ≤ 16, d² ≤ 256).
        Returns the Choi-like transfer matrix by applying the channel
        to every basis element of the d×d space.
        """
        d = 1 << self.n
        if d > 16:
            raise RuntimeError(
                f"Debug Liouvillian only for n<=4 (d<=16), got d={d}"
            )
        T = np.zeros((d * d, d * d), dtype=complex)
        for j in range(d * d):
            basis = np.zeros((d, d), dtype=complex)
            basis.flat[j] = 1.0
            out = self.step(basis, dt)
            T[:, j] = out.ravel()
        return T

    def validate_cptp_debug(self, dt: float, *, tol: float = 1e-9) -> dict:
        """Validate trace preservation and complete positivity for n<=4.

        Returns a dictionary with key diagnostics:
          - trace_deviation_max
          - choi_min_eig
          - is_trace_preserving
          - is_cp
        """
        d = 1 << self.n
        T = self._build_liouvillian_debug(dt)

        # Trace-preservation check: Tr(E(|i><j|)) = delta_ij.
        tp_dev = 0.0
        for i in range(d):
            for j in range(d):
                basis = np.zeros((d, d), dtype=complex)
                basis[i, j] = 1.0
                out = (T @ basis.ravel()).reshape(d, d)
                target = 1.0 if i == j else 0.0
                tp_dev = max(tp_dev, abs(np.trace(out) - target))

        # Choi eigenvalue check (CP if all >= 0 up to tolerance).
        choi = np.zeros((d * d, d * d), dtype=complex)
        for m in range(d):
            for n in range(d):
                basis = np.zeros((d, d), dtype=complex)
                basis[m, n] = 1.0
                out = (T @ basis.ravel()).reshape((d, d))
                choi += np.kron(out, basis)
        choi = 0.5 * (choi + choi.conj().T)
        min_eig = float(np.min(np.linalg.eigvalsh(choi)))

        return {
            "trace_deviation_max": float(tp_dev),
            "choi_min_eig": min_eig,
            "is_trace_preserving": bool(tp_dev <= tol),
            "is_cp": bool(min_eig >= -tol),
        }
