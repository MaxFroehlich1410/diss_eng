"""Small demo: edge 2q ancilla-reset can outperform do-nothing.

Run:
    python experiments/hydrogen_qalchemy/tail/example_edge_ancilla_reset.py
"""

from __future__ import annotations

import numpy as np
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from tail.constraint_dissipation import ConstraintConfig, ConstraintDissipation
from tail.fit_constraint_to_target import fit_to_teacher
from tail.metrics import fidelity_to_pure


def _bell_phi_plus() -> np.ndarray:
    return np.array([1.0, 0.0, 0.0, 1.0], dtype=complex) / np.sqrt(2.0)


def _target_state_n4() -> np.ndarray:
    # |Phi+>_(0,1) ⊗ |Phi+>_(2,3)
    b = _bell_phi_plus()
    psi = np.kron(b, b)
    return psi / np.linalg.norm(psi)


def _random_density_matrix(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def main() -> None:
    n = 4
    d = 1 << n
    psi_target = _target_state_n4()
    rho0 = _random_density_matrix(d, seed=7)

    cfg = ConstraintConfig(
        n_qubits=n,
        allow_2q=True,
        edges=[(0, 1), (2, 3)],
        gamma_max=2.0,
        # Disable other primitives to isolate the new one.
        enable_amp_damp=False,
        enable_dephasing=False,
        enable_depolarizing=False,
        enable_edge_pumping=False,
        allow_ancilla_reset=False,
        allow_2q_ancilla_reset=True,
        anc2q_n_params=15,
    )

    tmax = 5.0
    steps = 24

    # Baseline: do-nothing constrained tail (all params = 0).
    baseline = ConstraintDissipation(cfg)
    baseline.unpack_params(np.zeros(baseline.num_params()))
    rho_b = baseline.run_trajectory(rho0, tmax=tmax, steps=steps)[-1]
    f_b = fidelity_to_pure(rho_b, psi_target)

    # Fit to teacher target-cooling trajectory using only 2q ancilla-reset.
    fit = fit_to_teacher(
        psi_target=psi_target,
        rho_train=[rho0],
        cfg=cfg,
        gamma_teacher=1.0,
        tmax=tmax,
        steps=steps,
        fit_steps=10,
        loss="fidelity",
        max_iter=120,
        seed=3,
        verbose=True,
    )
    model = fit["model"]
    rho_m = model.run_trajectory(rho0, tmax=tmax, steps=steps)[-1]
    f_m = fidelity_to_pure(rho_m, psi_target)

    f0 = fidelity_to_pure(rho0, psi_target)
    print("\n=== edge 2q ancilla-reset demo ===")
    print(f"Initial fidelity                 : {f0:.6f}")
    print(f"Do-nothing constrained tail      : {f_b:.6f}")
    print(f"Optimized 2q ancilla-reset tail  : {f_m:.6f}")
    print(f"Gain over do-nothing             : {f_m - f_b:+.6f}")


if __name__ == "__main__":
    main()

