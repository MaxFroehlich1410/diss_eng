r"""Fit a lab-constrained dissipative tail to the ideal target-cooling teacher.

Overview
--------
The teacher (``target_cooling.py``) implements a global CPTP channel that
monotonically drives any state to |psi_*> with fidelity
    F(t) = 1 - (1 - F_0) exp(-gamma t).
This channel requires non-local jump operators and is not hardware-implementable.

This module finds the *best approximation* under laboratory constraints
(local 1q/2q channels, bounded rates) by minimising a loss between the
constrained evolution and the teacher evolution over a set of training
initial states and time points.

Two loss functions are provided:
1. **Frobenius (state-matching)**: mean ||rho_constrained - rho_teacher||_F^2
2. **Fidelity-curve**: mean (F_constrained - F_teacher)^2
   (cheaper -- avoids storing full density-matrix snapshots for the teacher)

Optimisation uses gradient-free methods (Nelder-Mead / Powell / L-BFGS-B
with finite differences) from SciPy, with deterministic seeding and
sigmoid / clip-based bound enforcement.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .target_cooling import apply_target_cooling_step
from .metrics import fidelity_to_pure
from .constraint_dissipation import ConstraintConfig, ConstraintDissipation


# -----------------------------------------------------------------------
# Teacher snapshots
# -----------------------------------------------------------------------

def run_teacher_snapshots(
    rho0: np.ndarray,
    psi: np.ndarray,
    gamma: float,
    tmax: float,
    steps: int,
) -> list[np.ndarray]:
    """Run the ideal target-cooling channel and return full rho snapshots.

    This mirrors ``run_target_cooling_trajectory`` but stores full
    density matrices instead of only scalar diagnostics.

    Returns
    -------
    rhos : list of ndarray (d, d), length steps + 1
    """
    psi = np.asarray(psi, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    psi_conj = psi.conj()

    dt = tmax / steps if steps > 0 else 0.0

    rho = rho0.copy()
    rhos = [rho.copy()]
    for _ in range(steps):
        rho = apply_target_cooling_step(rho, psi, psi_conj, gamma, dt)
        rhos.append(rho.copy())
    return rhos


def run_teacher_fidelities(
    rho0: np.ndarray,
    psi: np.ndarray,
    gamma: float,
    tmax: float,
    steps: int,
) -> np.ndarray:
    """Return the teacher fidelity curve (cheaper than full snapshots)."""
    psi = np.asarray(psi, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    psi_conj = psi.conj()

    dt = tmax / steps if steps > 0 else 0.0
    fids = np.empty(steps + 1)
    rho = rho0.copy()
    for i in range(steps + 1):
        fids[i] = fidelity_to_pure(rho, psi)
        if i < steps:
            rho = apply_target_cooling_step(rho, psi, psi_conj, gamma, dt)
    return fids


# -----------------------------------------------------------------------
# Training-state generators
# -----------------------------------------------------------------------

def generate_training_states(
    d: int,
    n_states: int,
    psi_target: np.ndarray,
    rho_circuit: Optional[np.ndarray] = None,
    seed: int = 0,
) -> list[np.ndarray]:
    """Generate a diverse set of training density matrices.

    Includes:
    - the circuit output state (if provided)
    - random pure states
    - computational-basis states (up to budget)
    - random mixed states
    """
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []

    # 1. Circuit output (most important training state).
    if rho_circuit is not None:
        states.append(rho_circuit.copy())

    # 2. Target state itself (should be a fixed point for the teacher).
    psi_t = np.asarray(psi_target, dtype=complex).ravel()
    psi_t = psi_t / np.linalg.norm(psi_t)
    states.append(np.outer(psi_t, psi_t.conj()))

    # 3. Random pure states.
    n_pure = max(1, (n_states - len(states)) // 2)
    for _ in range(n_pure):
        psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
        psi /= np.linalg.norm(psi)
        states.append(np.outer(psi, psi.conj()))

    # 4. Computational basis states (up to budget).
    remaining = n_states - len(states)
    if remaining > 0:
        basis_indices = rng.choice(d, size=min(remaining, d), replace=False)
        for idx in basis_indices:
            rho = np.zeros((d, d), dtype=complex)
            rho[idx, idx] = 1.0
            states.append(rho)

    # 5. Random mixed states if still under budget.
    while len(states) < n_states:
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        rho = A @ A.conj().T
        rho /= np.trace(rho)
        states.append(rho)

    return states[:n_states]


# -----------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------

def _loss_frobenius(
    model: ConstraintDissipation,
    teacher_snapshots: list[list[np.ndarray]],
    rho_trains: list[np.ndarray],
    tmax: float,
    steps: int,
) -> float:
    r"""State-matching loss: mean Frobenius-norm squared difference.

    L = (1/SJ) sum_{s,j} ||rho_model(t_j) - rho_teacher(t_j)||_F^2
    """
    S = len(rho_trains)
    J = steps  # compare at steps 1..steps (skip t=0 which is identical)
    total = 0.0
    dt = tmax / steps if steps > 0 else 0.0

    model.prepare_step(dt)
    for s in range(S):
        rho = rho_trains[s].copy()
        for j in range(1, steps + 1):
            rho = model.cached_step(rho)
            diff = rho - teacher_snapshots[s][j]
            total += float(np.real(np.vdot(diff, diff)))

    return total / max(S * J, 1)


def _loss_fidelity(
    model: ConstraintDissipation,
    teacher_fids: list[np.ndarray],
    rho_trains: list[np.ndarray],
    psi_target: np.ndarray,
    tmax: float,
    steps: int,
) -> float:
    r"""Fidelity-curve loss: mean squared fidelity difference.

    L = (1/SJ) sum_{s,j} (F_model(t_j) - F_teacher(t_j))^2
    """
    S = len(rho_trains)
    J = steps
    total = 0.0
    dt = tmax / steps if steps > 0 else 0.0

    model.prepare_step(dt)
    for s in range(S):
        rho = rho_trains[s].copy()
        for j in range(1, steps + 1):
            rho = model.cached_step(rho)
            f_model = fidelity_to_pure(rho, psi_target)
            f_teacher = teacher_fids[s][j]
            total += (f_model - f_teacher) ** 2

    return total / max(S * J, 1)


def _regularization(
    params: np.ndarray,
    cfg: ConstraintConfig,
    l2_weight: float = 0.01,
) -> float:
    """L2 penalty on rates scaled by gamma_max."""
    # Caller should pass only rate parameters (not ancilla angles).
    return l2_weight * float(np.sum((params / cfg.gamma_max) ** 2))


# -----------------------------------------------------------------------
# Sigmoid bound enforcement
# -----------------------------------------------------------------------

def _sigmoid(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map R -> (lo, hi) via sigmoid."""
    return lo + (hi - lo) / (1.0 + np.exp(-x))


def _inv_sigmoid(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Inverse sigmoid: (lo, hi) -> R."""
    y = np.clip(y, lo + 1e-12, hi - 1e-12)
    return np.log((y - lo) / (hi - y))


# -----------------------------------------------------------------------
# Main fitting routine
# -----------------------------------------------------------------------

def fit_to_teacher(
    psi_target: np.ndarray,
    rho_train: list[np.ndarray],
    cfg: ConstraintConfig,
    gamma_teacher: float,
    tmax: float,
    steps: int,
    loss: str = "fidelity",
    max_iter: int = 200,
    seed: int = 0,
    l2_weight: float = 0.01,
    verbose: bool = True,
    fit_steps: Optional[int] = None,
) -> dict:
    """Fit the constrained dissipation to the ideal teacher tail.

    Parameters
    ----------
    psi_target : ndarray (d,)
        Target pure state.
    rho_train : list of ndarray (d, d)
        Training initial states.
    cfg : ConstraintConfig
        Hardware constraint specification.
    gamma_teacher : float
        Teacher dissipation rate.
    tmax, steps : float, int
        Time grid for the tail (used for final evaluation / display).
    loss : {"fidelity", "frobenius"}
        Which loss function to minimise.
    max_iter : int
        Maximum optimiser iterations.
    seed : int
        Deterministic random seed.
    l2_weight : float
        L2 regularisation weight.
    verbose : bool
        Print progress.
    fit_steps : int or None
        If set, use a *coarser* time grid during fitting for speed.
        The teacher trajectories are computed on this coarser grid,
        reducing the cost per objective evaluation from O(steps) to
        O(fit_steps).  After fitting, the model is evaluated on the
        full ``steps`` grid.

    Returns
    -------
    result : dict with keys
        "best_params", "best_loss", "history", "model"
        (the trained ConstraintDissipation instance)
    """
    import time as _time

    psi = np.asarray(psi_target, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)

    # Use coarser grid for optimisation if requested.
    opt_steps = fit_steps if fit_steps is not None else steps

    # ---- Precompute teacher trajectories ----
    if verbose:
        print(f"  Precomputing teacher trajectories for {len(rho_train)} "
              f"training states ({opt_steps} steps) ...")

    if loss == "frobenius":
        teacher_snaps = [
            run_teacher_snapshots(rho, psi, gamma_teacher, tmax, opt_steps)
            for rho in rho_train
        ]
        teacher_fids = None
    else:
        teacher_snaps = None
        teacher_fids = [
            run_teacher_fidelities(rho, psi, gamma_teacher, tmax, opt_steps)
            for rho in rho_train
        ]

    # ---- Build model ----
    model = ConstraintDissipation(cfg)
    n_params = model.num_params()
    if verbose:
        print(f"  Constrained model: {n_params} parameters")

    # ---- Identify rate vs angle parameter indices ----
    rate_mask = np.ones(n_params, dtype=bool)
    for name, start, count in model._layout:
        if name.startswith("ancilla_reset"):
            rate_mask[start:start + count] = False

    # ---- Objective function (in sigmoid space for rates) ----
    history: list[float] = []
    t_start = _time.perf_counter()

    def objective(x_raw: np.ndarray) -> float:
        # Decode: rates via sigmoid, angles pass through.
        x = np.empty(n_params, dtype=float)
        x[rate_mask] = _sigmoid(x_raw[rate_mask], 0.0, cfg.gamma_max)
        x[~rate_mask] = x_raw[~rate_mask]

        model.unpack_params(x)

        if loss == "frobenius":
            val = _loss_frobenius(
                model, teacher_snaps, rho_train, tmax, opt_steps,
            )
        else:
            val = _loss_fidelity(
                model, teacher_fids, rho_train, psi, tmax, opt_steps,
            )
        val += _regularization(x[rate_mask], cfg, l2_weight)
        history.append(val)

        if verbose and len(history) % 10 == 0:
            elapsed = _time.perf_counter() - t_start
            print(f"    eval {len(history):4d}  loss = {val:.6e}"
                  f"  ({elapsed:.1f}s)")
        return val

    # ---- Initial point (in sigmoid space) ----
    rng = np.random.default_rng(seed)
    x0_real = np.full(n_params, 0.3)
    # Small random perturbation for diversity.
    x0_real += rng.uniform(-0.1, 0.1, size=n_params)
    x0_real = np.clip(x0_real, 1e-3, cfg.gamma_max - 1e-3)
    x0_raw = np.empty(n_params, dtype=float)
    x0_raw[rate_mask] = _inv_sigmoid(x0_real[rate_mask], 0.0, cfg.gamma_max)
    # Ancilla-reset angles: start near identity (small angles → U ≈ I,
    # so the channel is close to "do nothing" initially, giving the
    # optimizer a mild starting point instead of a random scramble).
    x0_raw[~rate_mask] = rng.uniform(-0.05, 0.05,
                                      size=int((~rate_mask).sum()))

    # ---- Run optimiser ----
    if verbose:
        print(f"  Optimising ({loss} loss, max_iter={max_iter}) ...")

    res = minimize(
        objective,
        x0_raw,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "maxfun": max_iter * 10},
    )

    # ---- Decode best parameters ----
    x_best = np.empty(n_params, dtype=float)
    x_best[rate_mask] = _sigmoid(res.x[rate_mask], 0.0, cfg.gamma_max)
    x_best[~rate_mask] = res.x[~rate_mask]
    model.unpack_params(x_best)

    elapsed = _time.perf_counter() - t_start
    if verbose:
        print(f"  Optimisation finished: {res.message}")
        print(f"  Best loss = {res.fun:.6e}  ({len(history)} evaluations, "
              f"{elapsed:.1f}s)")

    return {
        "best_params": model.pack_params(),
        "best_loss": float(res.fun),
        "history": history,
        "model": model,
        "scipy_result": res,
    }


# -----------------------------------------------------------------------
# Evaluation / reporting helper
# -----------------------------------------------------------------------

def evaluate_and_report(
    model: ConstraintDissipation,
    psi_target: np.ndarray,
    rho_valid: list[np.ndarray],
    gamma_teacher: float,
    tmax: float,
    steps: int,
    verbose: bool = True,
) -> dict:
    """Evaluate a trained model on a validation set and print a report.

    Returns
    -------
    info : dict with "val_fid_mse", "teacher_final_fids", "model_final_fids",
           "param_summary"
    """
    psi = np.asarray(psi_target, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)

    teacher_final_fids = []
    model_final_fids = []
    fid_mse_total = 0.0
    dt = tmax / steps if steps > 0 else 0.0

    for rho0 in rho_valid:
        # Teacher fidelity curve.
        t_fids = run_teacher_fidelities(rho0, psi, gamma_teacher, tmax, steps)

        # Model fidelity curve.
        rho = rho0.copy()
        m_fids = np.empty(steps + 1)
        m_fids[0] = fidelity_to_pure(rho, psi)
        for j in range(steps):
            rho = model.step(rho, dt)
            m_fids[j + 1] = fidelity_to_pure(rho, psi)

        teacher_final_fids.append(t_fids[-1])
        model_final_fids.append(m_fids[-1])
        fid_mse_total += float(np.mean((m_fids - t_fids) ** 2))

    val_fid_mse = fid_mse_total / max(len(rho_valid), 1)

    # Parameter summary.
    params = model.pack_params()
    rate_params = []
    for name, start, count in model._layout:
        if name != "ancilla_reset":
            rate_params.extend(params[start:start + count].tolist())
    rate_params = np.array(rate_params)

    if verbose:
        print("\n  === Validation report ===")
        print(f"  Validation fidelity MSE  : {val_fid_mse:.6e}")
        print(f"  Teacher final fids (mean): "
              f"{np.mean(teacher_final_fids):.6f}")
        print(f"  Model   final fids (mean): "
              f"{np.mean(model_final_fids):.6f}")
        if rate_params.size > 0:
            print(f"  Rate params: min={rate_params.min():.4f}, "
                  f"max={rate_params.max():.4f}, "
                  f"mean={rate_params.mean():.4f}")
            n_sat = int(np.sum(rate_params >= model.cfg.gamma_max - 1e-6))
            print(f"  Saturated at gamma_max: {n_sat}/{rate_params.size}")

    return {
        "val_fid_mse": val_fid_mse,
        "teacher_final_fids": teacher_final_fids,
        "model_final_fids": model_final_fids,
        "param_summary": {
            "min": float(rate_params.min()) if rate_params.size else 0.0,
            "max": float(rate_params.max()) if rate_params.size else 0.0,
            "mean": float(rate_params.mean()) if rate_params.size else 0.0,
        },
    }
