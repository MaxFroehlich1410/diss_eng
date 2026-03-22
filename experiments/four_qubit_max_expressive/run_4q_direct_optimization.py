r"""4-qubit maximally expressive constrained dissipation -- direct-to-target.

This experiment pushes the constrained dissipative tail to its most expressive
configuration on a 4-qubit system:

  * **all-to-all connectivity** (6 edges on 4 qubits)
  * every 1-qubit channel family: amplitude damping, dephasing, depolarising
  * 2-qubit channel families: ZZ dephasing, Bell pumping, parity pumping
    (collective decay disabled -- its superop-to-Kraus conversion can break
    trace preservation numerically)
  * 1-qubit ancilla-reset gadgets (15 params/qubit)
  * 2-qubit ancilla-reset gadgets (15 params/edge)

The optimization is **direct-to-target**: we maximize fidelity to the target
state without any teacher supervision.  The target is a GHZ state on 4 qubits:

    |GHZ_4> = (|0000> + |1111>) / sqrt(2)

Optimization strategy
---------------------
Two-stage approach to handle the high parameter count (186 total):

  **Stage 1** -- Optimize ONLY the 36 rate parameters (angles fixed to 0)
  using L-BFGS-B.  With 36 bounded parameters, BFGS converges quickly.

  **Stage 2** -- Jointly optimize ALL 186 parameters using COBYLA
  (derivative-free), warm-started from Stage 1.  COBYLA handles the
  non-convex landscape of the full model better than BFGS.

This staged approach is far more effective than optimizing all parameters
from random initialization, because Stage 1 finds a good basin for the rate
parameters, and Stage 2 only needs to refine the ancilla angles.

Run
---
    python3 experiments/four_qubit_max_expressive/run_4q_direct_optimization.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "hydrogen_qalchemy"))

from tail.constraint_dissipation import ConstraintConfig, ConstraintDissipation
from tail.fit_constraint_direct_to_target import (
    run_model_fidelities,
    evaluate_and_report_direct,
    _sigmoid,
    _inv_sigmoid,
    _build_curve_weights,
)
from tail.metrics import fidelity_to_pure


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def ghz_state(n: int) -> np.ndarray:
    d = 1 << n
    psi = np.zeros(d, dtype=complex)
    psi[0] = 1.0
    psi[-1] = 1.0
    return psi / np.linalg.norm(psi)


def random_density_matrix(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
    rho = A @ A.conj().T
    return rho / np.trace(rho)


def random_pure_state_dm(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    psi /= np.linalg.norm(psi)
    return np.outer(psi, psi.conj())


def make_max_expressive_config(n_qubits: int = 4) -> ConstraintConfig:
    return ConstraintConfig(
        n_qubits=n_qubits,
        allow_2q=True,
        connectivity="all_to_all",
        gamma_max=2.0,
        enable_amp_damp=True,
        enable_dephasing=True,
        enable_depolarizing=True,
        enable_edge_pumping=True,
        edge_pump_target="phi_plus",
        edge_parity_target="even",
        edge_parity_basis="ZZ",
        # Note: collective decay disabled because its Choi-based Kraus
        # extraction (_superop_to_kraus) can introduce trace-preservation
        # errors that accumulate over many time steps.
        enable_edge_collective_decay=False,
        allow_ancilla_reset=True,
        allow_2q_ancilla_reset=True,
        anc2q_n_params=15,
    )


def _eval_mean_fT(model, rho_list, psi, tmax, steps):
    fTs = []
    for rho0 in rho_list:
        fids = run_model_fidelities(model, rho0, psi, tmax, steps)
        fTs.append(fids[-1])
    return float(np.mean(fTs))


# ---------------------------------------------------------------------------
# Stage 1: rate-only optimization via L-BFGS-B
# ---------------------------------------------------------------------------

def stage1_rates_only(
    model: ConstraintDissipation,
    rho_train: list[np.ndarray],
    psi: np.ndarray,
    cfg: ConstraintConfig,
    tmax: float,
    steps: int,
    weights: np.ndarray,
    *,
    max_iter: int = 200,
    seed: int = 0,
    l2_weight: float = 0.005,
    verbose: bool = True,
) -> np.ndarray:
    """Optimize only rate params (angles=0). Returns best full parameter vector."""

    rate_mask = model.rate_mask()
    n_rates = int(rate_mask.sum())
    n_params = model.num_params()

    # Set all angles to zero
    x_full = model.pack_params()
    x_full[~rate_mask] = 0.0
    model.unpack_params(x_full)

    history: list[float] = []
    t0 = time.perf_counter()

    def objective(x_raw_rates: np.ndarray) -> float:
        rates = _sigmoid(x_raw_rates, 0.0, cfg.gamma_max)
        x = np.zeros(n_params, dtype=float)
        x[rate_mask] = rates
        model.unpack_params(x)

        vals = []
        for rho0 in rho_train:
            fids = run_model_fidelities(model, rho0, psi, tmax, steps)
            vals.append(float(np.sum(weights * (1.0 - fids[1:]))))

        val = float(np.mean(vals))
        val += l2_weight * float(np.sum((rates / cfg.gamma_max) ** 2))
        history.append(val)

        if verbose and len(history) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    eval {len(history):4d}  loss = {val:.6e}  ({elapsed:.1f}s)")
        return val

    rng = np.random.default_rng(seed)
    x0_rates = np.full(n_rates, 0.2)
    x0_rates += rng.uniform(-0.05, 0.05, size=n_rates)
    x0_rates = np.clip(x0_rates, 1e-3, cfg.gamma_max - 1e-3)
    x0_raw = _inv_sigmoid(x0_rates, 0.0, cfg.gamma_max)

    res = minimize(
        objective, x0_raw,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "maxfun": max_iter * 10},
    )

    rates_best = _sigmoid(res.x, 0.0, cfg.gamma_max)
    x_full = np.zeros(n_params, dtype=float)
    x_full[rate_mask] = rates_best
    model.unpack_params(x_full)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"    Stage 1 done: loss = {res.fun:.6e}  "
              f"({len(history)} evals, {elapsed:.1f}s)")

    return model.pack_params()


# ---------------------------------------------------------------------------
# Stage 2: full optimization via COBYLA (warm-started from stage 1)
# ---------------------------------------------------------------------------

def stage2_full_cobyla(
    model: ConstraintDissipation,
    x_warm: np.ndarray,
    rho_train: list[np.ndarray],
    psi: np.ndarray,
    cfg: ConstraintConfig,
    tmax: float,
    steps: int,
    weights: np.ndarray,
    *,
    max_iter: int = 800,
    l2_weight: float = 0.005,
    angle_l2_weight: float = 0.01,
    verbose: bool = True,
) -> np.ndarray:
    """Optimize all params jointly with COBYLA, warm-started."""

    rate_mask = model.rate_mask()
    angle_mask = ~rate_mask
    n_params = model.num_params()

    # Convert warm start to raw space
    x0_raw = np.empty(n_params, dtype=float)
    x0_raw[rate_mask] = _inv_sigmoid(
        np.clip(x_warm[rate_mask], 1e-6, cfg.gamma_max - 1e-6),
        0.0, cfg.gamma_max,
    )
    # Initialize angles to small random values (not zero) so COBYLA
    # has a gradient direction to explore the angle subspace.
    rng = np.random.default_rng(42)
    x0_raw[angle_mask] = rng.uniform(-0.3, 0.3, size=int(angle_mask.sum()))

    history: list[float] = []
    t0 = time.perf_counter()

    def objective(x_raw: np.ndarray) -> float:
        x = np.empty(n_params, dtype=float)
        x[rate_mask] = _sigmoid(x_raw[rate_mask], 0.0, cfg.gamma_max)
        x[angle_mask] = np.clip(x_raw[angle_mask], -2.0, 2.0)
        model.unpack_params(x)

        vals = []
        for rho0 in rho_train:
            fids = run_model_fidelities(model, rho0, psi, tmax, steps)
            vals.append(float(np.sum(weights * (1.0 - fids[1:]))))

        val = float(np.mean(vals))
        rates = x[rate_mask]
        angles = x[angle_mask]
        val += l2_weight * float(np.sum((rates / cfg.gamma_max) ** 2))
        val += angle_l2_weight * float(np.sum(angles ** 2))

        history.append(val)
        if verbose and len(history) % 25 == 0:
            elapsed = time.perf_counter() - t0
            print(f"    eval {len(history):5d}  loss = {val:.6e}  ({elapsed:.1f}s)")
        return val

    res = minimize(
        objective, x0_raw, method="COBYLA",
        options={"maxiter": max_iter, "rhobeg": 0.5},
    )

    x_best = np.empty(n_params, dtype=float)
    x_best[rate_mask] = _sigmoid(res.x[rate_mask], 0.0, cfg.gamma_max)
    x_best[angle_mask] = np.clip(res.x[angle_mask], -2.0, 2.0)
    model.unpack_params(x_best)

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"    Stage 2 done: loss = {res.fun:.6e}  "
              f"({len(history)} evals, {elapsed:.1f}s)")

    return model.pack_params()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    n = 4
    d = 1 << n

    psi_target = ghz_state(n)
    psi = psi_target / np.linalg.norm(psi_target)
    print(f"Target state: 4-qubit GHZ  (d = {d})")

    rho_train = [
        random_density_matrix(d, seed=0),
        random_density_matrix(d, seed=1),
        np.eye(d, dtype=complex) / d,
    ]

    rho_valid = [
        random_density_matrix(d, seed=50),
        random_density_matrix(d, seed=51),
        random_density_matrix(d, seed=52),
        random_pure_state_dm(d, seed=200),
    ]

    print(f"\nTraining initial fidelities ({len(rho_train)} states):")
    for i, rho0 in enumerate(rho_train):
        print(f"  state {i}: F_0 = {fidelity_to_pure(rho0, psi):.6f}")

    cfg = make_max_expressive_config(n)
    model = ConstraintDissipation(cfg)
    print(f"\nMaximally expressive config:")
    print(f"  Connectivity: all-to-all ({len(cfg.edges)} edges)")
    print(f"  Total parameters: {model.num_params()}")
    print(f"  Parameter layout:")
    for name, start, count in model._layout:
        print(f"    {name:30s}: {count:4d} params  (idx {start}..{start+count-1})")

    tmax = 5.0
    steps = 20
    print(f"\nSimulation: tmax={tmax}, steps={steps}")

    # Pre-optimization CPTP sanity check with default parameters
    print("\n  --- Pre-optimization CPTP check (default params) ---")
    dt_pre = tmax / steps
    cptp_pre = model.validate_cptp_debug(dt_pre)
    print(f"  TP: {cptp_pre['is_trace_preserving']}  "
          f"(dev={cptp_pre['trace_deviation_max']:.2e}),  "
          f"CP: {cptp_pre['is_cp']}  "
          f"(min_eig={cptp_pre['choi_min_eig']:.2e})")
    if not cptp_pre['is_trace_preserving']:
        print("  WARNING: channel is not TP even with default params!")

    weights = _build_curve_weights(tmax=tmax, steps=steps, mode="exp", alpha=None)

    t0_total = time.perf_counter()

    # ---- Stage 1: rates only ----
    print("\n" + "=" * 60)
    print("  STAGE 1: Rate-only optimization (L-BFGS-B)")
    print("=" * 60)

    x_stage1 = stage1_rates_only(
        model, rho_train, psi, cfg, tmax, steps, weights,
        max_iter=200, seed=42, verbose=True,
    )
    fT_stage1 = _eval_mean_fT(model, rho_train, psi, tmax, steps)
    print(f"  Stage 1 mean F(T) on train: {fT_stage1:.6f}")

    # ---- Stage 2: full COBYLA ----
    print("\n" + "=" * 60)
    print("  STAGE 2: Full optimization (COBYLA, warm-started)")
    print("=" * 60)

    x_stage2 = stage2_full_cobyla(
        model, x_stage1, rho_train, psi, cfg, tmax, steps, weights,
        max_iter=2000, verbose=True,
    )
    fT_stage2 = _eval_mean_fT(model, rho_train, psi, tmax, steps)
    print(f"  Stage 2 mean F(T) on train: {fT_stage2:.6f}")

    wall_time = time.perf_counter() - t0_total

    # ---- Validation ----
    print("\n" + "=" * 60)
    print("  VALIDATION on held-out states")
    print("=" * 60)

    report = evaluate_and_report_direct(
        model=model, psi_target=psi, rho_valid=rho_valid,
        tmax=tmax, steps=steps, verbose=True,
    )

    print("\n  --- Training set (full resolution) ---")
    report_train = evaluate_and_report_direct(
        model=model, psi_target=psi, rho_valid=rho_train,
        tmax=tmax, steps=steps, verbose=True,
    )

    # ---- CPTP sanity check ----
    print("\n  --- CPTP validation (debug Liouvillian) ---")
    dt_check = tmax / steps
    cptp = model.validate_cptp_debug(dt_check)
    print(f"  Trace-preserving: {cptp['is_trace_preserving']}  "
          f"(max dev = {cptp['trace_deviation_max']:.2e})")
    print(f"  Completely positive: {cptp['is_cp']}  "
          f"(min Choi eigenvalue = {cptp['choi_min_eig']:.2e})")

    # ---- Compare with teacher ----
    print("\n  --- Ideal teacher (target cooling) upper bound ---")
    from tail.target_cooling import run_target_cooling_trajectory
    gamma_teacher = 1.0
    teacher_fTs = []
    for rho0 in rho_valid:
        _, fids_t, _, _ = run_target_cooling_trajectory(
            rho0, psi, gamma_teacher, tmax, steps,
        )
        teacher_fTs.append(fids_t[-1])
    mean_teacher = float(np.mean(teacher_fTs))
    print(f"  Teacher mean F(T)    : {mean_teacher:.6f}  (gamma={gamma_teacher})")
    ratio = report['mean_final_fidelity'] / max(mean_teacher, 1e-12)
    print(f"  Constrained / Teacher: {ratio:.4f}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"  Target state         : 4-qubit GHZ")
    print(f"  Connectivity         : all-to-all ({len(cfg.edges)} edges)")
    print(f"  Total parameters     : {model.num_params()}")
    print(f"  Optimizer            : Stage 1 (L-BFGS-B rates) + Stage 2 (COBYLA full)")
    print(f"  Stage 1 mean F(T)    : {fT_stage1:.6f}  (rates only)")
    print(f"  Stage 2 mean F(T)    : {fT_stage2:.6f}  (rates + angles)")
    print(f"  Training mean F(T)   : {report_train['mean_final_fidelity']:.6f}")
    print(f"  Validation mean F(T) : {report['mean_final_fidelity']:.6f}")
    print(f"  Validation mean gain : {report['mean_gain']:+.6f}")
    print(f"  Teacher mean F(T)    : {mean_teacher:.6f}")
    print(f"  Constrained/Teacher  : {ratio:.4f}")
    print(f"  CPTP                 : TP={cptp['is_trace_preserving']}, "
          f"CP={cptp['is_cp']}")
    print(f"  Wall time            : {wall_time:.1f}s")
    print()


if __name__ == "__main__":
    main()
