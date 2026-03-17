r"""Direct optimization of constrained dissipation toward a target state.

Unlike ``fit_constraint_to_target.py``, this module does NOT use the ideal
teacher tail as supervision.  Instead, it directly optimizes constrained-tail
parameters to maximize fidelity to ``psi_target`` along the dissipative
trajectory from provided initial states.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.optimize import minimize

from .constraint_dissipation import ConstraintConfig, ConstraintDissipation
from .metrics import fidelity_to_pure


def _sigmoid(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Map R -> (lo, hi) via sigmoid."""
    return lo + (hi - lo) / (1.0 + np.exp(-x))


def _inv_sigmoid(y: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Inverse sigmoid: (lo, hi) -> R."""
    y = np.clip(y, lo + 1e-12, hi - 1e-12)
    return np.log((y - lo) / (hi - y))


def _build_curve_weights(
    tmax: float,
    steps: int,
    mode: str,
    alpha: Optional[float],
) -> np.ndarray:
    """Build normalized positive weights for curve loss."""
    if steps <= 0:
        return np.array([], dtype=float)
    times = np.linspace(0.0, tmax, steps + 1)[1:]  # exclude t=0

    if mode == "uniform":
        w = np.ones(steps, dtype=float)
    elif mode == "exp":
        a = (2.0 / max(tmax, 1e-12)) if alpha is None else float(alpha)
        w = np.exp(-a * times)
    elif mode == "early":
        cutoff = max(1, int(0.4 * steps))
        w = np.ones(steps, dtype=float)
        w[:cutoff] = 3.0
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    w = np.clip(w, 1e-16, None)
    return w / np.sum(w)


def _regularization(
    rate_params: np.ndarray,
    angle_params: np.ndarray,
    cfg: ConstraintConfig,
    l2_weight: float,
    angle_l2_weight: float,
) -> float:
    """L2 regularization on rates and optional angles."""
    reg = 0.0
    if rate_params.size > 0 and l2_weight > 0.0:
        reg += l2_weight * float(np.sum((rate_params / cfg.gamma_max) ** 2))
    if angle_params.size > 0 and angle_l2_weight > 0.0:
        reg += angle_l2_weight * float(np.sum(angle_params ** 2))
    return reg


def run_model_fidelities(
    model: ConstraintDissipation,
    rho0: np.ndarray,
    psi_target: np.ndarray,
    tmax: float,
    steps: int,
) -> np.ndarray:
    """Run constrained tail and return fidelity curve [F(0), ..., F(T)]."""
    psi = np.asarray(psi_target, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)

    dt = tmax / steps if steps > 0 else 0.0
    model.prepare_step(dt)
    rho = rho0.copy()
    fids = np.empty(steps + 1, dtype=float)
    fids[0] = fidelity_to_pure(rho, psi)
    for j in range(steps):
        rho = model.cached_step(rho)
        fids[j + 1] = fidelity_to_pure(rho, psi)
    return np.clip(fids, -1e-12, 1.0 + 1e-12)


def _loss_terminal(
    model: ConstraintDissipation,
    rho_train: list[np.ndarray],
    psi_target: np.ndarray,
    tmax: float,
    steps: int,
) -> float:
    """Terminal-fidelity loss: mean_s (1 - F_s(T))."""
    vals = []
    for rho0 in rho_train:
        fids = run_model_fidelities(model, rho0, psi_target, tmax, steps)
        vals.append(1.0 - fids[-1])
    return float(np.mean(vals)) if vals else 0.0


def _loss_curve(
    model: ConstraintDissipation,
    rho_train: list[np.ndarray],
    psi_target: np.ndarray,
    tmax: float,
    steps: int,
    weights: np.ndarray,
) -> float:
    """Weighted curve loss: mean_s sum_j w_j * (1 - F_s(t_j))."""
    vals = []
    for rho0 in rho_train:
        fids = run_model_fidelities(model, rho0, psi_target, tmax, steps)
        vals.append(float(np.sum(weights * (1.0 - fids[1:]))))
    return float(np.mean(vals)) if vals else 0.0


def fit_to_target(
    psi_target: np.ndarray,
    rho_train: list[np.ndarray],
    cfg: ConstraintConfig,
    tmax: float,
    steps: int,
    loss: str = "curve",                 # {"curve","terminal"}
    weight_mode: str = "exp",            # {"uniform","exp","early"}
    weight_alpha: float | None = None,   # used if weight_mode="exp"
    max_iter: int = 200,
    seed: int = 0,
    l2_weight: float = 0.01,
    angle_l2_weight: float = 0.0,
    verbose: bool = True,
    fit_steps: int | None = None,        # optional coarser grid during optimization
) -> dict:
    """Fit constrained dissipation directly to maximize target fidelity."""
    import time as _time

    psi = np.asarray(psi_target, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)
    opt_steps = fit_steps if fit_steps is not None else steps

    model = ConstraintDissipation(cfg)
    n_params = model.num_params()
    if verbose:
        print(f"  Direct-to-target constrained model: {n_params} parameters")
        print(f"  Optimisation grid steps: {opt_steps}")

    # Rate vs angle parameter indices.
    rate_mask = np.ones(n_params, dtype=bool)
    for name, start, count in model._layout:
        lname = name.lower()
        if ("ancilla" in lname) and ("reset" in lname):
            rate_mask[start:start + count] = False
    angle_mask = ~rate_mask

    weights = _build_curve_weights(
        tmax=tmax, steps=opt_steps, mode=weight_mode, alpha=weight_alpha
    )
    history: list[float] = []
    t_start = _time.perf_counter()
    warned_trace = False

    def objective(x_raw: np.ndarray) -> float:
        nonlocal warned_trace

        x = np.empty(n_params, dtype=float)
        x[rate_mask] = _sigmoid(x_raw[rate_mask], 0.0, cfg.gamma_max)
        x[angle_mask] = x_raw[angle_mask]
        model.unpack_params(x)

        if loss == "terminal":
            val = _loss_terminal(model, rho_train, psi, tmax, opt_steps)
        elif loss == "curve":
            val = _loss_curve(model, rho_train, psi, tmax, opt_steps, weights)
        else:
            raise ValueError("loss must be 'curve' or 'terminal'")

        val += _regularization(
            rate_params=x[rate_mask],
            angle_params=x[angle_mask],
            cfg=cfg,
            l2_weight=l2_weight,
            angle_l2_weight=angle_l2_weight,
        )
        history.append(float(val))

        # Lightweight trace sanity check once in a while.
        if (not warned_trace) and (len(history) % 25 == 0) and rho_train:
            dt = tmax / opt_steps if opt_steps > 0 else 0.0
            model.prepare_step(dt)
            rho = rho_train[0].copy()
            max_dev = 0.0
            for _ in range(opt_steps):
                rho = model.cached_step(rho)
                max_dev = max(max_dev, abs(np.trace(rho) - 1.0))
            if max_dev > 1e-6 and verbose:
                print(f"  [warn] trace drift detected: max |Tr(rho)-1|={max_dev:.2e}")
                warned_trace = True

        if verbose and len(history) % 10 == 0:
            elapsed = _time.perf_counter() - t_start
            print(f"    eval {len(history):4d}  loss = {val:.6e}  ({elapsed:.1f}s)")
        return float(val)

    rng = np.random.default_rng(seed)
    x0_real = np.full(n_params, 0.2)
    x0_real += rng.uniform(-0.05, 0.05, size=n_params)
    x0_real = np.clip(x0_real, 1e-3, cfg.gamma_max - 1e-3)
    x0_raw = np.empty(n_params, dtype=float)
    x0_raw[rate_mask] = _inv_sigmoid(x0_real[rate_mask], 0.0, cfg.gamma_max)
    x0_raw[angle_mask] = rng.uniform(-0.05, 0.05, size=int(angle_mask.sum()))

    if verbose:
        print(f"  Optimising direct loss='{loss}' (max_iter={max_iter}) ...")

    res = minimize(
        objective,
        x0_raw,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "maxfun": max_iter * 10},
    )

    x_best = np.empty(n_params, dtype=float)
    x_best[rate_mask] = _sigmoid(res.x[rate_mask], 0.0, cfg.gamma_max)
    x_best[angle_mask] = res.x[angle_mask]
    model.unpack_params(x_best)

    elapsed = _time.perf_counter() - t_start
    if verbose:
        print(f"  Optimisation finished: {res.message}")
        print(f"  Best loss = {res.fun:.6e}  ({len(history)} evaluations, {elapsed:.1f}s)")

    return {
        "best_params": model.pack_params(),
        "best_loss": float(res.fun),
        "history": history,
        "model": model,
        "scipy_result": res,
    }


def evaluate_and_report_direct(
    model: ConstraintDissipation,
    psi_target: np.ndarray,
    rho_valid: list[np.ndarray],
    tmax: float,
    steps: int,
    verbose: bool = True,
) -> dict:
    """Evaluate direct-optimized model on validation states."""
    psi = np.asarray(psi_target, dtype=complex).ravel()
    psi = psi / np.linalg.norm(psi)

    f0 = []
    fT = []
    gains = []
    trace_devs = []

    dt = tmax / steps if steps > 0 else 0.0
    model.prepare_step(dt)
    for rho0 in rho_valid:
        rho = rho0.copy()
        f_init = fidelity_to_pure(rho, psi)
        max_dev = abs(np.trace(rho) - 1.0)
        for _ in range(steps):
            rho = model.cached_step(rho)
            max_dev = max(max_dev, abs(np.trace(rho) - 1.0))
        f_fin = fidelity_to_pure(rho, psi)
        f0.append(float(f_init))
        fT.append(float(f_fin))
        gains.append(float(f_fin - f_init))
        trace_devs.append(float(max_dev))

    f0_arr = np.asarray(f0, dtype=float)
    fT_arr = np.asarray(fT, dtype=float)
    g_arr = np.asarray(gains, dtype=float)
    tr_arr = np.asarray(trace_devs, dtype=float)

    if verbose:
        print("\n  === Direct-to-target validation report ===")
        print(f"  Mean initial fidelity : {np.mean(f0_arr):.6f}")
        print(f"  Mean final fidelity   : {np.mean(fT_arr):.6f}")
        print(f"  Mean fidelity gain    : {np.mean(g_arr):+.6f}")
        print(f"  Gain min/med/max      : {np.min(g_arr):+.6f} / "
              f"{np.median(g_arr):+.6f} / {np.max(g_arr):+.6f}")
        print(f"  Max trace deviation   : {np.max(tr_arr):.2e}")

    return {
        "initial_fidelities": f0_arr.tolist(),
        "final_fidelities": fT_arr.tolist(),
        "gains": g_arr.tolist(),
        "mean_initial_fidelity": float(np.mean(f0_arr)),
        "mean_final_fidelity": float(np.mean(fT_arr)),
        "mean_gain": float(np.mean(g_arr)),
        "max_trace_deviation": float(np.max(tr_arr)),
    }

