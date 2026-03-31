"""Optimizers for the two-moons VQC benchmark.

Implemented optimizers
----------------------
- ``krotov_online``: original single-sample sequential gate update baseline
- ``krotov_batch``: full-batch Krotov-inspired update aligned with mean BCE
- ``krotov_minibatch``: optional mini-batch variant of the batch update
- ``adam``: Adam with full-batch parameter-shift gradients
- ``lbfgs``: L-BFGS-B with parameter-shift gradients

Accounting convention
---------------------
Every trace logs the following cumulative counters:

- ``sample_forward_passes``: number of single-sample forward propagations
- ``sample_backward_passes``: number of single-sample adjoint propagations
- ``full_loss_evaluations``: number of mean-loss evaluations
- ``gradient_evaluations``: number of optimizer update-direction evaluations
- ``cost_units``: ``sample_forward_passes + sample_backward_passes``

``cost_units`` is the main fair comparison axis across optimizers. The legacy
``func_evals`` field is retained as an alias of ``cost_units`` so existing
downstream code does not break.
"""

import time
import numpy as np
from scipy.optimize import minimize

from model import EPS


TRACE_FIELDS = (
    "loss",
    "train_acc",
    "test_acc",
    "step",
    "phase",
    "wall_time",
    "func_evals",
    "grad_evals",
    "sample_forward_passes",
    "sample_backward_passes",
    "full_loss_evaluations",
    "gradient_evaluations",
    "cost_units",
    "step_size",
    "update_norm",
    "gradient_norm",
    "contribution_variance",
)


def _init_trace():
    return {key: [] for key in TRACE_FIELDS}


def _init_counters():
    return {
        "sample_forward_passes": 0,
        "sample_backward_passes": 0,
        "full_loss_evaluations": 0,
        "gradient_evaluations": 0,
    }


def _apply_counter_delta(counters, delta):
    for key in counters:
        counters[key] += int(delta.get(key, 0))


def _current_cost_units(counters):
    return counters["sample_forward_passes"] + counters["sample_backward_passes"]


def _compute_metrics(model, params, X_train, y_train, X_test, y_test, counters):
    """Compute train/test metrics and account for forward-only evaluations."""
    train_loss = model.loss(params, X_train, y_train)
    _apply_counter_delta(
        counters,
        {
            "sample_forward_passes": len(X_train),
            "full_loss_evaluations": 1,
        },
    )

    train_acc = model.accuracy(params, X_train, y_train)
    _apply_counter_delta(counters, {"sample_forward_passes": len(X_train)})

    test_acc = model.accuracy(params, X_test, y_test)
    _apply_counter_delta(counters, {"sample_forward_passes": len(X_test)})

    return train_loss, train_acc, test_acc


def _append_trace(
    trace,
    counters,
    step,
    phase,
    wall_time,
    train_loss,
    train_acc,
    test_acc,
    step_size=np.nan,
    update_norm=np.nan,
    gradient_norm=np.nan,
    contribution_variance=np.nan,
):
    """Append a single checkpoint to the optimizer trace."""
    trace["loss"].append(float(train_loss))
    trace["train_acc"].append(float(train_acc))
    trace["test_acc"].append(float(test_acc))
    trace["step"].append(int(step))
    trace["phase"].append(str(phase))
    trace["wall_time"].append(float(wall_time))
    trace["sample_forward_passes"].append(int(counters["sample_forward_passes"]))
    trace["sample_backward_passes"].append(int(counters["sample_backward_passes"]))
    trace["full_loss_evaluations"].append(int(counters["full_loss_evaluations"]))
    trace["gradient_evaluations"].append(int(counters["gradient_evaluations"]))
    trace["cost_units"].append(int(_current_cost_units(counters)))
    trace["func_evals"].append(int(_current_cost_units(counters)))
    trace["grad_evals"].append(int(counters["gradient_evaluations"]))
    trace["step_size"].append(float(step_size))
    trace["update_norm"].append(float(update_norm))
    trace["gradient_norm"].append(float(gradient_norm))
    trace["contribution_variance"].append(float(contribution_variance))


def _krotov_step_size(base_step_size, iteration, schedule="constant", decay=0.0):
    """Return the scheduled Krotov step size for iteration ``iteration``."""
    exponent = max(iteration - 1, 0)
    if schedule == "constant":
        return base_step_size
    if schedule == "inverse":
        return base_step_size / (1.0 + decay * exponent)
    if schedule == "exp":
        return base_step_size * np.exp(-decay * exponent)
    raise ValueError(f"Unknown Krotov learning-rate schedule: {schedule}")


def _should_early_stop(trace, patience, min_delta, warmup):
    """Return ``True`` when the loss has plateaued for long enough."""
    if patience <= 0 or len(trace["step"]) <= 1:
        return False

    current_step = trace["step"][-1]
    if current_step < warmup:
        return False

    best_loss = trace["loss"][0]
    best_step = trace["step"][0]
    for step, loss in zip(trace["step"], trace["loss"]):
        if best_loss - loss > min_delta:
            best_loss = loss
            best_step = step
    return (current_step - best_step) >= patience


def _krotov_terminal_costate(model, final_state, y):
    """Return the BCE terminal co-state for a single sample.

    Let

    - ``z = <psi|O|psi>``
    - ``p = (z + 1) / 2``
    - ``L = -y log p - (1-y) log(1-p)``

    The Wirtinger derivative with respect to the complex-conjugated state is

    ``dL/dpsi* = (dL/dp) * (dp/dz) * (dz/dpsi*)``

    where ``dp/dz = 1/2`` and, for Hermitian ``O``,
    ``dz/dpsi* = O|psi>``. Therefore

    ``chi_T = dL/dpsi* = 0.5 * (dL/dp) * O|psi>``.

    The previous implementation missed the factor ``1/2``. That only rescales
    the resulting Krotov update direction, so it is equivalent to using a step
    size twice as large. We fix the co-state here and keep the learning rate
    explicit and configurable.
    """
    z = np.real(final_state.conj() @ model.obs @ final_state)
    p = np.clip((z + 1.0) / 2.0, EPS, 1.0 - EPS)
    dloss_dp = -y / p + (1.0 - y) / (1.0 - p)
    chi_terminal = 0.5 * dloss_dp * (model.obs @ final_state)
    return chi_terminal


def _build_costates(gates, chi_terminal):
    """Backward-propagate the terminal co-state through the gate sequence."""
    n_gates = len(gates)
    chi_states = [None] * (n_gates + 1)
    chi_states[n_gates] = chi_terminal
    for k in range(n_gates - 1, -1, -1):
        gate_mat = gates[k][0]
        chi_states[k] = gate_mat.conj().T @ chi_states[k + 1]
    return chi_states


def _sample_krotov_contribution(model, params, x, y):
    """Compute one sample's gate-wise Krotov contribution at fixed params."""
    gates, fwd_states = model.get_gate_sequence_and_states(params, x)
    chi_states = _build_costates(
        gates, _krotov_terminal_costate(model, fwd_states[-1], y)
    )

    contribution = np.zeros_like(params)
    for k, (gate_mat, pidx) in enumerate(gates):
        if pidx is None:
            continue
        gen = model.gate_derivative_generator(pidx, x)
        grad_vec = gen @ fwd_states[k + 1]
        contribution[pidx] = 2.0 * np.real(chi_states[k + 1].conj() @ grad_vec)

    return contribution


def _krotov_contribution_batch(model, params, X, y):
    """Average per-sample Krotov contributions over a batch."""
    contributions = np.zeros((len(X), len(params)), dtype=float)
    for idx, (x_i, y_i) in enumerate(zip(X, y)):
        contributions[idx] = _sample_krotov_contribution(model, params, x_i, y_i)

    mean_contribution = np.mean(contributions, axis=0)
    variance = 0.0 if len(X) <= 1 else float(np.mean(np.var(contributions, axis=0)))
    stats = {
        "sample_forward_passes": len(X),
        "sample_backward_passes": len(X),
        "full_loss_evaluations": 0,
        "gradient_evaluations": 1,
    }
    return mean_contribution, variance, stats


def _krotov_single_sample_update(model, params, x, y, step_size):
    """Original sequential single-sample Krotov update.

    Backward states are computed once with the old parameters and then reused
    while parameters are updated gate-by-gate. This preserves the original
    stale-adjoint online update as the benchmark baseline.
    """
    gates, fwd_states = model.get_gate_sequence_and_states(params, x)
    chi_states = _build_costates(
        gates, _krotov_terminal_costate(model, fwd_states[-1], y)
    )

    new_params = params.copy()
    current_fwd = fwd_states[0].copy()
    sample_contribution = np.zeros_like(params)

    for k, (gate_mat, pidx) in enumerate(gates):
        if pidx is not None:
            gen = model.gate_derivative_generator(pidx, x)
            gate_output = gate_mat @ current_fwd
            grad_vec = gen @ gate_output
            gradient = 2.0 * np.real(chi_states[k + 1].conj() @ grad_vec)
            sample_contribution[pidx] = gradient

            new_params[pidx] -= step_size * gradient
            gate_mat = model.rebuild_param_gate(pidx, new_params, x)

        current_fwd = gate_mat @ current_fwd

    stats = {
        "sample_forward_passes": 1,
        "sample_backward_passes": 1,
        "full_loss_evaluations": 0,
        "gradient_evaluations": 1,
    }
    return new_params, sample_contribution, stats


def _print_progress(name, step, loss, train_acc, test_acc, step_size, cost_units):
    print(
        f"  {name} step {step:3d}: loss={loss:.4f} acc={train_acc:.3f} "
        f"test_acc={test_acc:.3f} step_size={step_size:.4f} cost={cost_units}"
    )


def _resolve_krotov_settings(config, mode):
    """Resolve per-optimizer Krotov settings while preserving legacy fallbacks."""
    if mode == "online":
        return (
            getattr(config, "krotov_online_step_size", config.krotov_step_size),
            getattr(config, "krotov_online_schedule", config.krotov_lr_schedule),
            getattr(config, "krotov_online_decay", config.krotov_decay),
        )
    if mode == "batch":
        return (
            getattr(config, "krotov_batch_step_size", config.krotov_step_size),
            getattr(config, "krotov_batch_schedule", config.krotov_lr_schedule),
            getattr(config, "krotov_batch_decay", config.krotov_decay),
        )
    if mode == "hybrid_online":
        return (
            getattr(config, "hybrid_online_step_size", config.krotov_step_size),
            getattr(config, "hybrid_online_schedule", config.krotov_lr_schedule),
            getattr(config, "hybrid_online_decay", config.krotov_decay),
        )
    if mode == "hybrid_batch":
        return (
            getattr(config, "hybrid_batch_step_size", config.krotov_step_size),
            getattr(config, "hybrid_batch_schedule", config.krotov_lr_schedule),
            getattr(config, "hybrid_batch_decay", config.krotov_decay),
        )
    raise ValueError(f"Unknown Krotov settings mode: {mode}")


def _run_krotov_online_epoch(model, params, X_train, y_train, step_size, iteration_seed):
    """Apply one online Krotov pass over the full training set."""
    params_before = params.copy()
    rng = np.random.RandomState(iteration_seed)
    perm = rng.permutation(len(X_train))
    sample_contributions = []
    counter_delta = _init_counters()

    for idx in perm:
        params, contribution, stats = _krotov_single_sample_update(
            model, params, X_train[idx], y_train[idx], step_size
        )
        _apply_counter_delta(counter_delta, stats)
        sample_contributions.append(contribution)

    sample_contributions = np.asarray(sample_contributions)
    mean_contribution = np.mean(sample_contributions, axis=0)
    diagnostics = {
        "phase": "online",
        "step_size": float(step_size),
        "update_norm": float(np.linalg.norm(params - params_before)),
        "gradient_norm": float(np.linalg.norm(mean_contribution)),
        "contribution_variance": float(np.mean(np.var(sample_contributions, axis=0))),
    }
    return params, diagnostics, counter_delta


def _run_krotov_batch_epoch(model, params, X_train, y_train, step_size):
    """Apply one full-batch Krotov-inspired update."""
    mean_contribution, contribution_variance, counter_delta = _krotov_contribution_batch(
        model, params, X_train, y_train
    )
    update = step_size * mean_contribution
    params = params - update
    diagnostics = {
        "phase": "batch",
        "step_size": float(step_size),
        "update_norm": float(np.linalg.norm(update)),
        "gradient_norm": float(np.linalg.norm(mean_contribution)),
        "contribution_variance": float(contribution_variance),
    }
    return params, diagnostics, counter_delta


def train_krotov_online(
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    max_iterations=200,
    step_size=0.1,
    lr_schedule="constant",
    decay=0.0,
    early_stopping=False,
    early_stopping_patience=0,
    early_stopping_min_delta=0.0,
    early_stopping_warmup=0,
):
    """Train with the original online stale-adjoint Krotov update."""
    trace = _init_trace()
    counters = _init_counters()
    t0 = time.time()

    tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test, counters)
    _append_trace(
        trace,
        counters,
        0,
        "init",
        0.0,
        tl,
        ta,
        tea,
        step_size=0.0,
        update_norm=0.0,
        gradient_norm=0.0,
        contribution_variance=0.0,
    )

    for it in range(1, max_iterations + 1):
        current_step = _krotov_step_size(step_size, it, lr_schedule, decay)
        params, diagnostics, counter_delta = _run_krotov_online_epoch(
            model, params, X_train, y_train, current_step, it
        )
        _apply_counter_delta(counters, counter_delta)

        tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test, counters)
        _append_trace(
            trace,
            counters,
            it,
            diagnostics["phase"],
            time.time() - t0,
            tl,
            ta,
            tea,
            step_size=diagnostics["step_size"],
            update_norm=diagnostics["update_norm"],
            gradient_norm=diagnostics["gradient_norm"],
            contribution_variance=diagnostics["contribution_variance"],
        )

        if it % 20 == 0 or it == 1:
            _print_progress("Krotov-online", it, tl, ta, tea, current_step, trace["cost_units"][-1])

        if early_stopping and _should_early_stop(
            trace,
            early_stopping_patience,
            early_stopping_min_delta,
            early_stopping_warmup,
        ):
            print(f"  Krotov-online early stop at step {it}: loss plateau")
            break

    return params, trace


def train_krotov_batch(
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    max_iterations=200,
    step_size=0.1,
    lr_schedule="constant",
    decay=0.0,
    early_stopping=False,
    early_stopping_patience=0,
    early_stopping_min_delta=0.0,
    early_stopping_warmup=0,
):
    """Train with a full-batch Krotov-inspired update.

    Each outer iteration computes the gate-wise contribution for every training
    sample at the current parameters, averages those contributions, and updates
    all parameters once. This aligns the update direction with the benchmark's
    full-batch mean BCE objective and avoids the stale-adjoint sequential drift
    of the online baseline.
    """
    trace = _init_trace()
    counters = _init_counters()
    t0 = time.time()

    tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test, counters)
    _append_trace(
        trace,
        counters,
        0,
        "init",
        0.0,
        tl,
        ta,
        tea,
        step_size=0.0,
        update_norm=0.0,
        gradient_norm=0.0,
        contribution_variance=0.0,
    )

    for it in range(1, max_iterations + 1):
        current_step = _krotov_step_size(step_size, it, lr_schedule, decay)
        params, diagnostics, counter_delta = _run_krotov_batch_epoch(
            model, params, X_train, y_train, current_step
        )
        _apply_counter_delta(counters, counter_delta)

        tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test, counters)
        _append_trace(
            trace,
            counters,
            it,
            diagnostics["phase"],
            time.time() - t0,
            tl,
            ta,
            tea,
            step_size=diagnostics["step_size"],
            update_norm=diagnostics["update_norm"],
            gradient_norm=diagnostics["gradient_norm"],
            contribution_variance=diagnostics["contribution_variance"],
        )

        if it % 20 == 0 or it == 1:
            _print_progress("Krotov-batch", it, tl, ta, tea, current_step, trace["cost_units"][-1])

        if early_stopping and _should_early_stop(
            trace,
            early_stopping_patience,
            early_stopping_min_delta,
            early_stopping_warmup,
        ):
            print(f"  Krotov-batch early stop at step {it}: loss plateau")
            break

    return params, trace


def train_krotov_minibatch(
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    max_iterations=200,
    step_size=0.1,
    batch_size=32,
    lr_schedule="constant",
    decay=0.0,
    early_stopping=False,
    early_stopping_patience=0,
    early_stopping_min_delta=0.0,
    early_stopping_warmup=0,
):
    """Train with a mini-batch Krotov-inspired update."""
    trace = _init_trace()
    counters = _init_counters()
    t0 = time.time()

    tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test, counters)
    _append_trace(
        trace,
        counters,
        0,
        "init",
        0.0,
        tl,
        ta,
        tea,
        step_size=0.0,
        update_norm=0.0,
        gradient_norm=0.0,
        contribution_variance=0.0,
    )

    batch_size = min(batch_size, len(X_train))

    for it in range(1, max_iterations + 1):
        current_step = _krotov_step_size(step_size, it, lr_schedule, decay)
        params_before = params.copy()

        rng = np.random.RandomState(it)
        perm = rng.permutation(len(X_train))
        gradient_norms = []
        variances = []

        for start in range(0, len(X_train), batch_size):
            batch_idx = perm[start:start + batch_size]
            mean_contribution, variance, stats = _krotov_contribution_batch(
                model, params, X_train[batch_idx], y_train[batch_idx]
            )
            _apply_counter_delta(counters, stats)
            params = params - current_step * mean_contribution
            gradient_norms.append(np.linalg.norm(mean_contribution))
            variances.append(variance)

        tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test, counters)
        _append_trace(
            trace,
            counters,
            it,
            "minibatch",
            time.time() - t0,
            tl,
            ta,
            tea,
            step_size=current_step,
            update_norm=float(np.linalg.norm(params - params_before)),
            gradient_norm=float(np.mean(gradient_norms)),
            contribution_variance=float(np.mean(variances)),
        )

        if it % 20 == 0 or it == 1:
            _print_progress("Krotov-minibatch", it, tl, ta, tea, current_step, trace["cost_units"][-1])

        if early_stopping and _should_early_stop(
            trace,
            early_stopping_patience,
            early_stopping_min_delta,
            early_stopping_warmup,
        ):
            print(f"  Krotov-minibatch early stop at step {it}: loss plateau")
            break

    return params, trace


def train_adam(
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    max_iterations=200,
    lr=0.05,
):
    """Train using Adam with exact full-batch BCE gradients."""
    trace = _init_trace()
    counters = _init_counters()

    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    t0 = time.time()
    tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test, counters)
    _append_trace(
        trace,
        counters,
        0,
        "init",
        0.0,
        tl,
        ta,
        tea,
        step_size=0.0,
        update_norm=0.0,
        gradient_norm=0.0,
        contribution_variance=np.nan,
    )

    for it in range(1, max_iterations + 1):
        grad, grad_stats = model.loss_gradient(params, X_train, y_train)
        _apply_counter_delta(counters, grad_stats)
        counters["gradient_evaluations"] += 1

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad**2
        m_hat = m / (1.0 - beta1**it)
        v_hat = v / (1.0 - beta2**it)
        update = lr * m_hat / (np.sqrt(v_hat) + eps_adam)
        params = params - update

        tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test, counters)
        _append_trace(
            trace,
            counters,
            it,
            "adam",
            time.time() - t0,
            tl,
            ta,
            tea,
            step_size=lr,
            update_norm=float(np.linalg.norm(update)),
            gradient_norm=float(np.linalg.norm(grad)),
            contribution_variance=np.nan,
        )

        if it % 20 == 0 or it == 1:
            _print_progress("Adam", it, tl, ta, tea, lr, trace["cost_units"][-1])

    return params, trace


def train_lbfgs(model, params, X_train, y_train, X_test, y_test, max_iterations=200):
    """Train using L-BFGS-B with exact full-batch BCE gradients."""
    trace = _init_trace()
    counters = _init_counters()
    step_counter = [0]
    previous_point = [params.copy()]
    t0 = time.time()

    tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test, counters)
    _append_trace(
        trace,
        counters,
        0,
        "init",
        0.0,
        tl,
        ta,
        tea,
        step_size=0.0,
        update_norm=0.0,
        gradient_norm=0.0,
        contribution_variance=np.nan,
    )

    def objective_and_grad(p):
        loss = model.loss(p, X_train, y_train)
        _apply_counter_delta(
            counters,
            {
                "sample_forward_passes": len(X_train),
                "full_loss_evaluations": 1,
            },
        )

        grad, grad_stats = model.loss_gradient(p, X_train, y_train)
        _apply_counter_delta(counters, grad_stats)
        counters["gradient_evaluations"] += 1

        step_counter[0] += 1
        ta = model.accuracy(p, X_train, y_train)
        _apply_counter_delta(counters, {"sample_forward_passes": len(X_train)})
        tea = model.accuracy(p, X_test, y_test)
        _apply_counter_delta(counters, {"sample_forward_passes": len(X_test)})

        update_norm = float(np.linalg.norm(p - previous_point[0]))
        previous_point[0] = p.copy()
        _append_trace(
            trace,
            counters,
            step_counter[0],
            "lbfgs",
            time.time() - t0,
            loss,
            ta,
            tea,
            step_size=np.nan,
            update_norm=update_norm,
            gradient_norm=float(np.linalg.norm(grad)),
            contribution_variance=np.nan,
        )

        if step_counter[0] % 20 == 0 or step_counter[0] == 1:
            _print_progress("L-BFGS-B", step_counter[0], loss, ta, tea, np.nan, trace["cost_units"][-1])

        return loss, grad

    result = minimize(
        objective_and_grad,
        params,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max_iterations,
            "maxfun": max_iterations * 20,
            "maxls": 40,
            "maxcor": 20,
            "gtol": 1e-7,
            "ftol": 1e-12,
        },
    )

    return result.x, trace


def train_krotov_hybrid(
    model,
    params,
    X_train,
    y_train,
    X_test,
    y_test,
    max_iterations=200,
    switch_iteration=20,
    online_step_size=0.3,
    batch_step_size=1.0,
    online_schedule="constant",
    batch_schedule="constant",
    online_decay=0.0,
    batch_decay=0.0,
    early_stopping=False,
    early_stopping_patience=0,
    early_stopping_min_delta=0.0,
    early_stopping_warmup=0,
):
    """Train with an online-then-batch Krotov schedule.

    The online phase uses the original sequential stale-adjoint rule for the
    first ``switch_iteration`` outer iterations. The batch phase then switches
    to the full-batch objective-aligned Krotov update. Each phase keeps its own
    schedule counter, so the batch phase starts at its configured step size when
    the switch happens.
    """
    trace = _init_trace()
    counters = _init_counters()
    t0 = time.time()

    tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test, counters)
    _append_trace(
        trace,
        counters,
        0,
        "init",
        0.0,
        tl,
        ta,
        tea,
        step_size=0.0,
        update_norm=0.0,
        gradient_norm=0.0,
        contribution_variance=0.0,
    )

    for it in range(1, max_iterations + 1):
        if it <= switch_iteration:
            phase_iteration = it
            current_step = _krotov_step_size(
                online_step_size, phase_iteration, online_schedule, online_decay
            )
            params, diagnostics, counter_delta = _run_krotov_online_epoch(
                model, params, X_train, y_train, current_step, it
            )
        else:
            phase_iteration = it - switch_iteration
            current_step = _krotov_step_size(
                batch_step_size, phase_iteration, batch_schedule, batch_decay
            )
            params, diagnostics, counter_delta = _run_krotov_batch_epoch(
                model, params, X_train, y_train, current_step
            )

        _apply_counter_delta(counters, counter_delta)
        tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test, counters)
        _append_trace(
            trace,
            counters,
            it,
            diagnostics["phase"],
            time.time() - t0,
            tl,
            ta,
            tea,
            step_size=diagnostics["step_size"],
            update_norm=diagnostics["update_norm"],
            gradient_norm=diagnostics["gradient_norm"],
            contribution_variance=diagnostics["contribution_variance"],
        )

        if it % 20 == 0 or it == 1 or it == switch_iteration or it == switch_iteration + 1:
            _print_progress(
                "Krotov-hybrid",
                it,
                tl,
                ta,
                tea,
                current_step,
                trace["cost_units"][-1],
            )

        if early_stopping and _should_early_stop(
            trace,
            early_stopping_patience,
            early_stopping_min_delta,
            early_stopping_warmup,
        ):
            print(f"  Krotov-hybrid early stop at step {it}: loss plateau")
            break

    return params, trace


def run_optimizer(name, model, params, X_train, y_train, X_test, y_test, config):
    """Run a named optimizer and return ``(final_params, trace)``."""
    if name == "krotov_online":
        step_size, lr_schedule, decay = _resolve_krotov_settings(config, "online")
        return train_krotov_online(
            model,
            params,
            X_train,
            y_train,
            X_test,
            y_test,
            max_iterations=config.max_iterations,
            step_size=step_size,
            lr_schedule=lr_schedule,
            decay=decay,
            early_stopping=getattr(config, "early_stopping_enabled", False),
            early_stopping_patience=getattr(config, "early_stopping_patience", 0),
            early_stopping_min_delta=getattr(config, "early_stopping_min_delta", 0.0),
            early_stopping_warmup=getattr(config, "early_stopping_warmup", 0),
        )

    if name == "krotov_batch":
        step_size, lr_schedule, decay = _resolve_krotov_settings(config, "batch")
        return train_krotov_batch(
            model,
            params,
            X_train,
            y_train,
            X_test,
            y_test,
            max_iterations=config.max_iterations,
            step_size=step_size,
            lr_schedule=lr_schedule,
            decay=decay,
            early_stopping=getattr(config, "early_stopping_enabled", False),
            early_stopping_patience=getattr(config, "early_stopping_patience", 0),
            early_stopping_min_delta=getattr(config, "early_stopping_min_delta", 0.0),
            early_stopping_warmup=getattr(config, "early_stopping_warmup", 0),
        )

    if name == "krotov_minibatch":
        if config.krotov_batch_size is None:
            raise ValueError("krotov_minibatch requires config.krotov_batch_size")
        return train_krotov_minibatch(
            model,
            params,
            X_train,
            y_train,
            X_test,
            y_test,
            max_iterations=config.max_iterations,
            step_size=config.krotov_step_size,
            batch_size=config.krotov_batch_size,
            lr_schedule=config.krotov_lr_schedule,
            decay=config.krotov_decay,
            early_stopping=getattr(config, "early_stopping_enabled", False),
            early_stopping_patience=getattr(config, "early_stopping_patience", 0),
            early_stopping_min_delta=getattr(config, "early_stopping_min_delta", 0.0),
            early_stopping_warmup=getattr(config, "early_stopping_warmup", 0),
        )

    if name == "krotov_hybrid":
        online_step_size, online_schedule, online_decay = _resolve_krotov_settings(
            config, "hybrid_online"
        )
        batch_step_size, batch_schedule, batch_decay = _resolve_krotov_settings(
            config, "hybrid_batch"
        )
        return train_krotov_hybrid(
            model,
            params,
            X_train,
            y_train,
            X_test,
            y_test,
            max_iterations=config.max_iterations,
            switch_iteration=config.hybrid_switch_iteration,
            online_step_size=online_step_size,
            batch_step_size=batch_step_size,
            online_schedule=online_schedule,
            batch_schedule=batch_schedule,
            online_decay=online_decay,
            batch_decay=batch_decay,
            early_stopping=getattr(config, "early_stopping_enabled", False),
            early_stopping_patience=getattr(config, "early_stopping_patience", 0),
            early_stopping_min_delta=getattr(config, "early_stopping_min_delta", 0.0),
            early_stopping_warmup=getattr(config, "early_stopping_warmup", 0),
        )

    if name == "adam":
        return train_adam(
            model,
            params,
            X_train,
            y_train,
            X_test,
            y_test,
            max_iterations=config.max_iterations,
            lr=config.adam_lr,
        )

    if name == "lbfgs":
        return train_lbfgs(
            model,
            params,
            X_train,
            y_train,
            X_test,
            y_test,
            max_iterations=config.lbfgs_maxiter,
        )

    raise ValueError(f"Unknown optimizer: {name}")
