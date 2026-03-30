"""Optimizers for the VQC benchmark.

Implements:
    - Krotov (sequential gate-by-gate parameter update)
    - Adam (with parameter-shift gradients)
    - L-BFGS-B (via scipy.optimize.minimize)

Each optimizer returns a trace dict with per-step metrics.
"""

import time
import numpy as np
from scipy.optimize import minimize

from model import VQCModel, EPS


def _compute_metrics(model, params, X_train, y_train, X_test, y_test):
    """Compute loss and accuracy on train and test sets."""
    train_loss = model.loss(params, X_train, y_train)
    train_acc = model.accuracy(params, X_train, y_train)
    test_acc = model.accuracy(params, X_test, y_test)
    return train_loss, train_acc, test_acc


# ---------------------------------------------------------------------------
# Krotov optimizer
# ---------------------------------------------------------------------------

def _krotov_single_sample_update(model, params, x, y, step_size):
    """Krotov sequential update for a single training sample.

    For each parameterized gate (in circuit order):
      1. Compute gradient using current forward state and old backward state
      2. Update the parameter
      3. Re-propagate forward state with updated parameter

    Returns updated params and number of effective circuit evaluations.
    """
    gates, fwd_states = model.get_gate_sequence_and_states(params, x)

    # Compute output and cost gradient
    final_state = fwd_states[-1]
    z = np.real(final_state.conj() @ model.obs @ final_state)
    p = np.clip((z + 1) / 2, EPS, 1 - EPS)

    # d(BCE)/dp = -y/p + (1-y)/(1-p), dp/dz = 0.5
    dp = -y / p + (1 - y) / (1 - p)
    # chi = d(loss)/d(psi*) = dp * 0.5 * 2 * obs @ psi = dp * obs @ psi
    chi = dp * (model.obs @ final_state)

    # Backward propagate chi through all gates (using OLD parameters)
    n_gates = len(gates)
    chi_states = [None] * (n_gates + 1)
    chi_states[n_gates] = chi
    for k in range(n_gates - 1, -1, -1):
        gate_mat = gates[k][0]
        chi_states[k] = gate_mat.conj().T @ chi_states[k + 1]

    # Sequential Krotov update
    new_params = params.copy()
    # Track current forward state (will be updated after each param change)
    current_fwd = fwd_states[0].copy()  # |0000> state

    for k in range(n_gates):
        gate_mat, pidx = gates[k]
        if pidx is not None:
            # Compute gradient: Re(chi_k^dag @ dG/dtheta @ fwd_{k})
            gen = model.gate_derivative_generator(pidx)
            # dG/dtheta @ |fwd> = gen @ gate @ |fwd>
            # But we need gen applied to the gate output:
            # d(G|psi>)/dtheta = gen @ (G|psi>)
            gate_output = gate_mat @ current_fwd
            grad_vec = gen @ gate_output
            gradient = 2 * np.real(chi_states[k + 1].conj() @ grad_vec)

            # Update parameter
            new_params[pidx] -= step_size * gradient

            # Rebuild gate with new parameter and propagate forward
            gate_mat = _rebuild_single_gate(model, pidx, new_params, x)

        current_fwd = gate_mat @ current_fwd

    # Count: 1 full forward + 1 full backward = ~2 circuit evaluations
    n_evals = 2
    return new_params, n_evals


def _rebuild_single_gate(model, pidx, params, x):
    """Rebuild a single parameterized gate after parameter update."""
    from model import _ry, _rz, _single_qubit_gate
    nq = model.n_qubits
    n_per_layer = nq * 2
    pos_in_layer = pidx % n_per_layer

    if pos_in_layer < nq:
        qubit = pos_in_layer
        return _single_qubit_gate(_ry(params[pidx]), qubit, nq)
    else:
        qubit = pos_in_layer - nq
        return _single_qubit_gate(_rz(params[pidx]), qubit, nq)


def train_krotov(model, params, X_train, y_train, X_test, y_test,
                 max_iterations=200, step_size=0.3):
    """Train using Krotov's sequential update method.

    Each iteration sweeps over the full training set, updating parameters
    sample-by-sample with sequential gate updates.
    """
    trace = {
        "loss": [], "train_acc": [], "test_acc": [],
        "step": [], "func_evals": [], "grad_evals": [], "wall_time": []
    }

    total_evals = 0
    t0 = time.time()

    # Initial metrics
    tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test)
    total_evals += len(X_train) + len(X_test)
    trace["loss"].append(tl)
    trace["train_acc"].append(ta)
    trace["test_acc"].append(tea)
    trace["step"].append(0)
    trace["func_evals"].append(total_evals)
    trace["grad_evals"].append(0)
    trace["wall_time"].append(0.0)

    for it in range(1, max_iterations + 1):
        # Shuffle training data
        rng = np.random.RandomState(it)
        perm = rng.permutation(len(X_train))

        iter_evals = 0
        for idx in perm:
            params, n_ev = _krotov_single_sample_update(
                model, params, X_train[idx], y_train[idx], step_size
            )
            iter_evals += n_ev

        total_evals += iter_evals

        # Compute metrics
        tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test)
        total_evals += len(X_train) + len(X_test)

        trace["loss"].append(tl)
        trace["train_acc"].append(ta)
        trace["test_acc"].append(tea)
        trace["step"].append(it)
        trace["func_evals"].append(total_evals)
        trace["grad_evals"].append(it)
        trace["wall_time"].append(time.time() - t0)

        if it % 20 == 0 or it == 1:
            print(f"  Krotov step {it:3d}: loss={tl:.4f} acc={ta:.3f} "
                  f"test_acc={tea:.3f} evals={total_evals}")

    return params, trace


# ---------------------------------------------------------------------------
# Adam optimizer
# ---------------------------------------------------------------------------

def train_adam(model, params, X_train, y_train, X_test, y_test,
              max_iterations=200, lr=0.05):
    """Train using Adam with parameter-shift gradients."""
    trace = {
        "loss": [], "train_acc": [], "test_acc": [],
        "step": [], "func_evals": [], "grad_evals": [], "wall_time": []
    }

    # Adam state
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2, eps_adam = 0.9, 0.999, 1e-8

    total_evals = 0
    total_grad_evals = 0
    t0 = time.time()

    # Initial metrics
    tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test)
    total_evals += len(X_train) + len(X_test)
    trace["loss"].append(tl)
    trace["train_acc"].append(ta)
    trace["test_acc"].append(tea)
    trace["step"].append(0)
    trace["func_evals"].append(total_evals)
    trace["grad_evals"].append(0)
    trace["wall_time"].append(0.0)

    for it in range(1, max_iterations + 1):
        grad, n_evals = model.param_shift_gradient(params, X_train, y_train)
        total_evals += n_evals
        total_grad_evals += 1

        # Adam update
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m / (1 - beta1**it)
        v_hat = v / (1 - beta2**it)
        params = params - lr * m_hat / (np.sqrt(v_hat) + eps_adam)

        # Metrics
        tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test)
        total_evals += len(X_train) + len(X_test)

        trace["loss"].append(tl)
        trace["train_acc"].append(ta)
        trace["test_acc"].append(tea)
        trace["step"].append(it)
        trace["func_evals"].append(total_evals)
        trace["grad_evals"].append(total_grad_evals)
        trace["wall_time"].append(time.time() - t0)

        if it % 20 == 0 or it == 1:
            print(f"  Adam step {it:3d}: loss={tl:.4f} acc={ta:.3f} "
                  f"test_acc={tea:.3f} evals={total_evals}")

    return params, trace


# ---------------------------------------------------------------------------
# L-BFGS-B optimizer
# ---------------------------------------------------------------------------

def train_lbfgs(model, params, X_train, y_train, X_test, y_test,
                max_iterations=200):
    """Train using L-BFGS-B with parameter-shift gradients."""
    trace = {
        "loss": [], "train_acc": [], "test_acc": [],
        "step": [], "func_evals": [], "grad_evals": [], "wall_time": []
    }

    total_evals = [0]
    total_grad_evals = [0]
    step_counter = [0]
    t0 = time.time()

    # Initial metrics
    tl, ta, tea = _compute_metrics(model, params, X_train, y_train, X_test, y_test)
    total_evals[0] += len(X_train) + len(X_test)
    trace["loss"].append(tl)
    trace["train_acc"].append(ta)
    trace["test_acc"].append(tea)
    trace["step"].append(0)
    trace["func_evals"].append(total_evals[0])
    trace["grad_evals"].append(0)
    trace["wall_time"].append(0.0)

    def objective_and_grad(p):
        loss = model.loss(p, X_train, y_train)
        total_evals[0] += len(X_train)  # one full batch forward

        grad, n_ev = model.param_shift_gradient(p, X_train, y_train)
        total_evals[0] += n_ev
        total_grad_evals[0] += 1

        step_counter[0] += 1
        s = step_counter[0]

        ta = model.accuracy(p, X_train, y_train)
        tea = model.accuracy(p, X_test, y_test)
        total_evals[0] += len(X_train) + len(X_test)

        trace["loss"].append(loss)
        trace["train_acc"].append(ta)
        trace["test_acc"].append(tea)
        trace["step"].append(s)
        trace["func_evals"].append(total_evals[0])
        trace["grad_evals"].append(total_grad_evals[0])
        trace["wall_time"].append(time.time() - t0)

        if s % 20 == 0 or s == 1:
            print(f"  L-BFGS-B step {s:3d}: loss={loss:.4f} acc={ta:.3f} "
                  f"test_acc={tea:.3f} evals={total_evals[0]}")

        return loss, grad

    result = minimize(
        objective_and_grad,
        params,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations, "maxfun": max_iterations * 10},
    )

    return result.x, trace


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_optimizer(name, model, params, X_train, y_train, X_test, y_test, config):
    """Run a named optimizer and return (final_params, trace)."""
    if name == "krotov":
        return train_krotov(
            model, params, X_train, y_train, X_test, y_test,
            max_iterations=config.max_iterations,
            step_size=config.krotov_step_size,
        )
    elif name == "adam":
        return train_adam(
            model, params, X_train, y_train, X_test, y_test,
            max_iterations=config.max_iterations,
            lr=config.adam_lr,
        )
    elif name == "lbfgs":
        return train_lbfgs(
            model, params, X_train, y_train, X_test, y_test,
            max_iterations=config.lbfgs_maxiter,
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")
