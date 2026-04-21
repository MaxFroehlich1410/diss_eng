#!/usr/bin/env python3
"""Large hyperparameter sweeps for parity classification optimizers.

This benchmark uses the corrected leakage-free parity split with 10 unique
training bitstrings and 6 disjoint test bitstrings by default. Each
hyperparameter combination is evaluated on three seeds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
VENDOR_DIR = os.path.join(REPO_ROOT, "vendor", "pennylane")
MPLCONFIGDIR = os.path.join(REPO_ROOT, ".mplconfig")

os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
os.makedirs(MPLCONFIGDIR, exist_ok=True)

if VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from datasets import generate_parity_4bit_unique_split
from optimizers.runner import run_optimizer
from qml_models.variants import ParityRotClassifierModel
from run_configurable_qml_test import RunnerConfig


RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")
OPTIMIZER_ORDER = (
    "krotov_hybrid",
    "pennylane_adam",
    "pennylane_adagrad",
    "pennylane_gradient_descent",
    "pennylane_momentum",
    "pennylane_nesterov",
    "pennylane_rmsprop",
    "pennylane_spsa",
    "pennylane_qng",
    "pennylane_rotosolve",
)
OPTIMIZER_LABELS = {
    "krotov_hybrid": "Hybrid Krotov",
    "pennylane_adam": "PennyLane Adam",
    "pennylane_adagrad": "PennyLane Adagrad",
    "pennylane_gradient_descent": "PennyLane Gradient Descent",
    "pennylane_momentum": "PennyLane Momentum",
    "pennylane_nesterov": "PennyLane Nesterov",
    "pennylane_rmsprop": "PennyLane RMSProp",
    "pennylane_spsa": "PennyLane SPSA",
    "pennylane_qng": "PennyLane QNG",
    "pennylane_rotosolve": "PennyLane Rotosolve",
}
OPTIMIZER_COLORS = {
    "krotov_hybrid": "#8b1e3f",
    "pennylane_adam": "#2563eb",
    "pennylane_adagrad": "#1d4ed8",
    "pennylane_gradient_descent": "#6d28d9",
    "pennylane_momentum": "#7c3aed",
    "pennylane_nesterov": "#a21caf",
    "pennylane_rmsprop": "#0f766e",
    "pennylane_spsa": "#b45309",
    "pennylane_qng": "#d97706",
    "pennylane_rotosolve": "#15803d",
}
SCANNED_HYPERPARAMETERS = {
    "krotov_hybrid": [
        "hybrid_switch_iteration",
        "hybrid_online_step_size",
        "hybrid_batch_step_size",
        "hybrid_online_schedule",
        "hybrid_batch_schedule",
        "hybrid_online_decay",
        "hybrid_batch_decay",
        "hybrid_scaling_mode",
        "hybrid_scaling_apply_phase",
        "hybrid_scaling_config",
        "max_iterations",
    ],
    "pennylane_adam": ["stepsize", "beta1", "beta2", "eps", "max_iterations"],
    "pennylane_adagrad": ["stepsize", "eps", "max_iterations"],
    "pennylane_gradient_descent": ["stepsize", "max_iterations"],
    "pennylane_momentum": ["stepsize", "momentum", "max_iterations"],
    "pennylane_nesterov": ["stepsize", "momentum", "max_iterations"],
    "pennylane_rmsprop": ["stepsize", "decay", "eps", "max_iterations"],
    "pennylane_spsa": ["alpha", "gamma", "c", "A", "a", "maxiter"],
    "pennylane_qng": ["stepsize", "approx", "lam", "max_iterations"],
    "pennylane_rotosolve": ["substep_optimizer", "substep_kwargs", "max_iterations"],
}
SWEEP_GRIDS = {
    "krotov_hybrid": OrderedDict(
        [
            ("hybrid_switch_iteration", [5, 10, 15]),
            ("hybrid_online_step_size", [0.1, 0.3]),
            ("hybrid_batch_step_size", [0.3, 1.0]),
            ("hybrid_online_schedule", ["constant", "inverse"]),
            ("hybrid_batch_schedule", ["constant", "inverse"]),
        ]
    ),
    "pennylane_adam": OrderedDict(
        [
            ("adam_lr", [0.01, 0.03, 0.1]),
            ("adam_beta1", [0.9, 0.95]),
            ("adam_beta2", [0.99, 0.999]),
        ]
    ),
    "pennylane_adagrad": OrderedDict(
        [
            ("adagrad_lr", [0.01, 0.03, 0.1, 0.3]),
            ("adagrad_eps", [1e-8, 1e-6]),
        ]
    ),
    "pennylane_gradient_descent": OrderedDict(
        [
            ("gd_lr", [0.01, 0.03, 0.1, 0.3]),
        ]
    ),
    "pennylane_momentum": OrderedDict(
        [
            ("momentum_lr", [0.01, 0.03, 0.1]),
            ("momentum_beta", [0.8, 0.9, 0.95]),
        ]
    ),
    "pennylane_nesterov": OrderedDict(
        [
            ("nesterov_lr", [0.01, 0.03, 0.1]),
            ("nesterov_beta", [0.8, 0.9, 0.95]),
        ]
    ),
    "pennylane_rmsprop": OrderedDict(
        [
            ("rmsprop_lr", [0.01, 0.03, 0.1]),
            ("rmsprop_decay", [0.8, 0.9, 0.99]),
        ]
    ),
    "pennylane_spsa": OrderedDict(
        [
            ("spsa_c", [0.05, 0.1, 0.2]),
            ("spsa_a", [0.02, 0.05, 0.1]),
            ("spsa_alpha", [0.602, 1.0]),
        ]
    ),
    "pennylane_qng": OrderedDict(
        [
            ("qng_lr", [0.03, 0.1, 0.3]),
            ("qng_lam", [1e-4, 1e-3, 1e-2]),
            ("qng_approx", ["full", "block-diag"]),
        ]
    ),
    "pennylane_rotosolve": OrderedDict(
        [
            ("rotosolve_mode", ["analytic"]),
        ]
    ),
}
BASELINES = {
    "krotov_hybrid": {
        "hybrid_switch_iteration": 10,
        "hybrid_online_step_size": 0.3,
        "hybrid_batch_step_size": 1.0,
        "hybrid_online_schedule": "constant",
        "hybrid_batch_schedule": "constant",
    },
    "pennylane_adam": {"adam_lr": 0.05, "adam_beta1": 0.9, "adam_beta2": 0.99},
    "pennylane_adagrad": {"adagrad_lr": 0.05, "adagrad_eps": 1e-8},
    "pennylane_gradient_descent": {"gd_lr": 0.05},
    "pennylane_momentum": {"momentum_lr": 0.05, "momentum_beta": 0.9},
    "pennylane_nesterov": {"nesterov_lr": 0.05, "nesterov_beta": 0.9},
    "pennylane_rmsprop": {"rmsprop_lr": 0.05, "rmsprop_decay": 0.9},
    "pennylane_spsa": {"spsa_c": 0.2, "spsa_a": None, "spsa_alpha": 0.602},
    "pennylane_qng": {"qng_lr": 0.1, "qng_lam": 1e-3, "qng_approx": "block-diag"},
    "pennylane_rotosolve": {"rotosolve_mode": "analytic"},
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


@dataclass(frozen=True)
class SweepConfig:
    optimizers: list[str]
    seeds: list[int]
    train_size: int
    test_size: int
    n_layers: int
    max_iterations: int
    num_shards: int
    shard_index: int
    results_root: str


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZER_ORDER, default=list(OPTIMIZER_ORDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train-size", type=int, default=10)
    parser.add_argument("--test-size", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _timestamped_dir(results_root: str, run_name: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_name or "parity_optimizer_sweeps"
    path = os.path.join(results_root, f"{timestamp}_{suffix}")
    os.makedirs(path, exist_ok=False)
    return path


def _save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _square_loss(targets, predictions):
    residual = targets - predictions
    return pnp.mean(residual**2)


def _accuracy_from_scores(scores, targets):
    preds = np.where(np.asarray(scores, dtype=float) >= 0.0, 1, -1)
    return float(np.mean(preds == np.asarray(targets, dtype=int)))


def _optimal_bias_from_quantum_scores(targets, quantum_scores):
    targets_arr = np.asarray(targets, dtype=float)
    scores_arr = np.asarray(quantum_scores, dtype=float)
    return float(np.mean(targets_arr - scores_arr))


def _serialize_trace(trace):
    payload = {}
    for key, values in trace.items():
        if key == "phase":
            payload[key] = [str(v) for v in values]
        else:
            payload[key] = [float(v) for v in values]
    return payload


def _interpolate_traces(x_traces, y_traces, n_points=300):
    x_min = max(trace[0] for trace in x_traces)
    x_max = min(trace[-1] for trace in x_traces)
    if x_max <= x_min:
        x_grid = np.asarray(x_traces[0], dtype=float)
        y_interp = np.array([np.asarray(trace, dtype=float) for trace in y_traces])
        return x_grid, y_interp
    x_grid = np.linspace(x_min, x_max, n_points)
    y_interp = np.array([np.interp(x_grid, tx, ty) for tx, ty in zip(x_traces, y_traces)])
    return x_grid, y_interp


def _save_fig(fig, name: str, results_dir: str) -> None:
    fig.savefig(os.path.join(results_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(results_dir, f"{name}.png"))
    plt.close(fig)


def _plot_best_metric(results, best_rows, x_key, y_key, title, xlabel, ylabel, results_dir, file_name):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for optimizer_name in OPTIMIZER_ORDER:
        if optimizer_name not in best_rows:
            continue
        hp_key = best_rows[optimizer_name]["hp_key"]
        runs = [run for run in results if run["optimizer"] == optimizer_name and run["hp_key"] == hp_key]
        xs = [np.asarray(run["trace"][x_key], dtype=float) for run in runs if run["trace"][x_key]]
        ys = [np.asarray(run["trace"][y_key], dtype=float) for run in runs if run["trace"][y_key]]
        if not xs:
            continue
        x_grid, y_interp = _interpolate_traces(xs, ys)
        mean = np.mean(y_interp, axis=0)
        std = np.std(y_interp, axis=0)
        label = OPTIMIZER_LABELS[optimizer_name]
        ax.plot(x_grid, mean, color=OPTIMIZER_COLORS[optimizer_name], lw=2.2, label=label)
        ax.fill_between(x_grid, mean - std, mean + std, color=OPTIMIZER_COLORS[optimizer_name], alpha=0.16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    _save_fig(fig, file_name, results_dir)


def _hp_key(hp_dict):
    parts = []
    for key, value in hp_dict.items():
        if value is None:
            value_str = "none"
        elif isinstance(value, float):
            value_str = f"{value:.12g}"
        else:
            value_str = str(value)
        parts.append(f"{key}={value_str}")
    return "|".join(parts)


def _hp_label(hp_dict):
    return ", ".join(f"{key}={value}" for key, value in hp_dict.items())


def _expand_grid(grid_dict):
    names = list(grid_dict.keys())
    values = [grid_dict[name] for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def _close(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < max(1e-12, 1e-6 * max(abs(float(a)), abs(float(b)), 1.0))
    return a == b


def _matches_config(row, reference, keys):
    return all(_close(row[key], reference.get(key)) for key in keys)


class PennyLaneParityClassifier:
    def __init__(self, n_layers: int, qng_approx: str):
        self.n_layers = int(n_layers)
        self.n_qubits = 4
        self.weights_shape = (self.n_layers, self.n_qubits, 3)
        self.dev = qml.device("default.qubit", wires=self.n_qubits + 1)

        @qml.qnode(self.dev, interface="autograd")
        def circuit(x_bits, weights):
            qml.BasisState(x_bits, wires=range(self.n_qubits))
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    phi, theta, omega = weights[layer, qubit]
                    qml.Rot(phi, theta, omega, wires=qubit)
                qml.CNOT(wires=[0, 1])
                qml.CNOT(wires=[1, 2])
                qml.CNOT(wires=[2, 3])
                qml.CNOT(wires=[3, 0])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit
        approx = None if qng_approx == "full" else qng_approx
        self.metric_tensor_fn = qml.metric_tensor(self.circuit, approx=approx)

    def init_params(self, seed: int):
        native = ParityRotClassifierModel(n_layers=self.n_layers).init_params(seed=seed)
        weights = pnp.array(native[:-1].reshape(self.weights_shape), requires_grad=True)
        bias = pnp.array(native[-1], requires_grad=True)
        return weights, bias, native

    def predict_scores(self, weights, bias, X):
        return pnp.array([self.circuit(x_bits, weights) + bias for x_bits in X])

    def loss(self, weights, bias, X, y):
        return _square_loss(y, self.predict_scores(weights, bias, X))

    def metric_tensor(self, weights, bias, X):
        metrics = [self.metric_tensor_fn(x_bits, weights) for x_bits in X]
        mean_metric = sum(metrics) / len(metrics)
        bias_metric = pnp.array([[1.0]])
        return mean_metric, bias_metric


def _make_krotov_config(hp_dict, max_iterations: int, n_layers: int) -> RunnerConfig:
    return RunnerConfig(
        n_samples=0,
        test_fraction=0.0,
        model_architecture="parity_rot",
        n_qubits=4,
        n_layers=n_layers,
        observable="Z0",
        max_iterations=max_iterations,
        hybrid_switch_iteration=int(hp_dict["hybrid_switch_iteration"]),
        hybrid_online_step_size=float(hp_dict["hybrid_online_step_size"]),
        hybrid_batch_step_size=float(hp_dict["hybrid_batch_step_size"]),
        hybrid_online_schedule=str(hp_dict["hybrid_online_schedule"]),
        hybrid_batch_schedule=str(hp_dict["hybrid_batch_schedule"]),
        hybrid_online_decay=0.05,
        hybrid_batch_decay=0.05,
        hybrid_scaling_mode="none",
        hybrid_scaling_apply_phase="both",
        hybrid_scaling_config=None,
        early_stopping_enabled=False,
        results_dir="results",
        plots_dir="plots",
        optimizers=["krotov_hybrid"],
        seeds=[0, 1, 2],
    )


def _run_krotov(seed: int, hp_dict, args, X_train, y_train, X_test, y_test):
    model = ParityRotClassifierModel(n_layers=args.n_layers)
    init_params = np.asarray(model.init_params(seed=seed), dtype=float)
    config = _make_krotov_config(hp_dict, args.max_iterations, args.n_layers)
    start = time.time()
    final_params, trace = run_optimizer(
        "krotov_hybrid",
        model,
        init_params.copy(),
        X_train,
        y_train,
        X_test,
        y_test,
        config,
    )
    wall = time.time() - start
    return {
        "optimizer": "krotov_hybrid",
        "seed": int(seed),
        "wall_time_total": float(wall),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "initial_params": init_params.tolist(),
        "final_params": np.asarray(final_params, dtype=float).tolist(),
        "trace": _serialize_trace(trace),
    }


def _run_pennylane_optimizer(optimizer_name: str, seed: int, hp_dict, args, X_train, y_train, X_test, y_test):
    qng_approx = hp_dict.get("qng_approx", BASELINES["pennylane_qng"]["qng_approx"])
    model = PennyLaneParityClassifier(n_layers=args.n_layers, qng_approx=qng_approx)
    weights, bias, native_init = model.init_params(seed=seed)
    X_train_bits = [np.asarray(x, dtype=int) for x in X_train]
    X_test_bits = [np.asarray(x, dtype=int) for x in X_test]
    y_train_arr = pnp.array(np.asarray(y_train, dtype=float), requires_grad=False)
    y_test_arr = np.asarray(y_test, dtype=int)

    metric_tensor_fn = None
    if optimizer_name == "pennylane_adam":
        optimizer = qml.AdamOptimizer(
            stepsize=float(hp_dict["adam_lr"]),
            beta1=float(hp_dict["adam_beta1"]),
            beta2=float(hp_dict["adam_beta2"]),
            eps=1e-8,
        )
    elif optimizer_name == "pennylane_adagrad":
        optimizer = qml.AdagradOptimizer(
            stepsize=float(hp_dict["adagrad_lr"]),
            eps=float(hp_dict["adagrad_eps"]),
        )
    elif optimizer_name == "pennylane_gradient_descent":
        optimizer = qml.GradientDescentOptimizer(stepsize=float(hp_dict["gd_lr"]))
    elif optimizer_name == "pennylane_momentum":
        optimizer = qml.MomentumOptimizer(
            stepsize=float(hp_dict["momentum_lr"]),
            momentum=float(hp_dict["momentum_beta"]),
        )
    elif optimizer_name == "pennylane_nesterov":
        optimizer = qml.NesterovMomentumOptimizer(
            stepsize=float(hp_dict["nesterov_lr"]),
            momentum=float(hp_dict["nesterov_beta"]),
        )
    elif optimizer_name == "pennylane_rmsprop":
        optimizer = qml.RMSPropOptimizer(
            stepsize=float(hp_dict["rmsprop_lr"]),
            decay=float(hp_dict["rmsprop_decay"]),
            eps=1e-8,
        )
    elif optimizer_name == "pennylane_spsa":
        optimizer = qml.SPSAOptimizer(
            maxiter=args.max_iterations,
            alpha=float(hp_dict["spsa_alpha"]),
            gamma=0.101,
            c=float(hp_dict["spsa_c"]),
            a=hp_dict["spsa_a"],
        )
    elif optimizer_name == "pennylane_qng":
        approx = None if hp_dict["qng_approx"] == "full" else hp_dict["qng_approx"]
        optimizer = qml.QNGOptimizer(
            stepsize=float(hp_dict["qng_lr"]),
            lam=float(hp_dict["qng_lam"]),
            approx=approx,
        )
        metric_tensor_fn = lambda w, b: model.metric_tensor(w, b, X_train_bits)
    elif optimizer_name == "pennylane_rotosolve":
        optimizer = qml.RotosolveOptimizer()
    else:
        raise ValueError(f"Unknown PennyLane optimizer: {optimizer_name}")

    cost_fn = lambda w, b: model.loss(w, b, X_train_bits, y_train_arr)
    grad_fn = qml.grad(cost_fn, argnums=(0, 1))
    trace = {"step": [], "loss": [], "train_acc": [], "test_acc": [], "wall_time": []}
    start = time.time()

    for step in range(1, args.max_iterations + 1):
        if optimizer_name == "pennylane_spsa":
            (weights, bias), _ = optimizer.step_and_cost(cost_fn, weights, bias)
        elif optimizer_name in {
            "pennylane_adam",
            "pennylane_adagrad",
            "pennylane_gradient_descent",
            "pennylane_momentum",
            "pennylane_nesterov",
            "pennylane_rmsprop",
        }:
            (weights, bias), _ = optimizer.step_and_cost(cost_fn, weights, bias, grad_fn=grad_fn)
        elif optimizer_name == "pennylane_rotosolve":
            nums_frequency = {"w": {idx: 1 for idx in np.ndindex(model.weights_shape)}}

            def rotosolve_objective(w):
                return model.loss(w, bias, X_train_bits, y_train_arr)

            weights, _ = optimizer.step_and_cost(
                rotosolve_objective,
                weights,
                nums_frequency=nums_frequency,
            )
            train_quantum_scores = np.array([model.circuit(x_bits, weights) for x_bits in X_train_bits], dtype=float)
            bias = pnp.array(
                _optimal_bias_from_quantum_scores(y_train_arr, train_quantum_scores),
                requires_grad=False,
            )
        else:
            (weights, bias), _ = optimizer.step_and_cost(
                cost_fn,
                weights,
                bias,
                grad_fn=grad_fn,
                metric_tensor_fn=metric_tensor_fn,
            )

        train_scores = model.predict_scores(weights, bias, X_train_bits)
        test_scores = model.predict_scores(weights, bias, X_test_bits)
        trace["step"].append(float(step))
        trace["loss"].append(float(_square_loss(y_train_arr, train_scores)))
        trace["train_acc"].append(_accuracy_from_scores(train_scores, y_train))
        trace["test_acc"].append(_accuracy_from_scores(test_scores, y_test_arr))
        trace["wall_time"].append(float(time.time() - start))

    flat_final = np.concatenate([np.asarray(weights, dtype=float).reshape(-1), [float(bias)]])
    return {
        "optimizer": optimizer_name,
        "seed": int(seed),
        "wall_time_total": float(trace["wall_time"][-1]),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "initial_params": np.asarray(native_init, dtype=float).tolist(),
        "final_params": flat_final.tolist(),
        "trace": trace,
    }


def _run_single(optimizer_name: str, hp_dict, seed: int, args):
    X_train, X_test, y_train, y_test = generate_parity_4bit_unique_split(
        train_size=args.train_size,
        test_size=args.test_size,
        seed=seed,
    )
    if optimizer_name == "krotov_hybrid":
        result = _run_krotov(seed, hp_dict, args, X_train, y_train, X_test, y_test)
    else:
        result = _run_pennylane_optimizer(optimizer_name, seed, hp_dict, args, X_train, y_train, X_test, y_test)
    result.update(
        {
            "hp": dict(hp_dict),
            "hp_key": _hp_key(hp_dict),
            "hp_label": _hp_label(hp_dict),
            "train_size": int(args.train_size),
            "test_size": int(args.test_size),
            "n_layers": int(args.n_layers),
            "max_iterations": int(args.max_iterations),
        }
    )
    return result


def _build_jobs(args):
    jobs = []
    for optimizer_name in args.optimizers:
        for hp_dict in _expand_grid(SWEEP_GRIDS[optimizer_name]):
            for seed in args.seeds:
                jobs.append(
                    {
                        "optimizer": optimizer_name,
                        "hp": hp_dict,
                        "seed": seed,
                    }
                )
    return jobs


def _summarize_optimizer(optimizer_name: str, results):
    param_names = list(SWEEP_GRIDS[optimizer_name].keys())
    grouped = OrderedDict()
    for row in results:
        if row["optimizer"] != optimizer_name:
            continue
        grouped.setdefault(row["hp_key"], []).append(row)

    rows = []
    for hp_key, runs in grouped.items():
        hp = dict(runs[0]["hp"])
        losses = np.asarray([run["final_loss"] for run in runs], dtype=float)
        train_accs = np.asarray([run["final_train_acc"] for run in runs], dtype=float)
        test_accs = np.asarray([run["final_test_acc"] for run in runs], dtype=float)
        walls = np.asarray([run["wall_time_total"] for run in runs], dtype=float)
        rows.append(
            {
                **hp,
                "hp_key": hp_key,
                "n_runs": len(runs),
                "final_loss_mean": float(np.mean(losses)),
                "final_loss_std": float(np.std(losses)),
                "final_train_acc_mean": float(np.mean(train_accs)),
                "final_train_acc_std": float(np.std(train_accs)),
                "final_test_acc_mean": float(np.mean(test_accs)),
                "final_test_acc_std": float(np.std(test_accs)),
                "wall_time_mean": float(np.mean(walls)),
                "wall_time_std": float(np.std(walls)),
            }
        )

    rows.sort(
        key=lambda row: (
            -row["final_test_acc_mean"],
            row["final_loss_mean"],
            row["wall_time_mean"],
        )
    )
    baseline = None
    for row in rows:
        if _matches_config(row, BASELINES[optimizer_name], param_names):
            baseline = row
            break
    return {
        "optimizer": optimizer_name,
        "label": OPTIMIZER_LABELS[optimizer_name],
        "scanned_hyperparameters": SCANNED_HYPERPARAMETERS[optimizer_name],
        "swept_hyperparameters": param_names,
        "n_configs": len(rows),
        "best": rows[0] if rows else None,
        "baseline": baseline,
        "top10": rows[:10],
        "all_configs": rows,
    }


def _write_report(path: str, config: SweepConfig, summaries) -> None:
    lines = [
        "# Parity Optimizer Hyperparameter Sweeps",
        "",
        "Leakage-free parity benchmark with a 10/6 unique split and three seeds per configuration by default.",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(asdict(config), indent=2),
        "```",
        "",
        "## Best Configurations",
        "",
        "| Optimizer | Best test acc | Best loss | Best wall (s) | Swept hyperparameters |",
        "|---|---:|---:|---:|---|",
    ]
    for optimizer_name in config.optimizers:
        summary = summaries[optimizer_name]
        best = summary["best"]
        if best is None:
            continue
        lines.append(
            f"| {summary['label']} | {best['final_test_acc_mean']:.4f} | {best['final_loss_mean']:.4f} | "
            f"{best['wall_time_mean']:.2f} | {', '.join(summary['swept_hyperparameters'])} |"
        )
    lines.extend(["", "## Per-Optimizer Notes", ""])
    for optimizer_name in config.optimizers:
        summary = summaries[optimizer_name]
        best = summary["best"]
        lines.append(f"### {summary['label']}")
        lines.append("")
        lines.append(f"- Scanned hyperparameters: `{', '.join(summary['scanned_hyperparameters'])}`")
        lines.append(f"- Swept hyperparameters: `{', '.join(summary['swept_hyperparameters'])}`")
        if best is not None:
            hp_text = ", ".join(f"`{key}={best[key]}`" for key in summary["swept_hyperparameters"])
            lines.append(
                f"- Best tested setting: {hp_text}; mean final test acc `{best['final_test_acc_mean']:.4f}`, "
                f"mean final loss `{best['final_loss_mean']:.4f}`, mean wall time `{best['wall_time_mean']:.2f}s`."
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = _parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    os.makedirs(args.results_root, exist_ok=True)
    results_dir = _timestamped_dir(args.results_root, args.run_name)
    config = SweepConfig(
        optimizers=list(args.optimizers),
        seeds=list(args.seeds),
        train_size=args.train_size,
        test_size=args.test_size,
        n_layers=args.n_layers,
        max_iterations=args.max_iterations,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        results_root=args.results_root,
    )

    all_jobs = _build_jobs(args)
    shard_jobs = [job for idx, job in enumerate(all_jobs) if idx % args.num_shards == args.shard_index]

    print(f"\n{'#' * 88}")
    print(
        "# Parity optimizer sweeps | "
        f"optimizers={len(args.optimizers)} | layers={args.n_layers} | "
        f"split={args.train_size}/{args.test_size} | seeds={args.seeds} | "
        f"max_iter={args.max_iterations} | shard={args.shard_index + 1}/{args.num_shards}"
    )
    print(f"{'#' * 88}")

    all_results = []
    total_runs = len(shard_jobs)
    print(f"Total jobs in full sweep: {len(all_jobs)}")
    print(f"Jobs assigned to this shard: {total_runs}")
    for completed, job in enumerate(shard_jobs, start=1):
        optimizer_name = job["optimizer"]
        hp_dict = job["hp"]
        seed = job["seed"]
        print(
            f"  [{completed:3d}/{total_runs}] {OPTIMIZER_LABELS[optimizer_name]} | "
            f"{_hp_label(hp_dict)} | seed={seed}",
            flush=True,
        )
        result = _run_single(optimizer_name, hp_dict, seed, args)
        all_results.append(result)
        print(
            f"       loss={result['final_loss']:.4f} "
            f"train_acc={result['final_train_acc']:.3f} "
            f"test_acc={result['final_test_acc']:.3f} "
            f"wall={result['wall_time_total']:.2f}s",
            flush=True,
        )

    summaries = OrderedDict()
    best_rows = {}
    for optimizer_name in args.optimizers:
        summary = _summarize_optimizer(optimizer_name, all_results)
        summaries[optimizer_name] = summary
        if summary["best"] is not None:
            best_rows[optimizer_name] = summary["best"]

    _save_json(os.path.join(results_dir, "raw_results.json"), all_results)
    _save_json(
        os.path.join(results_dir, "analysis.json"),
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(config),
            "scanned_hyperparameters": SCANNED_HYPERPARAMETERS,
            "sweep_grids": {name: {k: list(v) for k, v in SWEEP_GRIDS[name].items()} for name in args.optimizers},
            "baselines": {name: BASELINES[name] for name in args.optimizers},
            "summaries": summaries,
        },
    )

    title = f"Best tested configs | 4-bit parity | {args.n_layers} layers | unique split {args.train_size}/{args.test_size}"
    _plot_best_metric(
        all_results,
        best_rows,
        "step",
        "loss",
        f"{title}: loss vs iteration",
        "Iteration",
        "Training loss (MSE)",
        results_dir,
        "best_configs_loss_vs_iteration",
    )
    _plot_best_metric(
        all_results,
        best_rows,
        "wall_time",
        "loss",
        f"{title}: loss vs wall-clock time",
        "Wall-clock time (s)",
        "Training loss (MSE)",
        results_dir,
        "best_configs_loss_vs_time",
    )
    _write_report(os.path.join(results_dir, "experiment.md"), config, summaries)
    print(f"\nSaved parity optimizer sweeps to {results_dir}/")


if __name__ == "__main__":
    main()
