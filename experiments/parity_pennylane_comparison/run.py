#!/usr/bin/env python3
"""Compare hybrid Krotov against PennyLane optimizers on parity."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable

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

from datasets import generate_parity_4bit
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

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


@dataclass(frozen=True)
class ComparisonConfig:
    seeds: list[int]
    repeats: int
    n_layers: int
    max_iterations: int
    test_fraction: float
    adam_lr: float
    adagrad_lr: float
    gd_lr: float
    momentum_lr: float
    momentum_beta: float
    nesterov_lr: float
    nesterov_beta: float
    rmsprop_lr: float
    rmsprop_decay: float
    spsa_c: float
    spsa_a: float | None
    qng_lr: float
    qng_lam: float
    qng_approx: str
    hybrid_switch_iteration: int
    hybrid_online_step_size: float
    hybrid_batch_step_size: float
    hybrid_online_decay: float
    hybrid_batch_decay: float


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--adam-lr", type=float, default=0.05)
    parser.add_argument("--adagrad-lr", type=float, default=0.05)
    parser.add_argument("--gd-lr", type=float, default=0.05)
    parser.add_argument("--momentum-lr", type=float, default=0.05)
    parser.add_argument("--momentum-beta", type=float, default=0.9)
    parser.add_argument("--nesterov-lr", type=float, default=0.05)
    parser.add_argument("--nesterov-beta", type=float, default=0.9)
    parser.add_argument("--rmsprop-lr", type=float, default=0.05)
    parser.add_argument("--rmsprop-decay", type=float, default=0.9)
    parser.add_argument("--spsa-c", type=float, default=0.2)
    parser.add_argument("--spsa-a", type=float, default=None)
    parser.add_argument("--qng-lr", type=float, default=0.1)
    parser.add_argument("--qng-lam", type=float, default=1e-3)
    parser.add_argument("--qng-approx", choices=["full", "block-diag", "diag"], default="full")
    parser.add_argument("--hybrid-switch-iteration", type=int, default=10)
    parser.add_argument("--hybrid-online-step-size", type=float, default=0.3)
    parser.add_argument("--hybrid-batch-step-size", type=float, default=1.0)
    parser.add_argument("--hybrid-online-decay", type=float, default=0.05)
    parser.add_argument("--hybrid-batch-decay", type=float, default=0.05)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _timestamped_dir(results_root: str, run_name: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_name or "parity_pennylane_comparison"
    path = os.path.join(results_root, f"{timestamp}_{suffix}")
    os.makedirs(path, exist_ok=False)
    return path


def _make_krotov_config(args) -> RunnerConfig:
    return RunnerConfig(
        n_samples=args.repeats,
        test_fraction=args.test_fraction,
        model_architecture="parity_rot",
        n_qubits=4,
        n_layers=args.n_layers,
        observable="Z0",
        max_iterations=args.max_iterations,
        hybrid_switch_iteration=args.hybrid_switch_iteration,
        hybrid_online_step_size=args.hybrid_online_step_size,
        hybrid_batch_step_size=args.hybrid_batch_step_size,
        hybrid_online_decay=args.hybrid_online_decay,
        hybrid_batch_decay=args.hybrid_batch_decay,
        hybrid_online_schedule="constant",
        hybrid_batch_schedule="constant",
        hybrid_scaling_mode="none",
        hybrid_scaling_apply_phase="both",
        hybrid_scaling_config=None,
        results_dir="results",
        plots_dir="plots",
        optimizers=["krotov_hybrid"],
        seeds=list(args.seeds),
    )


def _save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _interpolate_traces(x_traces, y_traces, n_points=300):
    x_min = max(trace[0] for trace in x_traces)
    x_max = min(trace[-1] for trace in x_traces)
    x_grid = np.linspace(x_min, x_max, n_points)
    y_interp = np.array([np.interp(x_grid, tx, ty) for tx, ty in zip(x_traces, y_traces)])
    return x_grid, y_interp


def _mean_std_trace(grouped, optimizer_name, x_key, y_key):
    runs = grouped[optimizer_name]
    xs = [np.asarray(run["trace"][x_key], dtype=float) for run in runs if run["trace"][x_key]]
    ys = [np.asarray(run["trace"][y_key], dtype=float) for run in runs if run["trace"][y_key]]
    if not xs:
        return None, None, None
    x_grid, y_interp = _interpolate_traces(xs, ys)
    return x_grid, np.mean(y_interp, axis=0), np.std(y_interp, axis=0)


def _save_fig(fig, name: str, results_dir: str) -> None:
    fig.savefig(os.path.join(results_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(results_dir, f"{name}.png"))
    plt.close(fig)


def _plot_metric(results, x_key, y_key, title, xlabel, ylabel, results_dir, file_name):
    grouped = OrderedDict((name, [run for run in results if run["optimizer"] == name]) for name in OPTIMIZER_ORDER)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for optimizer_name in OPTIMIZER_ORDER:
        x_grid, mean, std = _mean_std_trace(grouped, optimizer_name, x_key, y_key)
        if x_grid is None:
            continue
        color = OPTIMIZER_COLORS[optimizer_name]
        ax.plot(x_grid, mean, color=color, lw=2.2, label=OPTIMIZER_LABELS[optimizer_name])
        ax.fill_between(x_grid, mean - std, mean + std, color=color, alpha=0.18)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    _save_fig(fig, file_name, results_dir)


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
        self.metric_tensor_fn = qml.metric_tensor(
            self.circuit,
            approx=approx,
        )

    def init_params(self, seed: int):
        native = ParityRotClassifierModel(n_layers=self.n_layers).init_params(seed=seed)
        weights = pnp.array(native[:-1].reshape(self.weights_shape), requires_grad=True)
        bias = pnp.array(native[-1], requires_grad=True)
        return weights, bias, native

    def predict_scores(self, weights, bias, X):
        return pnp.array([self.circuit(x_bits, weights) + bias for x_bits in X])

    def loss(self, weights, bias, X, y):
        scores = self.predict_scores(weights, bias, X)
        return _square_loss(y, scores)

    def metric_tensor(self, weights, bias, X):
        metrics = [self.metric_tensor_fn(x_bits, weights) for x_bits in X]
        mean_metric = sum(metrics) / len(metrics)
        bias_metric = pnp.array([[1.0]])
        return mean_metric, bias_metric


def _serialize_trace(trace):
    payload = {}
    for key, values in trace.items():
        if key == "phase":
            payload[key] = [str(v) for v in values]
        else:
            payload[key] = [float(v) for v in values]
    return payload


def _run_krotov(seed: int, args, X_train, y_train, X_test, y_test):
    model = ParityRotClassifierModel(n_layers=args.n_layers)
    init_params = np.asarray(model.init_params(seed=seed), dtype=float)
    config = _make_krotov_config(args)
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
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "wall_time_total": float(wall),
        "initial_params": init_params.tolist(),
        "final_params": np.asarray(final_params, dtype=float).tolist(),
        "trace": _serialize_trace(trace),
    }


def _run_pennylane_optimizer(
    optimizer_name: str,
    seed: int,
    args,
    X_train,
    y_train,
    X_test,
    y_test,
):
    model = PennyLaneParityClassifier(n_layers=args.n_layers, qng_approx=args.qng_approx)
    weights, bias, native_init = model.init_params(seed=seed)
    X_train_bits = [np.asarray(x, dtype=int) for x in X_train]
    X_test_bits = [np.asarray(x, dtype=int) for x in X_test]
    y_train_arr = pnp.array(np.asarray(y_train, dtype=float), requires_grad=False)
    y_test_arr = np.asarray(y_test, dtype=int)

    if optimizer_name == "pennylane_adam":
        optimizer = qml.AdamOptimizer(stepsize=args.adam_lr)
        metric_tensor_fn = None
    elif optimizer_name == "pennylane_adagrad":
        optimizer = qml.AdagradOptimizer(stepsize=args.adagrad_lr)
        metric_tensor_fn = None
    elif optimizer_name == "pennylane_gradient_descent":
        optimizer = qml.GradientDescentOptimizer(stepsize=args.gd_lr)
        metric_tensor_fn = None
    elif optimizer_name == "pennylane_momentum":
        optimizer = qml.MomentumOptimizer(stepsize=args.momentum_lr, momentum=args.momentum_beta)
        metric_tensor_fn = None
    elif optimizer_name == "pennylane_nesterov":
        optimizer = qml.NesterovMomentumOptimizer(
            stepsize=args.nesterov_lr,
            momentum=args.nesterov_beta,
        )
        metric_tensor_fn = None
    elif optimizer_name == "pennylane_rmsprop":
        optimizer = qml.RMSPropOptimizer(stepsize=args.rmsprop_lr, decay=args.rmsprop_decay)
        metric_tensor_fn = None
    elif optimizer_name == "pennylane_spsa":
        optimizer = qml.SPSAOptimizer(maxiter=args.max_iterations, c=args.spsa_c, a=args.spsa_a)
        metric_tensor_fn = None
    elif optimizer_name == "pennylane_qng":
        approx = None if args.qng_approx == "full" else args.qng_approx
        optimizer = qml.QNGOptimizer(stepsize=args.qng_lr, lam=args.qng_lam, approx=approx)
        metric_tensor_fn = lambda w, b: model.metric_tensor(w, b, X_train_bits)
    elif optimizer_name == "pennylane_rotosolve":
        optimizer = qml.RotosolveOptimizer()
        metric_tensor_fn = None
    else:
        raise ValueError(f"Unknown PennyLane optimizer: {optimizer_name}")

    cost_fn: Callable[[pnp.ndarray, pnp.ndarray], pnp.ndarray] = (
        lambda w, b: model.loss(w, b, X_train_bits, y_train_arr)
    )
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
        train_loss = float(_square_loss(y_train_arr, train_scores))
        train_acc = _accuracy_from_scores(train_scores, y_train)
        test_acc = _accuracy_from_scores(test_scores, y_test_arr)
        elapsed = time.time() - start

        trace["step"].append(float(step))
        trace["loss"].append(train_loss)
        trace["train_acc"].append(train_acc)
        trace["test_acc"].append(test_acc)
        trace["wall_time"].append(float(elapsed))

        print(
            f"  {OPTIMIZER_LABELS[optimizer_name]} step {step:>3}: "
            f"loss={train_loss:.4f} train_acc={train_acc:.3f} test_acc={test_acc:.3f}",
            flush=True,
        )

    flat_final = np.concatenate([np.asarray(weights, dtype=float).reshape(-1), [float(bias)]])
    return {
        "optimizer": optimizer_name,
        "seed": int(seed),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "wall_time_total": float(trace["wall_time"][-1]),
        "initial_params": np.asarray(native_init, dtype=float).tolist(),
        "final_params": flat_final.tolist(),
        "trace": trace,
    }


def _summarize(results):
    summary = OrderedDict()
    for optimizer_name in OPTIMIZER_ORDER:
        runs = [run for run in results if run["optimizer"] == optimizer_name]
        if not runs:
            continue
        losses = np.asarray([run["final_loss"] for run in runs], dtype=float)
        train_accs = np.asarray([run["final_train_acc"] for run in runs], dtype=float)
        test_accs = np.asarray([run["final_test_acc"] for run in runs], dtype=float)
        wall = np.asarray([run["wall_time_total"] for run in runs], dtype=float)
        summary[optimizer_name] = {
            "label": OPTIMIZER_LABELS[optimizer_name],
            "n_runs": len(runs),
            "final_loss_mean": float(np.mean(losses)),
            "final_loss_std": float(np.std(losses)),
            "final_train_acc_mean": float(np.mean(train_accs)),
            "final_train_acc_std": float(np.std(train_accs)),
            "final_test_acc_mean": float(np.mean(test_accs)),
            "final_test_acc_std": float(np.std(test_accs)),
            "wall_time_mean": float(np.mean(wall)),
            "wall_time_std": float(np.std(wall)),
        }
    return summary


def _write_report(path, config: ComparisonConfig, results_dir: str, results, summary):
    lines = [
        "# Parity PennyLane Comparison",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "results_dir": results_dir,
                "config": asdict(config),
                "optimizers": list(OPTIMIZER_ORDER),
                "vendor_dir": VENDOR_DIR,
            },
            indent=2,
        ),
        "```",
        "",
        "## Summary",
        "",
        "| Optimizer | Runs | Final loss | Final train acc | Final test acc | Wall time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for stats in summary.values():
        lines.append(
            f"| {stats['label']} | {stats['n_runs']} | "
            f"{stats['final_loss_mean']:.4f} ± {stats['final_loss_std']:.4f} | "
            f"{stats['final_train_acc_mean']:.4f} ± {stats['final_train_acc_std']:.4f} | "
            f"{stats['final_test_acc_mean']:.4f} ± {stats['final_test_acc_std']:.4f} | "
            f"{stats['wall_time_mean']:.2f} ± {stats['wall_time_std']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `config.json`",
            "- `summary.json`",
            "- `loss_vs_iteration.pdf` and `loss_vs_iteration.png`",
            "- `loss_vs_time.pdf` and `loss_vs_time.png`",
            "- `result_<optimizer>_seed<seed>.json`",
            "",
        ]
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = _parse_args()
    os.makedirs(args.results_root, exist_ok=True)
    results_dir = _timestamped_dir(args.results_root, args.run_name)
    config = ComparisonConfig(
        seeds=list(args.seeds),
        repeats=args.repeats,
        n_layers=args.n_layers,
        max_iterations=args.max_iterations,
        test_fraction=args.test_fraction,
        adam_lr=args.adam_lr,
        adagrad_lr=args.adagrad_lr,
        gd_lr=args.gd_lr,
        momentum_lr=args.momentum_lr,
        momentum_beta=args.momentum_beta,
        nesterov_lr=args.nesterov_lr,
        nesterov_beta=args.nesterov_beta,
        rmsprop_lr=args.rmsprop_lr,
        rmsprop_decay=args.rmsprop_decay,
        spsa_c=args.spsa_c,
        spsa_a=args.spsa_a,
        qng_lr=args.qng_lr,
        qng_lam=args.qng_lam,
        qng_approx=args.qng_approx,
        hybrid_switch_iteration=args.hybrid_switch_iteration,
        hybrid_online_step_size=args.hybrid_online_step_size,
        hybrid_batch_step_size=args.hybrid_batch_step_size,
        hybrid_online_decay=args.hybrid_online_decay,
        hybrid_batch_decay=args.hybrid_batch_decay,
    )

    all_results = []
    for seed in args.seeds:
        X_train, X_test, y_train, y_test = generate_parity_4bit(
            test_fraction=args.test_fraction,
            seed=seed,
            repeats=args.repeats,
        )

        print(f"\n{'=' * 88}")
        print(f"Parity comparison | seed={seed} | repeats={args.repeats} | layers={args.n_layers}")
        print(f"{'=' * 88}")

        result = _run_krotov(seed, args, X_train, y_train, X_test, y_test)
        all_results.append(result)
        _save_json(os.path.join(results_dir, f"result_krotov_hybrid_seed{seed}.json"), result)

        for optimizer_name in (
            "pennylane_adam",
            "pennylane_adagrad",
            "pennylane_gradient_descent",
            "pennylane_momentum",
            "pennylane_nesterov",
            "pennylane_rmsprop",
            "pennylane_spsa",
            "pennylane_qng",
            "pennylane_rotosolve",
        ):
            print(f"\n{OPTIMIZER_LABELS[optimizer_name]} | seed={seed}")
            result = _run_pennylane_optimizer(
                optimizer_name,
                seed,
                args,
                X_train,
                y_train,
                X_test,
                y_test,
            )
            all_results.append(result)
            _save_json(os.path.join(results_dir, f"result_{optimizer_name}_seed{seed}.json"), result)

    summary = _summarize(all_results)
    _save_json(os.path.join(results_dir, "config.json"), {"config": asdict(config)})
    _save_json(os.path.join(results_dir, "summary.json"), summary)

    title = f"4-bit parity, {args.n_layers} layers, repeats={args.repeats}"
    _plot_metric(
        all_results,
        "step",
        "loss",
        f"{title}: loss vs iteration",
        "Iteration",
        "Training loss (MSE)",
        results_dir,
        "loss_vs_iteration",
    )
    _plot_metric(
        all_results,
        "wall_time",
        "loss",
        f"{title}: loss vs wall-clock time",
        "Wall-clock time (s)",
        "Training loss (MSE)",
        results_dir,
        "loss_vs_time",
    )
    _write_report(os.path.join(results_dir, "experiment.md"), config, results_dir, all_results, summary)

    print(f"\nSaved parity PennyLane comparison to {results_dir}/")


if __name__ == "__main__":
    main()
