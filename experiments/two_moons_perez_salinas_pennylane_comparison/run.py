#!/usr/bin/env python3
"""Compare Hybrid Krotov against PennyLane optimizers on the Perez-Salinas model."""

from __future__ import annotations

import argparse
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

from datasets import (
    available_perez_salinas_problems,
    generate_perez_salinas_dataset,
    perez_salinas_problem_num_classes,
)
from optimizers.runner import run_optimizer
from qml_models.variants import (
    PennyLanePerezSalinasReuploadingModel,
    PerezSalinasReuploadingModel,
)
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
    problem: str
    seeds: list[int]
    n_qubits: int
    n_layers: int
    n_samples: int
    test_fraction: float
    use_entanglement: bool
    use_classical_head: bool
    max_iterations: int
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
    parser.add_argument("--problem", choices=available_perez_salinas_problems(), default="crown")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=240)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--no-entanglement", action="store_true")
    parser.add_argument("--no-affine-head", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--adam-lr", type=float, default=0.05)
    parser.add_argument("--adagrad-lr", type=float, default=0.1)
    parser.add_argument("--gd-lr", type=float, default=0.05)
    parser.add_argument("--momentum-lr", type=float, default=0.05)
    parser.add_argument("--momentum-beta", type=float, default=0.9)
    parser.add_argument("--nesterov-lr", type=float, default=0.05)
    parser.add_argument("--nesterov-beta", type=float, default=0.9)
    parser.add_argument("--rmsprop-lr", type=float, default=0.05)
    parser.add_argument("--rmsprop-decay", type=float, default=0.9)
    parser.add_argument("--spsa-c", type=float, default=0.2)
    parser.add_argument("--spsa-a", type=float, default=None)
    parser.add_argument("--qng-lr", type=float, default=0.5)
    parser.add_argument("--qng-lam", type=float, default=0.01)
    parser.add_argument("--qng-approx", choices=["full", "block-diag", "diag"], default="block-diag")
    parser.add_argument("--hybrid-switch-iteration", type=int, default=10)
    parser.add_argument("--hybrid-online-step-size", type=float, default=0.3)
    parser.add_argument("--hybrid-batch-step-size", type=float, default=0.5)
    parser.add_argument("--hybrid-online-decay", type=float, default=0.05)
    parser.add_argument("--hybrid-batch-decay", type=float, default=0.05)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _timestamped_dir(results_root: str, run_name: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_name or "perez_salinas_pennylane_comparison"
    path = os.path.join(results_root, f"{timestamp}_{suffix}")
    os.makedirs(path, exist_ok=False)
    return path


def _save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


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
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
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


def _make_krotov_config(args) -> RunnerConfig:
    return RunnerConfig(
        n_samples=args.n_samples,
        test_fraction=args.test_fraction,
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
        early_stopping_enabled=False,
        results_dir="results",
        plots_dir="plots",
        optimizers=["krotov_hybrid"],
        seeds=list(args.seeds),
    )


def _run_krotov(seed: int, args, X_train, y_train, X_test, y_test):
    model = PerezSalinasReuploadingModel(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_classes=perez_salinas_problem_num_classes(args.problem),
        use_entanglement=not args.no_entanglement,
        use_classical_head=not args.no_affine_head,
        loss_mode="weighted_fidelity",
    )
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


def _run_pennylane_optimizer(optimizer_name: str, seed: int, args, X_train, y_train, X_test, y_test):
    model = PennyLanePerezSalinasReuploadingModel(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_classes=perez_salinas_problem_num_classes(args.problem),
        use_entanglement=not args.no_entanglement,
        use_classical_head=not args.no_affine_head,
        loss_mode="weighted_fidelity",
    )
    params = pnp.array(model.init_params(seed=seed), requires_grad=True)
    X_train_arr = np.asarray(X_train, dtype=float)
    X_test_arr = np.asarray(X_test, dtype=float)
    y_train_arr = np.asarray(y_train, dtype=int)
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
        optimizer = qml.NesterovMomentumOptimizer(stepsize=args.nesterov_lr, momentum=args.nesterov_beta)
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
        metric_tensor_fn = lambda p: model.metric_tensor(p, X_train_arr, approx=args.qng_approx)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    cost_fn = lambda p: model.loss(p, X_train_arr, y_train_arr)
    grad_fn = qml.grad(cost_fn)
    trace = {"step": [], "loss": [], "train_acc": [], "test_acc": [], "wall_time": []}
    start = time.time()

    for step in range(1, args.max_iterations + 1):
        if optimizer_name == "pennylane_spsa":
            params, _ = optimizer.step_and_cost(cost_fn, params)
        elif optimizer_name == "pennylane_qng":
            params, _ = optimizer.step_and_cost(cost_fn, params, grad_fn=grad_fn, metric_tensor_fn=metric_tensor_fn)
        else:
            params, _ = optimizer.step_and_cost(cost_fn, params, grad_fn=grad_fn)

        train_loss = float(model.loss(params, X_train_arr, y_train_arr))
        train_acc = model.accuracy(params, X_train_arr, y_train_arr)
        test_acc = model.accuracy(params, X_test_arr, y_test_arr)
        elapsed = time.time() - start

        trace["step"].append(float(step))
        trace["loss"].append(train_loss)
        trace["train_acc"].append(float(train_acc))
        trace["test_acc"].append(float(test_acc))
        trace["wall_time"].append(float(elapsed))

        print(
            f"  {OPTIMIZER_LABELS[optimizer_name]} step {step:>3}: "
            f"loss={train_loss:.4f} train_acc={train_acc:.3f} test_acc={test_acc:.3f}",
            flush=True,
        )

    return {
        "optimizer": optimizer_name,
        "seed": int(seed),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "wall_time_total": float(trace["wall_time"][-1]),
        "initial_params": np.asarray(model.init_params(seed=seed), dtype=float).tolist(),
        "final_params": np.asarray(params, dtype=float).tolist(),
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
        walls = np.asarray([run["wall_time_total"] for run in runs], dtype=float)
        summary[optimizer_name] = {
            "label": OPTIMIZER_LABELS[optimizer_name],
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
    return summary


def _write_report(path, config: ComparisonConfig, results_dir: str, summary):
    lines = [
        "# Perez-Salinas PennyLane Comparison",
        "",
        "Rotosolve is intentionally omitted here. For this data-reuploading objective, the parameter frequencies depend on the sample-dependent input scaling, so we do not have reliable exact frequency metadata for a faithful Rotosolve baseline.",
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
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = _parse_args()
    os.makedirs(args.results_root, exist_ok=True)
    results_dir = _timestamped_dir(args.results_root, args.run_name)
    config = ComparisonConfig(
        problem=args.problem,
        seeds=list(args.seeds),
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_samples=args.n_samples,
        test_fraction=args.test_fraction,
        use_entanglement=not args.no_entanglement,
        use_classical_head=not args.no_affine_head,
        max_iterations=args.max_iterations,
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
        X_train, X_test, y_train, y_test = generate_perez_salinas_dataset(
            problem=args.problem,
            n_samples=args.n_samples,
            test_fraction=args.test_fraction,
            seed=seed,
        )

        print(f"\n{'=' * 92}")
        print(
            f"Perez-Salinas comparison | problem={args.problem} | seed={seed} | "
            f"layers={args.n_layers} | head={'on' if not args.no_affine_head else 'off'}"
        )
        print(f"{'=' * 92}")

        result = _run_krotov(seed, args, X_train, y_train, X_test, y_test)
        all_results.append(result)
        _save_json(os.path.join(results_dir, f"result_krotov_hybrid_seed{seed}.json"), result)

        for optimizer_name in OPTIMIZER_ORDER[1:]:
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

    title = (
        f"Perez-Salinas {args.problem}, {args.n_layers} layers, "
        f"{'affine head' if not args.no_affine_head else 'no affine head'}"
    )
    _plot_metric(
        all_results,
        "step",
        "loss",
        f"{title}: loss vs iteration",
        "Iteration",
        "Training loss (weighted fidelity)",
        results_dir,
        "loss_vs_iteration",
    )
    _plot_metric(
        all_results,
        "wall_time",
        "loss",
        f"{title}: loss vs wall-clock time",
        "Wall-clock time (s)",
        "Training loss (weighted fidelity)",
        results_dir,
        "loss_vs_time",
    )
    _write_report(os.path.join(results_dir, "experiment.md"), config, results_dir, summary)

    print(f"\nSaved Perez-Salinas comparison to {results_dir}/")


if __name__ == "__main__":
    main()
