#!/usr/bin/env python3
"""Run a configurable QML training test with Iris-style progress plots.

This script is meant as a single entry point for quick controlled experiments.
You can choose the model family, optimizer(s), dataset, loss function, and the
most relevant model/optimizer hyperparameters from the command line.

Examples
--------
HEA on two moons with Adam and hybrid Krotov:

    python3 run_configurable_qml_test.py \
        --model hea \
        --optimizers krotov_hybrid adam \
        --loss-function bce \
        --n-qubits 4 \
        --n-layers 3 \
        --max-iterations 40

Perez-Salinas model on the crown problem without affine head:

    python3 run_configurable_qml_test.py \
        --model perez_salinas \
        --optimizers krotov_hybrid qng \
        --loss-function weighted_fidelity \
        --perez-problem crown \
        --n-qubits 4 \
        --n-layers 8 \
        --no-affine-head \
        --max-iterations 40

4-bit parity classifier with Adam and L-BFGS-B:

    python3 run_configurable_qml_test.py \
        --model parity_rot \
        --dataset parity_4bit \
        --loss-function mse \
        --optimizers adam lbfgs \
        --n-qubits 4 \
        --n-layers 2 \
        --n-samples 20 \
        --max-iterations 40
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from typing import Optional

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from datasets import (
    available_perez_salinas_problems,
    generate_parity_4bit,
    generate_perez_salinas_dataset,
    generate_two_moons,
    perez_salinas_problem_num_classes,
)
from optimizers.runner import run_optimizer
from qml_models import VQCModel
from qml_models.variants import (
    ChenSUNVQCModel,
    ParityRotClassifierModel,
    PerezSalinasReuploadingModel,
    SimonettiHybridModel,
    SouzaSQQNNModel,
)


RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results_configurable_qml_tests")


@dataclass(frozen=True)
class RunnerConfig:
    n_samples: int = 500
    moon_noise: float = 0.15
    test_fraction: float = 0.3
    input_encoding: str = "tanh_0_pi"

    model_architecture: str = "hea"
    n_qubits: int = 4
    n_layers: int = 3
    entangler: str = "ring"
    observable: str = "Z0Z1"

    max_iterations: int = 100
    adam_lr: float = 0.05
    adam_batch_size: Optional[int] = None
    adam_switch_iteration: int = 0
    lbfgs_maxiter: int = 100
    lbfgs_maxcor: int = 20
    lbfgs_gtol: float = 1e-7
    qng_lr: float = 0.5
    qng_batch_size: Optional[int] = None
    qng_switch_iteration: int = 0
    qng_lam: float = 0.01
    qng_approx: Optional[str] = None

    early_stopping_enabled: bool = True
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-4
    early_stopping_warmup: int = 20

    krotov_step_size: float = 0.3
    krotov_lr_schedule: str = "constant"
    krotov_decay: float = 0.05
    krotov_batch_size: Optional[int] = None
    krotov_target_loss: float = 0.4

    krotov_online_step_size: float = 0.3
    krotov_online_schedule: str = "constant"
    krotov_online_decay: float = 0.05
    krotov_batch_step_size: float = 1.0
    krotov_batch_schedule: str = "constant"
    krotov_batch_decay: float = 0.05

    hybrid_switch_iteration: int = 20
    hybrid_online_step_size: float = 0.3
    hybrid_batch_step_size: float = 1.0
    hybrid_online_schedule: str = "constant"
    hybrid_batch_schedule: str = "constant"
    hybrid_online_decay: float = 0.05
    hybrid_batch_decay: float = 0.05
    hybrid_scaling_mode: str = "none"
    hybrid_scaling_apply_phase: str = "both"
    hybrid_scaling_config: Optional[dict] = None

    seeds: list[int] = field(default_factory=lambda: list(range(5)))
    optimizers: list[str] = field(default_factory=lambda: ["krotov_hybrid", "adam", "lbfgs", "qng"])
    run_krotov_batch_sweep: bool = False
    krotov_batch_sweep_step_sizes: list[float] = field(
        default_factory=lambda: [1.0, 0.3, 0.1, 0.05, 0.02]
    )
    krotov_batch_sweep_schedules: list[str] = field(default_factory=lambda: ["constant", "inverse"])
    run_krotov_hybrid_sweep: bool = True
    hybrid_switch_iterations: list[int] = field(default_factory=lambda: [5, 10, 20, 30, 50])
    loss_threshold: float = 0.4
    loss_thresholds: list[float] = field(default_factory=lambda: [0.45, 0.40, 0.38, 0.36])
    results_dir: str = "results"
    plots_dir: str = "plots"


DEFAULT_CONFIG = RunnerConfig()

MODEL_CHOICES = (
    "hea",
    "dense_angle",
    "data_reuploading",
    "perez_salinas",
    "simonetti_hybrid",
    "simonetti_explicit_angle",
    "chen_sun",
    "souza_sqqnn",
    "parity_rot",
)
DATASET_CHOICES = ("auto", "two_moons", "perez_salinas", "parity_4bit")
LOSS_CHOICES = ("bce", "weighted_fidelity", "mse")
OPTIMIZER_CHOICES = ("krotov_online", "krotov_batch", "krotov_hybrid", "adam", "lbfgs", "qng")

OPTIMIZER_LABELS = {
    "krotov_online": "Krotov online",
    "krotov_batch": "Krotov batch",
    "krotov_hybrid": "Hybrid Krotov",
    "adam": "Adam",
    "lbfgs": "L-BFGS-B",
    "qng": "QNG",
}
OPTIMIZER_COLORS = {
    "krotov_online": "#c94c4c",
    "krotov_batch": "#d17c00",
    "krotov_hybrid": "#8b1e3f",
    "adam": "#2563eb",
    "lbfgs": "#0f766e",
    "qng": "#d97706",
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


def _slugify(text):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_").lower() or "run"


def _jsonify_trace(trace):
    payload = {}
    for key, values in trace.items():
        payload[key] = [str(v) for v in values] if key == "phase" else [float(v) for v in values]
    return payload


def _resolve_dataset_name(args):
    if args.dataset != "auto":
        return args.dataset
    if args.model == "perez_salinas":
        return "perez_salinas"
    if args.model == "parity_rot":
        return "parity_4bit"
    return "two_moons"


def _resolve_loss_label(loss_function):
    if loss_function == "bce":
        return "BCE"
    if loss_function == "weighted_fidelity":
        return "weighted fidelity"
    return "mean squared error"


def _model_label(args):
    labels = {
        "hea": "HEA",
        "dense_angle": "Dense-angle VQC",
        "data_reuploading": "Data-reuploading VQC",
        "perez_salinas": "Perez-Salinas reuploading",
        "simonetti_hybrid": "Simonetti hybrid",
        "simonetti_explicit_angle": "Simonetti explicit-angle",
        "chen_sun": "Chen SUN-VQC",
        "souza_sqqnn": "Souza SQQNN",
        "parity_rot": "Parity Rot classifier",
    }
    return labels[args.model]


def _validate_args(args):
    dataset_name = _resolve_dataset_name(args)

    if args.model == "perez_salinas":
        if dataset_name != "perez_salinas":
            raise ValueError("The Perez-Salinas model must be used with the perez_salinas dataset option.")
        if args.loss_function != "weighted_fidelity":
            raise ValueError("The Perez-Salinas model only supports --loss-function weighted_fidelity.")
    elif args.model == "parity_rot":
        if dataset_name != "parity_4bit":
            raise ValueError("The parity_rot model must be used with the parity_4bit dataset option.")
        if args.loss_function != "mse":
            raise ValueError("The parity_rot model only supports --loss-function mse.")
        if args.n_qubits != 4:
            raise ValueError("The parity_rot model requires --n-qubits 4.")
    else:
        if dataset_name != "two_moons":
            raise ValueError("All non-Perez models in this script currently use the two_moons dataset.")
        if args.loss_function != "bce":
            raise ValueError("All non-Perez models in this script currently use --loss-function bce.")

    if args.model == "dense_angle" and args.n_qubits != 2:
        raise ValueError("The dense_angle architecture currently requires --n-qubits 2.")

    if args.progress_interval < 0:
        raise ValueError("--progress-interval must be non-negative.")

    if args.max_iterations <= 0:
        raise ValueError("--max-iterations must be positive.")

    if args.n_layers <= 0:
        raise ValueError("--n-layers must be positive.")

    if args.n_qubits <= 0:
        raise ValueError("--n-qubits must be positive.")

    if dataset_name == "parity_4bit" and args.n_samples <= 0:
        raise ValueError("--n-samples must be positive for the parity_4bit dataset.")

    return dataset_name


def _parse_scaling_config(raw_value):
    if raw_value is None:
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON for --hybrid-scaling-config: {exc}") from exc


def _build_train_config(args):
    qng_approx = None if args.qng_approx == "full" else args.qng_approx
    scaling_config = _parse_scaling_config(args.hybrid_scaling_config)

    return replace(
        DEFAULT_CONFIG,
        n_samples=args.n_samples,
        moon_noise=args.noise,
        test_fraction=args.test_fraction,
        input_encoding=args.input_encoding,
        model_architecture={
            "hea": "hea",
            "dense_angle": "two_moons_dense_angle",
            "data_reuploading": "data_reuploading",
            "parity_rot": "parity_rot",
        }.get(args.model, DEFAULT_CONFIG.model_architecture),
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        entangler=args.entangler,
        observable=args.observable,
        max_iterations=args.max_iterations,
        adam_lr=args.adam_lr,
        adam_batch_size=args.adam_batch_size,
        adam_switch_iteration=args.adam_switch_iteration,
        lbfgs_maxiter=args.lbfgs_maxiter,
        lbfgs_maxcor=args.lbfgs_maxcor,
        lbfgs_gtol=args.lbfgs_gtol,
        qng_lr=args.qng_lr,
        qng_batch_size=args.qng_batch_size,
        qng_switch_iteration=args.qng_switch_iteration,
        qng_lam=args.qng_lam,
        qng_approx=qng_approx,
        early_stopping_enabled=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
        early_stopping_warmup=args.early_stopping_warmup,
        krotov_step_size=args.krotov_step_size,
        krotov_lr_schedule=args.krotov_lr_schedule,
        krotov_decay=args.krotov_decay,
        krotov_batch_size=args.krotov_batch_size,
        krotov_online_step_size=args.krotov_online_step_size,
        krotov_online_schedule=args.krotov_online_schedule,
        krotov_online_decay=args.krotov_online_decay,
        krotov_batch_step_size=args.krotov_batch_step_size,
        krotov_batch_schedule=args.krotov_batch_schedule,
        krotov_batch_decay=args.krotov_batch_decay,
        hybrid_switch_iteration=args.hybrid_switch_iteration,
        hybrid_online_step_size=args.hybrid_online_step_size,
        hybrid_batch_step_size=args.hybrid_batch_step_size,
        hybrid_online_schedule=args.hybrid_online_schedule,
        hybrid_batch_schedule=args.hybrid_batch_schedule,
        hybrid_online_decay=args.hybrid_online_decay,
        hybrid_batch_decay=args.hybrid_batch_decay,
        hybrid_scaling_mode=args.hybrid_scaling_mode,
        hybrid_scaling_apply_phase=args.hybrid_scaling_apply_phase,
        hybrid_scaling_config=scaling_config,
        seeds=list(args.seeds),
        optimizers=list(args.optimizers),
        run_krotov_batch_sweep=False,
        run_krotov_hybrid_sweep=False,
        results_dir=os.path.basename(args.results_root),
    )


def _build_run_directory(args):
    run_tag = args.run_name or f"{args.model}_{'_'.join(args.optimizers)}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.results_root, f"{timestamp}_{_slugify(run_tag)}")
    os.makedirs(run_dir, exist_ok=False)
    return run_dir


def _load_dataset(args, dataset_name, seed):
    if dataset_name == "two_moons":
        X_train, X_test, y_train, y_test = generate_two_moons(
            n_samples=args.n_samples,
            noise=args.noise,
            test_fraction=args.test_fraction,
            seed=seed,
            encoding=args.input_encoding,
        )
        metadata = {
            "dataset": "two_moons",
            "n_samples": int(args.n_samples),
            "noise": float(args.noise),
            "test_fraction": float(args.test_fraction),
            "input_encoding": args.input_encoding,
        }
        return X_train, X_test, y_train, y_test, metadata

    if dataset_name == "parity_4bit":
        X_train, X_test, y_train, y_test = generate_parity_4bit(
            test_fraction=args.test_fraction,
            seed=seed,
            repeats=args.n_samples,
        )
        metadata = {
            "dataset": "parity_4bit",
            "test_fraction": float(args.test_fraction),
            "repeats_per_bitstring": int(args.n_samples),
            "n_unique_bitstrings": 16,
            "label_values": [-1, 1],
        }
        return X_train, X_test, y_train, y_test, metadata

    X_train, X_test, y_train, y_test = generate_perez_salinas_dataset(
        problem=args.perez_problem,
        n_samples=args.n_samples,
        test_fraction=args.test_fraction,
        seed=seed,
    )
    metadata = {
        "dataset": "perez_salinas",
        "problem": args.perez_problem,
        "n_samples": int(args.n_samples),
        "test_fraction": float(args.test_fraction),
        "n_classes": int(perez_salinas_problem_num_classes(args.perez_problem)),
    }
    return X_train, X_test, y_train, y_test, metadata


def _instantiate_model(args):
    if args.model == "hea":
        return VQCModel(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            entangler=args.entangler,
            architecture="hea",
            observable=args.observable,
        )

    if args.model == "dense_angle":
        return VQCModel(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            entangler=args.entangler,
            architecture="two_moons_dense_angle",
            observable=args.observable,
        )

    if args.model == "data_reuploading":
        return VQCModel(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            entangler=args.entangler,
            architecture="data_reuploading",
            observable=args.observable,
        )

    if args.model == "perez_salinas":
        return PerezSalinasReuploadingModel(
            n_qubits=args.n_qubits,
            n_layers=args.n_layers,
            n_classes=perez_salinas_problem_num_classes(args.perez_problem),
            use_entanglement=not args.no_entanglement,
            use_classical_head=not args.no_affine_head,
            loss_mode=args.loss_function,
        )

    if args.model == "simonetti_hybrid":
        return SimonettiHybridModel(
            mode="hybrid",
            n_sublayers=args.simonetti_sublayers,
            entangler=args.simonetti_entangler,
        )

    if args.model == "simonetti_explicit_angle":
        return SimonettiHybridModel(
            mode="explicit_angle",
            n_sublayers=args.simonetti_sublayers,
            entangler=args.simonetti_entangler,
        )

    if args.model == "chen_sun":
        return ChenSUNVQCModel(
            n_macro_layers=args.chen_macro_layers,
            encoding_axes=tuple(args.chen_encoding_axes),
            readout=args.chen_readout,
        )

    if args.model == "souza_sqqnn":
        return SouzaSQQNNModel(
            variant=args.souza_variant,
            n_neurons=args.souza_neurons,
        )

    if args.model == "parity_rot":
        return ParityRotClassifierModel(n_layers=args.n_layers)

    raise ValueError(f"Unknown model: {args.model}")


def _model_metadata(model):
    payload = OrderedDict()
    payload["class"] = model.__class__.__name__
    for field in (
        "n_qubits",
        "n_layers",
        "n_params",
        "n_quantum_params",
        "n_weight_params",
        "weights_shape",
        "n_classes",
        "architecture",
        "observable",
        "use_entanglement",
        "use_classical_head",
        "loss_mode",
        "mode",
        "n_macro_layers",
        "encoding_axes",
        "readout",
        "n_sublayers",
        "variant",
        "n_neurons",
    ):
        if hasattr(model, field):
            value = getattr(model, field)
            if isinstance(value, tuple):
                value = list(value)
            payload[field] = value
    return payload


def _run_single(optimizer_name, seed, args, dataset_name, config):
    X_train, X_test, y_train, y_test, dataset_metadata = _load_dataset(args, dataset_name, seed)
    model = _instantiate_model(args)
    init_params = np.asarray(model.init_params(seed=seed), dtype=float)

    print(f"\n{'=' * 88}")
    print(
        f"Model: {_model_label(args)} | Dataset: {dataset_metadata['dataset']} | "
        f"Loss: {args.loss_function} | Optimizer: {optimizer_name} | Seed: {seed}"
    )
    print(f"{'=' * 88}")

    start_time = time.time()
    final_params, trace = run_optimizer(
        optimizer_name,
        model,
        init_params.copy(),
        X_train,
        y_train,
        X_test,
        y_test,
        config,
    )
    wall_total = time.time() - start_time

    result = {
        "model_name": args.model,
        "model_label": _model_label(args),
        "dataset": dataset_metadata,
        "loss_function": args.loss_function,
        "optimizer": optimizer_name,
        "seed": int(seed),
        "model_metadata": _model_metadata(model),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "wall_time_total": float(wall_total),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "total_cost_units": int(trace["cost_units"][-1]),
        "total_steps": int(trace["step"][-1]),
        "trace": _jsonify_trace(trace),
        "initial_params": init_params.tolist(),
        "final_params": np.asarray(final_params, dtype=float).tolist(),
    }

    print(
        f"  Done: loss={result['final_loss']:.4f} "
        f"train_acc={result['final_train_acc']:.3f} "
        f"test_acc={result['final_test_acc']:.3f} "
        f"cost={result['total_cost_units']} wall={wall_total:.2f}s",
        flush=True,
    )
    return result


def _summarize_results(results, optimizer_order):
    summary = OrderedDict()
    for optimizer_name in optimizer_order:
        runs = [run for run in results if run["optimizer"] == optimizer_name]
        if not runs:
            continue
        final_losses = np.asarray([run["final_loss"] for run in runs], dtype=float)
        train_accs = np.asarray([run["final_train_acc"] for run in runs], dtype=float)
        test_accs = np.asarray([run["final_test_acc"] for run in runs], dtype=float)
        wall_times = np.asarray([run["wall_time_total"] for run in runs], dtype=float)
        costs = np.asarray([run["total_cost_units"] for run in runs], dtype=float)
        summary[optimizer_name] = {
            "label": OPTIMIZER_LABELS[optimizer_name],
            "n_runs": len(runs),
            "final_loss_mean": float(np.mean(final_losses)),
            "final_loss_std": float(np.std(final_losses)),
            "final_train_acc_mean": float(np.mean(train_accs)),
            "final_train_acc_std": float(np.std(train_accs)),
            "final_test_acc_mean": float(np.mean(test_accs)),
            "final_test_acc_std": float(np.std(test_accs)),
            "wall_time_mean": float(np.mean(wall_times)),
            "wall_time_std": float(np.std(wall_times)),
            "cost_mean": float(np.mean(costs)),
            "cost_std": float(np.std(costs)),
        }
    return summary


def _interpolate_traces(x_traces, y_traces, n_points=300):
    x_min = max(trace[0] for trace in x_traces)
    x_max = min(trace[-1] for trace in x_traces)
    x_grid = np.linspace(x_min, x_max, n_points)
    y_interp = np.array([np.interp(x_grid, tx, ty) for tx, ty in zip(x_traces, y_traces)])
    return x_grid, y_interp


def _grouped_by_optimizer(results, optimizer_order):
    return OrderedDict((name, [run for run in results if run["optimizer"] == name]) for name in optimizer_order)


def _mean_std_trace(grouped, optimizer_name, x_key, y_key):
    runs = grouped[optimizer_name]
    if not runs:
        return None, None, None
    xs = [np.asarray(run["trace"][x_key], dtype=float) for run in runs]
    ys = [np.asarray(run["trace"][y_key], dtype=float) for run in runs]
    x_grid, y_interp = _interpolate_traces(xs, ys)
    return x_grid, np.mean(y_interp, axis=0), np.std(y_interp, axis=0)


def _save_fig(fig, name, results_dir):
    fig.savefig(os.path.join(results_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(results_dir, f"{name}.png"))
    plt.close(fig)


def _plot_metric(results, optimizer_order, x_key, y_key, title, xlabel, ylabel, results_dir, file_name):
    grouped = _grouped_by_optimizer(results, optimizer_order)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for optimizer_name in optimizer_order:
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


def _write_markdown_report(path, args, resolved, config, results, summary):
    summary_lines = [
        "| Optimizer | Runs | Final loss | Final train acc | Final test acc | Wall time (s) | Cost units |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for stats in summary.values():
        summary_lines.append(
            f"| {stats['label']} | {stats['n_runs']} | "
            f"{stats['final_loss_mean']:.4f} ± {stats['final_loss_std']:.4f} | "
            f"{stats['final_train_acc_mean']:.4f} ± {stats['final_train_acc_std']:.4f} | "
            f"{stats['final_test_acc_mean']:.4f} ± {stats['final_test_acc_std']:.4f} | "
            f"{stats['wall_time_mean']:.2f} ± {stats['wall_time_std']:.2f} | "
            f"{stats['cost_mean']:.1f} ± {stats['cost_std']:.1f} |"
        )

    payload = {
        "timestamp": resolved["timestamp"],
        "command": resolved["command"],
        "results_dir": resolved["results_dir"],
        "dataset": resolved["dataset_name"],
        "model_label": resolved["model_label"],
        "loss_label": resolved["loss_label"],
        "args": vars(args),
        "config": asdict(config),
        "n_result_files": len(results),
    }

    lines = [
        "# Configurable QML Test",
        "",
        "## Overview",
        "",
        f"- Created: `{resolved['timestamp']}`",
        f"- Model: `{resolved['model_label']}`",
        f"- Dataset: `{resolved['dataset_name']}`",
        f"- Loss: `{resolved['loss_label']}`",
        f"- Optimizers: `{', '.join(args.optimizers)}`",
        f"- Seeds: `{args.seeds}`",
        "",
        "## Summary",
        "",
        *summary_lines,
        "",
        "## Experiment Parameters",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
        "## Generated Files",
        "",
        "- `config.json`",
        "- `summary.json`",
        "- `loss_vs_iteration.pdf` and `loss_vs_iteration.png`",
        "- `loss_vs_time.pdf` and `loss_vs_time.png`",
        "- `test_accuracy_vs_iteration.pdf` and `test_accuracy_vs_iteration.png`",
        "- `test_accuracy_vs_time.pdf` and `test_accuracy_vs_time.png`",
        "- `result_<optimizer>_seed<seed>.json` for each run",
        "",
    ]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--model", choices=MODEL_CHOICES, default="hea")
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="auto")
    parser.add_argument("--loss-function", choices=LOSS_CHOICES, default="bce")
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZER_CHOICES, default=["krotov_hybrid"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--results-root", default=RESULTS_ROOT)

    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--input-encoding", choices=["tanh_0_pi", "linear_pm_pi"], default="tanh_0_pi")
    parser.add_argument("--perez-problem", choices=available_perez_salinas_problems(), default="crown")

    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--entangler", choices=["ring", "chain", "none"], default="ring")
    parser.add_argument("--observable", choices=["Z0", "Z0Z1"], default="Z0Z1")
    parser.add_argument("--no-entanglement", action="store_true")
    parser.add_argument("--no-affine-head", action="store_true")

    parser.add_argument("--simonetti-sublayers", type=int, default=4)
    parser.add_argument(
        "--simonetti-entangler",
        choices=["cnot_01", "cnot_10", "bidirectional"],
        default="cnot_01",
    )
    parser.add_argument("--chen-macro-layers", type=int, default=2)
    parser.add_argument("--chen-encoding-axes", nargs="+", choices=["y", "z"], default=["y"])
    parser.add_argument("--chen-readout", choices=["simple_z0", "hybrid_linear"], default="simple_z0")
    parser.add_argument("--souza-variant", choices=["reduced", "full"], default="reduced")
    parser.add_argument("--souza-neurons", type=int, default=4)

    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--progress-interval", type=int, default=5)
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stopping-warmup", type=int, default=20)

    parser.add_argument("--adam-lr", type=float, default=0.05)
    parser.add_argument("--adam-batch-size", type=int, default=None)
    parser.add_argument("--adam-switch-iteration", type=int, default=0)

    parser.add_argument("--lbfgs-maxiter", type=int, default=40)
    parser.add_argument("--lbfgs-maxcor", type=int, default=20)
    parser.add_argument("--lbfgs-gtol", type=float, default=1e-7)

    parser.add_argument("--qng-lr", type=float, default=0.5)
    parser.add_argument("--qng-lam", type=float, default=0.01)
    parser.add_argument("--qng-approx", choices=["full", "diag"], default="full")
    parser.add_argument("--qng-batch-size", type=int, default=None)
    parser.add_argument("--qng-switch-iteration", type=int, default=0)

    parser.add_argument("--krotov-step-size", type=float, default=0.3)
    parser.add_argument("--krotov-lr-schedule", choices=["constant", "inverse", "exp"], default="constant")
    parser.add_argument("--krotov-decay", type=float, default=0.05)
    parser.add_argument("--krotov-batch-size", type=int, default=None)

    parser.add_argument("--krotov-online-step-size", type=float, default=0.3)
    parser.add_argument("--krotov-online-schedule", choices=["constant", "inverse", "exp"], default="constant")
    parser.add_argument("--krotov-online-decay", type=float, default=0.05)

    parser.add_argument("--krotov-batch-step-size", type=float, default=1.0)
    parser.add_argument("--krotov-batch-schedule", choices=["constant", "inverse", "exp"], default="constant")
    parser.add_argument("--krotov-batch-decay", type=float, default=0.05)

    parser.add_argument("--hybrid-switch-iteration", type=int, default=20)
    parser.add_argument("--hybrid-online-step-size", type=float, default=0.3)
    parser.add_argument("--hybrid-batch-step-size", type=float, default=1.0)
    parser.add_argument("--hybrid-online-schedule", choices=["constant", "inverse", "exp"], default="constant")
    parser.add_argument("--hybrid-batch-schedule", choices=["constant", "inverse", "exp"], default="constant")
    parser.add_argument("--hybrid-online-decay", type=float, default=0.05)
    parser.add_argument("--hybrid-batch-decay", type=float, default=0.05)
    parser.add_argument("--hybrid-scaling-mode", default="none")
    parser.add_argument("--hybrid-scaling-apply-phase", default="both")
    parser.add_argument("--hybrid-scaling-config", default=None)

    return parser.parse_args()


def main():
    args = parse_args()
    dataset_name = _validate_args(args)
    os.makedirs(args.results_root, exist_ok=True)
    results_dir = _build_run_directory(args)

    sys.stdout.reconfigure(line_buffering=True)
    config = _build_train_config(args)

    resolved = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset_name": dataset_name,
        "model_label": _model_label(args),
        "loss_label": _resolve_loss_label(args.loss_function),
        "results_dir": results_dir,
        "command": " ".join(sys.argv),
    }

    all_results = []
    for optimizer_name in args.optimizers:
        for seed in args.seeds:
            result = _run_single(optimizer_name, seed, args, dataset_name, config)
            all_results.append(result)
            out_name = f"result_{optimizer_name}_seed{seed}.json"
            with open(os.path.join(results_dir, out_name), "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)

    summary = _summarize_results(all_results, args.optimizers)

    with open(os.path.join(results_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "resolved": resolved,
                "args": vars(args),
                "config": asdict(config),
            },
            handle,
            indent=2,
        )
    with open(os.path.join(results_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    title_prefix = f"{_model_label(args)} on {dataset_name}"
    loss_ylabel = f"Training loss ({_resolve_loss_label(args.loss_function)})"

    _plot_metric(
        all_results,
        args.optimizers,
        "step",
        "loss",
        f"{title_prefix}: loss vs iteration",
        "Iteration",
        loss_ylabel,
        results_dir,
        "loss_vs_iteration",
    )
    _plot_metric(
        all_results,
        args.optimizers,
        "wall_time",
        "loss",
        f"{title_prefix}: loss vs wall-clock time",
        "Wall-clock time (s)",
        loss_ylabel,
        results_dir,
        "loss_vs_time",
    )
    _plot_metric(
        all_results,
        args.optimizers,
        "step",
        "test_acc",
        f"{title_prefix}: test accuracy vs iteration",
        "Iteration",
        "Test accuracy",
        results_dir,
        "test_accuracy_vs_iteration",
    )
    _plot_metric(
        all_results,
        args.optimizers,
        "wall_time",
        "test_acc",
        f"{title_prefix}: test accuracy vs wall-clock time",
        "Wall-clock time (s)",
        "Test accuracy",
        results_dir,
        "test_accuracy_vs_time",
    )

    _write_markdown_report(
        os.path.join(results_dir, "experiment.md"),
        args,
        resolved,
        config,
        all_results,
        summary,
    )

    print(f"\nConfigurable QML test saved to {results_dir}/")


if __name__ == "__main__":
    main()
