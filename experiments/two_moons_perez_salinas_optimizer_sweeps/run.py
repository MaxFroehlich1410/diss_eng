#!/usr/bin/env python3
"""Sharded hyperparameter sweeps for the Perez-Salinas classifier.

This sweep fixes the model to the strongest currently preferred benchmark setup:
- problem: crown
- 4 qubits
- 8 layers
- entangling circuit
- classical affine head enabled
- 600 total samples with a 70/30 split

It compares Hybrid Krotov against the PennyLane optimizer family already used in
the parity study, using three seeds per hyperparameter combination by default.
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

import numpy as np

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

import pennylane as qml
from pennylane import numpy as pnp

from datasets import available_perez_salinas_problems, generate_perez_salinas_dataset, perez_salinas_problem_num_classes
from optimizers.runner import run_optimizer
from qml_models.variants import PennyLanePerezSalinasReuploadingModel, PerezSalinasReuploadingModel
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
}
SWEEP_GRIDS = {
    "krotov_hybrid": OrderedDict(
        [
            ("hybrid_switch_iteration", [5, 10, 15]),
            ("hybrid_online_step_size", [0.1, 0.3]),
            ("hybrid_batch_step_size", [0.5, 1.0]),
        ]
    ),
    "pennylane_adam": OrderedDict(
        [
            ("adam_lr", [0.01, 0.03, 0.05, 0.1]),
        ]
    ),
    "pennylane_adagrad": OrderedDict(
        [
            ("adagrad_lr", [0.03, 0.1, 0.3]),
        ]
    ),
    "pennylane_gradient_descent": OrderedDict(
        [
            ("gd_lr", [0.01, 0.03, 0.1]),
        ]
    ),
    "pennylane_momentum": OrderedDict(
        [
            ("momentum_lr", [0.01, 0.03, 0.1]),
            ("momentum_beta", [0.8, 0.9]),
        ]
    ),
    "pennylane_nesterov": OrderedDict(
        [
            ("nesterov_lr", [0.01, 0.03, 0.1]),
            ("nesterov_beta", [0.8, 0.9]),
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
}
BASELINES = {
    "krotov_hybrid": {
        "hybrid_switch_iteration": 10,
        "hybrid_online_step_size": 0.3,
        "hybrid_batch_step_size": 0.5,
    },
    "pennylane_adam": {"adam_lr": 0.05},
    "pennylane_adagrad": {"adagrad_lr": 0.1},
    "pennylane_gradient_descent": {"gd_lr": 0.05},
    "pennylane_momentum": {"momentum_lr": 0.05, "momentum_beta": 0.9},
    "pennylane_nesterov": {"nesterov_lr": 0.05, "nesterov_beta": 0.9},
    "pennylane_rmsprop": {"rmsprop_lr": 0.05, "rmsprop_decay": 0.9},
    "pennylane_spsa": {"spsa_c": 0.2, "spsa_a": None, "spsa_alpha": 0.602},
}


@dataclass(frozen=True)
class SweepConfig:
    problem: str
    optimizers: list[str]
    seeds: list[int]
    n_qubits: int
    n_layers: int
    n_samples: int
    test_fraction: float
    use_entanglement: bool
    use_classical_head: bool
    max_iterations: int
    num_shards: int
    shard_index: int
    results_root: str


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=available_perez_salinas_problems(), default="crown")
    parser.add_argument("--optimizers", nargs="+", choices=OPTIMIZER_ORDER, default=list(OPTIMIZER_ORDER))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--no-entanglement", action="store_true")
    parser.add_argument("--no-affine-head", action="store_true")
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _timestamped_dir(results_root: str, run_name: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_name or "perez_salinas_optimizer_sweeps"
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


def _build_jobs(args):
    jobs = []
    for optimizer_name in args.optimizers:
        for hp_dict in _expand_grid(SWEEP_GRIDS[optimizer_name]):
            for seed in args.seeds:
                jobs.append({"optimizer": optimizer_name, "hp": hp_dict, "seed": seed})
    return jobs


def _assert_no_overlap(X_train, X_test) -> None:
    train_points = {tuple(np.round(row, 12)) for row in np.asarray(X_train, dtype=float)}
    test_points = {tuple(np.round(row, 12)) for row in np.asarray(X_test, dtype=float)}
    overlap = train_points & test_points
    if overlap:
        raise RuntimeError(
            f"Train/test split contains {len(overlap)} duplicate points; aborting to avoid leakage."
        )


def _make_krotov_config(hp_dict, args) -> RunnerConfig:
    return RunnerConfig(
        n_samples=args.n_samples,
        test_fraction=args.test_fraction,
        max_iterations=args.max_iterations,
        optimizers=["krotov_hybrid"],
        seeds=list(args.seeds),
        early_stopping_enabled=False,
        hybrid_switch_iteration=int(hp_dict["hybrid_switch_iteration"]),
        hybrid_online_step_size=float(hp_dict["hybrid_online_step_size"]),
        hybrid_batch_step_size=float(hp_dict["hybrid_batch_step_size"]),
        hybrid_online_schedule="constant",
        hybrid_batch_schedule="constant",
        hybrid_online_decay=0.05,
        hybrid_batch_decay=0.05,
        hybrid_scaling_mode="none",
        hybrid_scaling_apply_phase="both",
        hybrid_scaling_config=None,
        results_dir="results",
        plots_dir="plots",
    )


def _run_krotov(seed: int, hp_dict, args, X_train, y_train, X_test, y_test):
    model = PerezSalinasReuploadingModel(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_classes=perez_salinas_problem_num_classes(args.problem),
        use_entanglement=not args.no_entanglement,
        use_classical_head=not args.no_affine_head,
        loss_mode="weighted_fidelity",
    )
    init_params = np.asarray(model.init_params(seed=seed), dtype=float)
    config = _make_krotov_config(hp_dict, args)
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
        optimizer = qml.AdamOptimizer(stepsize=float(hp_dict["adam_lr"]))
    elif optimizer_name == "pennylane_adagrad":
        optimizer = qml.AdagradOptimizer(stepsize=float(hp_dict["adagrad_lr"]))
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
        )
    elif optimizer_name == "pennylane_spsa":
        optimizer = qml.SPSAOptimizer(
            maxiter=args.max_iterations,
            alpha=float(hp_dict["spsa_alpha"]),
            gamma=0.101,
            c=float(hp_dict["spsa_c"]),
            a=hp_dict["spsa_a"],
        )
    else:
        raise ValueError(f"Unknown PennyLane optimizer: {optimizer_name}")

    cost_fn = lambda p: model.loss(p, X_train_arr, y_train_arr)
    grad_fn = qml.grad(cost_fn)
    trace = {"step": [], "loss": [], "train_acc": [], "test_acc": [], "wall_time": []}
    start = time.time()

    for step in range(1, args.max_iterations + 1):
        if optimizer_name == "pennylane_spsa":
            params, _ = optimizer.step_and_cost(cost_fn, params)
        else:
            params, _ = optimizer.step_and_cost(cost_fn, params, grad_fn=grad_fn)

        trace["step"].append(float(step))
        trace["loss"].append(float(model.loss(params, X_train_arr, y_train_arr)))
        trace["train_acc"].append(float(model.accuracy(params, X_train_arr, y_train_arr)))
        trace["test_acc"].append(float(model.accuracy(params, X_test_arr, y_test_arr)))
        trace["wall_time"].append(float(time.time() - start))

    return {
        "optimizer": optimizer_name,
        "seed": int(seed),
        "wall_time_total": float(trace["wall_time"][-1]),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "initial_params": np.asarray(model.init_params(seed=seed), dtype=float).tolist(),
        "final_params": np.asarray(params, dtype=float).tolist(),
        "trace": trace,
    }


def _run_single(optimizer_name: str, hp_dict, seed: int, args):
    X_train, X_test, y_train, y_test = generate_perez_salinas_dataset(
        problem=args.problem,
        n_samples=args.n_samples,
        test_fraction=args.test_fraction,
        seed=seed,
    )
    _assert_no_overlap(X_train, X_test)
    if optimizer_name == "krotov_hybrid":
        result = _run_krotov(seed, hp_dict, args, X_train, y_train, X_test, y_test)
    else:
        result = _run_pennylane_optimizer(optimizer_name, seed, hp_dict, args, X_train, y_train, X_test, y_test)
    result.update(
        {
            "problem": args.problem,
            "hp": dict(hp_dict),
            "hp_key": _hp_key(hp_dict),
            "hp_label": _hp_label(hp_dict),
            "n_qubits": int(args.n_qubits),
            "n_layers": int(args.n_layers),
            "n_samples_total": int(args.n_samples),
            "test_fraction": float(args.test_fraction),
            "use_entanglement": bool(not args.no_entanglement),
            "use_classical_head": bool(not args.no_affine_head),
            "train_test_overlap": 0,
        }
    )
    return result


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

    rows.sort(key=lambda row: (-row["final_test_acc_mean"], row["final_loss_mean"], row["wall_time_mean"]))
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
        "# Perez-Salinas Optimizer Sweep",
        "",
        "Entangling 4-qubit, 8-layer, affine-head benchmark on the `crown` task with 600 total samples.",
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
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _checkpoint(results_dir: str, config: SweepConfig, all_results, completed_jobs: int, total_jobs: int) -> None:
    summaries = OrderedDict()
    for optimizer_name in config.optimizers:
        summaries[optimizer_name] = _summarize_optimizer(optimizer_name, all_results)

    _save_json(os.path.join(results_dir, "raw_results.json"), all_results)
    _save_json(
        os.path.join(results_dir, "analysis.json"),
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(config),
            "progress": {
                "completed_jobs": int(completed_jobs),
                "total_jobs": int(total_jobs),
            },
            "scanned_hyperparameters": {name: SCANNED_HYPERPARAMETERS[name] for name in config.optimizers},
            "sweep_grids": {name: {k: list(v) for k, v in SWEEP_GRIDS[name].items()} for name in config.optimizers},
            "baselines": {name: BASELINES[name] for name in config.optimizers},
            "summaries": summaries,
        },
    )
    _write_report(os.path.join(results_dir, "experiment.md"), config, summaries)


def main():
    args = _parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("--shard-index must satisfy 0 <= shard-index < num-shards")
    os.makedirs(args.results_root, exist_ok=True)
    results_dir = _timestamped_dir(args.results_root, args.run_name)
    config = SweepConfig(
        problem=args.problem,
        optimizers=list(args.optimizers),
        seeds=list(args.seeds),
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_samples=args.n_samples,
        test_fraction=args.test_fraction,
        use_entanglement=not args.no_entanglement,
        use_classical_head=not args.no_affine_head,
        max_iterations=args.max_iterations,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        results_root=args.results_root,
    )

    all_jobs = _build_jobs(args)
    shard_jobs = [job for idx, job in enumerate(all_jobs) if idx % args.num_shards == args.shard_index]

    print(f"\n{'#' * 92}")
    print(
        "# Perez-Salinas optimizer sweeps | "
        f"problem={args.problem} | samples={args.n_samples} | layers={args.n_layers} | "
        f"shard={args.shard_index + 1}/{args.num_shards} | jobs={len(shard_jobs)}"
    )
    print(f"{'#' * 92}")

    all_results = []
    total_jobs = len(shard_jobs)
    _checkpoint(results_dir, config, all_results, completed_jobs=0, total_jobs=total_jobs)

    for completed, job in enumerate(shard_jobs, start=1):
        optimizer_name = job["optimizer"]
        hp_dict = job["hp"]
        seed = job["seed"]
        print(
            f"  [{completed:3d}/{len(shard_jobs)}] {OPTIMIZER_LABELS[optimizer_name]} | "
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
        _checkpoint(results_dir, config, all_results, completed_jobs=completed, total_jobs=total_jobs)
    print(f"\nSaved Perez-Salinas sweep shard to {results_dir}/")


if __name__ == "__main__":
    main()
