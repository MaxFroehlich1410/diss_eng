#!/usr/bin/env python3
"""Small optimizer sweeps for the Perez-Salinas re-uploading classifier.

The default target is the paper-style binary concentric-circles task
(``crown`` / binary annulus) using the 4-qubit entangled 8-layer data
re-uploading circuit.

Run one optimizer at a time so the jobs can be launched in parallel:

    python -m experiments.two_moons_perez_salinas_sweep.run --optimizer krotov_hybrid
    python -m experiments.two_moons_perez_salinas_sweep.run --optimizer adam
    python -m experiments.two_moons_perez_salinas_sweep.run --optimizer qng
    python -m experiments.two_moons_perez_salinas_sweep.run --optimizer lbfgs

To rerun only the hybrid Krotov sweep without the classical affine readout
head, add ``--no-affine-head``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import asdict, replace

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

if __package__ in {None, ""}:
    from experiments.two_moons_common.config import DEFAULT_CONFIG
    from qml_models.variants import PerezSalinasReuploadingModel
    from optimizers.runner import run_optimizer
    from perez_salinas_dataset import (
        available_perez_salinas_problems,
        generate_perez_salinas_dataset,
        perez_salinas_benchmark_preset,
        perez_salinas_problem_num_classes,
    )
else:
    from experiments.two_moons_common.config import DEFAULT_CONFIG
    from qml_models.variants import PerezSalinasReuploadingModel
    from optimizers.runner import run_optimizer
    from datasets import (
        available_perez_salinas_problems,
        generate_perez_salinas_dataset,
        perez_salinas_benchmark_preset,
        perez_salinas_problem_num_classes,
    )


sys.stdout.reconfigure(line_buffering=True)

RESULTS_BASE = os.path.join(SCRIPT_DIR, "results_perez_salinas_sweeps")
OPTIMIZERS = ["krotov_hybrid", "adam", "qng", "lbfgs"]

SWEEP_GRIDS = {
    "krotov_hybrid": OrderedDict(
        [
            ("n_layers", [8]),
            ("hybrid_switch_iteration", [5, 10]),
            ("hybrid_online_step_size", [0.1, 0.3]),
            ("hybrid_batch_step_size", [0.5, 1.0]),
        ]
    ),
    "adam": OrderedDict(
        [
            ("n_layers", [8]),
            ("adam_lr", [0.01, 0.03, 0.05, 0.1]),
        ]
    ),
    "qng": OrderedDict(
        [
            ("n_layers", [8]),
            ("qng_lr", [0.1, 0.5]),
            ("qng_lam", [0.001, 0.01]),
        ]
    ),
    "lbfgs": OrderedDict(
        [
            ("n_layers", [8]),
            ("lbfgs_maxcor", [10, 20]),
            ("lbfgs_gtol", [1e-6, 1e-7]),
        ]
    ),
}

BASELINES = {
    "krotov_hybrid": {
        "n_layers": 8,
        "hybrid_switch_iteration": 10,
        "hybrid_online_step_size": 0.3,
        "hybrid_batch_step_size": 1.0,
    },
    "adam": {
        "n_layers": 8,
        "adam_lr": 0.05,
    },
    "qng": {
        "n_layers": 8,
        "qng_lr": 0.5,
        "qng_lam": 0.01,
    },
    "lbfgs": {
        "n_layers": 8,
        "lbfgs_maxcor": 20,
        "lbfgs_gtol": 1e-7,
    },
}


def build_train_config(optimizer_name, hp_dict, max_iterations):
    return replace(
        DEFAULT_CONFIG,
        optimizers=[optimizer_name],
        run_krotov_batch_sweep=False,
        run_krotov_hybrid_sweep=False,
        max_iterations=max_iterations,
        lbfgs_maxiter=max_iterations,
        early_stopping_enabled=False,
        hybrid_online_schedule="constant",
        hybrid_batch_schedule="constant",
        results_dir="results",
        **{key: value for key, value in hp_dict.items() if key != "n_layers"},
    )


def expand_grid(grid_dict):
    names = list(grid_dict.keys())
    value_lists = [grid_dict[name] for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def hp_label(hp_dict):
    parts = []
    for key, value in hp_dict.items():
        short = key.split("_", 1)[-1]
        parts.append(f"{short}={value}")
    return " ".join(parts)


def jsonify_trace(trace):
    json_trace = {}
    for key, values in trace.items():
        if key == "phase":
            json_trace[key] = [str(v) for v in values]
        else:
            json_trace[key] = [float(v) for v in values]
    return json_trace


def run_single(
    problem,
    n_qubits,
    use_entanglement,
    use_classical_head,
    n_samples,
    test_fraction,
    optimizer_name,
    hp_dict,
    seed,
    max_iterations,
):
    n_layers = int(hp_dict["n_layers"])
    X_train, X_test, y_train, y_test = generate_perez_salinas_dataset(
        problem=problem,
        n_samples=n_samples,
        test_fraction=test_fraction,
        seed=seed,
    )
    model = PerezSalinasReuploadingModel(
        n_qubits=n_qubits,
        n_layers=n_layers,
        n_classes=perez_salinas_problem_num_classes(problem),
        use_entanglement=use_entanglement,
        use_classical_head=use_classical_head,
        loss_mode="weighted_fidelity",
    )
    init_params = np.asarray(model.init_params(seed=seed), dtype=float)
    config = build_train_config(optimizer_name, hp_dict, max_iterations=max_iterations)

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

    return {
        "problem": problem,
        "optimizer": optimizer_name,
        "seed": seed,
        "n_qubits": int(n_qubits),
        "use_entanglement": bool(use_entanglement),
        "use_classical_head": bool(use_classical_head),
        "n_samples_total": int(n_samples),
        "test_fraction": float(test_fraction),
        "n_params": int(model.n_params),
        "n_quantum_params": int(model.n_quantum_params),
        "n_classical_params": int(model.n_weight_params),
        "max_iterations": int(max_iterations),
        "wall_time_total": float(wall_total),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "total_cost_units": int(trace["cost_units"][-1]),
        "total_steps": int(trace["step"][-1]),
        "trace": jsonify_trace(trace),
        "initial_params": init_params.tolist(),
        "final_params": np.asarray(final_params, dtype=float).tolist(),
        **hp_dict,
    }


def _close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < max(1e-12, 1e-6 * max(abs(float(a)), abs(float(b)), 1.0))
    return a == b


def _matches_config(row, reference, keys):
    return all(_close(row[key], reference[key]) for key in keys)


def analyse_results(optimizer_name, results, param_names):
    configs = OrderedDict()
    for row in results:
        key = tuple(row[param_name] for param_name in param_names)
        configs.setdefault(key, []).append(row)

    rows = []
    for key, runs in configs.items():
        hp = dict(zip(param_names, key))
        final_losses = np.array([run["final_loss"] for run in runs], dtype=float)
        final_test_accs = np.array([run["final_test_acc"] for run in runs], dtype=float)
        final_train_accs = np.array([run["final_train_acc"] for run in runs], dtype=float)
        wall_times = np.array([run["wall_time_total"] for run in runs], dtype=float)
        costs = np.array([run["total_cost_units"] for run in runs], dtype=float)
        rows.append(
            {
                **hp,
                "n_runs": len(runs),
                "final_loss_mean": float(np.mean(final_losses)),
                "final_loss_std": float(np.std(final_losses)),
                "final_test_acc_mean": float(np.mean(final_test_accs)),
                "final_test_acc_std": float(np.std(final_test_accs)),
                "final_train_acc_mean": float(np.mean(final_train_accs)),
                "wall_time_mean": float(np.mean(wall_times)),
                "wall_time_std": float(np.std(wall_times)),
                "cost_mean": float(np.mean(costs)),
                "cost_std": float(np.std(costs)),
            }
        )

    rows.sort(key=lambda row: (row["final_loss_mean"], -row["final_test_acc_mean"]))
    baseline = None
    for row in rows:
        if _matches_config(row, BASELINES[optimizer_name], param_names):
            baseline = row
            break

    return {
        "optimizer": optimizer_name,
        "param_names": param_names,
        "n_configs": len(rows),
        "best": rows[0],
        "top10": rows[:10],
        "baseline": baseline,
        "all_configs": rows,
    }


def write_report(problem, optimizer_name, summary, results_dir, use_classical_head):
    preset_8 = perez_salinas_benchmark_preset(problem=problem, n_qubits=4, n_layers=8)
    best = summary["best"]
    baseline = summary["baseline"]
    param_names = summary["param_names"]

    lines = [
        f"# Perez-Salinas Sweep Report: {optimizer_name}",
        "",
        f"Problem: `{problem}` (binary concentric circles / annulus benchmark).",
        "",
        "## Architecture",
        "",
        f"- 8-layer preset: `{preset_8}`",
        f"- Classical affine head: `{'enabled' if use_classical_head else 'disabled'}`",
        "",
        "## Best Configuration",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for param_name in param_names:
        lines.append(f"| {param_name} | {best[param_name]} |")
    lines.extend(
        [
            f"| mean final loss | {best['final_loss_mean']:.4f} ± {best['final_loss_std']:.4f} |",
            f"| mean final test acc | {best['final_test_acc_mean']:.4f} ± {best['final_test_acc_std']:.4f} |",
            f"| mean wall time | {best['wall_time_mean']:.2f}s |",
            f"| mean cost units | {best['cost_mean']:.1f} ± {best['cost_std']:.1f} |",
            "",
        ]
    )

    if baseline is not None:
        lines.extend(
            [
                "## Baseline Comparison",
                "",
                "| Metric | Baseline | Best | Delta |",
                "|---|---|---|---|",
                f"| final loss | {baseline['final_loss_mean']:.4f} | {best['final_loss_mean']:.4f} | {best['final_loss_mean'] - baseline['final_loss_mean']:+.4f} |",
                f"| final test acc | {baseline['final_test_acc_mean']:.4f} | {best['final_test_acc_mean']:.4f} | {best['final_test_acc_mean'] - baseline['final_test_acc_mean']:+.4f} |",
                f"| wall time | {baseline['wall_time_mean']:.2f}s | {best['wall_time_mean']:.2f}s | {best['wall_time_mean'] - baseline['wall_time_mean']:+.2f}s |",
                "",
            ]
        )

    lines.extend(
        [
            "## Top Configurations",
            "",
            "| Rank | " + " | ".join(param_names) + " | loss (mean±std) | test acc | wall (s) |",
            "|---|---" + "|---" * len(param_names) + "|---|---|",
        ]
    )
    for rank, row in enumerate(summary["top10"], start=1):
        vals = " | ".join(str(row[param_name]) for param_name in param_names)
        lines.append(
            f"| {rank} | {vals} | {row['final_loss_mean']:.4f}±{row['final_loss_std']:.4f} | "
            f"{row['final_test_acc_mean']:.4f} | {row['wall_time_mean']:.2f} |"
        )

    report_path = os.path.join(results_dir, f"sweep_{optimizer_name}_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", required=True, choices=OPTIMIZERS)
    parser.add_argument("--problem", choices=available_perez_salinas_problems(), default="crown")
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--no-entanglement", action="store_true")
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1])
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument("--no-affine-head", action="store_true")
    parser.add_argument("--results-dir", default=RESULTS_BASE)
    return parser.parse_args()


def main():
    args = parse_args()
    grid = SWEEP_GRIDS[args.optimizer]
    param_names = list(grid.keys())
    combos = expand_grid(grid)
    total_runs = len(combos) * len(args.seeds)

    results_dir = os.path.join(args.results_dir, args.optimizer)
    os.makedirs(results_dir, exist_ok=True)

    print(f"\n{'#' * 78}")
    print(
        f"# Perez-Salinas small sweep | optimizer={args.optimizer} | "
        f"problem={args.problem} | head={'off' if args.no_affine_head else 'on'} | "
        f"configs={len(combos)} | seeds={len(args.seeds)} | total runs={total_runs}"
    )
    print(f"{'#' * 78}")

    all_results = []
    for combo_idx, hp_dict in enumerate(combos):
        for seed_idx, seed in enumerate(args.seeds):
            run_idx = combo_idx * len(args.seeds) + seed_idx + 1
            print(f"  [{run_idx:3d}/{total_runs}] {hp_label(hp_dict)} seed={seed}", end="", flush=True)
            result = run_single(
                problem=args.problem,
                n_qubits=args.n_qubits,
                use_entanglement=not args.no_entanglement,
                use_classical_head=not args.no_affine_head,
                n_samples=args.n_samples,
                test_fraction=args.test_fraction,
                optimizer_name=args.optimizer,
                hp_dict=hp_dict,
                seed=seed,
                max_iterations=args.max_iterations,
            )
            all_results.append(result)
            print(
                f"  loss={result['final_loss']:.4f} "
                f"test_acc={result['final_test_acc']:.3f} "
                f"wall={result['wall_time_total']:.1f}s"
            )

    raw_path = os.path.join(results_dir, f"sweep_{args.optimizer}.json")
    with open(raw_path, "w", encoding="utf-8") as handle:
        json.dump(all_results, handle, indent=2)

    summary = analyse_results(args.optimizer, all_results, param_names)
    summary_payload = {
        **summary,
        "args": vars(args),
        "grid": {key: list(values) for key, values in grid.items()},
        "benchmark_preset_8": perez_salinas_benchmark_preset(
            problem=args.problem,
            n_qubits=args.n_qubits,
            n_layers=8,
            use_entanglement=not args.no_entanglement,
        ),
        "use_classical_head": not args.no_affine_head,
        "base_config": asdict(
            build_train_config(args.optimizer, BASELINES[args.optimizer], args.max_iterations)
        ),
    }
    summary_path = os.path.join(results_dir, f"analysis_{args.optimizer}.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)

    report_path = write_report(
        args.problem,
        args.optimizer,
        summary,
        results_dir,
        use_classical_head=not args.no_affine_head,
    )

    print(f"\nBest config for {args.optimizer}:")
    for key in param_names:
        print(f"  {key} = {summary['best'][key]}")
    print(
        f"  final_loss = {summary['best']['final_loss_mean']:.4f} ± {summary['best']['final_loss_std']:.4f}\n"
        f"  final_test_acc = {summary['best']['final_test_acc_mean']:.4f} ± {summary['best']['final_test_acc_std']:.4f}"
    )
    print(f"\nSaved raw results to {raw_path}")
    print(f"Saved analysis to {summary_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
