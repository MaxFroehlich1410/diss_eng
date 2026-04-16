#!/usr/bin/env python3
"""Run optimizer benchmarks on the Perez-Salinas re-uploading classifier."""

from __future__ import annotations

import argparse
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
        perez_salinas_4q8l_preset,
        perez_salinas_problem_num_classes,
    )
else:
    from experiments.two_moons_common.config import DEFAULT_CONFIG
    from qml_models.variants import PerezSalinasReuploadingModel
    from optimizers.runner import run_optimizer
    from datasets import (
        available_perez_salinas_problems,
        generate_perez_salinas_dataset,
        perez_salinas_4q8l_preset,
        perez_salinas_problem_num_classes,
    )


RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
OPTIMIZERS = ["krotov_hybrid", "adam", "qng", "lbfgs"]


def build_benchmark_config(max_iterations=None, optimizers=None, seeds=None):
    config = replace(
        DEFAULT_CONFIG,
        optimizers=list(OPTIMIZERS if optimizers is None else optimizers),
        run_krotov_batch_sweep=False,
        run_krotov_hybrid_sweep=False,
        results_dir="results",
    )
    if max_iterations is not None:
        config = replace(config, max_iterations=max_iterations, lbfgs_maxiter=max_iterations)
    if seeds is not None:
        config = replace(config, seeds=list(seeds))
    return config


def jsonify_trace(trace):
    json_trace = {}
    for key, values in trace.items():
        if key == "phase":
            json_trace[key] = [str(v) for v in values]
        else:
            json_trace[key] = [float(v) for v in values]
    return json_trace


def run_single(problem, n_qubits, n_layers, use_entanglement, n_samples, test_fraction, optimizer_name, seed, config):
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
        loss_mode="weighted_fidelity",
    )
    init_params = np.asarray(model.init_params(seed=seed), dtype=float)

    print(f"\n{'=' * 76}")
    print(
        f"Problem: {problem} | Model: Perez-Salinas | "
        f"Qubits: {n_qubits} | Layers: {n_layers} | "
        f"Entanglement: {use_entanglement} | Optimizer: {optimizer_name} | Seed: {seed}"
    )
    print(f"{'=' * 76}")

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
        "problem": problem,
        "model_name": "perez_salinas_reuploading",
        "loss_mode": "weighted_fidelity",
        "optimizer": optimizer_name,
        "seed": seed,
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "use_entanglement": bool(use_entanglement),
        "n_classes": int(model.n_classes),
        "n_samples_total": int(n_samples),
        "test_fraction": float(test_fraction),
        "n_params": int(model.n_params),
        "n_quantum_params": int(model.n_quantum_params),
        "n_classical_params": int(model.n_weight_params),
        "wall_time_total": float(wall_total),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "total_cost_units": int(trace["cost_units"][-1]),
        "total_steps": int(trace["step"][-1]),
        "trace": jsonify_trace(trace),
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


def summarise_results(results):
    summary = OrderedDict()
    for optimizer_name in OPTIMIZERS:
        runs = [run for run in results if run["optimizer"] == optimizer_name]
        if not runs:
            continue
        final_losses = np.array([run["final_loss"] for run in runs], dtype=float)
        final_test_accs = np.array([run["final_test_acc"] for run in runs], dtype=float)
        wall_times = np.array([run["wall_time_total"] for run in runs], dtype=float)
        costs = np.array([run["total_cost_units"] for run in runs], dtype=float)
        summary[optimizer_name] = {
            "n_runs": len(runs),
            "final_loss_mean": float(np.mean(final_losses)),
            "final_loss_std": float(np.std(final_losses)),
            "final_test_acc_mean": float(np.mean(final_test_accs)),
            "final_test_acc_std": float(np.std(final_test_accs)),
            "wall_time_mean": float(np.mean(wall_times)),
            "wall_time_std": float(np.std(wall_times)),
            "cost_mean": float(np.mean(costs)),
            "cost_std": float(np.std(costs)),
        }
    return summary


def parse_args():
    preset = perez_salinas_4q8l_preset()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", choices=available_perez_salinas_problems(), default=preset["problem"])
    parser.add_argument("--n-qubits", type=int, default=preset["n_qubits"])
    parser.add_argument("--n-layers", type=int, default=preset["n_layers"])
    parser.add_argument("--no-entanglement", action="store_true")
    parser.add_argument("--n-samples", type=int, default=600)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--optimizers", nargs="*", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(3)))
    parser.add_argument("--max-iterations", type=int, default=60)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    config = build_benchmark_config(
        max_iterations=args.max_iterations,
        optimizers=args.optimizers,
        seeds=args.seeds,
    )

    all_results = []
    for optimizer_name in args.optimizers:
        for seed in args.seeds:
            result = run_single(
                problem=args.problem,
                n_qubits=args.n_qubits,
                n_layers=args.n_layers,
                use_entanglement=not args.no_entanglement,
                n_samples=args.n_samples,
                test_fraction=args.test_fraction,
                optimizer_name=optimizer_name,
                seed=seed,
                config=config,
            )
            all_results.append(result)
            file_name = f"result_{args.problem}_{optimizer_name}_seed{seed}.json"
            with open(os.path.join(args.results_dir, file_name), "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)

    summary = summarise_results(all_results)
    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(os.path.join(args.results_dir, "config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "benchmark": "perez_salinas_reuploading",
                "preset": perez_salinas_4q8l_preset(args.problem),
                "config": asdict(config),
                "args": vars(args),
            },
            handle,
            indent=2,
        )

    print(f"\nPerez-Salinas benchmark saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
