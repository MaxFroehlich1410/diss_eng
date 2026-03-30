#!/usr/bin/env python3
"""Run the two-moons QML optimizer benchmark.

Usage:
    python run_experiment.py
"""

import json
import os
import sys
import time
import numpy as np

# Ensure this directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_CONFIG, ExperimentConfig
from dataset import generate_two_moons
from model import VQCModel
from optimizers import run_optimizer


def run_single(config, optimizer_name, seed):
    """Run a single optimizer on a single seed. Returns result dict."""
    print(f"\n{'='*60}")
    print(f"Optimizer: {optimizer_name} | Seed: {seed}")
    print(f"{'='*60}")

    # Dataset
    X_train, X_test, y_train, y_test = generate_two_moons(
        n_samples=config.n_samples,
        noise=config.moon_noise,
        test_fraction=config.test_fraction,
        seed=seed,
    )

    # Model
    model = VQCModel(
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        entangler=config.entangler,
    )

    # Initialize parameters (same init for all optimizers given same seed)
    init_params = model.init_params(seed=seed)

    t_start = time.time()
    final_params, trace = run_optimizer(
        optimizer_name, model, init_params.copy(),
        X_train, y_train, X_test, y_test, config,
    )
    wall_total = time.time() - t_start

    result = {
        "optimizer": optimizer_name,
        "seed": seed,
        "n_layers": config.n_layers,
        "n_params": model.n_params,
        "wall_time_total": wall_total,
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "total_func_evals": int(trace["func_evals"][-1]),
        "total_grad_evals": int(trace["grad_evals"][-1]),
        "total_steps": int(trace["step"][-1]),
        "trace": {k: [float(v) for v in vals] for k, vals in trace.items()},
        "final_params": final_params.tolist(),
    }

    print(f"  Done: loss={result['final_loss']:.4f} "
          f"train_acc={result['final_train_acc']:.3f} "
          f"test_acc={result['final_test_acc']:.3f} "
          f"wall={wall_total:.1f}s")

    return result


def main():
    config = DEFAULT_CONFIG

    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), config.results_dir
    )
    os.makedirs(results_dir, exist_ok=True)

    # Save config
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if not k.startswith("_")
    }
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    all_results = []

    for optimizer_name in config.optimizers:
        for seed in config.seeds:
            result = run_single(config, optimizer_name, seed)
            all_results.append(result)

            # Save incrementally
            fname = f"result_{optimizer_name}_seed{seed}.json"
            with open(os.path.join(results_dir, fname), "w") as f:
                json.dump(result, f, indent=2)

    # Save summary
    summary = []
    for r in all_results:
        summary.append({
            "optimizer": r["optimizer"],
            "seed": r["seed"],
            "final_loss": r["final_loss"],
            "final_train_acc": r["final_train_acc"],
            "final_test_acc": r["final_test_acc"],
            "total_func_evals": r["total_func_evals"],
            "wall_time_total": r["wall_time_total"],
        })

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {results_dir}/")
    print("\nSummary:")
    print(f"{'Optimizer':<12} {'Seed':>4} {'Loss':>8} {'TrainAcc':>8} "
          f"{'TestAcc':>8} {'Evals':>8} {'Time':>8}")
    print("-" * 64)
    for r in summary:
        print(f"{r['optimizer']:<12} {r['seed']:>4} {r['final_loss']:>8.4f} "
              f"{r['final_train_acc']:>8.3f} {r['final_test_acc']:>8.3f} "
              f"{r['total_func_evals']:>8} {r['wall_time_total']:>8.1f}")


if __name__ == "__main__":
    main()
