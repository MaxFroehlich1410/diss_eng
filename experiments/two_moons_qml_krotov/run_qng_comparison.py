#!/usr/bin/env python3
"""Run QNG on the improved Simonetti/Chen models and compare with existing results.

Reuses the exact same model configurations, datasets and seeds as the improved
benchmark (``run_improved_simonetti_chen_benchmark.py``), adding ``qng`` as a
fourth optimizer.  Existing results for krotov_hybrid / adam / lbfgs are loaded
from the saved JSON files so only the QNG runs need to be executed.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import replace

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from config import DEFAULT_CONFIG
from dataset import generate_two_moons
from models import ChenSUNVQCModel, SimonettiHybridModel
from optimizers import run_optimizer

EXISTING_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_improved_simonetti_chen")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_qng_comparison")

ALL_OPTIMIZERS = ["krotov_hybrid", "adam", "lbfgs", "qng"]
SEEDS = [0, 1, 2]

MODEL_SPECS = OrderedDict(
    [
        (
            "simonetti_full_hybrid",
            {
                "label": "Simonetti full hybrid",
                "builder": lambda: SimonettiHybridModel(mode="hybrid"),
                "config_overrides": {
                    "input_encoding": "linear_pm_pi",
                    "n_samples": 1000,
                    "moon_noise": 0.05,
                    "test_fraction": 0.2,
                    "max_iterations": 20,
                    "lbfgs_maxiter": 20,
                    "adam_lr": 0.03,
                    "qng_lr": 0.5,
                    "qng_lam": 0.01,
                    "hybrid_switch_iteration": 10,
                    "hybrid_online_step_size": 0.02,
                    "hybrid_batch_step_size": 0.05,
                    "hybrid_online_schedule": "constant",
                    "hybrid_batch_schedule": "constant",
                    "early_stopping_enabled": False,
                },
            },
        ),
        (
            "chen_sun_vqc_improved",
            {
                "label": "Chen SUN-VQC improved",
                "builder": lambda: ChenSUNVQCModel(
                    n_macro_layers=2,
                    encoding_axes=("y", "z"),
                    readout="simple_z0",
                ),
                "config_overrides": {
                    "input_encoding": "linear_pm_pi",
                    "n_samples": 300,
                    "moon_noise": 0.07,
                    "test_fraction": 0.2,
                    "max_iterations": 20,
                    "lbfgs_maxiter": 20,
                    "adam_lr": 0.02,
                    "qng_lr": 0.5,
                    "qng_lam": 0.01,
                    "hybrid_switch_iteration": 10,
                    "hybrid_online_step_size": 0.01,
                    "hybrid_batch_step_size": 0.03,
                    "hybrid_online_schedule": "constant",
                    "hybrid_batch_schedule": "constant",
                    "early_stopping_enabled": False,
                },
            },
        ),
    ]
)


def build_config(model_name):
    config = replace(
        DEFAULT_CONFIG,
        optimizers=list(ALL_OPTIMIZERS),
        run_krotov_batch_sweep=False,
        run_krotov_hybrid_sweep=False,
    )
    return replace(config, **MODEL_SPECS[model_name]["config_overrides"])


def jsonify_trace(trace):
    out = {}
    for key, values in trace.items():
        out[key] = [str(v) if key == "phase" else float(v) for v in values]
    return out


def compute_threshold_metrics(trace, thresholds):
    metrics = {}
    losses = np.asarray(trace["loss"], dtype=float)
    steps = np.asarray(trace["step"], dtype=int)
    wall_times = np.asarray(trace["wall_time"], dtype=float)
    cost_units = np.asarray(trace["cost_units"], dtype=int)
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        hits = np.where(losses <= threshold)[0]
        if len(hits) == 0:
            metrics[key] = {"threshold": float(threshold), "reached": False,
                            "step": None, "wall_time": None, "cost_units": None}
        else:
            idx = int(hits[0])
            metrics[key] = {"threshold": float(threshold), "reached": True,
                            "step": int(steps[idx]),
                            "wall_time": float(wall_times[idx]),
                            "cost_units": int(cost_units[idx])}
    return metrics


def run_qng_single(model_name, seed, config):
    spec = MODEL_SPECS[model_name]
    model = spec["builder"]()

    X_train, X_test, y_train, y_test = generate_two_moons(
        n_samples=config.n_samples,
        noise=config.moon_noise,
        test_fraction=config.test_fraction,
        seed=seed,
        encoding=config.input_encoding,
    )
    init_params = model.init_params(seed=seed)

    print(f"\n{'=' * 76}")
    print(f"Model: {spec['label']} | Optimizer: qng | Seed: {seed}")
    print(f"  n_params={model.n_params}  n_train={len(X_train)}  "
          f"lr={config.qng_lr}  lam={config.qng_lam}")
    print(f"{'=' * 76}")

    t0 = time.time()
    final_params, trace = run_optimizer(
        "qng", model, init_params.copy(),
        X_train, y_train, X_test, y_test, config,
    )
    wall_total = time.time() - t0

    result = {
        "model_name": model_name,
        "model_label": spec["label"],
        "optimizer": "qng",
        "seed": seed,
        "n_params": int(model.n_params),
        "wall_time_total": float(wall_total),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "total_cost_units": int(trace["cost_units"][-1]),
        "total_steps": int(trace["step"][-1]),
        "threshold_metrics": compute_threshold_metrics(trace, config.loss_thresholds),
        "trace": jsonify_trace(trace),
        "final_params": np.asarray(final_params, dtype=float).tolist(),
    }

    print(
        f"  Done: loss={result['final_loss']:.4f}  "
        f"train_acc={result['final_train_acc']:.3f}  "
        f"test_acc={result['final_test_acc']:.3f}  "
        f"cost={result['total_cost_units']}  wall={wall_total:.1f}s",
        flush=True,
    )
    return result


def load_existing_results():
    results = []
    if not os.path.isdir(EXISTING_RESULTS_DIR):
        print(f"Warning: existing results directory not found: {EXISTING_RESULTS_DIR}")
        return results
    for fname in os.listdir(EXISTING_RESULTS_DIR):
        if fname.startswith("result_") and fname.endswith(".json"):
            with open(os.path.join(EXISTING_RESULTS_DIR, fname)) as f:
                results.append(json.load(f))
    return results


def aggregate(runs):
    losses = np.array([r["final_loss"] for r in runs], dtype=float)
    train_accs = np.array([r["final_train_acc"] for r in runs], dtype=float)
    test_accs = np.array([r["final_test_acc"] for r in runs], dtype=float)
    walls = np.array([r["wall_time_total"] for r in runs], dtype=float)
    costs = np.array([r["total_cost_units"] for r in runs], dtype=float)
    return {
        "n": len(runs),
        "loss_mean": float(np.mean(losses)),
        "loss_std": float(np.std(losses)),
        "train_acc_mean": float(np.mean(train_accs)),
        "train_acc_std": float(np.std(train_accs)),
        "test_acc_mean": float(np.mean(test_accs)),
        "test_acc_std": float(np.std(test_accs)),
        "wall_mean": float(np.mean(walls)),
        "cost_mean": float(np.mean(costs)),
    }


def print_comparison(all_results):
    print("\n")
    print("=" * 100)
    print("  COMPARISON: Quantum Natural Gradient  vs  Hybrid Krotov / Adam / L-BFGS")
    print("=" * 100)

    for model_name, spec in MODEL_SPECS.items():
        print(f"\n{'─' * 100}")
        print(f"  Model: {spec['label']}  ({model_name})")
        print(f"{'─' * 100}")
        header = (f"  {'Optimizer':<20} {'Loss':>12} {'Train Acc':>12} "
                  f"{'Test Acc':>12} {'Cost Units':>12} {'Wall (s)':>10}")
        print(header)
        print(f"  {'─' * 78}")

        for opt in ALL_OPTIMIZERS:
            runs = [r for r in all_results
                    if r["model_name"] == model_name and r["optimizer"] == opt]
            if not runs:
                print(f"  {opt:<20} {'(no data)':>12}")
                continue
            agg = aggregate(runs)
            print(
                f"  {opt:<20} "
                f"{agg['loss_mean']:>7.4f}±{agg['loss_std']:.4f}"
                f"{agg['train_acc_mean']:>7.3f}±{agg['train_acc_std']:.3f}"
                f"{agg['test_acc_mean']:>7.3f}±{agg['test_acc_std']:.3f}"
                f"{agg['cost_mean']:>12.0f}"
                f"{agg['wall_mean']:>10.1f}"
            )

        print()
        print(f"  Loss convergence trace (mean over seeds):")
        print(f"  {'Step':>6}", end="")
        for opt in ALL_OPTIMIZERS:
            print(f"  {opt:>16}", end="")
        print()

        max_steps = 0
        traces_by_opt = {}
        for opt in ALL_OPTIMIZERS:
            runs = [r for r in all_results
                    if r["model_name"] == model_name and r["optimizer"] == opt]
            if not runs:
                continue
            trace_losses = []
            for r in runs:
                trace_losses.append(np.array(r["trace"]["loss"], dtype=float))
            min_len = min(len(t) for t in trace_losses)
            mean_trace = np.mean([t[:min_len] for t in trace_losses], axis=0)
            traces_by_opt[opt] = mean_trace
            max_steps = max(max_steps, min_len)

        for step in range(0, max_steps, max(1, max_steps // 10)):
            print(f"  {step:>6}", end="")
            for opt in ALL_OPTIMIZERS:
                if opt in traces_by_opt and step < len(traces_by_opt[opt]):
                    print(f"  {traces_by_opt[opt][step]:>16.4f}", end="")
                else:
                    print(f"  {'—':>16}", end="")
            print()
        if max_steps > 1:
            step = max_steps - 1
            print(f"  {step:>6}", end="")
            for opt in ALL_OPTIMIZERS:
                if opt in traces_by_opt and step < len(traces_by_opt[opt]):
                    print(f"  {traces_by_opt[opt][step]:>16.4f}", end="")
                else:
                    print(f"  {'—':>16}", end="")
            print()

    print(f"\n{'=' * 100}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    existing = load_existing_results()
    print(f"Loaded {len(existing)} existing results from {EXISTING_RESULTS_DIR}")

    qng_results = []
    for model_name in MODEL_SPECS:
        config = build_config(model_name)
        for seed in SEEDS:
            result = run_qng_single(model_name, seed, config)
            qng_results.append(result)
            fname = f"result_{model_name}_qng_seed{seed}.json"
            with open(os.path.join(RESULTS_DIR, fname), "w") as f:
                json.dump(result, f, indent=2)

    all_results = existing + qng_results

    summary = OrderedDict()
    for model_name in MODEL_SPECS:
        summary[model_name] = OrderedDict()
        for opt in ALL_OPTIMIZERS:
            runs = [r for r in all_results
                    if r["model_name"] == model_name and r["optimizer"] == opt]
            if runs:
                summary[model_name][opt] = aggregate(runs)

    with open(os.path.join(RESULTS_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print_comparison(all_results)


if __name__ == "__main__":
    main()
