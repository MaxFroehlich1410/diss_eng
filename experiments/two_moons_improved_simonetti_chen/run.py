#!/usr/bin/env python3
"""Run the improved Simonetti/Chen optimizer benchmark."""

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

from experiments.two_moons_common.config import DEFAULT_CONFIG
from datasets import generate_two_moons
from qml_models.variants import ChenSUNVQCModel, SimonettiHybridModel
from optimizers.runner import run_optimizer


RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
OPTIMIZERS = ["krotov_hybrid", "adam", "lbfgs"]

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


def build_train_config(model_name, seeds=None):
    config = replace(
        DEFAULT_CONFIG,
        optimizers=list(OPTIMIZERS),
        run_krotov_batch_sweep=False,
        run_krotov_hybrid_sweep=False,
        results_dir="results",
    )
    config = replace(config, **MODEL_SPECS[model_name]["config_overrides"])
    if seeds is not None:
        config = replace(config, seeds=list(seeds))
    return config


def jsonify_trace(trace):
    out = {}
    for key, values in trace.items():
        if key == "phase":
            out[key] = [str(v) for v in values]
        else:
            out[key] = [float(v) for v in values]
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
            metrics[key] = {
                "threshold": float(threshold),
                "reached": False,
                "step": None,
                "wall_time": None,
                "cost_units": None,
            }
            continue
        first_hit = int(hits[0])
        metrics[key] = {
            "threshold": float(threshold),
            "reached": True,
            "step": int(steps[first_hit]),
            "wall_time": float(wall_times[first_hit]),
            "cost_units": int(cost_units[first_hit]),
        }
    return metrics


def run_single(model_name, optimizer_name, seed, config):
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
    print(f"Model: {spec['label']} | Optimizer: {optimizer_name} | Seed: {seed}")
    print(f"{'=' * 76}")

    t0 = time.time()
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
    wall_total = time.time() - t0

    result = {
        "model_name": model_name,
        "model_label": spec["label"],
        "optimizer": optimizer_name,
        "seed": seed,
        "input_encoding": config.input_encoding,
        "n_samples": config.n_samples,
        "moon_noise": config.moon_noise,
        "test_fraction": config.test_fraction,
        "n_params": int(model.n_params),
        "wall_time_total": float(wall_total),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "total_cost_units": int(trace["cost_units"][-1]),
        "total_func_evals": int(trace["func_evals"][-1]),
        "total_grad_evals": int(trace["gradient_evaluations"][-1]),
        "total_sample_forward_passes": int(trace["sample_forward_passes"][-1]),
        "total_sample_backward_passes": int(trace["sample_backward_passes"][-1]),
        "total_steps": int(trace["step"][-1]),
        "threshold_metrics": compute_threshold_metrics(trace, config.loss_thresholds),
        "trace": jsonify_trace(trace),
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


def summarize_results(results, thresholds):
    summary = OrderedDict()
    for model_name in MODEL_SPECS:
        summary[model_name] = OrderedDict()
        for optimizer_name in OPTIMIZERS:
            runs = [
                result
                for result in results
                if result["model_name"] == model_name and result["optimizer"] == optimizer_name
            ]
            losses = np.asarray([run["final_loss"] for run in runs], dtype=float)
            train_accs = np.asarray([run["final_train_acc"] for run in runs], dtype=float)
            test_accs = np.asarray([run["final_test_acc"] for run in runs], dtype=float)
            wall_times = np.asarray([run["wall_time_total"] for run in runs], dtype=float)
            costs = np.asarray([run["total_cost_units"] for run in runs], dtype=float)
            summary[model_name][optimizer_name] = {
                "n_runs": len(runs),
                "final_loss_mean": float(np.mean(losses)),
                "final_loss_std": float(np.std(losses)),
                "final_train_acc_mean": float(np.mean(train_accs)),
                "final_train_acc_std": float(np.std(train_accs)),
                "final_test_acc_mean": float(np.mean(test_accs)),
                "final_test_acc_std": float(np.std(test_accs)),
                "wall_time_mean": float(np.mean(wall_times)),
                "wall_time_std": float(np.std(wall_times)),
                "cost_mean": float(np.mean(costs)),
                "cost_std": float(np.std(costs)),
                "thresholds": {
                    f"{threshold:.2f}": {
                        "success_rate": float(
                            sum(
                                run["threshold_metrics"][f"{threshold:.2f}"]["reached"]
                                for run in runs
                            )
                            / len(runs)
                        ),
                        "time_mean": (
                            None
                            if not any(run["threshold_metrics"][f"{threshold:.2f}"]["reached"] for run in runs)
                            else float(
                                np.mean(
                                    [
                                        run["threshold_metrics"][f"{threshold:.2f}"]["wall_time"]
                                        for run in runs
                                        if run["threshold_metrics"][f"{threshold:.2f}"]["reached"]
                                    ]
                                )
                            )
                        ),
                    }
                    for threshold in thresholds
                },
            }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    all_results = []
    all_configs = OrderedDict()
    for model_name in MODEL_SPECS:
        config = build_train_config(model_name, seeds=args.seeds)
        all_configs[model_name] = asdict(config)
        for optimizer_name in OPTIMIZERS:
            for seed in config.seeds:
                result = run_single(model_name, optimizer_name, seed, config)
                all_results.append(result)
                out_path = os.path.join(
                    args.results_dir,
                    f"result_{model_name}_{optimizer_name}_seed{seed}.json",
                )
                with open(out_path, "w") as handle:
                    json.dump(result, handle, indent=2)

    summary = summarize_results(all_results, DEFAULT_CONFIG.loss_thresholds)
    with open(os.path.join(args.results_dir, "config.json"), "w") as handle:
        json.dump({"benchmark_config": all_configs, "optimizers": OPTIMIZERS}, handle, indent=2)
    with open(os.path.join(args.results_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nImproved Simonetti/Chen benchmark saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
