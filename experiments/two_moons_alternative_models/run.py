#!/usr/bin/env python3
"""Run the hybrid-vs-gradient benchmark on the alternative QML models."""

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
from qml_models.variants import (
    ChenSUNVQCModel,
    ProjectedTrainableModel,
    SimonettiHybridModel,
    SouzaSQQNNModel,
)
from optimizers.runner import run_optimizer


RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

MODEL_SPECS = OrderedDict(
    [
        (
            "simonetti_hybrid",
            {
                "label": "Simonetti hybrid",
                "builder": lambda: SimonettiHybridModel(mode="hybrid"),
                "input_encoding": "linear_pm_pi",
                "optimizer_overrides": {
                    "max_iterations": 100,
                    "lbfgs_maxiter": 100,
                    "adam_lr": 0.04,
                    "hybrid_switch_iteration": 10,
                    "hybrid_online_step_size": 0.03,
                    "hybrid_batch_step_size": 0.08,
                    "hybrid_online_schedule": "constant",
                    "hybrid_batch_schedule": "constant",
                },
                "projection": "gate_only",
                "variant_label": "paper-faithful gate-supported subset",
            },
        ),
        (
            "souza_sqqnn",
            {
                "label": "Souza reduced SQQNN",
                "builder": lambda: SouzaSQQNNModel(variant="reduced", n_neurons=4),
                "input_encoding": "linear_pm_pi",
                "optimizer_overrides": {
                    "max_iterations": 100,
                    "lbfgs_maxiter": 100,
                    "adam_lr": 0.05,
                    "hybrid_switch_iteration": 10,
                    "hybrid_online_step_size": 0.08,
                    "hybrid_batch_step_size": 0.16,
                    "hybrid_online_schedule": "constant",
                    "hybrid_batch_schedule": "constant",
                },
                "projection": "gate_only",
                "variant_label": "reduced classifier",
            },
        ),
        (
            "chen_sun_vqc",
            {
                "label": "Chen SUN-VQC",
                "builder": lambda: ChenSUNVQCModel(n_macro_layers=2, readout="simple_z0"),
                "input_encoding": "linear_pm_pi",
                "optimizer_overrides": {
                    "max_iterations": 100,
                    "lbfgs_maxiter": 100,
                    "adam_lr": 0.03,
                    "hybrid_switch_iteration": 10,
                    "hybrid_online_step_size": 0.02,
                    "hybrid_batch_step_size": 0.05,
                    "hybrid_online_schedule": "constant",
                    "hybrid_batch_schedule": "constant",
                },
                "projection": "gate_only",
                "variant_label": "2-layer simple-Z0 readout",
            },
        ),
    ]
)

OPTIMIZERS = ["krotov_hybrid", "adam", "lbfgs"]


def build_train_config(
    model_name,
    max_iterations=None,
    seeds=None,
    n_samples=None,
    noise=None,
    test_fraction=None,
):
    spec = MODEL_SPECS[model_name]
    config = replace(
        DEFAULT_CONFIG,
        input_encoding=spec["input_encoding"],
        optimizers=list(OPTIMIZERS),
        run_krotov_batch_sweep=False,
        run_krotov_hybrid_sweep=False,
        results_dir="results",
    )
    config = replace(config, **spec["optimizer_overrides"])
    if max_iterations is not None:
        config = replace(config, max_iterations=max_iterations, lbfgs_maxiter=max_iterations)
    if seeds is not None:
        config = replace(config, seeds=list(seeds))
    if n_samples is not None:
        config = replace(config, n_samples=n_samples)
    if noise is not None:
        config = replace(config, moon_noise=noise)
    if test_fraction is not None:
        config = replace(config, test_fraction=test_fraction)
    return config


def instantiate_model(model_name, seed):
    spec = MODEL_SPECS[model_name]
    base_model = spec["builder"]()
    full_init_params = np.asarray(base_model.init_params(seed=seed), dtype=float)
    projection = spec.get("projection", "full")
    if projection == "full":
        train_model = base_model
        train_init = full_init_params.copy()
    elif projection == "gate_only":
        if not hasattr(base_model, "gate_parameter_indices"):
            raise ValueError(f"Model {model_name} does not expose gate_parameter_indices().")
        trainable_indices = np.asarray(base_model.gate_parameter_indices(), dtype=int)
        train_model = ProjectedTrainableModel(
            base_model,
            full_reference_params=full_init_params,
            trainable_indices=trainable_indices,
            label=f"{model_name}:gate_only",
        )
        train_init = full_init_params[trainable_indices].copy()
    else:
        raise ValueError(f"Unknown projection mode: {projection}")

    return base_model, full_init_params, train_model, train_init


def jsonify_trace(trace):
    json_trace = {}
    for key, values in trace.items():
        if key == "phase":
            json_trace[key] = [str(v) for v in values]
        else:
            json_trace[key] = [float(v) for v in values]
    return json_trace


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


def threshold_stats(runs, threshold_key):
    times = []
    costs = []
    for run in runs:
        metric = run["threshold_metrics"][threshold_key]
        if metric["reached"]:
            times.append(metric["wall_time"])
            costs.append(metric["cost_units"])
    return {
        "success_rate": float(sum(run["threshold_metrics"][threshold_key]["reached"] for run in runs) / len(runs)),
        "time_mean": None if not times else float(np.mean(times)),
        "cost_mean": None if not costs else float(np.mean(costs)),
    }


def summarize_runs(results, thresholds):
    grouped = OrderedDict()
    for model_name in MODEL_SPECS:
        grouped[model_name] = OrderedDict()
        for optimizer_name in OPTIMIZERS:
            runs = [
                run
                for run in results
                if run["model_name"] == model_name and run["optimizer"] == optimizer_name
            ]
            grouped[model_name][optimizer_name] = runs

    summary = OrderedDict()
    for model_name, optimizer_groups in grouped.items():
        model_summary = OrderedDict()
        for optimizer_name, runs in optimizer_groups.items():
            final_losses = np.array([run["final_loss"] for run in runs], dtype=float)
            final_test_accs = np.array([run["final_test_acc"] for run in runs], dtype=float)
            wall_times = np.array([run["wall_time_total"] for run in runs], dtype=float)
            costs = np.array([run["total_cost_units"] for run in runs], dtype=float)
            model_summary[optimizer_name] = {
                "n_runs": len(runs),
                "final_loss_mean": float(np.mean(final_losses)),
                "final_loss_std": float(np.std(final_losses)),
                "final_test_acc_mean": float(np.mean(final_test_accs)),
                "final_test_acc_std": float(np.std(final_test_accs)),
                "wall_time_mean": float(np.mean(wall_times)),
                "wall_time_std": float(np.std(wall_times)),
                "cost_mean": float(np.mean(costs)),
                "cost_std": float(np.std(costs)),
                "thresholds": {
                    f"{threshold:.2f}": threshold_stats(runs, f"{threshold:.2f}")
                    for threshold in thresholds
                },
            }
        summary[model_name] = model_summary
    return summary


def run_single(model_name, optimizer_name, seed, config):
    spec = MODEL_SPECS[model_name]
    X_train, X_test, y_train, y_test = generate_two_moons(
        n_samples=config.n_samples,
        noise=config.moon_noise,
        test_fraction=config.test_fraction,
        seed=seed,
        encoding=config.input_encoding,
    )

    base_model, full_init_params, train_model, train_init = instantiate_model(model_name, seed)

    print(f"\n{'=' * 76}")
    print(
        f"Model: {spec['label']} | Optimizer: {optimizer_name} | Seed: {seed} | "
        f"Projection: {spec['projection']}"
    )
    print(f"{'=' * 76}")

    t_start = time.time()
    final_projected_params, trace = run_optimizer(
        optimizer_name,
        train_model,
        train_init.copy(),
        X_train,
        y_train,
        X_test,
        y_test,
        config,
    )
    wall_total = time.time() - t_start

    if isinstance(train_model, ProjectedTrainableModel):
        final_full_params = train_model.expand_params(final_projected_params)
    else:
        final_full_params = np.asarray(final_projected_params, dtype=float)

    result = {
        "model_name": model_name,
        "model_label": spec["label"],
        "model_variant": spec["variant_label"],
        "optimizer": optimizer_name,
        "seed": seed,
        "input_encoding": config.input_encoding,
        "projection": spec["projection"],
        "base_n_params": int(base_model.n_params),
        "optimized_n_params": int(train_model.n_params),
        "optimized_param_fraction": float(train_model.n_params / base_model.n_params),
        "wall_time_total": float(wall_total),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "total_cost_units": int(trace["cost_units"][-1]),
        "total_func_evals": int(trace["func_evals"][-1]),
        "total_grad_evals": int(trace["gradient_evaluations"][-1]),
        "total_sample_forward_passes": int(trace["sample_forward_passes"][-1]),
        "total_sample_backward_passes": int(trace["sample_backward_passes"][-1]),
        "total_full_loss_evaluations": int(trace["full_loss_evaluations"][-1]),
        "total_steps": int(trace["step"][-1]),
        "threshold_metrics": compute_threshold_metrics(trace, config.loss_thresholds),
        "trace": jsonify_trace(trace),
        "final_projected_params": np.asarray(final_projected_params, dtype=float).tolist(),
        "final_params": final_full_params.tolist(),
        "initial_params": full_init_params.tolist(),
    }

    print(
        f"  Done: loss={result['final_loss']:.4f} "
        f"train_acc={result['final_train_acc']:.3f} "
        f"test_acc={result['final_test_acc']:.3f} "
        f"cost={result['total_cost_units']} wall={wall_total:.2f}s"
    , flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", choices=list(MODEL_SPECS), default=list(MODEL_SPECS))
    parser.add_argument("--optimizers", nargs="*", choices=OPTIMIZERS, default=list(OPTIMIZERS))
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--noise", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    all_results = []
    benchmark_config = OrderedDict()

    for model_name in args.models:
        config = build_train_config(
            model_name,
            max_iterations=args.max_iterations,
            seeds=args.seeds,
            n_samples=args.n_samples,
            noise=args.noise,
            test_fraction=args.test_fraction,
        )
        benchmark_config[model_name] = asdict(config)
        for optimizer_name in args.optimizers:
            for seed in config.seeds:
                result = run_single(model_name, optimizer_name, seed, config)
                all_results.append(result)
                fname = f"result_{model_name}_{optimizer_name}_seed{seed}.json"
                with open(os.path.join(args.results_dir, fname), "w") as handle:
                    json.dump(result, handle, indent=2)

    summary = summarize_runs(all_results, DEFAULT_CONFIG.loss_thresholds)
    with open(os.path.join(args.results_dir, "config.json"), "w") as handle:
        json.dump(
            {
                "benchmark": "alternative_models_hybrid_vs_gradient",
                "models": list(args.models),
                "optimizers": list(args.optimizers),
                "benchmark_config": benchmark_config,
            },
            handle,
            indent=2,
        )
    with open(os.path.join(args.results_dir, "summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"\nAlternative-model benchmark saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
