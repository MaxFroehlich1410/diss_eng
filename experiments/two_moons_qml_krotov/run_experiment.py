#!/usr/bin/env python3
"""Run the two-moons QML optimizer benchmark."""

import json
import os
import sys
import time
from dataclasses import asdict, replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_CONFIG
from dataset import generate_two_moons
from model import VQCModel
from optimizers import run_optimizer


def build_experiment_specs(config):
    """Build the list of benchmark runs for the configured phases."""
    specs = []
    for optimizer_name in config.optimizers:
        specs.append(
            {
                "run_name": optimizer_name,
                "optimizer_name": optimizer_name,
                "optimizer_family": optimizer_name,
                "config_overrides": {},
                "is_sweep": False,
                "sweep_group": None,
            }
        )

    if config.run_krotov_batch_sweep:
        seen = {
            (
                config.krotov_batch_step_size,
                config.krotov_batch_schedule,
                config.krotov_batch_decay,
            )
        }
        for step_size in config.krotov_batch_sweep_step_sizes:
            for schedule in config.krotov_batch_sweep_schedules:
                key = (step_size, schedule, config.krotov_batch_decay)
                if key in seen:
                    continue
                seen.add(key)
                specs.append(
                    {
                        "run_name": f"krotov_batch_lr{step_size:.3f}_{schedule}",
                        "optimizer_name": "krotov_batch",
                        "optimizer_family": "krotov_batch",
                        "config_overrides": {
                            "krotov_batch_step_size": step_size,
                            "krotov_batch_schedule": schedule,
                        },
                        "is_sweep": True,
                        "sweep_group": "krotov_batch",
                    }
                )

    if config.run_krotov_hybrid_sweep:
        seen = {config.hybrid_switch_iteration}
        for switch_iteration in config.hybrid_switch_iterations:
            if switch_iteration in seen:
                continue
            seen.add(switch_iteration)
            specs.append(
                {
                    "run_name": f"krotov_hybrid_sw{switch_iteration:03d}",
                    "optimizer_name": "krotov_hybrid",
                    "optimizer_family": "krotov_hybrid",
                    "config_overrides": {
                        "hybrid_switch_iteration": switch_iteration,
                    },
                    "is_sweep": True,
                    "sweep_group": "krotov_hybrid",
                }
            )

    return specs


def _jsonify_trace(trace):
    json_trace = {}
    for key, values in trace.items():
        if key == "phase":
            json_trace[key] = [str(v) for v in values]
        else:
            json_trace[key] = [float(v) for v in values]
    return json_trace


def _compute_threshold_metrics(trace, thresholds):
    """Return first-hit metrics for each configured loss threshold."""
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


def run_single(config, spec, seed):
    """Run one optimizer variant on one seed and return a JSON-serializable dict."""
    run_name = spec["run_name"]
    optimizer_name = spec["optimizer_name"]
    run_config = replace(config, **spec["config_overrides"])

    print(f"\n{'=' * 72}")
    print(f"Optimizer: {run_name} | Base: {optimizer_name} | Seed: {seed}")
    print(f"{'=' * 72}")

    X_train, X_test, y_train, y_test = generate_two_moons(
        n_samples=run_config.n_samples,
        noise=run_config.moon_noise,
        test_fraction=run_config.test_fraction,
        seed=seed,
        encoding=getattr(run_config, "input_encoding", "tanh_0_pi"),
    )

    model = VQCModel(
        n_qubits=run_config.n_qubits,
        n_layers=run_config.n_layers,
        entangler=run_config.entangler,
        architecture=getattr(run_config, "model_architecture", "hea"),
        observable=getattr(run_config, "observable", "Z0Z1"),
    )
    init_params = model.init_params(seed=seed)

    t_start = time.time()
    final_params, trace = run_optimizer(
        optimizer_name,
        model,
        init_params.copy(),
        X_train,
        y_train,
        X_test,
        y_test,
        run_config,
    )
    wall_total = time.time() - t_start

    result = {
        "optimizer": run_name,
        "optimizer_name": optimizer_name,
        "optimizer_family": spec["optimizer_family"],
        "is_sweep": spec["is_sweep"],
        "sweep_group": spec.get("sweep_group"),
        "config_overrides": spec["config_overrides"],
        "seed": seed,
        "n_layers": run_config.n_layers,
        "model_architecture": getattr(run_config, "model_architecture", "hea"),
        "input_encoding": getattr(run_config, "input_encoding", "tanh_0_pi"),
        "n_params": model.n_params,
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
        "threshold_metrics": _compute_threshold_metrics(
            trace,
            getattr(run_config, "loss_thresholds", [run_config.loss_threshold]),
        ),
        "trace": _jsonify_trace(trace),
        "final_params": final_params.tolist(),
    }

    print(
        f"  Done: loss={result['final_loss']:.4f} "
        f"train_acc={result['final_train_acc']:.3f} "
        f"test_acc={result['final_test_acc']:.3f} "
        f"cost={result['total_cost_units']} wall={wall_total:.1f}s"
    )

    return result


def main():
    config = DEFAULT_CONFIG
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, config.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    config_dict = asdict(config)
    config_dict["experiment_specs"] = build_experiment_specs(config)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    all_results = []
    for spec in config_dict["experiment_specs"]:
        for seed in config.seeds:
            result = run_single(config, spec, seed)
            all_results.append(result)

            fname = f"result_{result['optimizer']}_seed{seed}.json"
            with open(os.path.join(results_dir, fname), "w") as f:
                json.dump(result, f, indent=2)

    summary = []
    for r in all_results:
        summary.append(
            {
                "optimizer": r["optimizer"],
                "optimizer_name": r["optimizer_name"],
                "optimizer_family": r["optimizer_family"],
                "is_sweep": r["is_sweep"],
                "sweep_group": r["sweep_group"],
                "seed": r["seed"],
                "final_loss": r["final_loss"],
                "final_train_acc": r["final_train_acc"],
                "final_test_acc": r["final_test_acc"],
                "total_cost_units": r["total_cost_units"],
                "total_sample_forward_passes": r["total_sample_forward_passes"],
                "total_sample_backward_passes": r["total_sample_backward_passes"],
                "total_full_loss_evaluations": r["total_full_loss_evaluations"],
                "total_grad_evals": r["total_grad_evals"],
                "wall_time_total": r["wall_time_total"],
                "threshold_metrics": r["threshold_metrics"],
            }
        )

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {results_dir}/")
    print("\nSummary:")
    print(
        f"{'Optimizer':<30} {'Seed':>4} {'Loss':>8} {'TrainAcc':>8} "
        f"{'TestAcc':>8} {'Cost':>10} {'Time':>8}"
    )
    print("-" * 86)
    for r in summary:
        print(
            f"{r['optimizer']:<30} {r['seed']:>4} {r['final_loss']:>8.4f} "
            f"{r['final_train_acc']:>8.3f} {r['final_test_acc']:>8.3f} "
            f"{r['total_cost_units']:>10} {r['wall_time_total']:>8.1f}"
        )


if __name__ == "__main__":
    main()
