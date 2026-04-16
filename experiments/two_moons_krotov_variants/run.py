#!/usr/bin/env python3
"""Run a Krotov-focused benchmark into a clean results directory."""

import json
import os
from dataclasses import asdict, replace

from experiments.two_moons_common.config import DEFAULT_CONFIG
from run_experiment import run_single


def build_krotov_specs():
    """Return the Krotov variants used in the focused analysis."""
    return [
        {
            "run_name": "krotov_online",
            "optimizer_name": "krotov_online",
            "optimizer_family": "krotov_online",
            "config_overrides": {
                "krotov_online_step_size": 0.3,
                "krotov_online_schedule": "constant",
            },
            "is_sweep": False,
        },
        {
            "run_name": "krotov_batch",
            "optimizer_name": "krotov_batch",
            "optimizer_family": "krotov_batch",
            "config_overrides": {
                "krotov_batch_step_size": 0.3,
                "krotov_batch_schedule": "constant",
            },
            "is_sweep": False,
        },
        {
            "run_name": "krotov_batch_lr1.000_constant",
            "optimizer_name": "krotov_batch",
            "optimizer_family": "krotov_batch",
            "config_overrides": {
                "krotov_batch_step_size": 1.0,
                "krotov_batch_schedule": "constant",
            },
            "is_sweep": True,
        },
        {
            "run_name": "krotov_batch_lr0.100_constant",
            "optimizer_name": "krotov_batch",
            "optimizer_family": "krotov_batch",
            "config_overrides": {
                "krotov_batch_step_size": 0.1,
                "krotov_batch_schedule": "constant",
            },
            "is_sweep": True,
        },
        {
            "run_name": "krotov_batch_lr0.050_constant",
            "optimizer_name": "krotov_batch",
            "optimizer_family": "krotov_batch",
            "config_overrides": {
                "krotov_batch_step_size": 0.05,
                "krotov_batch_schedule": "constant",
            },
            "is_sweep": True,
        },
        {
            "run_name": "krotov_batch_lr0.020_constant",
            "optimizer_name": "krotov_batch",
            "optimizer_family": "krotov_batch",
            "config_overrides": {
                "krotov_batch_step_size": 0.02,
                "krotov_batch_schedule": "constant",
            },
            "is_sweep": True,
        },
        {
            "run_name": "krotov_batch_lr0.300_inverse",
            "optimizer_name": "krotov_batch",
            "optimizer_family": "krotov_batch",
            "config_overrides": {
                "krotov_batch_step_size": 0.3,
                "krotov_batch_schedule": "inverse",
            },
            "is_sweep": True,
        },
        {
            "run_name": "krotov_batch_lr1.000_inverse",
            "optimizer_name": "krotov_batch",
            "optimizer_family": "krotov_batch",
            "config_overrides": {
                "krotov_batch_step_size": 1.0,
                "krotov_batch_schedule": "inverse",
            },
            "is_sweep": True,
        },
    ]


def main():
    config = replace(
        DEFAULT_CONFIG,
        seeds=list(range(5)),
        optimizers=["krotov_online", "krotov_batch"],
        run_krotov_batch_sweep=False,
        results_dir="results",
        plots_dir="results/plots",
    )
    specs = build_krotov_specs()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, config.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    config_dict = asdict(config)
    config_dict["experiment_specs"] = specs
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    all_results = []
    for spec in specs:
        for seed in config.seeds:
            result = run_single(config, spec, seed)
            all_results.append(result)
            out_path = os.path.join(results_dir, f"result_{result['optimizer']}_seed{seed}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

    summary = []
    for result in all_results:
        summary.append(
            {
                "optimizer": result["optimizer"],
                "optimizer_name": result["optimizer_name"],
                "optimizer_family": result["optimizer_family"],
                "is_sweep": result["is_sweep"],
                "seed": result["seed"],
                "final_loss": result["final_loss"],
                "final_train_acc": result["final_train_acc"],
                "final_test_acc": result["final_test_acc"],
                "total_cost_units": result["total_cost_units"],
                "total_sample_forward_passes": result["total_sample_forward_passes"],
                "total_sample_backward_passes": result["total_sample_backward_passes"],
                "total_full_loss_evaluations": result["total_full_loss_evaluations"],
                "total_grad_evals": result["total_grad_evals"],
                "wall_time_total": result["wall_time_total"],
            }
        )

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved Krotov variant results to {results_dir}/")


if __name__ == "__main__":
    main()
