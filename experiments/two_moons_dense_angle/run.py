#!/usr/bin/env python3
"""Run the online-derived dense-angle two-moons benchmark."""

import json
import os
from dataclasses import asdict, replace

from experiments.two_moons_common.config import DEFAULT_CONFIG
from run_experiment import build_experiment_specs, run_single


def build_config():
    return replace(
        DEFAULT_CONFIG,
        input_encoding="linear_pm_pi",
        model_architecture="two_moons_dense_angle",
        n_qubits=2,
        n_layers=4,
        entangler="none",
        observable="Z0",
        optimizers=["krotov_hybrid", "adam", "lbfgs"],
        run_krotov_batch_sweep=False,
        run_krotov_hybrid_sweep=False,
        hybrid_switch_iteration=10,
        hybrid_online_step_size=0.05,
        hybrid_batch_step_size=0.1,
        hybrid_online_schedule="constant",
        hybrid_batch_schedule="constant",
        adam_lr=0.05,
        results_dir="results",
    )


def main():
    config = build_config()
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
                "seed": r["seed"],
                "final_loss": r["final_loss"],
                "final_train_acc": r["final_train_acc"],
                "final_test_acc": r["final_test_acc"],
                "total_cost_units": r["total_cost_units"],
                "wall_time_total": r["wall_time_total"],
                "threshold_metrics": r["threshold_metrics"],
            }
        )

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDense-angle benchmark saved to {results_dir}/")


if __name__ == "__main__":
    main()
