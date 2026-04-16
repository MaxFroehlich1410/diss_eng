#!/usr/bin/env python3
"""Compare Chen SUN-VQC with and without a classical readout head.

The script uses the best hybrid-Krotov hyperparameters found for the Chen model
in the repository's prior sweep:

- ``hybrid_switch_iteration = 15``
- ``hybrid_online_step_size = 0.05``
- ``hybrid_batch_step_size = 0.1``

It then runs two matched variants on the same two-moons splits and seeds:

1. ``simple_z0``: pure quantum readout
2. ``hybrid_linear``: classical affine head over ``[<Z0>, <Z1>, <Z2>, <Z3>]``

By default the comparison uses three seeds so the outcome is less brittle than
single-seed anecdotal evidence, while still remaining lightweight.
"""

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
    from datasets import generate_two_moons
    from qml_models.variants import ChenSUNVQCModel
    from optimizers.runner import run_optimizer
else:
    from experiments.two_moons_common.config import DEFAULT_CONFIG
    from datasets import generate_two_moons
    from qml_models.variants import ChenSUNVQCModel
    from optimizers.runner import run_optimizer


RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
BEST_HYBRID_KROTOV_CONFIG = {
    "input_encoding": "linear_pm_pi",
    "n_samples": 300,
    "moon_noise": 0.07,
    "test_fraction": 0.2,
    "max_iterations": 20,
    "lbfgs_maxiter": 20,
    "hybrid_switch_iteration": 15,
    "hybrid_online_step_size": 0.05,
    "hybrid_batch_step_size": 0.1,
    "hybrid_online_schedule": "constant",
    "hybrid_batch_schedule": "constant",
    "early_stopping_enabled": False,
}

VARIANTS = OrderedDict(
    [
        (
            "simple_z0",
            {
                "label": "Chen SUN-VQC without classical head",
                "builder": lambda: ChenSUNVQCModel(
                    n_macro_layers=2,
                    encoding_axes=("y", "z"),
                    readout="simple_z0",
                ),
            },
        ),
        (
            "hybrid_linear",
            {
                "label": "Chen SUN-VQC with classical affine head",
                "builder": lambda: ChenSUNVQCModel(
                    n_macro_layers=2,
                    encoding_axes=("y", "z"),
                    readout="hybrid_linear",
                ),
            },
        ),
    ]
)


def build_config(seeds):
    return replace(
        DEFAULT_CONFIG,
        optimizers=["krotov_hybrid"],
        run_krotov_batch_sweep=False,
        run_krotov_hybrid_sweep=False,
        results_dir="results",
        seeds=list(seeds),
        **BEST_HYBRID_KROTOV_CONFIG,
    )


def jsonify_trace(trace):
    out = {}
    for key, values in trace.items():
        if key == "phase":
            out[key] = [str(v) for v in values]
        else:
            out[key] = [float(v) for v in values]
    return out


def run_single(variant_name, seed, config):
    variant = VARIANTS[variant_name]
    model = variant["builder"]()
    X_train, X_test, y_train, y_test = generate_two_moons(
        n_samples=config.n_samples,
        noise=config.moon_noise,
        test_fraction=config.test_fraction,
        seed=seed,
        encoding=config.input_encoding,
    )
    init_params = np.asarray(model.init_params(seed=seed), dtype=float)

    print(f"\n{'=' * 80}")
    print(f"Variant: {variant['label']} | Seed: {seed}")
    print(f"{'=' * 80}")

    start_time = time.time()
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
    wall_total = time.time() - start_time

    result = {
        "variant": variant_name,
        "variant_label": variant["label"],
        "seed": seed,
        "n_params": int(model.n_params),
        "n_quantum_params": int(model.n_quantum_params),
        "n_classical_params": int(model.n_output_params),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "wall_time_total": float(wall_total),
        "total_cost_units": int(trace["cost_units"][-1]),
        "total_steps": int(trace["step"][-1]),
        "initial_params": init_params.tolist(),
        "final_params": np.asarray(final_params, dtype=float).tolist(),
        "trace": jsonify_trace(trace),
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
    for variant_name in VARIANTS:
        runs = [result for result in results if result["variant"] == variant_name]
        losses = np.asarray([run["final_loss"] for run in runs], dtype=float)
        train_accs = np.asarray([run["final_train_acc"] for run in runs], dtype=float)
        test_accs = np.asarray([run["final_test_acc"] for run in runs], dtype=float)
        wall_times = np.asarray([run["wall_time_total"] for run in runs], dtype=float)
        costs = np.asarray([run["total_cost_units"] for run in runs], dtype=float)
        summary[variant_name] = {
            "label": VARIANTS[variant_name]["label"],
            "n_runs": len(runs),
            "n_params": int(runs[0]["n_params"]),
            "n_quantum_params": int(runs[0]["n_quantum_params"]),
            "n_classical_params": int(runs[0]["n_classical_params"]),
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
        }
    return summary


def compare_variants(summary):
    base = summary["simple_z0"]
    head = summary["hybrid_linear"]
    return {
        "loss_delta": head["final_loss_mean"] - base["final_loss_mean"],
        "test_acc_delta": head["final_test_acc_mean"] - base["final_test_acc_mean"],
        "train_acc_delta": head["final_train_acc_mean"] - base["final_train_acc_mean"],
        "wall_time_delta": head["wall_time_mean"] - base["wall_time_mean"],
        "cost_delta": head["cost_mean"] - base["cost_mean"],
    }


def write_report(summary, comparison, config, results_dir):
    base = summary["simple_z0"]
    head = summary["hybrid_linear"]
    lines = [
        "# Chen Readout Head Comparison",
        "",
        "## Setup",
        "",
        "Hybrid Krotov configuration taken from the best prior Chen sweep:",
        "",
        f"- `hybrid_switch_iteration = {config.hybrid_switch_iteration}`",
        f"- `hybrid_online_step_size = {config.hybrid_online_step_size}`",
        f"- `hybrid_batch_step_size = {config.hybrid_batch_step_size}`",
        f"- `max_iterations = {config.max_iterations}`",
        f"- `n_samples = {config.n_samples}`",
        f"- `moon_noise = {config.moon_noise}`",
        f"- `test_fraction = {config.test_fraction}`",
        f"- `seeds = {config.seeds}`",
        "",
        "## Results",
        "",
        "| Variant | Params | Classical head params | Final loss | Final test acc | Wall time (s) |",
        "|---|---:|---:|---|---|---|",
        (
            f"| {base['label']} | {base['n_params']} | {base['n_classical_params']} | "
            f"{base['final_loss_mean']:.4f} ± {base['final_loss_std']:.4f} | "
            f"{base['final_test_acc_mean']:.4f} ± {base['final_test_acc_std']:.4f} | "
            f"{base['wall_time_mean']:.2f} ± {base['wall_time_std']:.2f} |"
        ),
        (
            f"| {head['label']} | {head['n_params']} | {head['n_classical_params']} | "
            f"{head['final_loss_mean']:.4f} ± {head['final_loss_std']:.4f} | "
            f"{head['final_test_acc_mean']:.4f} ± {head['final_test_acc_std']:.4f} | "
            f"{head['wall_time_mean']:.2f} ± {head['wall_time_std']:.2f} |"
        ),
        "",
        "## Delta: classical head minus no-head",
        "",
        f"- Final loss delta: `{comparison['loss_delta']:+.4f}`",
        f"- Final test accuracy delta: `{comparison['test_acc_delta']:+.4f}`",
        f"- Final train accuracy delta: `{comparison['train_acc_delta']:+.4f}`",
        f"- Wall time delta: `{comparison['wall_time_delta']:+.2f}s`",
        f"- Cost units delta: `{comparison['cost_delta']:+.1f}`",
        "",
    ]

    if comparison["loss_delta"] < 0.0 and comparison["test_acc_delta"] >= 0.0:
        verdict = (
            "Under this matched configuration, the classical affine head helps: "
            "it lowers loss without hurting test accuracy."
        )
    elif comparison["loss_delta"] > 0.0 and comparison["test_acc_delta"] <= 0.0:
        verdict = (
            "Under this matched configuration, the classical affine head hurts: "
            "it increases loss and does not improve accuracy."
        )
    else:
        verdict = (
            "Under this matched configuration, the effect of the classical affine head is mixed: "
            "it changes loss and accuracy in different directions or only weakly."
        )
    lines.extend(["## Verdict", "", verdict, ""])

    report_path = os.path.join(results_dir, "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    config = build_config(args.seeds)

    all_results = []
    for variant_name in VARIANTS:
        for seed in config.seeds:
            result = run_single(variant_name, seed, config)
            all_results.append(result)
            out_path = os.path.join(args.results_dir, f"result_{variant_name}_seed{seed}.json")
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)

    summary = summarise_results(all_results)
    comparison = compare_variants(summary)
    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(config),
                "summary": summary,
                "comparison": comparison,
            },
            handle,
            indent=2,
        )
    report_path = write_report(summary, comparison, config, args.results_dir)

    print(f"\nComparison report written to {report_path}")
    print(json.dumps({"summary": summary, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
