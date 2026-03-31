#!/usr/bin/env python3
"""Analyze benchmark results and print summary statistics."""

import glob
import json
import os

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def load_results(config):
    allowed = {spec["run_name"] for spec in config.get("experiment_specs", [])}
    allowed_seeds = set(config.get("seeds", []))
    results = []
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*.json"))):
        with open(fpath) as f:
            result = json.load(f)
        if allowed and result["optimizer"] not in allowed:
            continue
        if allowed_seeds and result["seed"] not in allowed_seeds:
            continue
        results.append(result)
    return results


def group_results(results):
    grouped = {}
    for result in results:
        grouped.setdefault(result["optimizer"], []).append(result)
    return grouped


def _threshold_key(threshold):
    return f"{threshold:.2f}"


def _threshold_metric_stats(runs, threshold):
    key = _threshold_key(threshold)
    reached = [run["threshold_metrics"][key]["reached"] for run in runs]
    step_vals = [run["threshold_metrics"][key]["step"] for run in runs if run["threshold_metrics"][key]["reached"]]
    time_vals = [run["threshold_metrics"][key]["wall_time"] for run in runs if run["threshold_metrics"][key]["reached"]]
    cost_vals = [run["threshold_metrics"][key]["cost_units"] for run in runs if run["threshold_metrics"][key]["reached"]]
    return {
        "success_rate": float(np.mean(reached)),
        "step_mean": float(np.mean(step_vals)) if step_vals else np.nan,
        "time_mean": float(np.mean(time_vals)) if time_vals else np.nan,
        "cost_mean": float(np.mean(cost_vals)) if cost_vals else np.nan,
    }


def print_group_summary(name, runs, threshold, thresholds):
    losses = [run["final_loss"] for run in runs]
    train_accs = [run["final_train_acc"] for run in runs]
    test_accs = [run["final_test_acc"] for run in runs]
    costs = [run["total_cost_units"] for run in runs]
    forwards = [run["total_sample_forward_passes"] for run in runs]
    backwards = [run["total_sample_backward_passes"] for run in runs]
    full_losses = [run["total_full_loss_evaluations"] for run in runs]
    grad_evals = [run["total_grad_evals"] for run in runs]
    times = [run["wall_time_total"] for run in runs]
    success = sum(loss < threshold for loss in losses)

    print(f"--- {name} ---")
    print(
        f"  Final loss:      {np.mean(losses):.4f} +/- {np.std(losses):.4f} "
        f"(min={np.min(losses):.4f}, max={np.max(losses):.4f})"
    )
    print(f"  Train accuracy:  {np.mean(train_accs):.3f} +/- {np.std(train_accs):.3f}")
    print(f"  Test accuracy:   {np.mean(test_accs):.3f} +/- {np.std(test_accs):.3f}")
    print(f"  Cost units:      {np.mean(costs):.0f} +/- {np.std(costs):.0f}")
    print(f"  Sample forwards: {np.mean(forwards):.0f} +/- {np.std(forwards):.0f}")
    print(f"  Sample backwards:{np.mean(backwards):.0f} +/- {np.std(backwards):.0f}")
    print(f"  Full losses:     {np.mean(full_losses):.0f} +/- {np.std(full_losses):.0f}")
    print(f"  Gradient evals:  {np.mean(grad_evals):.0f} +/- {np.std(grad_evals):.0f}")
    print(f"  Wall time:       {np.mean(times):.1f}s +/- {np.std(times):.1f}s")
    print(f"  Success rate:    {success}/{len(runs)} ({100 * success / len(runs):.0f}%)")
    for threshold_value in thresholds:
        stats = _threshold_metric_stats(runs, threshold_value)
        print(
            f"  Reach loss<={threshold_value:.2f}: "
            f"{100 * stats['success_rate']:.0f}% "
            f"(step={stats['step_mean']:.1f}, time={stats['time_mean']:.1f}s, cost={stats['cost_mean']:.0f})"
        )
    print()


def main():
    with open(os.path.join(RESULTS_DIR, "config.json")) as f:
        config = json.load(f)

    results = load_results(config)
    if not results:
        print("No results found.")
        return

    threshold = config.get("loss_threshold", 0.4)
    thresholds = config.get("loss_thresholds", [threshold])
    main_results = [result for result in results if not result.get("is_sweep", False)]
    sweep_results = [result for result in results if result.get("is_sweep", False)]

    print("=" * 78)
    print("TWO-MOONS QML BENCHMARK RESULTS")
    print("=" * 78)
    print(
        f"Model: {config['n_qubits']}-qubit HEA, {config['n_layers']} layers, "
        f"{config['entangler']} entangler"
    )
    print(
        f"Dataset: {config['n_samples']} samples, noise={config['moon_noise']}, "
        f"test_frac={config['test_fraction']}"
    )
    print(f"Seeds: {len(config['seeds'])}")
    print()

    print("Main comparison")
    print("-" * 78)
    for name, runs in group_results(main_results).items():
        print_group_summary(name, runs, threshold, thresholds)

    if sweep_results:
        print("Sweep results")
        print("-" * 78)
        for name, runs in group_results(sweep_results).items():
            print_group_summary(name, runs, threshold, thresholds)

    hybrid_groups = {
        name: runs
        for name, runs in group_results(results).items()
        if runs[0].get("optimizer_family") == "krotov_hybrid"
    }
    if hybrid_groups:
        best_hybrid = min(
            hybrid_groups,
            key=lambda name: np.mean([run["final_loss"] for run in hybrid_groups[name]]),
        )
        best_threshold = _threshold_metric_stats(hybrid_groups[best_hybrid], threshold)
        print("Best hybrid variant")
        print("-" * 78)
        print(
            f"{best_hybrid}: final_loss={np.mean([run['final_loss'] for run in hybrid_groups[best_hybrid]]):.4f} "
            f"time_to_{threshold:.2f}={best_threshold['time_mean']:.1f}s "
            f"cost_to_{threshold:.2f}={best_threshold['cost_mean']:.0f}"
        )
        print()

    print("=" * 78)
    print("ACCOUNTING CONVENTION")
    print("=" * 78)
    print("Cost units = sample forward passes + sample backward passes.")
    print("Full-loss evaluations are logged separately because parameter-shift")
    print("gradients consume many full-batch losses but no sample backward passes.")
    print("Krotov online computes one local update direction per sample.")
    print("Krotov batch computes one full-batch update direction per outer step.")


if __name__ == "__main__":
    main()
