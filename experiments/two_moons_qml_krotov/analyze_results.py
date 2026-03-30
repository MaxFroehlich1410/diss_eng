#!/usr/bin/env python3
"""Analyze benchmark results and print summary statistics."""

import json
import os
import sys
import glob
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def load_results():
    results = {}
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*.json"))):
        with open(fpath) as f:
            r = json.load(f)
        results.setdefault(r["optimizer"], []).append(r)
    return results


def main():
    results = load_results()
    if not results:
        print("No results found.")
        return

    with open(os.path.join(RESULTS_DIR, "config.json")) as f:
        config = json.load(f)

    threshold = config.get("loss_threshold", 0.3)

    print("=" * 70)
    print("TWO-MOONS QML BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Model: {config['n_qubits']}-qubit HEA, {config['n_layers']} layers, "
          f"{config['entangler']} entangler")
    print(f"Dataset: {config['n_samples']} samples, noise={config['moon_noise']}, "
          f"test_frac={config['test_fraction']}")
    print(f"Seeds: {len(config['seeds'])}")
    print()

    for opt_name in results:
        runs = results[opt_name]
        losses = [r["final_loss"] for r in runs]
        train_accs = [r["final_train_acc"] for r in runs]
        test_accs = [r["final_test_acc"] for r in runs]
        evals = [r["total_func_evals"] for r in runs]
        times = [r["wall_time_total"] for r in runs]
        success = sum(1 for l in losses if l < threshold)

        print(f"--- {opt_name.upper()} ---")
        print(f"  Final loss:      {np.mean(losses):.4f} +/- {np.std(losses):.4f} "
              f"(min={np.min(losses):.4f}, max={np.max(losses):.4f})")
        print(f"  Train accuracy:  {np.mean(train_accs):.3f} +/- {np.std(train_accs):.3f}")
        print(f"  Test accuracy:   {np.mean(test_accs):.3f} +/- {np.std(test_accs):.3f}")
        print(f"  Func evals:      {np.mean(evals):.0f} +/- {np.std(evals):.0f}")
        print(f"  Wall time:       {np.mean(times):.1f}s +/- {np.std(times):.1f}s")
        print(f"  Success rate:    {success}/{len(runs)} "
              f"({100*success/len(runs):.0f}%) below threshold={threshold}")
        print()

    # Counting convention
    print("=" * 70)
    print("FUNCTION EVALUATION COUNTING CONVENTION")
    print("=" * 70)
    print("- Krotov: each sample update = 2 circuit evaluations (fwd + bwd)")
    print("  Plus metric evaluations at each epoch (N_train + N_test)")
    print("- Adam: parameter-shift gradient = 2 * n_params circuit evals per step")
    print("  Plus metric evaluations (loss + accuracy on train + test)")
    print("- L-BFGS-B: same as Adam (parameter-shift for gradients)")
    print("  L-BFGS-B may also do extra function-only evaluations for line search")
    print()
    print("NOTE: The 'function evaluations' count includes metric evaluations,")
    print("so it is an upper bound on the true optimization cost. The primary")
    print("fair comparison is 'loss vs optimization step' and 'loss vs wall time'.")


if __name__ == "__main__":
    main()
