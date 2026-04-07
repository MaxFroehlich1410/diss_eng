#!/usr/bin/env python3
"""Binary Iris classification benchmark using the HEA model.

Follows the same methodology as the two-moons HEA benchmark described in
Krotov_paper.pdf, adapted for the Iris dataset (setosa vs. versicolor,
first two features).

Four optimizers are compared:
  1. Hybrid Krotov  (online → batch, switch at iteration 10)
  2. Adam
  3. L-BFGS-B
  4. Quantum Natural Gradient (QNG)

Usage:
    python experiments/iris_hea_benchmark.py
    python experiments/iris_hea_benchmark.py --seeds 0 1 2 3 4 --threshold 0.40
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
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ---------------------------------------------------------------------------
# Path setup — reuse two-moons codebase
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TWO_MOONS_DIR = os.path.join(SCRIPT_DIR, "two_moons_qml_krotov")
sys.path.insert(0, TWO_MOONS_DIR)

from config import ExperimentConfig
from model import VQCModel
from optimizers import run_optimizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_DIR = os.path.join(SCRIPT_DIR, "iris_results")

OPTIMIZERS = ["krotov_hybrid", "adam", "lbfgs", "qng"]
OPTIMIZER_LABELS = {
    "krotov_hybrid": "Hybrid Krotov",
    "adam": "Adam",
    "lbfgs": "L-BFGS-B",
    "qng": "QNG",
}
OPTIMIZER_COLORS = {
    "krotov_hybrid": "#8b1e3f",
    "adam": "#2563eb",
    "lbfgs": "#0f766e",
    "qng": "#d97706",
}

LOSS_THRESHOLD = 0.40
TAIL_WINDOW = 10

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def load_iris_binary(test_fraction=0.2, seed=42):
    """Load Iris (setosa vs. versicolor), first 2 features, scaled to [0, pi].

    Returns X_train, X_test, y_train, y_test, and the scaling parameters
    (x_min, span) needed to map raw features to the encoded space.
    """
    iris = load_iris()
    mask = iris.target < 2
    X = iris.data[mask, :2].astype(float)
    y = iris.target[mask].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y,
    )

    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    span = np.where(x_max > x_min, x_max - x_min, 1.0)
    X_train_enc = np.pi * (X_train - x_min) / span
    X_test_enc = np.pi * (X_test - x_min) / span

    return X_train_enc, X_test_enc, y_train, y_test, x_min, span


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
def build_config(seeds=None, threshold=LOSS_THRESHOLD):
    return replace(
        ExperimentConfig(),
        n_qubits=4,
        n_layers=3,
        entangler="ring",
        model_architecture="hea",
        observable="Z0Z1",
        max_iterations=100,
        lbfgs_maxiter=100,
        adam_lr=0.05,
        qng_lr=0.5,
        qng_lam=0.01,
        qng_approx=None,
        hybrid_switch_iteration=10,
        hybrid_online_step_size=0.3,
        hybrid_batch_step_size=1.0,
        hybrid_online_schedule="constant",
        hybrid_batch_schedule="constant",
        early_stopping_enabled=False,
        optimizers=list(OPTIMIZERS),
        run_krotov_batch_sweep=False,
        run_krotov_hybrid_sweep=False,
        loss_threshold=threshold,
        seeds=list(seeds) if seeds is not None else list(range(5)),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def jsonify_trace(trace):
    out = {}
    for key, values in trace.items():
        out[key] = [str(v) for v in values] if key == "phase" else [float(v) for v in values]
    return out


def tail_std(trace, window=TAIL_WINDOW):
    losses = np.asarray(trace["loss"], dtype=float)
    return float(np.std(losses[-window:])) if len(losses) > window else float(np.std(losses))


def time_to_threshold(trace, threshold):
    losses = np.asarray(trace["loss"], dtype=float)
    wall = np.asarray(trace["wall_time"], dtype=float)
    hits = np.where(losses <= threshold)[0]
    return float(wall[hits[0]]) if len(hits) else None


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(optimizer_name, seed, config):
    model = VQCModel(
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        entangler=config.entangler,
        architecture=config.model_architecture,
        observable=config.observable,
    )
    X_train, X_test, y_train, y_test, _, _ = load_iris_binary(
        test_fraction=0.2, seed=seed,
    )
    init_params = model.init_params(seed=seed)

    print(f"\n{'=' * 70}")
    print(f"Optimizer: {OPTIMIZER_LABELS[optimizer_name]} | Seed: {seed}")
    print(f"  n_train={len(X_train)}  n_test={len(X_test)}  n_params={model.n_params}")
    print(f"{'=' * 70}")

    t0 = time.time()
    final_params, trace = run_optimizer(
        optimizer_name, model, init_params.copy(),
        X_train, y_train, X_test, y_test, config,
    )
    wall_total = time.time() - t0

    result = {
        "optimizer": optimizer_name,
        "seed": seed,
        "n_params": int(model.n_params),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "wall_time_total": float(wall_total),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "total_cost_units": int(trace["cost_units"][-1]),
        "total_steps": int(trace["step"][-1]),
        "tail_loss_std": tail_std(trace),
        "time_to_threshold": time_to_threshold(trace, config.loss_threshold),
        "trace": jsonify_trace(trace),
        "final_params": np.asarray(final_params, dtype=float).tolist(),
    }

    print(
        f"  Done: loss={result['final_loss']:.4f}  "
        f"train_acc={result['final_train_acc']:.3f}  "
        f"test_acc={result['final_test_acc']:.3f}  "
        f"cost={result['total_cost_units']}  wall={wall_total:.1f}s"
    )
    return result


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(results, threshold):
    summary = OrderedDict()
    for opt in OPTIMIZERS:
        runs = [r for r in results if r["optimizer"] == opt]
        if not runs:
            continue
        losses = np.array([r["final_loss"] for r in runs])
        train_accs = np.array([r["final_train_acc"] for r in runs])
        test_accs = np.array([r["final_test_acc"] for r in runs])
        walls = np.array([r["wall_time_total"] for r in runs])
        tail_stds = np.array([r["tail_loss_std"] for r in runs])
        reached = [r["time_to_threshold"] for r in runs if r["time_to_threshold"] is not None]

        summary[opt] = {
            "label": OPTIMIZER_LABELS[opt],
            "n_runs": len(runs),
            "final_loss_mean": float(np.mean(losses)),
            "final_loss_std": float(np.std(losses)),
            "final_train_acc_mean": float(np.mean(train_accs)),
            "final_train_acc_std": float(np.std(train_accs)),
            "final_test_acc_mean": float(np.mean(test_accs)),
            "final_test_acc_std": float(np.std(test_accs)),
            "wall_time_mean": float(np.mean(walls)),
            "wall_time_std": float(np.std(walls)),
            "success_rate": len(reached) / len(runs),
            "time_to_threshold_mean": float(np.mean(reached)) if reached else None,
            "tail_loss_std_mean": float(np.mean(tail_stds)),
            "threshold": threshold,
        }
    return summary


def print_table(summary):
    thr = next(iter(summary.values()))["threshold"]
    header = (
        f"{'Optimizer':<16} {'Final loss':>16} {'Final test acc':>16} "
        f"{'Wall (s)':>10} {'Time≤%.2f (s)':>14} {'Success':>8} {'Tail std':>10}"
    ) % thr
    sep = "-" * len(header)
    print(f"\n{header}\n{sep}")
    for s in summary.values():
        ttt = f"{s['time_to_threshold_mean']:.1f}" if s["time_to_threshold_mean"] else "--"
        print(
            f"{s['label']:<16} "
            f"{s['final_loss_mean']:.3f} ± {s['final_loss_std']:.3f}   "
            f"{s['final_test_acc_mean']:.3f} ± {s['final_test_acc_std']:.3f}   "
            f"{s['wall_time_mean']:8.1f}   "
            f"{ttt:>12}   "
            f"{s['success_rate']:6.2f}   "
            f"{s['tail_loss_std_mean']:.4f}"
        )


def write_latex_table(summary, path):
    thr = next(iter(summary.values()))["threshold"]
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        (r"Optimizer & Final loss & Final test acc. & Wall time (s) "
         r"& Time to $\leq %.2f$ (s) & Success @ %.2f & Tail std. \\") % (thr, thr),
        r"\midrule",
    ]
    for s in summary.values():
        ttt = f"{s['time_to_threshold_mean']:.1f}" if s['time_to_threshold_mean'] else "--"
        lines.append(
            f"{s['label']} & "
            f"{s['final_loss_mean']:.3f} $\\pm$ {s['final_loss_std']:.3f} & "
            f"{s['final_test_acc_mean']:.3f} $\\pm$ {s['final_test_acc_std']:.3f} & "
            f"{s['wall_time_mean']:.1f} $\\pm$ {s['wall_time_std']:.1f} & "
            f"{ttt} & "
            f"{s['success_rate']:.2f} & "
            f"{s['tail_loss_std_mean']:.4f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}}",
        r"\caption{Iris binary classification benchmark (5 seeds, 100 iterations).}",
        r"\end{table}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def interpolate_traces(x_traces, y_traces, n_points=300):
    x_min = max(t[0] for t in x_traces)
    x_max = min(t[-1] for t in x_traces)
    x_grid = np.linspace(x_min, x_max, n_points)
    y_interp = np.array(
        [np.interp(x_grid, x, y) for x, y in zip(x_traces, y_traces)]
    )
    return x_grid, y_interp


def save_fig(fig, name, results_dir):
    fig.savefig(os.path.join(results_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(results_dir, f"{name}.png"))
    plt.close(fig)


def grouped_by_optimizer(results):
    return OrderedDict(
        (opt, [r for r in results if r["optimizer"] == opt]) for opt in OPTIMIZERS
    )


def _mean_std_trace(grouped, opt, x_key, y_key="loss"):
    runs = grouped[opt]
    if not runs:
        return None, None, None
    xs = [np.asarray(r["trace"][x_key], dtype=float) for r in runs]
    ys = [np.asarray(r["trace"][y_key], dtype=float) for r in runs]
    x_grid, y_interp = interpolate_traces(xs, ys)
    return x_grid, np.mean(y_interp, axis=0), np.std(y_interp, axis=0)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------
def plot_loss_vs_iteration(results, results_dir):
    grouped = grouped_by_optimizer(results)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for opt in OPTIMIZERS:
        x, mean, std = _mean_std_trace(grouped, opt, "step")
        if x is None:
            continue
        c = OPTIMIZER_COLORS[opt]
        ax.plot(x, mean, color=c, lw=2.2, label=OPTIMIZER_LABELS[opt])
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.18)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss (BCE)")
    ax.set_title("Iris HEA benchmark: loss vs iteration")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, "loss_vs_iteration", results_dir)


def plot_loss_vs_time(results, results_dir):
    grouped = grouped_by_optimizer(results)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for opt in OPTIMIZERS:
        x, mean, std = _mean_std_trace(grouped, opt, "wall_time")
        if x is None:
            continue
        c = OPTIMIZER_COLORS[opt]
        ax.plot(x, mean, color=c, lw=2.2, label=OPTIMIZER_LABELS[opt])
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.18)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Training loss (BCE)")
    ax.set_title("Iris HEA benchmark: loss vs wall-clock time")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, "loss_vs_time", results_dir)


def plot_loss_vs_cost(results, results_dir):
    grouped = grouped_by_optimizer(results)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for opt in OPTIMIZERS:
        x, mean, std = _mean_std_trace(grouped, opt, "cost_units")
        if x is None:
            continue
        c = OPTIMIZER_COLORS[opt]
        ax.plot(x, mean, color=c, lw=2.2, label=OPTIMIZER_LABELS[opt])
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.18)
    ax.set_xlabel("Cost units (forward + backward passes)")
    ax.set_ylabel("Training loss (BCE)")
    ax.set_title("Iris HEA benchmark: loss vs propagation cost")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, "loss_vs_cost", results_dir)


def plot_update_norm(results, results_dir):
    grouped = grouped_by_optimizer(results)
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for opt in OPTIMIZERS:
        x, mean, std = _mean_std_trace(grouped, opt, "step", "update_norm")
        if x is None:
            continue
        c = OPTIMIZER_COLORS[opt]
        ax.plot(x, mean, color=c, lw=2.2, label=OPTIMIZER_LABELS[opt])
        ax.fill_between(x, mean - std, mean + std, color=c, alpha=0.18)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter update norm")
    ax.set_title("Iris HEA benchmark: update norm vs iteration")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, "update_norm_vs_iteration", results_dir)


def plot_decision_boundaries(results, config, results_dir):
    grouped = grouped_by_optimizer(results)
    model = VQCModel(
        n_qubits=config.n_qubits,
        n_layers=config.n_layers,
        entangler=config.entangler,
        architecture=config.model_architecture,
        observable=config.observable,
    )

    n_opt = len(OPTIMIZERS)
    fig, axes = plt.subplots(1, n_opt, figsize=(4.8 * n_opt, 4.2))
    if n_opt == 1:
        axes = [axes]

    cmap_bg = ListedColormap(["#fee2e2", "#dbeafe"])
    grid_axis = np.linspace(0, np.pi, 80)
    xx, yy = np.meshgrid(grid_axis, grid_axis)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    for ax, opt in zip(axes, OPTIMIZERS):
        runs = grouped[opt]
        if not runs:
            continue
        losses = np.array([r["final_loss"] for r in runs])
        median_idx = int(np.argsort(losses)[len(losses) // 2])
        run = runs[median_idx]

        params = np.asarray(run["final_params"], dtype=float)
        probs = model.forward_batch(params, grid_points).reshape(xx.shape)

        ax.contourf(xx, yy, probs, levels=np.linspace(0, 1, 11), cmap=cmap_bg, alpha=0.78)
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.0)

        X_train, X_test, y_train, y_test, _, _ = load_iris_binary(
            test_fraction=0.2, seed=run["seed"],
        )
        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                   c="#b91c1c", s=14, alpha=0.5)
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                   c="#1d4ed8", s=14, alpha=0.5)
        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
                   c="#7f1d1d", s=30, marker="x", alpha=0.85)
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
                   c="#1e3a8a", s=30, marker="x", alpha=0.85)
        ax.set_title(
            f"{OPTIMIZER_LABELS[opt]}\n"
            f"loss={run['final_loss']:.3f}, test acc={run['final_test_acc']:.3f}"
        )
        ax.set_xlabel("Sepal length (scaled)")
        ax.set_ylabel("Sepal width (scaled)")

    fig.suptitle("Iris HEA: representative decision boundaries", y=1.03)
    plt.tight_layout()
    save_fig(fig, "decision_boundaries", results_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--threshold", type=float, default=LOSS_THRESHOLD)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    sys.stdout.reconfigure(line_buffering=True)

    config = build_config(seeds=args.seeds, threshold=args.threshold)

    all_results = []
    for opt in OPTIMIZERS:
        for seed in config.seeds:
            result = run_single(opt, seed, config)
            all_results.append(result)
            out_path = os.path.join(results_dir, f"result_{opt}_seed{seed}.json")
            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)

    # Summary
    summary = summarize(all_results, config.loss_threshold)
    print_table(summary)

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)
    write_latex_table(summary, os.path.join(results_dir, "results_table.tex"))

    # Plots
    plot_loss_vs_iteration(all_results, results_dir)
    plot_loss_vs_time(all_results, results_dir)
    plot_loss_vs_cost(all_results, results_dir)
    plot_update_norm(all_results, results_dir)
    plot_decision_boundaries(all_results, config, results_dir)

    print(f"\nAll results and plots saved to {results_dir}/")


if __name__ == "__main__":
    main()
