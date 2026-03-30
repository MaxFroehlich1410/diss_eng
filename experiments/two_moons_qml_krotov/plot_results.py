#!/usr/bin/env python3
"""Generate publication-quality plots from benchmark results.

Usage:
    python plot_results.py
"""

import json
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import VQCModel
from dataset import generate_two_moons

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")

COLORS = {"krotov": "#e41a1c", "adam": "#377eb8", "lbfgs": "#4daf4a"}
LABELS = {"krotov": "Krotov", "adam": "Adam", "lbfgs": "L-BFGS-B"}

plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def load_results():
    """Load all result JSON files."""
    results = {}
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*.json"))):
        with open(fpath) as f:
            r = json.load(f)
        key = r["optimizer"]
        results.setdefault(key, []).append(r)
    return results


def _interp_traces(traces_x, traces_y, n_points=500):
    """Interpolate traces to common x-grid for averaging."""
    x_min = max(t[0] for t in traces_x)
    x_max = min(t[-1] for t in traces_x)
    x_grid = np.linspace(x_min, x_max, n_points)
    y_interp = []
    for tx, ty in zip(traces_x, traces_y):
        y_interp.append(np.interp(x_grid, tx, ty))
    return x_grid, np.array(y_interp)


def plot_loss_vs_evals(results):
    """Plot 1: Mean training loss vs function evaluations."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for opt_name, runs in results.items():
        traces_x = [r["trace"]["func_evals"] for r in runs]
        traces_y = [r["trace"]["loss"] for r in runs]
        x_grid, y_arr = _interp_traces(traces_x, traces_y)
        mean = np.mean(y_arr, axis=0)
        std = np.std(y_arr, axis=0)
        ax.plot(x_grid, mean, color=COLORS[opt_name], label=LABELS[opt_name], lw=2)
        ax.fill_between(x_grid, mean - std, mean + std,
                        color=COLORS[opt_name], alpha=0.15)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Training loss (BCE)")
    ax.set_title("Training loss vs function evaluations")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "loss_vs_evals")


def plot_loss_vs_step(results):
    """Plot 2: Training loss vs optimization step."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for opt_name, runs in results.items():
        traces_x = [r["trace"]["step"] for r in runs]
        traces_y = [r["trace"]["loss"] for r in runs]
        # Pad to common length
        max_steps = max(len(t) for t in traces_x)
        for i, (tx, ty) in enumerate(zip(traces_x, traces_y)):
            ax.plot(tx, ty, color=COLORS[opt_name], alpha=0.15, lw=0.5)
        # Mean trace (use interp on steps)
        x_grid, y_arr = _interp_traces(traces_x, traces_y)
        mean = np.mean(y_arr, axis=0)
        ax.plot(x_grid, mean, color=COLORS[opt_name], label=LABELS[opt_name], lw=2)
    ax.set_xlabel("Optimization step")
    ax.set_ylabel("Training loss (BCE)")
    ax.set_title("Training loss vs optimization step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "loss_vs_step")


def plot_loss_vs_time(results):
    """Plot 3: Training loss vs wall-clock time."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for opt_name, runs in results.items():
        traces_x = [r["trace"]["wall_time"] for r in runs]
        traces_y = [r["trace"]["loss"] for r in runs]
        x_grid, y_arr = _interp_traces(traces_x, traces_y)
        mean = np.mean(y_arr, axis=0)
        std = np.std(y_arr, axis=0)
        ax.plot(x_grid, mean, color=COLORS[opt_name], label=LABELS[opt_name], lw=2)
        ax.fill_between(x_grid, mean - std, mean + std,
                        color=COLORS[opt_name], alpha=0.15)
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Training loss (BCE)")
    ax.set_title("Training loss vs wall-clock time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "loss_vs_time")


def plot_boxplot_loss(results):
    """Plot 4: Boxplot of final training loss by optimizer."""
    fig, ax = plt.subplots(figsize=(6, 5))
    data = []
    labels = []
    colors = []
    for opt_name in results:
        vals = [r["final_loss"] for r in results[opt_name]]
        data.append(vals)
        labels.append(LABELS[opt_name])
        colors.append(COLORS[opt_name])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Final training loss (BCE)")
    ax.set_title("Final training loss by optimizer")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "boxplot_loss")


def plot_boxplot_accuracy(results):
    """Plot 5: Boxplot of final test accuracy by optimizer."""
    fig, ax = plt.subplots(figsize=(6, 5))
    data = []
    labels = []
    colors = []
    for opt_name in results:
        vals = [r["final_test_acc"] for r in results[opt_name]]
        data.append(vals)
        labels.append(LABELS[opt_name])
        colors.append(COLORS[opt_name])
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Final test accuracy by optimizer")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "boxplot_accuracy")


def plot_success_rate(results):
    """Plot 6: Success rate (fraction reaching loss threshold) vs evaluations."""
    with open(os.path.join(RESULTS_DIR, "config.json")) as f:
        config = json.load(f)
    threshold = config.get("loss_threshold", 0.3)

    fig, ax = plt.subplots(figsize=(8, 5))
    for opt_name, runs in results.items():
        traces_x = [r["trace"]["func_evals"] for r in runs]
        traces_y = [r["trace"]["loss"] for r in runs]
        x_grid, y_arr = _interp_traces(traces_x, traces_y)
        success_frac = np.mean(y_arr <= threshold, axis=0)
        ax.plot(x_grid, success_frac, color=COLORS[opt_name],
                label=LABELS[opt_name], lw=2)
    ax.axhline(1.0, color="gray", ls="--", lw=0.5)
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel(f"Fraction of runs with loss < {threshold}")
    ax.set_title("Success rate vs function evaluations")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "success_rate")


def plot_decision_boundaries(results):
    """Plot 7: Decision boundaries for one representative model per optimizer."""
    with open(os.path.join(RESULTS_DIR, "config.json")) as f:
        config = json.load(f)

    # Pick the run with median final loss for each optimizer
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5))
    if len(results) == 1:
        axes = [axes]

    for ax, (opt_name, runs) in zip(axes, results.items()):
        losses = [r["final_loss"] for r in runs]
        median_idx = np.argsort(losses)[len(losses) // 2]
        run = runs[median_idx]

        seed = run["seed"]
        X_train, X_test, y_train, y_test = generate_two_moons(
            n_samples=config["n_samples"], noise=config["moon_noise"],
            test_fraction=config["test_fraction"], seed=seed,
        )

        model = VQCModel(
            n_qubits=config["n_qubits"], n_layers=config["n_layers"],
            entangler=config["entangler"],
        )
        params = np.array(run["final_params"])

        # Grid
        x1_range = np.linspace(0, np.pi, 60)
        x2_range = np.linspace(0, np.pi, 60)
        xx1, xx2 = np.meshgrid(x1_range, x2_range)
        grid = np.c_[xx1.ravel(), xx2.ravel()]
        probs = model.forward_batch(params, grid).reshape(xx1.shape)

        cmap_bg = ListedColormap(["#FFAAAA", "#AAAAFF"])
        ax.contourf(xx1, xx2, probs, levels=[0, 0.5, 1], cmap=cmap_bg, alpha=0.4)
        ax.contour(xx1, xx2, probs, levels=[0.5], colors="k", linewidths=1)

        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                   c="red", s=8, alpha=0.5, label="Train 0")
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                   c="blue", s=8, alpha=0.5, label="Train 1")
        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
                   c="red", s=15, marker="x", alpha=0.7)
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
                   c="blue", s=15, marker="x", alpha=0.7)

        ax.set_title(f"{LABELS[opt_name]} (loss={run['final_loss']:.3f})")
        ax.set_xlabel("$x_1$ (encoded)")
        ax.set_ylabel("$x_2$ (encoded)")

    fig.suptitle("Decision boundaries (median-loss run)", y=1.02)
    plt.tight_layout()
    _save(fig, "decision_boundaries")


def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.png"))
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  Saved {name}.png/.pdf")


def main():
    results = load_results()
    if not results:
        print("No results found. Run run_experiment.py first.")
        return

    print(f"Loaded results for: {list(results.keys())}")
    print(f"Seeds per optimizer: {[len(v) for v in results.values()]}")

    plot_loss_vs_evals(results)
    plot_loss_vs_step(results)
    plot_loss_vs_time(results)
    plot_boxplot_loss(results)
    plot_boxplot_accuracy(results)
    plot_success_rate(results)
    plot_decision_boundaries(results)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
