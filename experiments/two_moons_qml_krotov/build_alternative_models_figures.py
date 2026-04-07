#!/usr/bin/env python3
"""Build per-model figures for the alternative-model hybrid report section."""

from __future__ import annotations

import glob
import json
import os
import sys
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dataset import generate_two_moons
from run_alternative_models_benchmark import MODEL_SPECS


RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_alternative_models_120x20")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report_hybrid")
FIG_DIR = os.path.join(REPORT_DIR, "figures")

OPTIMIZER_ORDER = ["krotov_hybrid", "adam", "lbfgs"]
OPTIMIZER_LABELS = {
    "krotov_hybrid": "Hybrid Krotov",
    "adam": "Adam",
    "lbfgs": "L-BFGS-B",
}
OPTIMIZER_COLORS = {
    "krotov_hybrid": "#6b46c1",
    "adam": "#2b6cb0",
    "lbfgs": "#2f855a",
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_results():
    results = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*.json"))):
        with open(path) as handle:
            results.append(json.load(handle))
    return results


def group_results_by_model(results):
    grouped = OrderedDict()
    for model_name in MODEL_SPECS:
        grouped[model_name] = OrderedDict()
        for optimizer_name in OPTIMIZER_ORDER:
            grouped[model_name][optimizer_name] = [
                result
                for result in results
                if result["model_name"] == model_name and result["optimizer"] == optimizer_name
            ]
    return grouped


def interp_traces(traces_x, traces_y, n_points=300):
    x_min = max(trace[0] for trace in traces_x)
    x_max = min(trace[-1] for trace in traces_x)
    x_grid = np.linspace(x_min, x_max, n_points)
    y_interp = np.array([np.interp(x_grid, tx, ty) for tx, ty in zip(traces_x, traces_y)])
    return x_grid, y_interp


def save_fig(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"))
    plt.close(fig)


def representative_run(runs):
    losses = np.array([run["final_loss"] for run in runs], dtype=float)
    order = np.argsort(losses)
    return runs[int(order[len(order) // 2])]


def build_model(model_name):
    return MODEL_SPECS[model_name]["builder"]()


def plot_loss_vs_iteration(model_name, grouped_runs):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for optimizer_name in OPTIMIZER_ORDER:
        runs = grouped_runs[optimizer_name]
        traces_x = [np.asarray(run["trace"]["step"], dtype=float) for run in runs]
        traces_y = [np.asarray(run["trace"]["loss"], dtype=float) for run in runs]
        x_grid, y_interp = interp_traces(traces_x, traces_y)
        mean = np.mean(y_interp, axis=0)
        std = np.std(y_interp, axis=0)
        ax.plot(
            x_grid,
            mean,
            color=OPTIMIZER_COLORS[optimizer_name],
            lw=2,
            label=OPTIMIZER_LABELS[optimizer_name],
        )
        ax.fill_between(
            x_grid,
            mean - std,
            mean + std,
            color=OPTIMIZER_COLORS[optimizer_name],
            alpha=0.18,
        )
    ax.set_title(f"{MODEL_SPECS[model_name]['label']}: loss vs iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss (BCE)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, f"alt_{model_name}_loss_vs_iteration")


def plot_decision_boundaries(model_name, grouped_runs, benchmark_config):
    fig, axes = plt.subplots(1, len(OPTIMIZER_ORDER), figsize=(4.6 * len(OPTIMIZER_ORDER), 4.2))
    if len(OPTIMIZER_ORDER) == 1:
        axes = [axes]

    cmap_bg = ListedColormap(["#ffd7d7", "#dbe9ff"])
    spec = MODEL_SPECS[model_name]
    model = build_model(model_name)
    grid_size = 35 if model_name == "chen_sun_vqc" else 90
    x_grid = np.linspace(-np.pi, np.pi, grid_size)
    y_grid = np.linspace(-np.pi, np.pi, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    for ax, optimizer_name in zip(axes, OPTIMIZER_ORDER):
        run = representative_run(grouped_runs[optimizer_name])
        params = np.asarray(run["final_params"], dtype=float)
        probs = model.forward_batch(params, grid_points).reshape(xx.shape)

        ax.contourf(xx, yy, probs, levels=np.linspace(0.0, 1.0, 11), cmap=cmap_bg, alpha=0.72)
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.0)

        X_train, X_test, y_train, y_test = generate_two_moons(
            n_samples=benchmark_config["n_samples"],
            noise=benchmark_config["moon_noise"],
            test_fraction=benchmark_config["test_fraction"],
            seed=run["seed"],
            encoding=benchmark_config["input_encoding"],
        )
        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="#d94841", s=8, alpha=0.4)
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="#3366cc", s=8, alpha=0.4)
        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#8c2d19", s=22, marker="x", alpha=0.8)
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#1d4ed8", s=22, marker="x", alpha=0.8)

        ax.set_title(
            f"{OPTIMIZER_LABELS[optimizer_name]}\n"
            f"loss={run['final_loss']:.3f}, test={run['final_test_acc']:.3f}"
        )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    fig.suptitle(f"{spec['label']}: representative learned decision boundaries", y=1.03)
    plt.tight_layout()
    save_fig(fig, f"alt_{model_name}_decision_boundaries")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    results = load_results()
    grouped = group_results_by_model(results)
    with open(os.path.join(RESULTS_DIR, "config.json")) as handle:
        config = json.load(handle)

    for model_name, grouped_runs in grouped.items():
        plot_loss_vs_iteration(model_name, grouped_runs)
        plot_decision_boundaries(
            model_name,
            grouped_runs,
            config["benchmark_config"][model_name],
        )

    print(f"Alternative-model figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
