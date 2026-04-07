#!/usr/bin/env python3
"""Build report figures for the improved Simonetti/Chen benchmark."""

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
from run_improved_simonetti_chen_benchmark import MODEL_SPECS


RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_improved_simonetti_chen")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report_hybrid")
FIG_DIR = os.path.join(REPORT_DIR, "figures")

OPTIMIZER_ORDER = ["krotov_hybrid", "adam", "lbfgs"]
OPTIMIZER_LABELS = {
    "krotov_hybrid": "Hybrid Krotov",
    "adam": "Adam",
    "lbfgs": "L-BFGS-B",
}
OPTIMIZER_COLORS = {
    "krotov_hybrid": "#8b1e3f",
    "adam": "#2563eb",
    "lbfgs": "#0f766e",
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


def representative_run(runs):
    losses = np.array([run["final_loss"] for run in runs], dtype=float)
    order = np.argsort(losses)
    return runs[int(order[len(order) // 2])]


def interpolate_traces(trace_steps, trace_losses, n_points=300):
    x_min = max(step_trace[0] for step_trace in trace_steps)
    x_max = min(step_trace[-1] for step_trace in trace_steps)
    x_grid = np.linspace(x_min, x_max, n_points)
    y_interp = np.array(
        [np.interp(x_grid, x_vals, y_vals) for x_vals, y_vals in zip(trace_steps, trace_losses)]
    )
    return x_grid, y_interp


def save_figure(fig, stem):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{stem}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"{stem}.png"))
    plt.close(fig)


def build_model(model_name):
    return MODEL_SPECS[model_name]["builder"]()


def plot_loss_vs_iteration(model_name, grouped_runs):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for optimizer_name in OPTIMIZER_ORDER:
        runs = grouped_runs[optimizer_name]
        step_traces = [np.asarray(run["trace"]["step"], dtype=float) for run in runs]
        loss_traces = [np.asarray(run["trace"]["loss"], dtype=float) for run in runs]
        x_grid, y_interp = interpolate_traces(step_traces, loss_traces)
        mean = np.mean(y_interp, axis=0)
        std = np.std(y_interp, axis=0)
        ax.plot(
            x_grid,
            mean,
            color=OPTIMIZER_COLORS[optimizer_name],
            lw=2.2,
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
    save_figure(fig, f"improved_{model_name}_loss_vs_iteration")


def plot_loss_vs_time(model_name, grouped_runs):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for optimizer_name in OPTIMIZER_ORDER:
        runs = grouped_runs[optimizer_name]
        time_traces = [np.asarray(run["trace"]["wall_time"], dtype=float) for run in runs]
        loss_traces = [np.asarray(run["trace"]["loss"], dtype=float) for run in runs]
        x_grid, y_interp = interpolate_traces(time_traces, loss_traces)
        mean = np.mean(y_interp, axis=0)
        std = np.std(y_interp, axis=0)
        ax.plot(
            x_grid,
            mean,
            color=OPTIMIZER_COLORS[optimizer_name],
            lw=2.2,
            label=OPTIMIZER_LABELS[optimizer_name],
        )
        ax.fill_between(
            x_grid,
            mean - std,
            mean + std,
            color=OPTIMIZER_COLORS[optimizer_name],
            alpha=0.18,
        )
    ax.set_title(f"{MODEL_SPECS[model_name]['label']}: loss vs wall-clock time")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Training loss (BCE)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_figure(fig, f"improved_{model_name}_loss_vs_time")


def plot_loss_vs_cost(model_name, grouped_runs):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for optimizer_name in OPTIMIZER_ORDER:
        runs = grouped_runs[optimizer_name]
        cost_traces = [np.asarray(run["trace"]["cost_units"], dtype=float) for run in runs]
        loss_traces = [np.asarray(run["trace"]["loss"], dtype=float) for run in runs]
        x_grid, y_interp = interpolate_traces(cost_traces, loss_traces)
        mean = np.mean(y_interp, axis=0)
        std = np.std(y_interp, axis=0)
        ax.plot(
            x_grid,
            mean,
            color=OPTIMIZER_COLORS[optimizer_name],
            lw=2.2,
            label=OPTIMIZER_LABELS[optimizer_name],
        )
        ax.fill_between(
            x_grid,
            mean - std,
            mean + std,
            color=OPTIMIZER_COLORS[optimizer_name],
            alpha=0.18,
        )
    ax.set_title(f"{MODEL_SPECS[model_name]['label']}: loss vs propagation cost")
    ax.set_xlabel("Cost units (forward + backward passes)")
    ax.set_ylabel("Training loss (BCE)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_figure(fig, f"improved_{model_name}_loss_vs_cost")


def plot_decision_boundaries(model_name, grouped_runs, benchmark_config):
    fig, axes = plt.subplots(1, len(OPTIMIZER_ORDER), figsize=(4.8 * len(OPTIMIZER_ORDER), 4.2))
    if len(OPTIMIZER_ORDER) == 1:
        axes = [axes]

    cmap_bg = ListedColormap(["#fee2e2", "#dbeafe"])
    model = build_model(model_name)
    grid_size = 60 if model_name == "chen_sun_vqc_improved" else 100
    grid_axis = np.linspace(-np.pi, np.pi, grid_size)
    xx, yy = np.meshgrid(grid_axis, grid_axis)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    for ax, optimizer_name in zip(axes, OPTIMIZER_ORDER):
        run = representative_run(grouped_runs[optimizer_name])
        params = np.asarray(run["final_params"], dtype=float)
        probs = model.forward_batch(params, grid_points).reshape(xx.shape)

        ax.contourf(xx, yy, probs, levels=np.linspace(0.0, 1.0, 11), cmap=cmap_bg, alpha=0.78)
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.0)

        X_train, X_test, y_train, y_test = generate_two_moons(
            n_samples=benchmark_config["n_samples"],
            noise=benchmark_config["moon_noise"],
            test_fraction=benchmark_config["test_fraction"],
            seed=run["seed"],
            encoding=benchmark_config["input_encoding"],
        )
        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="#b91c1c", s=8, alpha=0.35)
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="#1d4ed8", s=8, alpha=0.35)
        ax.scatter(
            X_test[y_test == 0, 0],
            X_test[y_test == 0, 1],
            c="#7f1d1d",
            s=22,
            marker="x",
            alpha=0.85,
        )
        ax.scatter(
            X_test[y_test == 1, 0],
            X_test[y_test == 1, 1],
            c="#1e3a8a",
            s=22,
            marker="x",
            alpha=0.85,
        )
        ax.set_title(
            f"{OPTIMIZER_LABELS[optimizer_name]}\n"
            f"loss={run['final_loss']:.3f}, test={run['final_test_acc']:.3f}"
        )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    fig.suptitle(
        f"{MODEL_SPECS[model_name]['label']}: representative learned decision boundaries",
        y=1.03,
    )
    plt.tight_layout()
    save_figure(fig, f"improved_{model_name}_decision_boundaries")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    results = load_results()
    grouped = group_results_by_model(results)
    with open(os.path.join(RESULTS_DIR, "config.json")) as handle:
        config = json.load(handle)

    for model_name, grouped_runs in grouped.items():
        plot_loss_vs_iteration(model_name, grouped_runs)
        plot_loss_vs_time(model_name, grouped_runs)
        plot_loss_vs_cost(model_name, grouped_runs)
        plot_decision_boundaries(model_name, grouped_runs, config["benchmark_config"][model_name])

    print(f"Improved benchmark figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
