#!/usr/bin/env python3
"""Build report figures comparing QNG with Hybrid Krotov, Adam and L-BFGS-B."""

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

from datasets import generate_two_moons
from qml_models.variants import ChenSUNVQCModel, SimonettiHybridModel

EXISTING_DIR = os.path.join(SCRIPT_DIR, "results_improved_simonetti_chen")
QNG_DIR = os.path.join(SCRIPT_DIR, "results_qng_comparison")
FIG_DIR = os.path.join(SCRIPT_DIR, "report_hybrid", "figures")

MODEL_SPECS = OrderedDict(
    [
        (
            "simonetti_full_hybrid",
            {
                "label": "Simonetti Full Hybrid",
                "builder": lambda: SimonettiHybridModel(mode="hybrid"),
                "config": {
                    "input_encoding": "linear_pm_pi",
                    "n_samples": 1000,
                    "moon_noise": 0.05,
                    "test_fraction": 0.2,
                },
            },
        ),
        (
            "chen_sun_vqc_improved",
            {
                "label": "Chen SUN-VQC",
                "builder": lambda: ChenSUNVQCModel(
                    n_macro_layers=2, encoding_axes=("y", "z"), readout="simple_z0"
                ),
                "config": {
                    "input_encoding": "linear_pm_pi",
                    "n_samples": 300,
                    "moon_noise": 0.07,
                    "test_fraction": 0.2,
                },
            },
        ),
    ]
)

OPTIMIZER_ORDER = ["krotov_hybrid", "adam", "lbfgs", "qng"]
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
OPTIMIZER_LINESTYLES = {
    "krotov_hybrid": "-",
    "adam": "--",
    "lbfgs": "-.",
    "qng": "-",
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


def load_all_results():
    results = []
    for directory in (EXISTING_DIR, QNG_DIR):
        if not os.path.isdir(directory):
            continue
        for path in sorted(glob.glob(os.path.join(directory, "result_*.json"))):
            with open(path) as f:
                results.append(json.load(f))
    return results


def group_results(results):
    grouped = OrderedDict()
    for model_name in MODEL_SPECS:
        grouped[model_name] = OrderedDict()
        for opt in OPTIMIZER_ORDER:
            grouped[model_name][opt] = [
                r for r in results
                if r["model_name"] == model_name and r["optimizer"] == opt
            ]
    return grouped


def representative_run(runs):
    losses = np.array([r["final_loss"] for r in runs], dtype=float)
    order = np.argsort(losses)
    return runs[int(order[len(order) // 2])]


def interpolate_traces(step_lists, value_lists, n_points=300):
    x_min = max(s[0] for s in step_lists)
    x_max = min(s[-1] for s in step_lists)
    x_grid = np.linspace(x_min, x_max, n_points)
    y_interp = np.array(
        [np.interp(x_grid, x, y) for x, y in zip(step_lists, value_lists)]
    )
    return x_grid, y_interp


def save_fig(fig, stem):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, f"{stem}.pdf"))
    fig.savefig(os.path.join(FIG_DIR, f"{stem}.png"))
    plt.close(fig)
    print(f"  {stem}.pdf / .png")


def plot_loss_vs_iteration(model_name, grouped_runs):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for opt in OPTIMIZER_ORDER:
        runs = grouped_runs[opt]
        if not runs:
            continue
        steps = [np.asarray(r["trace"]["step"], dtype=float) for r in runs]
        losses = [np.asarray(r["trace"]["loss"], dtype=float) for r in runs]
        x, y = interpolate_traces(steps, losses)
        mean, std = np.mean(y, axis=0), np.std(y, axis=0)
        ax.plot(x, mean, color=OPTIMIZER_COLORS[opt], ls=OPTIMIZER_LINESTYLES[opt],
                lw=2.2, label=OPTIMIZER_LABELS[opt])
        ax.fill_between(x, mean - std, mean + std, color=OPTIMIZER_COLORS[opt], alpha=0.15)

    ax.set_title(f"{MODEL_SPECS[model_name]['label']}: Loss vs Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Loss (BCE)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, f"qng_{model_name}_loss_vs_iteration")


def plot_loss_vs_cost(model_name, grouped_runs):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for opt in OPTIMIZER_ORDER:
        runs = grouped_runs[opt]
        if not runs:
            continue
        costs = [np.asarray(r["trace"]["cost_units"], dtype=float) for r in runs]
        losses = [np.asarray(r["trace"]["loss"], dtype=float) for r in runs]
        x, y = interpolate_traces(costs, losses)
        mean, std = np.mean(y, axis=0), np.std(y, axis=0)
        ax.plot(x, mean, color=OPTIMIZER_COLORS[opt], ls=OPTIMIZER_LINESTYLES[opt],
                lw=2.2, label=OPTIMIZER_LABELS[opt])
        ax.fill_between(x, mean - std, mean + std, color=OPTIMIZER_COLORS[opt], alpha=0.15)

    ax.set_title(f"{MODEL_SPECS[model_name]['label']}: Loss vs Propagation Cost")
    ax.set_xlabel("Cost Units (forward + backward passes)")
    ax.set_ylabel("Training Loss (BCE)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, f"qng_{model_name}_loss_vs_cost")


def plot_loss_vs_time(model_name, grouped_runs):
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for opt in OPTIMIZER_ORDER:
        runs = grouped_runs[opt]
        if not runs:
            continue
        times = [np.asarray(r["trace"]["wall_time"], dtype=float) for r in runs]
        losses = [np.asarray(r["trace"]["loss"], dtype=float) for r in runs]
        x, y = interpolate_traces(times, losses)
        mean, std = np.mean(y, axis=0), np.std(y, axis=0)
        ax.plot(x, mean, color=OPTIMIZER_COLORS[opt], ls=OPTIMIZER_LINESTYLES[opt],
                lw=2.2, label=OPTIMIZER_LABELS[opt])
        ax.fill_between(x, mean - std, mean + std, color=OPTIMIZER_COLORS[opt], alpha=0.15)

    ax.set_title(f"{MODEL_SPECS[model_name]['label']}: Loss vs Wall-Clock Time")
    ax.set_xlabel("Wall-clock time (s)")
    ax.set_ylabel("Training Loss (BCE)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save_fig(fig, f"qng_{model_name}_loss_vs_time")


def plot_decision_boundaries(model_name, grouped_runs):
    model = MODEL_SPECS[model_name]["builder"]()
    cfg = MODEL_SPECS[model_name]["config"]
    n_opt = len([o for o in OPTIMIZER_ORDER if grouped_runs[o]])

    fig, axes = plt.subplots(1, n_opt, figsize=(4.4 * n_opt, 4.2))
    if n_opt == 1:
        axes = [axes]

    cmap_bg = ListedColormap(["#fee2e2", "#dbeafe"])
    grid_size = 60 if model_name == "chen_sun_vqc_improved" else 100
    grid_axis = np.linspace(-np.pi, np.pi, grid_size)
    xx, yy = np.meshgrid(grid_axis, grid_axis)
    grid_pts = np.column_stack([xx.ravel(), yy.ravel()])

    ax_idx = 0
    for opt in OPTIMIZER_ORDER:
        runs = grouped_runs[opt]
        if not runs:
            continue
        ax = axes[ax_idx]
        ax_idx += 1
        run = representative_run(runs)
        params = np.asarray(run["final_params"], dtype=float)
        probs = model.forward_batch(params, grid_pts).reshape(xx.shape)

        ax.contourf(xx, yy, probs, levels=np.linspace(0.0, 1.0, 11),
                     cmap=cmap_bg, alpha=0.78)
        ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=1.0)

        X_train, X_test, y_train, y_test = generate_two_moons(
            n_samples=cfg["n_samples"], noise=cfg["moon_noise"],
            test_fraction=cfg["test_fraction"], seed=run["seed"],
            encoding=cfg["input_encoding"],
        )
        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
                   c="#b91c1c", s=8, alpha=0.35)
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1],
                   c="#1d4ed8", s=8, alpha=0.35)
        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1],
                   c="#7f1d1d", s=22, marker="x", alpha=0.85)
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1],
                   c="#1e3a8a", s=22, marker="x", alpha=0.85)
        ax.set_title(
            f"{OPTIMIZER_LABELS[opt]}\n"
            f"loss={run['final_loss']:.3f}, test={run['final_test_acc']:.3f}"
        )
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    fig.suptitle(
        f"{MODEL_SPECS[model_name]['label']}: Decision Boundaries (median seed)",
        y=1.03,
    )
    plt.tight_layout()
    save_fig(fig, f"qng_{model_name}_decision_boundaries")


def plot_final_accuracy_bar(grouped_all):
    """Grouped bar chart of final test accuracy across models and optimizers."""
    model_names = list(MODEL_SPECS.keys())
    model_labels = [MODEL_SPECS[m]["label"] for m in model_names]
    n_models = len(model_names)
    n_opts = len(OPTIMIZER_ORDER)
    bar_w = 0.18
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, opt in enumerate(OPTIMIZER_ORDER):
        means, stds = [], []
        for m in model_names:
            runs = grouped_all[m][opt]
            if runs:
                accs = np.array([r["final_test_acc"] for r in runs], dtype=float)
                means.append(np.mean(accs))
                stds.append(np.std(accs))
            else:
                means.append(0)
                stds.append(0)
        offset = (i - (n_opts - 1) / 2) * bar_w
        ax.bar(x + offset, means, bar_w, yerr=stds, capsize=3,
               color=OPTIMIZER_COLORS[opt], label=OPTIMIZER_LABELS[opt],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel("Final Test Accuracy")
    ax.set_ylim(0.85, 1.01)
    ax.set_title("Final Test Accuracy by Model and Optimizer")
    ax.legend(frameon=False, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, "qng_comparison_test_accuracy_bar")


def plot_final_loss_bar(grouped_all):
    """Grouped bar chart of final loss across models and optimizers."""
    model_names = list(MODEL_SPECS.keys())
    model_labels = [MODEL_SPECS[m]["label"] for m in model_names]
    n_models = len(model_names)
    n_opts = len(OPTIMIZER_ORDER)
    bar_w = 0.18
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, opt in enumerate(OPTIMIZER_ORDER):
        means, stds = [], []
        for m in model_names:
            runs = grouped_all[m][opt]
            if runs:
                losses = np.array([r["final_loss"] for r in runs], dtype=float)
                means.append(np.mean(losses))
                stds.append(np.std(losses))
            else:
                means.append(0)
                stds.append(0)
        offset = (i - (n_opts - 1) / 2) * bar_w
        ax.bar(x + offset, means, bar_w, yerr=stds, capsize=3,
               color=OPTIMIZER_COLORS[opt], label=OPTIMIZER_LABELS[opt],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel("Final Training Loss (BCE)")
    ax.set_title("Final Training Loss by Model and Optimizer")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, "qng_comparison_final_loss_bar")


def plot_cost_efficiency(grouped_all):
    """Scatter: final loss vs total cost units, one marker per optimizer per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, model_name in zip(axes, MODEL_SPECS):
        spec = MODEL_SPECS[model_name]
        for opt in OPTIMIZER_ORDER:
            runs = grouped_all[model_name][opt]
            if not runs:
                continue
            costs = [r["total_cost_units"] for r in runs]
            losses = [r["final_loss"] for r in runs]
            ax.scatter(costs, losses, c=OPTIMIZER_COLORS[opt], s=60, alpha=0.8,
                       edgecolors="white", linewidths=0.5, label=OPTIMIZER_LABELS[opt],
                       zorder=3)
        ax.set_xlabel("Total Cost Units")
        ax.set_ylabel("Final Loss (BCE)")
        ax.set_title(f"{spec['label']}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Cost-Efficiency: Final Loss vs Total Propagation Cost", y=1.02)
    plt.tight_layout()
    save_fig(fig, "qng_comparison_cost_efficiency")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    results = load_all_results()
    print(f"Loaded {len(results)} result files")
    grouped = group_results(results)

    for model_name in MODEL_SPECS:
        print(f"\n{MODEL_SPECS[model_name]['label']}:")
        plot_loss_vs_iteration(model_name, grouped[model_name])
        plot_loss_vs_time(model_name, grouped[model_name])
        plot_loss_vs_cost(model_name, grouped[model_name])
        plot_decision_boundaries(model_name, grouped[model_name])

    print("\nCross-model comparisons:")
    plot_final_accuracy_bar(grouped)
    plot_final_loss_bar(grouped)
    plot_cost_efficiency(grouped)

    print(f"\nAll figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
