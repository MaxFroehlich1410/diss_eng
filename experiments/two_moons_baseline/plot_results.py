#!/usr/bin/env python3
"""Generate plots for the two-moons QML optimizer benchmark."""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from datasets import generate_two_moons
from qml_models import VQCModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "results", "plots")

COLORS = {
    "krotov_online": "#c94c4c",
    "krotov_batch": "#d17c00",
    "adam": "#2b6cb0",
    "lbfgs": "#2f855a",
}
LABELS = {
    "krotov_online": "Krotov online",
    "krotov_batch": "Krotov batch",
    "adam": "Adam",
    "lbfgs": "L-BFGS-B",
}
MAIN_ORDER = ["krotov_online", "krotov_batch", "adam", "lbfgs"]

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def load_config():
    with open(os.path.join(RESULTS_DIR, "config.json")) as f:
        return json.load(f)


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


def group_results(results, use_optimizer_name=False):
    grouped = {}
    for result in results:
        key = result["optimizer_name"] if use_optimizer_name else result["optimizer"]
        grouped.setdefault(key, []).append(result)
    return grouped


def _ordered_groups(grouped):
    ordered = {}
    for key in MAIN_ORDER:
        if key in grouped:
            ordered[key] = grouped[key]
    for key in grouped:
        if key not in ordered:
            ordered[key] = grouped[key]
    return ordered


def _interp_traces(traces_x, traces_y, n_points=500):
    x_min = max(trace[0] for trace in traces_x)
    x_max = min(trace[-1] for trace in traces_x)
    x_grid = np.linspace(x_min, x_max, n_points)
    y_interp = np.array([np.interp(x_grid, tx, ty) for tx, ty in zip(traces_x, traces_y)])
    return x_grid, y_interp


def _plot_mean_with_band(ax, runs, x_key, y_key, title, xlabel):
    grouped = _ordered_groups(group_results(runs))
    for opt_name, opt_runs in grouped.items():
        traces_x = [run["trace"][x_key] for run in opt_runs]
        traces_y = [run["trace"][y_key] for run in opt_runs]
        x_grid, y_arr = _interp_traces(traces_x, traces_y)
        mean = np.mean(y_arr, axis=0)
        std = np.std(y_arr, axis=0)
        color = COLORS.get(opt_name, "#444444")
        label = LABELS.get(opt_name, opt_name)
        ax.plot(x_grid, mean, color=color, label=label, lw=2)
        ax.fill_between(x_grid, mean - std, mean + std, color=color, alpha=0.15)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Training loss (BCE)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_loss_vs_cost(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_mean_with_band(
        ax,
        results,
        "cost_units",
        "loss",
        "Training loss vs fair propagation cost",
        "Propagation cost = sample forwards + sample backwards",
    )
    _save(fig, "loss_vs_cost")


def plot_loss_vs_step(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_mean_with_band(ax, results, "step", "loss", "Training loss vs optimization step", "Optimization step")
    _save(fig, "loss_vs_step")


def plot_loss_vs_time(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_mean_with_band(ax, results, "wall_time", "loss", "Training loss vs wall-clock time", "Wall-clock time (s)")
    _save(fig, "loss_vs_time")


def plot_boxplot_loss(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    grouped = _ordered_groups(group_results(results))
    data = [[run["final_loss"] for run in runs] for runs in grouped.values()]
    labels = [LABELS.get(name, name) for name in grouped]
    colors = [COLORS.get(name, "#888888") for name in grouped]
    boxplot = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Final training loss (BCE)")
    ax.set_title("Final training loss by optimizer")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "boxplot_final_loss")


def plot_boxplot_accuracy(results):
    fig, ax = plt.subplots(figsize=(7, 5))
    grouped = _ordered_groups(group_results(results))
    data = [[run["final_test_acc"] for run in runs] for runs in grouped.values()]
    labels = [LABELS.get(name, name) for name in grouped]
    colors = [COLORS.get(name, "#888888") for name in grouped]
    boxplot = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Final test accuracy")
    ax.set_title("Final test accuracy by optimizer")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "boxplot_final_test_accuracy")


def plot_success_rate(results, threshold):
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = _ordered_groups(group_results(results))
    for opt_name, opt_runs in grouped.items():
        traces_x = [run["trace"]["cost_units"] for run in opt_runs]
        traces_y = [run["trace"]["loss"] for run in opt_runs]
        x_grid, y_arr = _interp_traces(traces_x, traces_y)
        success_frac = np.mean(y_arr <= threshold, axis=0)
        ax.plot(
            x_grid,
            success_frac,
            color=COLORS.get(opt_name, "#444444"),
            label=LABELS.get(opt_name, opt_name),
            lw=2,
        )
    ax.set_xlabel("Propagation cost = sample forwards + sample backwards")
    ax.set_ylabel(f"Fraction of runs with loss < {threshold}")
    ax.set_title("Success rate vs fair propagation cost")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "success_rate_vs_cost")


def plot_krotov_step_size(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = _ordered_groups(group_results(results))
    for opt_name in ("krotov_online", "krotov_batch"):
        if opt_name not in grouped:
            continue
        traces_x = [run["trace"]["step"] for run in grouped[opt_name]]
        traces_y = [run["trace"]["step_size"] for run in grouped[opt_name]]
        x_grid, y_arr = _interp_traces(traces_x, traces_y)
        ax.plot(x_grid, np.mean(y_arr, axis=0), color=COLORS[opt_name], label=LABELS[opt_name], lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Step size")
    ax.set_title("Krotov step size vs iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "krotov_step_size_vs_iteration")


def plot_krotov_update_norm(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = _ordered_groups(group_results(results))
    for opt_name in ("krotov_online", "krotov_batch"):
        if opt_name not in grouped:
            continue
        traces_x = [run["trace"]["step"] for run in grouped[opt_name]]
        traces_y = [run["trace"]["update_norm"] for run in grouped[opt_name]]
        x_grid, y_arr = _interp_traces(traces_x, traces_y)
        mean = np.mean(y_arr, axis=0)
        std = np.std(y_arr, axis=0)
        ax.plot(x_grid, mean, color=COLORS[opt_name], label=LABELS[opt_name], lw=2)
        ax.fill_between(x_grid, mean - std, mean + std, color=COLORS[opt_name], alpha=0.15)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Update norm")
    ax.set_title("Krotov update norm vs iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "krotov_update_norm_vs_iteration")


def plot_krotov_loss_comparison(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = _ordered_groups(group_results(results))
    for opt_name in ("krotov_online", "krotov_batch"):
        if opt_name not in grouped:
            continue
        traces_x = [run["trace"]["step"] for run in grouped[opt_name]]
        traces_y = [run["trace"]["loss"] for run in grouped[opt_name]]
        x_grid, y_arr = _interp_traces(traces_x, traces_y)
        mean = np.mean(y_arr, axis=0)
        std = np.std(y_arr, axis=0)
        ax.plot(x_grid, mean, color=COLORS[opt_name], label=LABELS[opt_name], lw=2)
        ax.fill_between(x_grid, mean - std, mean + std, color=COLORS[opt_name], alpha=0.15)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss (BCE)")
    ax.set_title("Krotov online vs batch loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "krotov_loss_vs_iteration")


def plot_krotov_variance(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    grouped = _ordered_groups(group_results(results))
    for opt_name in ("krotov_online", "krotov_batch"):
        if opt_name not in grouped:
            continue
        traces_x = [run["trace"]["step"] for run in grouped[opt_name]]
        traces_y = [run["trace"]["contribution_variance"] for run in grouped[opt_name]]
        x_grid, y_arr = _interp_traces(traces_x, traces_y)
        ax.plot(x_grid, np.mean(y_arr, axis=0), color=COLORS[opt_name], label=LABELS[opt_name], lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean per-parameter contribution variance")
    ax.set_title("Krotov contribution variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, "krotov_contribution_variance")


def plot_decision_boundaries(results, config):
    grouped = _ordered_groups(group_results(results))
    present = [name for name in MAIN_ORDER if name in grouped]
    if not present:
        return

    fig, axes = plt.subplots(1, len(present), figsize=(5 * len(present), 4.5))
    if len(present) == 1:
        axes = [axes]

    for ax, opt_name in zip(axes, present):
        runs = grouped[opt_name]
        losses = [run["final_loss"] for run in runs]
        median_idx = np.argsort(losses)[len(losses) // 2]
        run = runs[median_idx]

        X_train, X_test, y_train, y_test = generate_two_moons(
            n_samples=config["n_samples"],
            noise=config["moon_noise"],
            test_fraction=config["test_fraction"],
            seed=run["seed"],
        )
        model = VQCModel(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            entangler=config["entangler"],
        )
        params = np.array(run["final_params"])

        x1_range = np.linspace(0.0, np.pi, 80)
        x2_range = np.linspace(0.0, np.pi, 80)
        xx1, xx2 = np.meshgrid(x1_range, x2_range)
        grid = np.c_[xx1.ravel(), xx2.ravel()]
        probs = model.forward_batch(params, grid).reshape(xx1.shape)

        cmap_bg = ListedColormap(["#ffd7d7", "#dbe9ff"])
        ax.contourf(xx1, xx2, probs, levels=[0.0, 0.5, 1.0], cmap=cmap_bg, alpha=0.6)
        ax.contour(xx1, xx2, probs, levels=[0.5], colors="k", linewidths=1.2)
        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="#d94841", s=9, alpha=0.55)
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="#3366cc", s=9, alpha=0.55)
        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#8c2d19", s=18, marker="x", alpha=0.7)
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#1d4ed8", s=18, marker="x", alpha=0.7)
        ax.set_title(f"{LABELS.get(opt_name, opt_name)}\n(loss={run['final_loss']:.3f})")
        ax.set_xlabel("$x_1$ (encoded)")
        ax.set_ylabel("$x_2$ (encoded)")

    fig.suptitle("Decision boundaries for median-loss runs", y=1.02)
    plt.tight_layout()
    _save(fig, "decision_boundaries")


def plot_krotov_sweep(sweep_results):
    if not sweep_results:
        return

    grouped = group_results(sweep_results)
    ordered_names = sorted(grouped)

    fig, ax = plt.subplots(figsize=(max(8, len(ordered_names) * 1.25), 5))
    data = [[run["final_loss"] for run in grouped[name]] for name in ordered_names]
    boxplot = ax.boxplot(data, tick_labels=ordered_names, patch_artist=True, widths=0.55)
    for patch in boxplot["boxes"]:
        patch.set_facecolor("#d17c00")
        patch.set_alpha(0.45)
    ax.set_ylabel("Final training loss (BCE)")
    ax.set_title("Krotov batch hyperparameter sweep")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, "krotov_batch_sweep_final_loss")


def _save(fig, name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.png"))
    fig.savefig(os.path.join(PLOTS_DIR, f"{name}.pdf"))
    plt.close(fig)
    print(f"  Saved {name}.png/.pdf")


def main():
    config = load_config()
    results = load_results(config)
    if not results:
        print("No results found. Run run_experiment.py first.")
        return

    main_results = [result for result in results if not result.get("is_sweep", False)]
    sweep_results = [result for result in results if result.get("is_sweep", False)]

    print(f"Loaded {len(results)} result files")
    print(f"Main comparison runs: {len(main_results)}")
    print(f"Sweep runs: {len(sweep_results)}")

    plot_loss_vs_cost(main_results)
    plot_loss_vs_time(main_results)
    plot_loss_vs_step(main_results)
    plot_boxplot_loss(main_results)
    plot_boxplot_accuracy(main_results)
    plot_success_rate(main_results, config["loss_threshold"])
    plot_krotov_step_size(main_results)
    plot_krotov_update_norm(main_results)
    plot_krotov_loss_comparison(main_results)
    plot_krotov_variance(main_results)
    plot_decision_boundaries(main_results, config)
    plot_krotov_sweep(sweep_results)

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
