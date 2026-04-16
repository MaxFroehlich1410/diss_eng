#!/usr/bin/env python3
"""Build summary figures for the completed Perez-Salinas optimizer sweeps."""

from __future__ import annotations

import argparse
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
if __package__ in {None, ""}:
    from qml_models.variants import PerezSalinasReuploadingModel
    from perez_salinas_dataset import generate_perez_salinas_dataset, perez_salinas_problem_num_classes
else:
    from qml_models.variants import PerezSalinasReuploadingModel
    from datasets import (
        generate_perez_salinas_dataset,
        perez_salinas_problem_num_classes,
    )


RESULTS_BASE = os.path.join(SCRIPT_DIR, "results_perez_salinas_sweeps")
FIG_DIR = os.path.join(RESULTS_BASE, "figures")
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


def _close(a, b):
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < max(1e-12, 1e-6 * max(abs(float(a)), abs(float(b)), 1.0))
    return a == b


def _matches_config(row, reference, keys):
    return all(_close(row[key], reference[key]) for key in keys)


def _interp_traces(traces_x, traces_y, n_points=500):
    x_min = max(trace[0] for trace in traces_x)
    x_max = min(trace[-1] for trace in traces_x)
    x_grid = np.linspace(x_min, x_max, n_points)
    y_interp = np.array([np.interp(x_grid, tx, ty) for tx, ty in zip(traces_x, traces_y)])
    return x_grid, y_interp


def _representative_run(runs):
    ordered = sorted(runs, key=lambda run: run["final_loss"])
    return ordered[len(ordered) // 2]


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    png_path = os.path.join(FIG_DIR, f"{name}.png")
    pdf_path = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")


def load_best_runs(results_base):
    grouped = OrderedDict()
    benchmark_config = None

    for optimizer_name in OPTIMIZER_ORDER:
        analysis_path = os.path.join(results_base, optimizer_name, f"analysis_{optimizer_name}.json")
        raw_path = os.path.join(results_base, optimizer_name, f"sweep_{optimizer_name}.json")
        with open(analysis_path, encoding="utf-8") as handle:
            analysis = json.load(handle)
        with open(raw_path, encoding="utf-8") as handle:
            raw_runs = json.load(handle)

        param_names = analysis["param_names"]
        best = analysis["best"]
        selected = [
            run for run in raw_runs if _matches_config(run, best, param_names)
        ]
        if not selected:
            raise RuntimeError(f"No runs matched best config for {optimizer_name}.")
        grouped[optimizer_name] = selected

        if benchmark_config is None:
            args = analysis["args"]
            benchmark_config = {
                "problem": args["problem"],
                "n_qubits": args["n_qubits"],
                "use_entanglement": not args["no_entanglement"],
                "n_samples": args["n_samples"],
                "test_fraction": args["test_fraction"],
                "max_iterations": args["max_iterations"],
            }

    return grouped, benchmark_config


def plot_mean_band(grouped_runs, x_key, y_key, title, xlabel, out_name):
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    for optimizer_name in OPTIMIZER_ORDER:
        runs = grouped_runs[optimizer_name]
        traces_x = [np.asarray(run["trace"][x_key], dtype=float) for run in runs]
        traces_y = [np.asarray(run["trace"][y_key], dtype=float) for run in runs]
        x_grid, y_arr = _interp_traces(traces_x, traces_y)
        mean = np.mean(y_arr, axis=0)
        std = np.std(y_arr, axis=0)
        color = OPTIMIZER_COLORS[optimizer_name]
        label = OPTIMIZER_LABELS[optimizer_name]
        ax.plot(x_grid, mean, color=color, label=label, lw=2.2)
        ax.fill_between(x_grid, mean - std, mean + std, color=color, alpha=0.16)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Training loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.28)
    _save(fig, out_name)


def _score_margin(model, params, x):
    scores = model.class_scores(params, x)
    return float(scores[1] - scores[0])


def plot_decision_boundaries(grouped_runs, benchmark_config, out_name):
    fig, axes = plt.subplots(1, len(OPTIMIZER_ORDER), figsize=(5.0 * len(OPTIMIZER_ORDER), 4.2))
    cmap_bg = ListedColormap(["#f7d6d6", "#dbe8ff"])

    x_grid = np.linspace(-1.0, 1.0, 80)
    y_grid = np.linspace(-1.0, 1.0, 80)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    for ax, optimizer_name in zip(axes, OPTIMIZER_ORDER):
        run = _representative_run(grouped_runs[optimizer_name])
        model = PerezSalinasReuploadingModel(
            n_qubits=benchmark_config["n_qubits"],
            n_layers=int(run["n_layers"]),
            n_classes=perez_salinas_problem_num_classes(benchmark_config["problem"]),
            use_entanglement=benchmark_config["use_entanglement"],
            loss_mode="weighted_fidelity",
        )
        params = np.asarray(run["final_params"], dtype=float)

        margins = np.array([_score_margin(model, params, point) for point in grid_points], dtype=float)
        preds = (margins >= 0.0).astype(float).reshape(xx.shape)
        margin_grid = margins.reshape(xx.shape)

        ax.contourf(xx, yy, preds, levels=[-0.5, 0.5, 1.5], cmap=cmap_bg, alpha=0.75)
        ax.contour(xx, yy, margin_grid, levels=[0.0], colors="black", linewidths=1.1)

        X_train, X_test, y_train, y_test = generate_perez_salinas_dataset(
            problem=benchmark_config["problem"],
            n_samples=benchmark_config["n_samples"],
            test_fraction=benchmark_config["test_fraction"],
            seed=run["seed"],
        )
        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="#c44e52", s=10, alpha=0.55)
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="#4c72b0", s=10, alpha=0.55)
        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#7f1d1d", s=18, marker="x", alpha=0.75)
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#1d4ed8", s=18, marker="x", alpha=0.75)
        ax.set_title(
            f"{OPTIMIZER_LABELS[optimizer_name]}\n"
            f"loss={run['final_loss']:.3f}, test acc={run['final_test_acc']:.3f}"
        )
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-1.0, 1.0)

    fig.suptitle("Perez-Salinas crown: representative decision boundaries", y=1.02)
    plt.tight_layout()
    _save(fig, out_name)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-base", default=RESULTS_BASE)
    return parser.parse_args()


def main():
    args = parse_args()
    grouped_runs, benchmark_config = load_best_runs(args.results_base)

    plot_mean_band(
        grouped_runs,
        "cost_units",
        "loss",
        "Perez-Salinas crown benchmark: loss vs propagation cost",
        "Cost units (forward + backward passes)",
        "loss_vs_cost",
    )
    plot_mean_band(
        grouped_runs,
        "wall_time",
        "loss",
        "Perez-Salinas crown benchmark: loss vs wall-clock time",
        "Wall time (s)",
        "loss_vs_time",
    )
    plot_decision_boundaries(grouped_runs, benchmark_config, "decision_boundaries")


if __name__ == "__main__":
    main()
