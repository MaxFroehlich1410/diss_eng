#!/usr/bin/env python3
"""Plot the best parity methods across the default and custom PennyLane studies."""

from __future__ import annotations

import json
import os
from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_DIR = os.path.join(
    SCRIPT_DIR,
    "results",
    "20260416_232247_parity_vs_all_applicable_pennylane_optimizers",
)
DEFAULT_DIR = os.path.join(
    SCRIPT_DIR,
    "..",
    "parity_pennylane_defaults_comparison",
    "results",
    "20260417_002147_parity_pennylane_defaults_full",
)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results_best_methods")

COLORS = {
    "krotov_hybrid": "#8b1e3f",
    "pennylane_rmsprop": "#0f766e",
    "pennylane_rotosolve": "#15803d",
}
BASE_LABELS = {
    "krotov_hybrid": "Hybrid Krotov",
    "pennylane_rmsprop": "PennyLane RMSProp",
    "pennylane_rotosolve": "PennyLane Rotosolve",
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


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _interpolate_traces(x_traces, y_traces, n_points=300):
    x_min = max(trace[0] for trace in x_traces)
    x_max = min(trace[-1] for trace in x_traces)
    x_grid = np.linspace(x_min, x_max, n_points)
    y_interp = np.array([np.interp(x_grid, tx, ty) for tx, ty in zip(x_traces, y_traces)])
    return x_grid, y_interp


def _save_fig(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"))
    plt.close(fig)


def _load_experiment(results_dir, tag):
    summary = _load_json(os.path.join(results_dir, "summary.json"))
    runs = {}
    for name in summary:
        result_path = os.path.join(results_dir, f"result_{name}_seed0.json")
        if os.path.exists(result_path):
            runs[name] = _load_json(result_path)
    return {"tag": tag, "summary": summary, "runs": runs, "results_dir": results_dir}


def _select_best_variants(custom_exp, default_exp):
    selected = OrderedDict()
    candidates = set(custom_exp["summary"]) | set(default_exp["summary"])
    for name in sorted(candidates):
        custom_stats = custom_exp["summary"].get(name)
        default_stats = default_exp["summary"].get(name)
        options = []
        if custom_stats and name in custom_exp["runs"]:
            options.append(("custom", custom_stats, custom_exp["runs"][name]))
        if default_stats and name in default_exp["runs"]:
            options.append(("default", default_stats, default_exp["runs"][name]))
        if not options:
            continue

        max_acc = max(item[1]["final_test_acc_mean"] for item in options)
        if max_acc < 0.999:
            continue

        best_variant = min(
            [item for item in options if item[1]["final_test_acc_mean"] == max_acc],
            key=lambda item: (item[1]["final_loss_mean"], item[1]["wall_time_mean"]),
        )
        selected[name] = {
            "variant": best_variant[0],
            "stats": best_variant[1],
            "run": best_variant[2],
        }
    return selected


def _plot_metric(selected, x_key, y_key, title, xlabel, ylabel, file_name):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for name, payload in selected.items():
        trace = payload["run"]["trace"]
        x = np.asarray(trace[x_key], dtype=float)
        y = np.asarray(trace[y_key], dtype=float)
        label = f"{BASE_LABELS[name]} ({payload['variant']} better)"
        ax.plot(x, y, color=COLORS[name], lw=2.4, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="best")
    _save_fig(fig, file_name)


def _write_summary(selected):
    payload = OrderedDict()
    for name, info in selected.items():
        payload[name] = {
            "label": BASE_LABELS[name],
            "better_variant": info["variant"],
            "final_loss_mean": info["stats"]["final_loss_mean"],
            "final_test_acc_mean": info["stats"]["final_test_acc_mean"],
            "wall_time_mean": info["stats"]["wall_time_mean"],
        }
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    custom_exp = _load_experiment(CUSTOM_DIR, "custom")
    default_exp = _load_experiment(DEFAULT_DIR, "default")
    selected = _select_best_variants(custom_exp, default_exp)

    _plot_metric(
        selected,
        "step",
        "loss",
        "Best Parity Methods: loss vs iteration",
        "Iteration",
        "Training loss (MSE)",
        "best_methods_loss_vs_iteration",
    )
    _plot_metric(
        selected,
        "wall_time",
        "loss",
        "Best Parity Methods: loss vs wall-clock time",
        "Wall-clock time (s)",
        "Training loss (MSE)",
        "best_methods_loss_vs_time",
    )
    _write_summary(selected)


if __name__ == "__main__":
    main()
