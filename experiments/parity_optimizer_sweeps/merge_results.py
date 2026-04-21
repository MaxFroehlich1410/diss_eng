#!/usr/bin/env python3
"""Merge sharded parity optimizer sweep outputs into one combined report."""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.parity_optimizer_sweeps.run import (
    BASELINES,
    OPTIMIZER_COLORS,
    OPTIMIZER_LABELS,
    OPTIMIZER_ORDER,
    SCANNED_HYPERPARAMETERS,
    SWEEP_GRIDS,
)


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")
EXTRA_OPTIMIZER_ORDER = ("lbfgs_native",)
MERGED_OPTIMIZER_ORDER = EXTRA_OPTIMIZER_ORDER + OPTIMIZER_ORDER
MERGED_OPTIMIZER_LABELS = {
    **OPTIMIZER_LABELS,
    "lbfgs_native": "Native L-BFGS",
}
MERGED_OPTIMIZER_COLORS = {
    **OPTIMIZER_COLORS,
    "lbfgs_native": "#111827",
}
MERGED_SCANNED_HYPERPARAMETERS = {
    **SCANNED_HYPERPARAMETERS,
    "lbfgs_native": ["lbfgs_maxcor", "lbfgs_gtol", "max_iterations"],
}
MERGED_SWEEP_GRIDS = {
    **SWEEP_GRIDS,
    "lbfgs_native": OrderedDict(
        [
            ("lbfgs_maxcor", [10, 20]),
            ("lbfgs_gtol", [1e-6, 1e-7]),
        ]
    ),
}
MERGED_BASELINES = {
    **BASELINES,
    "lbfgs_native": {"lbfgs_maxcor": 20, "lbfgs_gtol": 1e-7},
}

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 8.5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Shard result directories containing raw_results.json",
    )
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _timestamped_dir() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_ROOT, f"{timestamp}_merged_summary")
    os.makedirs(path, exist_ok=False)
    return path


def _interpolate_traces(x_traces, y_traces, n_points=300):
    x_min = max(trace[0] for trace in x_traces)
    x_max = min(trace[-1] for trace in x_traces)
    if x_max <= x_min:
        x_grid = np.asarray(x_traces[0], dtype=float)
        y_interp = np.array([np.asarray(trace, dtype=float) for trace in y_traces])
        return x_grid, y_interp
    x_grid = np.linspace(x_min, x_max, n_points)
    y_interp = np.array([np.interp(x_grid, tx, ty) for tx, ty in zip(x_traces, y_traces)])
    return x_grid, y_interp


def _initial_loss_by_seed(results):
    mapping = {}
    for row in results:
        if row["optimizer"] != "krotov_hybrid":
            continue
        steps = list(np.asarray(row["trace"]["step"], dtype=float))
        losses = list(np.asarray(row["trace"]["loss"], dtype=float))
        if steps and steps[0] == 0.0:
            mapping[int(row["seed"])] = float(losses[0])
    return mapping


def _prepend_initial_point_if_missing(row, initial_loss_map):
    steps = list(np.asarray(row["trace"]["step"], dtype=float))
    losses = list(np.asarray(row["trace"]["loss"], dtype=float))
    if steps and steps[0] == 0.0:
        return row

    updated = dict(row)
    updated_trace = dict(row["trace"])
    updated_trace["step"] = [0.0] + steps
    updated_trace["loss"] = [float(initial_loss_map[int(row["seed"])])] + losses
    updated["trace"] = updated_trace
    return updated


def _save_fig(fig, name: str, results_dir: str) -> None:
    fig.savefig(os.path.join(results_dir, f"{name}.pdf"))
    fig.savefig(os.path.join(results_dir, f"{name}.png"))
    plt.close(fig)


def _plot_best_metric(results, best_rows, x_key, y_key, title, xlabel, ylabel, results_dir, file_name):
    initial_loss_map = _initial_loss_by_seed(results)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for optimizer_name in MERGED_OPTIMIZER_ORDER:
        if optimizer_name not in best_rows:
            continue
        hp_key = best_rows[optimizer_name]["hp_key"]
        runs = [run for run in results if run["optimizer"] == optimizer_name and run["hp_key"] == hp_key]
        if x_key == "step" and y_key == "loss":
            runs = [_prepend_initial_point_if_missing(run, initial_loss_map) for run in runs]
        xs = [np.asarray(run["trace"][x_key], dtype=float) for run in runs if run["trace"][x_key]]
        ys = [np.asarray(run["trace"][y_key], dtype=float) for run in runs if run["trace"][y_key]]
        if not xs:
            continue
        x_grid, y_interp = _interpolate_traces(xs, ys)
        mean = np.mean(y_interp, axis=0)
        std = np.std(y_interp, axis=0)
        ax.plot(x_grid, mean, color=MERGED_OPTIMIZER_COLORS[optimizer_name], lw=2.2, label=MERGED_OPTIMIZER_LABELS[optimizer_name])
        ax.fill_between(x_grid, mean - std, mean + std, color=MERGED_OPTIMIZER_COLORS[optimizer_name], alpha=0.16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=2)
    _save_fig(fig, file_name, results_dir)


def _close(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < max(1e-12, 1e-6 * max(abs(float(a)), abs(float(b)), 1.0))
    return a == b


def _matches_config(row, reference, keys):
    return all(_close(row[key], reference.get(key)) for key in keys)


def _summarize_optimizer(optimizer_name: str, results):
    param_names = list(MERGED_SWEEP_GRIDS[optimizer_name].keys())
    grouped = OrderedDict()
    for row in results:
        if row["optimizer"] != optimizer_name:
            continue
        grouped.setdefault(row["hp_key"], []).append(row)

    rows = []
    for hp_key, runs in grouped.items():
        hp = dict(runs[0]["hp"])
        losses = np.asarray([run["final_loss"] for run in runs], dtype=float)
        train_accs = np.asarray([run["final_train_acc"] for run in runs], dtype=float)
        test_accs = np.asarray([run["final_test_acc"] for run in runs], dtype=float)
        walls = np.asarray([run["wall_time_total"] for run in runs], dtype=float)
        rows.append(
            {
                **hp,
                "hp_key": hp_key,
                "n_runs": len(runs),
                "final_loss_mean": float(np.mean(losses)),
                "final_loss_std": float(np.std(losses)),
                "final_train_acc_mean": float(np.mean(train_accs)),
                "final_train_acc_std": float(np.std(train_accs)),
                "final_test_acc_mean": float(np.mean(test_accs)),
                "final_test_acc_std": float(np.std(test_accs)),
                "wall_time_mean": float(np.mean(walls)),
                "wall_time_std": float(np.std(walls)),
            }
        )

    rows.sort(
        key=lambda row: (
            -row["final_test_acc_mean"],
            row["final_loss_mean"],
            row["wall_time_mean"],
        )
    )
    baseline = None
    for row in rows:
        if _matches_config(row, MERGED_BASELINES[optimizer_name], param_names):
            baseline = row
            break
    return {
        "optimizer": optimizer_name,
        "label": MERGED_OPTIMIZER_LABELS[optimizer_name],
        "scanned_hyperparameters": MERGED_SCANNED_HYPERPARAMETERS[optimizer_name],
        "swept_hyperparameters": param_names,
        "n_configs": len(rows),
        "best": rows[0] if rows else None,
        "baseline": baseline,
        "top10": rows[:10],
        "all_configs": rows,
    }


def _write_overview(path: str, summaries, source_dirs):
    lines = [
        "# Parity Optimizer Sweep Overview",
        "",
        "Merged from shard outputs:",
        "",
    ]
    lines.extend([f"- `{src}`" for src in source_dirs])
    lines.extend(
        [
            "",
            "## Best Per Optimizer",
            "",
            "| Optimizer | Swept hyperparameters | Best tested setting | Final test acc | Final loss | Wall time (s) |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for optimizer_name in MERGED_OPTIMIZER_ORDER:
        summary = summaries.get(optimizer_name)
        if not summary or not summary["best"]:
            continue
        best = summary["best"]
        best_setting = ", ".join(f"{name}={best[name]}" for name in summary["swept_hyperparameters"])
        lines.append(
            f"| {summary['label']} | {', '.join(summary['swept_hyperparameters'])} | "
            f"{best_setting} | {best['final_test_acc_mean']:.4f} | {best['final_loss_mean']:.4f} | {best['wall_time_mean']:.2f} |"
        )
    lines.extend(["", "## Hyperparameters Scanned", ""])
    for optimizer_name in MERGED_OPTIMIZER_ORDER:
        summary = summaries.get(optimizer_name)
        if not summary:
            continue
        lines.append(f"### {summary['label']}")
        lines.append("")
        lines.append(f"- Scanned in code: `{', '.join(summary['scanned_hyperparameters'])}`")
        lines.append(f"- Swept in this study: `{', '.join(summary['swept_hyperparameters'])}`")
        if summary["baseline"] is not None and summary["best"] is not None:
            baseline = summary["baseline"]
            best = summary["best"]
            lines.append(
                f"- Baseline: final test acc `{baseline['final_test_acc_mean']:.4f}`, "
                f"final loss `{baseline['final_loss_mean']:.4f}`, wall `{baseline['wall_time_mean']:.2f}s`."
            )
            lines.append(
                f"- Best: final test acc `{best['final_test_acc_mean']:.4f}`, "
                f"final loss `{best['final_loss_mean']:.4f}`, wall `{best['wall_time_mean']:.2f}s`."
            )
        lines.append("")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = _parse_args()
    output_dir = args.output_dir or _timestamped_dir()
    os.makedirs(output_dir, exist_ok=True)

    source_dirs = []
    all_results = []
    for src in args.inputs:
        raw_path = os.path.join(src, "raw_results.json")
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Missing raw_results.json in {src}")
        source_dirs.append(src)
        all_results.extend(_load_json(raw_path))

    summaries = OrderedDict()
    best_rows = {}
    for optimizer_name in MERGED_OPTIMIZER_ORDER:
        summary = _summarize_optimizer(optimizer_name, all_results)
        if summary["best"] is not None:
            summaries[optimizer_name] = summary
            best_rows[optimizer_name] = summary["best"]

    _save_json(os.path.join(output_dir, "raw_results_merged.json"), all_results)
    _save_json(
        os.path.join(output_dir, "analysis_merged.json"),
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "source_dirs": source_dirs,
            "summaries": summaries,
        },
    )

    title = "Best tested configs | 4-bit parity | unique split 10/6"
    _plot_best_metric(
        all_results,
        best_rows,
        "step",
        "loss",
        f"{title}: loss vs iteration",
        "Iteration",
        "Training loss (MSE)",
        output_dir,
        "best_candidates_loss_vs_iteration",
    )
    _plot_best_metric(
        all_results,
        best_rows,
        "wall_time",
        "loss",
        f"{title}: loss vs wall-clock time",
        "Wall-clock time (s)",
        "Training loss (MSE)",
        output_dir,
        "best_candidates_loss_vs_time",
    )
    _write_overview(os.path.join(output_dir, "overview.md"), summaries, source_dirs)
    print(output_dir)


if __name__ == "__main__":
    main()
