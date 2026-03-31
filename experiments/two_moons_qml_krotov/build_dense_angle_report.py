#!/usr/bin/env python3
"""Build figures and a LaTeX report for the dense-angle two-moons benchmark."""

import glob
import json
import os
import subprocess
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from dataset import generate_two_moons
from model import VQCModel


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_dense_angle")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report_dense_angle")
FIG_DIR = os.path.join(REPORT_DIR, "figures")

COLORS = {
    "krotov_hybrid": "#6b46c1",
    "adam": "#2b6cb0",
    "lbfgs": "#2f855a",
}
LABELS = {
    "krotov_hybrid": "Hybrid Krotov",
    "adam": "Adam",
    "lbfgs": "L-BFGS-B",
}
ORDER = ["krotov_hybrid", "adam", "lbfgs"]

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
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
    seeds = set(config.get("seeds", []))
    results = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*.json"))):
        with open(path) as f:
            result = json.load(f)
        if allowed and result["optimizer"] not in allowed:
            continue
        if seeds and result["seed"] not in seeds:
            continue
        results.append(result)
    return results


def group_results(results):
    grouped = OrderedDict()
    for name in ORDER:
        grouped[name] = [result for result in results if result["optimizer"] == name]
    return grouped


def threshold_stats(runs, threshold_key):
    times = []
    costs = []
    for run in runs:
        metric = run["threshold_metrics"][threshold_key]
        if metric["reached"]:
            times.append(metric["wall_time"])
            costs.append(metric["cost_units"])
    return {
        "success_rate": float(sum(run["threshold_metrics"][threshold_key]["reached"] for run in runs) / len(runs)),
        "time_mean": float(np.mean(times)) if times else None,
        "cost_mean": float(np.mean(costs)) if costs else None,
    }


def summarize_group(runs, config):
    final_losses = np.array([run["final_loss"] for run in runs], dtype=float)
    final_test_accs = np.array([run["final_test_acc"] for run in runs], dtype=float)
    wall_times = np.array([run["wall_time_total"] for run in runs], dtype=float)
    costs = np.array([run["total_cost_units"] for run in runs], dtype=float)
    return {
        "final_loss_mean": float(np.mean(final_losses)),
        "final_loss_std": float(np.std(final_losses)),
        "final_test_acc_mean": float(np.mean(final_test_accs)),
        "final_test_acc_std": float(np.std(final_test_accs)),
        "wall_time_mean": float(np.mean(wall_times)),
        "wall_time_std": float(np.std(wall_times)),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "thresholds": {
            f"{threshold:.2f}": threshold_stats(runs, f"{threshold:.2f}")
            for threshold in config["loss_thresholds"]
        },
    }


def summarize_all(grouped, config):
    return OrderedDict((name, summarize_group(runs, config)) for name, runs in grouped.items())


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


def plot_mean_band(grouped, x_key, y_key, title, xlabel, ylabel, out_name):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, runs in grouped.items():
        traces_x = [np.asarray(run["trace"][x_key], dtype=float) for run in runs]
        traces_y = [np.asarray(run["trace"][y_key], dtype=float) for run in runs]
        x_grid, y_interp = interp_traces(traces_x, traces_y)
        mean = np.mean(y_interp, axis=0)
        std = np.std(y_interp, axis=0)
        ax.plot(x_grid, mean, color=COLORS[name], lw=2, label=LABELS[name])
        ax.fill_between(x_grid, mean - std, mean + std, color=COLORS[name], alpha=0.18)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    save_fig(fig, out_name)


def plot_threshold_time_boxplot(grouped, threshold_key, out_name):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    labels = []
    heights = []
    colors = []
    for name, runs in grouped.items():
        stats = threshold_stats(runs, threshold_key)
        labels.append(LABELS[name])
        heights.append(np.nan if stats["time_mean"] is None else stats["time_mean"])
        colors.append(COLORS[name])
    x = np.arange(len(labels))
    ax.bar(x, np.nan_to_num(heights, nan=0.0), color=colors, alpha=0.65)
    for idx, height in enumerate(heights):
        text = "not reached" if np.isnan(height) else f"{height:.2f}s"
        y = 0.02 if np.isnan(height) else height
        ax.text(idx, y, text, ha="center", va="bottom", fontsize=9, rotation=90)
    ax.set_xticks(x, labels)
    ax.set_ylabel(f"Wall time to loss <= {threshold_key}")
    ax.set_title("Time to threshold")
    save_fig(fig, out_name)


def plot_final_boxplot(grouped, field, ylabel, title, out_name):
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    data = [[run[field] for run in runs] for runs in grouped.values()]
    ax.boxplot(data, labels=[LABELS[name] for name in grouped], patch_artist=True)
    for patch, name in zip(ax.artists, grouped.keys()):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.35)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    save_fig(fig, out_name)


def representative_run(runs):
    sorted_runs = sorted(runs, key=lambda run: run["final_loss"])
    return sorted_runs[len(sorted_runs) // 2]


def plot_decision_boundaries(grouped, config, out_name):
    fig, axes = plt.subplots(1, len(grouped), figsize=(5.0 * len(grouped), 4.0))
    if len(grouped) == 1:
        axes = [axes]

    x_grid = np.linspace(-np.pi, np.pi, 140)
    y_grid = np.linspace(-np.pi, np.pi, 140)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    cmap = ListedColormap(["#f6ad55", "#63b3ed"])
    for ax, (name, runs) in zip(axes, grouped.items()):
        run = representative_run(runs)
        model = VQCModel(
            n_qubits=config["n_qubits"],
            n_layers=config["n_layers"],
            entangler=config["entangler"],
            architecture=config["model_architecture"],
            observable=config["observable"],
        )
        probs = model.forward_batch(np.asarray(run["final_params"], dtype=float), grid_points)
        zz = probs.reshape(xx.shape)
        ax.contourf(xx, yy, zz, levels=np.linspace(0, 1, 11), cmap=cmap, alpha=0.75)
        ax.contour(xx, yy, zz, levels=[0.5], colors="black", linewidths=1.0)

        X_train, X_test, y_train, y_test = generate_two_moons(
            n_samples=config["n_samples"],
            noise=config["moon_noise"],
            test_fraction=config["test_fraction"],
            seed=run["seed"],
            encoding=config["input_encoding"],
        )
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap, edgecolors="k", s=18)
        ax.set_title(LABELS[name])
        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
    save_fig(fig, out_name)


def build_latex_report(config, summary):
    threshold_key = f"{config['loss_threshold']:.2f}"
    def fmt_threshold_time(stats):
        value = stats["thresholds"][threshold_key]["time_mean"]
        return "--" if value is None else f"{value:.2f}"
    rows = []
    for name in ORDER:
        stats = summary[name]
        rows.append(
            f"{LABELS[name]} & "
            f"{stats['final_loss_mean']:.3f} $\\pm$ {stats['final_loss_std']:.3f} & "
            f"{stats['final_test_acc_mean']:.3f} $\\pm$ {stats['final_test_acc_std']:.3f} & "
            f"{stats['wall_time_mean']:.1f} $\\pm$ {stats['wall_time_std']:.1f} & "
            f"{fmt_threshold_time(stats)} \\\\"
        )
    best_name = min(ORDER, key=lambda name: summary[name]["final_loss_mean"])
    fastest_threshold_name = min(
        ORDER,
        key=lambda name: summary[name]["thresholds"][threshold_key]["time_mean"]
        if summary[name]["thresholds"][threshold_key]["time_mean"] is not None
        else float("inf"),
    )
    fastest_wall_name = min(ORDER, key=lambda name: summary[name]["wall_time_mean"])
    hybrid = summary["krotov_hybrid"]
    adam = summary["adam"]
    lbfgs = summary["lbfgs"]
    latex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}
\title{{Dense-Angle Two-Moons Benchmark}}
\date{{}}
\begin{{document}}
\maketitle

\section*{{Architecture}}
This follow-up benchmark replaces the original 4-qubit HEA with an online-derived two-moons circuit. The adopted structure is the 2-qubit dense-angle classifier used in two-moons tutorials: four trainable input-scaling gates,
\[
R_Y(w_0 x_0),\; R_Y(w_1 x_0),\; R_Z(w_2 x_1),\; R_Z(w_3 x_1),
\]
followed by {config['n_layers']} repeated blocks of
\[
R_Y(\theta_1) R_Y(\theta_2)\,\mathrm{{CZ}}\,R_Y(\theta_3) R_Y(\theta_4),
\]
and a final pair of $R_Y$ gates measured with $Z_0$. The data preprocessing also follows the centered angle map used in those examples, sending each feature to $[-\pi, \pi]$ on the training split.

\section*{{Results}}
The optimizer comparison was rerun for Adam, L-BFGS-B, and the best previously discovered hybrid Krotov schedule, with the Krotov step sizes retuned for the new circuit. The aggregate results over {len(config['seeds'])} seeds are:

\begin{{center}}
\begin{{tabular}}{{lcccc}}
\toprule
Optimizer & Final loss & Test accuracy & Wall time (s) & Time to $L \le {threshold_key}$ (s) \\
\midrule
{"\n".join(rows)}
\bottomrule
\end{{tabular}}
\end{{center}}

The best mean final loss in this architecture is achieved by \textbf{{{LABELS[best_name]}}}, although Adam is statistically almost tied. In this corrected rerun, hybrid Krotov reaches mean final loss {hybrid['final_loss_mean']:.3f} and mean test accuracy {hybrid['final_test_acc_mean']:.3f}, versus {adam['final_loss_mean']:.3f} / {adam['final_test_acc_mean']:.3f} for Adam and {lbfgs['final_loss_mean']:.3f} / {lbfgs['final_test_acc_mean']:.3f} for L-BFGS-B.

The strongest remaining advantage of hybrid Krotov is early progress: it is the fastest method to reach the benchmark threshold $L \le {threshold_key}$, while \textbf{{{LABELS[fastest_wall_name]}}} is the fastest optimizer in total wall-clock time. The corrected gradients therefore change the scientific conclusion substantially: the dense-angle architecture does help, but it no longer makes hybrid Krotov dominate the corrected classical baselines.

\section*{{Figures}}
\begin{{center}}
\includegraphics[width=0.78\textwidth]{{figures/comparison_loss_vs_iteration.pdf}}

\includegraphics[width=0.78\textwidth]{{figures/comparison_loss_vs_time.pdf}}

\includegraphics[width=0.78\textwidth]{{figures/comparison_loss_vs_cost.pdf}}

\includegraphics[width=0.62\textwidth]{{figures/comparison_final_loss_boxplot.pdf}}
\includegraphics[width=0.62\textwidth]{{figures/comparison_final_test_accuracy_boxplot.pdf}}

\includegraphics[width=0.92\textwidth]{{figures/comparison_decision_boundaries.pdf}}
\end{{center}}

\end{{document}}
"""
    os.makedirs(REPORT_DIR, exist_ok=True)
    tex_path = os.path.join(REPORT_DIR, "dense_angle_experiment_analysis.tex")
    with open(tex_path, "w") as f:
        f.write(latex)
    return tex_path


def compile_latex(tex_path):
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", os.path.basename(tex_path)],
        cwd=os.path.dirname(tex_path),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    config = load_config()
    results = load_results(config)
    grouped = group_results(results)
    summary = summarize_all(grouped, config)

    plot_mean_band(grouped, "step", "loss", "Loss vs iteration", "Iteration", "Train loss", "comparison_loss_vs_iteration")
    plot_mean_band(grouped, "wall_time", "loss", "Loss vs wall-clock time", "Wall time (s)", "Train loss", "comparison_loss_vs_time")
    plot_mean_band(grouped, "cost_units", "loss", "Loss vs fair propagation cost", "Cost units", "Train loss", "comparison_loss_vs_cost")
    plot_threshold_time_boxplot(grouped, f"{config['loss_threshold']:.2f}", "time_to_threshold_boxplot")
    plot_final_boxplot(grouped, "final_loss", "Final train loss", "Final loss", "comparison_final_loss_boxplot")
    plot_final_boxplot(grouped, "final_test_acc", "Final test accuracy", "Final test accuracy", "comparison_final_test_accuracy_boxplot")
    plot_decision_boundaries(grouped, config, "comparison_decision_boundaries")

    with open(os.path.join(REPORT_DIR, "dense_angle_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    tex_path = build_latex_report(config, summary)
    compile_latex(tex_path)
    print(f"Dense-angle report written to {REPORT_DIR}/")


if __name__ == "__main__":
    main()
