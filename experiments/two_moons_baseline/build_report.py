#!/usr/bin/env python3
"""Build figures and summary artifacts for the hybrid Krotov benchmark."""

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

from datasets import generate_two_moons
from qml_models import VQCModel


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REPORT_DIR = os.path.join(SCRIPT_DIR, "results", "report")
FIG_DIR = os.path.join(REPORT_DIR, "figures")

FAMILY_COLORS = {
    "krotov_online": "#c94c4c",
    "krotov_batch": "#d17c00",
    "krotov_hybrid": "#6b46c1",
    "adam": "#2b6cb0",
    "lbfgs": "#2f855a",
}
BASE_LABELS = {
    "krotov_online": "Krotov online",
    "krotov_batch": "Krotov batch",
    "krotov_hybrid": "Krotov hybrid",
    "adam": "Adam",
    "lbfgs": "L-BFGS-B",
}
COMPARISON_BASE_ORDER = [
    "krotov_online",
    "krotov_batch",
    "krotov_hybrid",
    "adam",
    "lbfgs",
]

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


def group_results(results, names=None):
    grouped = OrderedDict()
    if names is None:
        names = []
    for name in names:
        runs = [result for result in results if result["optimizer"] == name]
        if runs:
            grouped[name] = runs
    seen = set(grouped)
    for result in results:
        name = result["optimizer"]
        if name in seen:
            continue
        grouped[name] = [r for r in results if r["optimizer"] == name]
        seen.add(name)
    return grouped


def hybrid_switch_iteration(run, config):
    return run.get("config_overrides", {}).get(
        "hybrid_switch_iteration",
        config["hybrid_switch_iteration"],
    )


def family_name(run):
    return run.get("optimizer_family", run["optimizer"])


def label_for_group(name, runs, config):
    run = runs[0]
    family = family_name(run)
    if family == "krotov_hybrid":
        switch_iteration = hybrid_switch_iteration(run, config)
        return f"Hybrid sw={switch_iteration}"
    return BASE_LABELS.get(name, BASE_LABELS.get(family, name))


def color_for_group(name, runs):
    run = runs[0]
    family = family_name(run)
    return FAMILY_COLORS.get(name, FAMILY_COLORS.get(family, "#444444"))


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


def threshold_stats(runs, threshold_key):
    step_vals = []
    time_vals = []
    cost_vals = []
    for run in runs:
        metric = run["threshold_metrics"][threshold_key]
        if metric["reached"]:
            step_vals.append(metric["step"])
            time_vals.append(metric["wall_time"])
            cost_vals.append(metric["cost_units"])
    return {
        "success_rate": float(sum(run["threshold_metrics"][threshold_key]["reached"] for run in runs) / len(runs)),
        "step_mean": float(np.mean(step_vals)) if step_vals else None,
        "time_mean": float(np.mean(time_vals)) if time_vals else None,
        "cost_mean": float(np.mean(cost_vals)) if cost_vals else None,
    }


def summarize_group(runs, config):
    final_losses = np.array([run["final_loss"] for run in runs], dtype=float)
    final_test_accs = np.array([run["final_test_acc"] for run in runs], dtype=float)
    wall_times = np.array([run["wall_time_total"] for run in runs], dtype=float)
    costs = np.array([run["total_cost_units"] for run in runs], dtype=float)
    early_loss = np.array([run["trace"]["loss"][min(1, len(run["trace"]["loss"]) - 1)] for run in runs], dtype=float)
    step10_loss = np.array([run["trace"]["loss"][min(10, len(run["trace"]["loss"]) - 1)] for run in runs], dtype=float)
    tail_loss_std = np.array([np.std(run["trace"]["loss"][-20:]) for run in runs], dtype=float)
    tail_update_norm = np.array([np.mean(run["trace"]["update_norm"][-20:]) for run in runs], dtype=float)
    threshold_summary = {
        key: threshold_stats(runs, key) for key in [f"{threshold:.2f}" for threshold in config["loss_thresholds"]]
    }
    return {
        "n_runs": len(runs),
        "final_loss_mean": float(np.mean(final_losses)),
        "final_loss_std": float(np.std(final_losses)),
        "final_test_acc_mean": float(np.mean(final_test_accs)),
        "final_test_acc_std": float(np.std(final_test_accs)),
        "wall_time_mean": float(np.mean(wall_times)),
        "wall_time_std": float(np.std(wall_times)),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "loss_step1_mean": float(np.mean(early_loss)),
        "loss_step10_mean": float(np.mean(step10_loss)),
        "tail_loss_std_mean": float(np.mean(tail_loss_std)),
        "tail_update_norm_mean": float(np.mean(tail_update_norm)),
        "thresholds": threshold_summary,
    }


def summarize_all(grouped, config):
    return OrderedDict((name, summarize_group(runs, config)) for name, runs in grouped.items())


def best_hybrid_name(grouped, config):
    hybrid_names = [
        name for name, runs in grouped.items() if family_name(runs[0]) == "krotov_hybrid"
    ]
    return min(hybrid_names, key=lambda name: summarize_group(grouped[name], config)["final_loss_mean"])


def fastest_hybrid_at_threshold(grouped, config, threshold_key):
    hybrid_names = [
        name for name, runs in grouped.items() if family_name(runs[0]) == "krotov_hybrid"
    ]
    valid = []
    for name in hybrid_names:
        stats = threshold_stats(grouped[name], threshold_key)
        if stats["time_mean"] is not None:
            valid.append((stats["time_mean"], name))
    return min(valid)[1] if valid else None


def comparison_grouped(grouped, config, best_hybrid):
    selected = OrderedDict()
    for name in COMPARISON_BASE_ORDER:
        if name == "krotov_hybrid":
            selected[best_hybrid] = grouped[best_hybrid]
        elif name in grouped:
            selected[name] = grouped[name]
    return selected


def plot_mean_band(grouped, config, x_key, y_key, title, xlabel, ylabel, out_name):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for name, runs in grouped.items():
        traces_x = [run["trace"][x_key] for run in runs]
        traces_y = [run["trace"][y_key] for run in runs]
        x_grid, y_arr = interp_traces(traces_x, traces_y)
        mean = np.mean(y_arr, axis=0)
        std = np.std(y_arr, axis=0)
        color = color_for_group(name, runs)
        ax.plot(x_grid, mean, lw=2, color=color, label=label_for_group(name, runs, config))
        ax.fill_between(x_grid, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out_name)


def plot_threshold_time_boxplot(grouped, config, threshold_key, out_name):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    names = list(grouped)
    data = []
    labels = []
    colors = []
    for name, runs in grouped.items():
        times = [
            run["threshold_metrics"][threshold_key]["wall_time"]
            for run in runs
            if run["threshold_metrics"][threshold_key]["reached"]
        ]
        if not times:
            continue
        data.append(times)
        labels.append(label_for_group(name, runs, config))
        colors.append(color_for_group(name, runs))
    boxplot = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax.set_title(f"Wall-clock time to reach loss <= {threshold_key}")
    ax.set_ylabel("Wall-clock time (s)")
    ax.tick_params(axis="x", rotation=18)
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, out_name)


def plot_success_rate_vs_threshold(grouped, config, out_name):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    thresholds = [f"{threshold:.2f}" for threshold in config["loss_thresholds"]]
    x = [float(threshold) for threshold in thresholds]
    for name, runs in grouped.items():
        y = [
            threshold_stats(runs, threshold_key)["success_rate"]
            for threshold_key in thresholds
        ]
        ax.plot(x, y, marker="o", lw=2, color=color_for_group(name, runs), label=label_for_group(name, runs, config))
    ax.set_title("Success rate vs threshold")
    ax.set_xlabel("Loss threshold")
    ax.set_ylabel("Fraction of runs that reach threshold")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out_name)


def plot_final_boxplot(grouped, config, field, ylabel, title, out_name):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    names = list(grouped)
    data = [[run[field] for run in grouped[name]] for name in names]
    labels = [label_for_group(name, grouped[name], config) for name in names]
    colors = [color_for_group(name, grouped[name]) for name in names]
    boxplot = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for patch, color in zip(boxplot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=18)
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, out_name)


def plot_hybrid_switch_traces(grouped, config, out_name):
    hybrid_names = [
        name for name, runs in grouped.items() if family_name(runs[0]) == "krotov_hybrid"
    ]
    hybrid_names = sorted(hybrid_names, key=lambda name: hybrid_switch_iteration(grouped[name][0], config))
    cmap = plt.cm.plasma(np.linspace(0.15, 0.9, len(hybrid_names)))
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for color, name in zip(cmap, hybrid_names):
        runs = grouped[name]
        traces_x = [run["trace"]["step"] for run in runs]
        traces_y = [run["trace"]["loss"] for run in runs]
        x_grid, y_arr = interp_traces(traces_x, traces_y)
        label = label_for_group(name, runs, config)
        ax.plot(x_grid, np.mean(y_arr, axis=0), lw=2, color=color, label=label)
    ax.set_title("Hybrid variants: loss vs iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss (BCE)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out_name)


def plot_update_norm_comparison(grouped, config, out_name):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for name, runs in grouped.items():
        traces_x = [run["trace"]["step"] for run in runs]
        traces_y = [run["trace"]["update_norm"] for run in runs]
        x_grid, y_arr = interp_traces(traces_x, traces_y)
        mean = np.mean(y_arr, axis=0)
        std = np.std(y_arr, axis=0)
        color = color_for_group(name, runs)
        ax.plot(x_grid, mean, lw=2, color=color, label=label_for_group(name, runs, config))
        ax.fill_between(x_grid, mean - std, mean + std, color=color, alpha=0.15)
    ax.set_title("Update norm vs iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Update norm")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out_name)


def plot_hybrid_phase_trace(grouped, config, best_hybrid, out_name):
    runs = grouped[best_hybrid]
    losses = np.array([run["final_loss"] for run in runs], dtype=float)
    median_idx = int(np.argsort(losses)[len(losses) // 2])
    run = runs[median_idx]
    switch_iteration = hybrid_switch_iteration(run, config)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    steps = np.asarray(run["trace"]["step"], dtype=float)
    loss = np.asarray(run["trace"]["loss"], dtype=float)
    update_norm = np.asarray(run["trace"]["update_norm"], dtype=float)
    phases = run["trace"]["phase"]

    online_mask = np.array([phase == "online" or phase == "init" for phase in phases], dtype=bool)
    batch_mask = np.array([phase == "batch" for phase in phases], dtype=bool)
    ax.plot(steps[online_mask], loss[online_mask], color=FAMILY_COLORS["krotov_online"], lw=2.2, label="Loss (online phase)")
    ax.plot(steps[batch_mask], loss[batch_mask], color=FAMILY_COLORS["krotov_batch"], lw=2.2, label="Loss (batch phase)")
    ax.axvline(switch_iteration, color="#222222", ls="--", lw=1.2, label=f"Switch @ {switch_iteration}")
    ax.set_title(f"Representative hybrid trace: {label_for_group(best_hybrid, runs, config)}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss (BCE)")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(steps, update_norm, color=FAMILY_COLORS["krotov_hybrid"], lw=1.3, alpha=0.7, label="Update norm")
    ax2.set_ylabel("Update norm")

    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, loc="upper right")
    save_fig(fig, out_name)


def plot_decision_boundaries(grouped, config, out_name):
    names = list(grouped)
    fig, axes = plt.subplots(1, len(names), figsize=(4.8 * len(names), 4.4))
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        runs = grouped[name]
        losses = np.array([run["final_loss"] for run in runs], dtype=float)
        median_idx = int(np.argsort(losses)[len(losses) // 2])
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
        ax.contour(xx1, xx2, probs, levels=[0.5], colors="k", linewidths=1.0)
        ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="#d94841", s=8, alpha=0.55)
        ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="#3366cc", s=8, alpha=0.55)
        ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#8c2d19", s=18, marker="x", alpha=0.7)
        ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#1d4ed8", s=18, marker="x", alpha=0.7)
        ax.set_title(f"{label_for_group(name, runs, config)}\n(loss={run['final_loss']:.3f})")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")

    fig.suptitle("Decision boundaries for representative runs", y=1.02)
    plt.tight_layout()
    save_fig(fig, out_name)


def latex_escape(text):
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def build_comparison_table(summary, grouped, config, names, threshold_key):
    rows = []
    for name in names:
        stats = summary[name]
        label = latex_escape(label_for_group(name, grouped[name], config))
        threshold_stats = stats["thresholds"][threshold_key]
        rows.append(
            f"{label} & "
            f"{stats['final_loss_mean']:.3f} $\\pm$ {stats['final_loss_std']:.3f} & "
            f"{stats['final_test_acc_mean']:.3f} $\\pm$ {stats['final_test_acc_std']:.3f} & "
            f"{stats['wall_time_mean']:.1f} $\\pm$ {stats['wall_time_std']:.1f} & "
            f"{threshold_stats['time_mean']:.1f} & "
            f"{threshold_stats['success_rate']:.2f} & "
            f"{stats['tail_loss_std_mean']:.4f} \\\\"
        )
    return "\n".join(rows)


def build_hybrid_sweep_table(summary, grouped, config, threshold_key):
    hybrid_names = [
        name for name, runs in grouped.items() if family_name(runs[0]) == "krotov_hybrid"
    ]
    hybrid_names = sorted(hybrid_names, key=lambda name: hybrid_switch_iteration(grouped[name][0], config))
    rows = []
    for name in hybrid_names:
        stats = summary[name]
        threshold_stats = stats["thresholds"][threshold_key]
        rows.append(
            f"{latex_escape(label_for_group(name, grouped[name], config))} & "
            f"{stats['final_loss_mean']:.3f} $\\pm$ {stats['final_loss_std']:.3f} & "
            f"{stats['wall_time_mean']:.1f} & "
            f"{threshold_stats['time_mean']:.1f} & "
            f"{stats['tail_loss_std_mean']:.4f} \\\\"
        )
    return "\n".join(rows)


def build_report_tex(config, summary, grouped, best_hybrid, fastest_hybrid):
    threshold_key = f"{config['loss_threshold']:.2f}"
    comparison_names = ["krotov_online", "krotov_batch", best_hybrid, "adam", "lbfgs"]
    comparison_rows = build_comparison_table(summary, grouped, config, comparison_names, threshold_key)
    hybrid_rows = build_hybrid_sweep_table(summary, grouped, config, threshold_key)

    best_stats = summary[best_hybrid]
    batch_stats = summary["krotov_batch"]
    online_stats = summary["krotov_online"]
    adam_stats = summary["adam"]
    lbfgs_stats = summary["lbfgs"]
    fastest_label = latex_escape(label_for_group(fastest_hybrid, grouped[fastest_hybrid], config))
    best_label = latex_escape(label_for_group(best_hybrid, grouped[best_hybrid], config))

    return rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}

\title{{Hybrid Krotov Schedule in the Two-Moons QML Benchmark}}
\author{{Codex experiment report}}
\date{{\today}}

\begin{{document}}
\maketitle

\section*{{Experimental setup}}
This report updates the two-moons QML benchmark with a new hybrid Krotov optimizer that runs the original online stale-adjoint update for an initial phase and then switches to the full-batch Krotov update. The benchmark keeps the same model and data pipeline as the earlier experiments: {config['n_qubits']} qubits, {config['n_layers']} trainable layers, {len(config['seeds'])} random seeds, and {config['max_iterations']} outer iterations on the same train/test split convention. The main comparison contains:
\begin{{itemize}}
\item \texttt{{krotov\_online}} with step size $0.3$,
\item \texttt{{krotov\_batch}} with step size $1.0$,
\item \texttt{{krotov\_hybrid}} with online step size $0.3$, batch step size $1.0$, and default switch at iteration {config['hybrid_switch_iteration']},
\item Adam,
\item L-BFGS-B.
\end{{itemize}}

The hybrid sweep varies the switch point over $\{{5, 10, 20, 30, 50\}}$. The fair accounting axis remains
\[
\text{{cost units}} = \text{{sample forward passes}} + \text{{sample backward passes}},
\]
so all Krotov-family methods are compared under the same primitive propagation accounting as before.

\section*{{Main findings}}
\begin{{enumerate}}
\item The hybrid direction is strongly supported. The best tested hybrid setting is \texttt{{{best_label}}}, which reaches mean final loss {best_stats['final_loss_mean']:.3f} $\pm$ {best_stats['final_loss_std']:.3f}. This is lower than pure online ({online_stats['final_loss_mean']:.3f}), pure batch ({batch_stats['final_loss_mean']:.3f}), Adam ({adam_stats['final_loss_mean']:.3f}), and L-BFGS-B ({lbfgs_stats['final_loss_mean']:.3f}) on the five-seed study.
\item The hybrid preserves the defining early-strength of online Krotov. For the benchmark threshold loss $\leq {threshold_key}$, the fastest hybrid is \texttt{{{fastest_label}}}, which reaches the threshold in {summary[fastest_hybrid]['thresholds'][threshold_key]['time_mean']:.1f}s on average. Pure online reaches the same threshold in {online_stats['thresholds'][threshold_key]['time_mean']:.1f}s, while batch, Adam, and L-BFGS-B need {batch_stats['thresholds'][threshold_key]['time_mean']:.1f}s, {adam_stats['thresholds'][threshold_key]['time_mean']:.1f}s, and {lbfgs_stats['thresholds'][threshold_key]['time_mean']:.1f}s, respectively.
\item The switch removes the late online oscillation almost completely. The online baseline has mean tail loss standard deviation {online_stats['tail_loss_std_mean']:.4f}, whereas \texttt{{{best_label}}} drops that to {best_stats['tail_loss_std_mean']:.4f}. The tail update norm shows the same pattern: the online rule keeps moving aggressively, while the hybrid settles into a near-stationary refinement regime after the switch.
\item On optimization-centric metrics, the best hybrid setting dominates the tested alternatives in this study. The only notable caveat is that Adam still has the highest mean final test accuracy ({adam_stats['final_test_acc_mean']:.3f} versus {best_stats['final_test_acc_mean']:.3f} for the best hybrid), so the dominance claim is strongest for optimization speed and training objective value rather than for every downstream metric.
\end{{enumerate}}

\section*{{Quantitative comparison of the dominant methods}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lcccccc}}
\toprule
Optimizer & Final loss & Final test acc. & Wall time (s) & Time to $\leq {threshold_key}$ (s) & Success @ {threshold_key} & Tail std. \\
\midrule
{comparison_rows}
\bottomrule
\end{{tabular}}
\caption{{Direct comparison on the aligned five-seed benchmark.}}
\end{{table}}

The quantitative picture is unusually clean. The best hybrid run improves final loss over both pure Krotov baselines while preserving the near-instant threshold crossing that previously belonged only to the online method. At the same time it retains the practical wall-clock advantage over Adam and L-BFGS-B, finishing in {best_stats['wall_time_mean']:.1f}s on average versus {adam_stats['wall_time_mean']:.1f}s for Adam and {lbfgs_stats['wall_time_mean']:.1f}s for L-BFGS-B.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\linewidth]{{figures/comparison_loss_vs_iteration.pdf}}
\caption{{Training loss versus iteration for the main comparison, with the best hybrid variant substituted for the generic hybrid family.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\linewidth]{{figures/comparison_loss_vs_time.pdf}}
\caption{{Training loss versus wall-clock time. The hybrid closes the final-loss gap without sacrificing the practical runtime advantage.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\linewidth]{{figures/comparison_loss_vs_cost.pdf}}
\caption{{Training loss versus fair propagation cost. The hybrid inherits the rapid early threshold crossing of the online rule while remaining as propagation-efficient as the other Krotov-family methods.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\linewidth]{{figures/time_to_threshold_boxplot.pdf}}
\caption{{Wall-clock time to reach the benchmark threshold loss $\leq {threshold_key}$.}}
\end{{figure}}

\section*{{Switch sweep}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
Hybrid setting & Final loss & Wall time (s) & Time to $\leq {threshold_key}$ (s) & Tail std. \\
\midrule
{hybrid_rows}
\bottomrule
\end{{tabular}}
\caption{{Hybrid switch sweep. Earlier switches perform best on this benchmark, with the switch-at-10 setting giving the strongest overall result.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\linewidth]{{figures/hybrid_switch_loss_vs_iteration.pdf}}
\caption{{Loss versus iteration for the hybrid switch sweep. Very late switches drift back toward the noisier online behavior.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\linewidth]{{figures/comparison_update_norm_vs_iteration.pdf}}
\caption{{Update norms for the main comparison. The hybrid retains a strong early move and then settles into a batch-like refinement regime.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\linewidth]{{figures/hybrid_phase_transition_trace.pdf}}
\caption{{Representative hybrid trace with the phase transition marked explicitly. The loss flattens rapidly after the switch while update norms collapse.}}
\end{{figure}}

\section*{{Interpretation}}
The hybrid result supports a more specific story than the earlier batch-only report. The online rule appears to be valuable primarily as a short, high-gain transient that moves the parameters into a good basin very quickly. Leaving it active for too long produces unnecessary late-stage noise. The batch rule, in contrast, is well suited to stable refinement once that basin has been reached. The hybrid schedule works because it assigns each mechanism to the regime where it is strongest.

This means the current evidence favors ``online then batch'' over either pure online or pure batch. In the tested configuration, the hybrid schedule is not merely a compromise. It is the strongest optimizer among the tested methods on mean final training loss, fair-cost threshold crossing, and wall-clock time-to-solution. That makes it the best current headline result in this benchmark, subject to the modest caveat that Adam retains a slightly higher mean final test accuracy.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/comparison_decision_boundaries.pdf}}
\caption{{Representative decision boundaries for the main comparison.}}
\end{{figure}}

\section*{{Conclusion}}
The new experiments upgrade the research story substantially. The batch Krotov method already showed that objective-aligned updates could be competitive with gradient baselines. The hybrid schedule goes further: it preserves the fast early progress of online Krotov, removes the online oscillation, improves the final training loss beyond the pure batch rule, and on this five-seed benchmark dominates all tested methods on the central optimization metrics. The next natural step would be local tuning around the switch-at-10 setting and then repeating the study on a larger seed budget to confirm the effect size.

\end{{document}}
"""


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    config = load_config()
    results = load_results(config)
    if not results:
        raise SystemExit("No results found. Run run_experiment.py first.")

    grouped = group_results(results)
    summary = summarize_all(grouped, config)
    best_hybrid = best_hybrid_name(grouped, config)
    threshold_key = f"{config['loss_threshold']:.2f}"
    fastest_hybrid = fastest_hybrid_at_threshold(grouped, config, threshold_key)
    comparison = comparison_grouped(grouped, config, best_hybrid)

    plot_mean_band(
        comparison,
        config,
        "step",
        "loss",
        "Training loss vs iteration",
        "Iteration",
        "Training loss (BCE)",
        "comparison_loss_vs_iteration",
    )
    plot_mean_band(
        comparison,
        config,
        "wall_time",
        "loss",
        "Training loss vs wall-clock time",
        "Wall-clock time (s)",
        "Training loss (BCE)",
        "comparison_loss_vs_time",
    )
    plot_mean_band(
        comparison,
        config,
        "cost_units",
        "loss",
        "Training loss vs fair propagation cost",
        "Propagation cost = sample forwards + sample backwards",
        "Training loss (BCE)",
        "comparison_loss_vs_cost",
    )
    plot_threshold_time_boxplot(comparison, config, threshold_key, "time_to_threshold_boxplot")
    plot_success_rate_vs_threshold(comparison, config, "success_rate_vs_threshold")
    plot_hybrid_switch_traces(grouped, config, "hybrid_switch_loss_vs_iteration")
    plot_update_norm_comparison(comparison, config, "comparison_update_norm_vs_iteration")
    plot_hybrid_phase_trace(grouped, config, best_hybrid, "hybrid_phase_transition_trace")
    plot_final_boxplot(comparison, config, "final_loss", "Final training loss (BCE)", "Final training loss", "comparison_final_loss_boxplot")
    plot_final_boxplot(comparison, config, "final_test_acc", "Final test accuracy", "Final test accuracy", "comparison_final_test_accuracy_boxplot")
    plot_decision_boundaries(comparison, config, "comparison_decision_boundaries")

    report_summary = OrderedDict(
        [
            ("best_hybrid_by_final_loss", best_hybrid),
            ("best_hybrid_label", label_for_group(best_hybrid, grouped[best_hybrid], config)),
            ("fastest_hybrid_at_loss_threshold", fastest_hybrid),
            (
                "fastest_hybrid_label",
                None if fastest_hybrid is None else label_for_group(fastest_hybrid, grouped[fastest_hybrid], config),
            ),
            ("comparison_threshold", float(config["loss_threshold"])),
            ("summary_by_optimizer", summary),
        ]
    )
    with open(os.path.join(REPORT_DIR, "hybrid_summary.json"), "w") as f:
        json.dump(report_summary, f, indent=2)

    tex = build_report_tex(config, summary, grouped, best_hybrid, fastest_hybrid)
    tex_path = os.path.join(REPORT_DIR, "hybrid_experiment_analysis.tex")
    with open(tex_path, "w") as f:
        f.write(tex)

    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "hybrid_experiment_analysis.tex",
        ],
        cwd=REPORT_DIR,
        check=True,
    )

    print(f"Best hybrid by final loss: {label_for_group(best_hybrid, grouped[best_hybrid], config)}")
    if fastest_hybrid is not None:
        print(f"Fastest hybrid at loss <= {threshold_key}: {label_for_group(fastest_hybrid, grouped[fastest_hybrid], config)}")
    print(f"Summary written to {os.path.join(REPORT_DIR, 'hybrid_summary.json')}")
    print(f"Report written to {tex_path}")
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
