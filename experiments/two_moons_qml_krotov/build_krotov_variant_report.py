#!/usr/bin/env python3
"""Build Krotov-only figures and a compiled LaTeX report."""

import glob
import json
import os
import subprocess
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_krotov_variants")
BASELINE_RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
REPORT_DIR = os.path.join(SCRIPT_DIR, "report_krotov_variants")
FIG_DIR = os.path.join(REPORT_DIR, "figures")

BASE_ORDER = ["krotov_online", "krotov_batch"]
COMPARISON_VARIANT = "krotov_batch_lr1.000_constant"
COMPARISON_ORDER = [COMPARISON_VARIANT, "adam", "lbfgs"]
SWEEP_ORDER = [
    "krotov_batch",
    "krotov_batch_lr1.000_constant",
    "krotov_batch_lr0.100_constant",
    "krotov_batch_lr0.050_constant",
    "krotov_batch_lr0.020_constant",
    "krotov_batch_lr0.300_inverse",
    "krotov_batch_lr1.000_inverse",
]
COLORS = {
    "krotov_online": "#c94c4c",
    "krotov_batch": "#d17c00",
    "krotov_batch_lr1.000_constant": "#8c510a",
    "krotov_batch_lr0.100_constant": "#bf812d",
    "krotov_batch_lr0.050_constant": "#dfc27d",
    "krotov_batch_lr0.020_constant": "#f6e8c3",
    "krotov_batch_lr0.300_inverse": "#5ab4ac",
    "krotov_batch_lr1.000_inverse": "#01665e",
    "adam": "#2b6cb0",
    "lbfgs": "#2f855a",
}
LABELS = {
    "krotov_online": "online, lr=0.3",
    "krotov_batch": "batch, lr=0.3",
    "krotov_batch_lr1.000_constant": "batch, lr=1.0",
    "krotov_batch_lr0.100_constant": "batch, lr=0.1",
    "krotov_batch_lr0.050_constant": "batch, lr=0.05",
    "krotov_batch_lr0.020_constant": "batch, lr=0.02",
    "krotov_batch_lr0.300_inverse": "batch, lr=0.3 inverse",
    "krotov_batch_lr1.000_inverse": "batch, lr=1.0 inverse",
    "adam": "Adam",
    "lbfgs": "L-BFGS-B",
}

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
    allowed = {spec["run_name"] for spec in config["experiment_specs"]}
    results = []
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, "result_*.json"))):
        with open(fpath) as f:
            result = json.load(f)
        if result["optimizer"] in allowed:
            results.append(result)
    return results


def load_baseline_results(seeds):
    allowed = {"adam", "lbfgs"}
    seed_set = set(seeds)
    results = []
    for fpath in sorted(glob.glob(os.path.join(BASELINE_RESULTS_DIR, "result_*.json"))):
        with open(fpath) as f:
            result = json.load(f)
        if result["optimizer"] in allowed and result["seed"] in seed_set:
            results.append(result)
    return results


def group_results(results, ordered_names):
    grouped = OrderedDict()
    for name in ordered_names:
        runs = [result for result in results if result["optimizer"] == name]
        if runs:
            grouped[name] = runs
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


def plot_mean_band(grouped, x_key, y_key, names, title, xlabel, ylabel, out_name):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name in names:
        if name not in grouped:
            continue
        traces_x = [run["trace"][x_key] for run in grouped[name]]
        traces_y = [run["trace"][y_key] for run in grouped[name]]
        x_grid, y_arr = interp_traces(traces_x, traces_y)
        mean = np.mean(y_arr, axis=0)
        std = np.std(y_arr, axis=0)
        ax.plot(x_grid, mean, lw=2, color=COLORS[name], label=LABELS[name])
        ax.fill_between(x_grid, mean - std, mean + std, color=COLORS[name], alpha=0.16)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, out_name)


def plot_selected_batch_vs_baselines(grouped, names):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for name in names:
        if name not in grouped:
            continue

        traces_step = [run["trace"]["step"] for run in grouped[name]]
        traces_loss = [run["trace"]["loss"] for run in grouped[name]]
        step_grid, loss_step_arr = interp_traces(traces_step, traces_loss)
        step_mean = np.mean(loss_step_arr, axis=0)
        step_std = np.std(loss_step_arr, axis=0)
        axes[0].plot(step_grid, step_mean, lw=2, color=COLORS[name], label=LABELS[name])
        axes[0].fill_between(step_grid, step_mean - step_std, step_mean + step_std, color=COLORS[name], alpha=0.16)

        traces_time = [run["trace"]["wall_time"] for run in grouped[name]]
        time_grid, loss_time_arr = interp_traces(traces_time, traces_loss)
        time_mean = np.mean(loss_time_arr, axis=0)
        time_std = np.std(loss_time_arr, axis=0)
        axes[1].plot(time_grid, time_mean, lw=2, color=COLORS[name], label=LABELS[name])
        axes[1].fill_between(time_grid, time_mean - time_std, time_mean + time_std, color=COLORS[name], alpha=0.16)

    axes[0].set_title("Loss vs iteration")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Training loss (BCE)")
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Loss vs wall-clock time")
    axes[1].set_xlabel("Wall-clock time (s)")
    axes[1].set_ylabel("Training loss (BCE)")
    axes[1].grid(True, alpha=0.3)

    box_names = [name for name in names if name in grouped]
    boxplot = axes[2].boxplot(
        [[run["final_loss"] for run in grouped[name]] for name in box_names],
        tick_labels=[LABELS[name] for name in box_names],
        patch_artist=True,
        widths=0.55,
    )
    for patch, name in zip(boxplot["boxes"], box_names):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.45)
    axes[2].set_title("Final loss distribution")
    axes[2].set_ylabel("Final training loss (BCE)")
    axes[2].tick_params(axis="x", rotation=18)
    axes[2].grid(True, axis="y", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, "selected_batch_vs_baselines")


def plot_sweep_boxplot(grouped):
    names = [name for name in SWEEP_ORDER if name in grouped and name != "krotov_online"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    data = [[run["final_loss"] for run in grouped[name]] for name in names]
    labels = [LABELS[name] for name in names]
    boxplot = ax.boxplot(data, tick_labels=labels, patch_artist=True, widths=0.55)
    for patch, name in zip(boxplot["boxes"], names):
        patch.set_facecolor(COLORS[name])
        patch.set_alpha(0.45)
    ax.set_title("Krotov batch final loss across learning-rate variants")
    ax.set_ylabel("Final training loss (BCE)")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(True, axis="y", alpha=0.3)
    save_fig(fig, "krotov_batch_sweep_boxplot")


def plot_selected_batch_variants(grouped):
    names = [
        "krotov_batch",
        "krotov_batch_lr1.000_constant",
        "krotov_batch_lr0.300_inverse",
        "krotov_batch_lr1.000_inverse",
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for name in names:
        if name not in grouped:
            continue
        traces_x = [run["trace"]["step"] for run in grouped[name]]
        traces_y = [run["trace"]["loss"] for run in grouped[name]]
        x_grid, y_arr = interp_traces(traces_x, traces_y)
        ax.plot(x_grid, np.mean(y_arr, axis=0), lw=2, color=COLORS[name], label=LABELS[name])
    ax.set_title("Selected batch variants: loss vs iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training loss (BCE)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    save_fig(fig, "krotov_batch_selected_variants")


def summarize_runs(grouped, threshold):
    summary = OrderedDict()
    for name, runs in grouped.items():
        final_losses = np.array([run["final_loss"] for run in runs], dtype=float)
        test_accs = np.array([run["final_test_acc"] for run in runs], dtype=float)
        costs = np.array([run["total_cost_units"] for run in runs], dtype=float)
        early_losses = np.array([run["trace"]["loss"][1] for run in runs], dtype=float)
        step10_losses = np.array([run["trace"]["loss"][min(10, len(run["trace"]["loss"]) - 1)] for run in runs], dtype=float)
        tail_stds = np.array([np.std(run["trace"]["loss"][-20:]) for run in runs], dtype=float)
        tail_update = np.array([np.mean(run["trace"]["update_norm"][-20:]) for run in runs], dtype=float)
        threshold_costs = []
        for run in runs:
            loss_trace = np.array(run["trace"]["loss"], dtype=float)
            cost_trace = np.array(run["trace"]["cost_units"], dtype=float)
            hit = np.where(loss_trace <= threshold)[0]
            threshold_costs.append(float(cost_trace[hit[0]]) if len(hit) else np.nan)
        threshold_costs = np.array(threshold_costs, dtype=float)

        mean_threshold_cost = float(np.nanmean(threshold_costs)) if np.any(~np.isnan(threshold_costs)) else float("nan")

        summary[name] = {
            "n_runs": len(runs),
            "final_loss_mean": float(np.mean(final_losses)),
            "final_loss_std": float(np.std(final_losses)),
            "final_test_acc_mean": float(np.mean(test_accs)),
            "final_test_acc_std": float(np.std(test_accs)),
            "total_cost_mean": float(np.mean(costs)),
            "loss_step1_mean": float(np.mean(early_losses)),
            "loss_step10_mean": float(np.mean(step10_losses)),
            "tail_loss_std_mean": float(np.mean(tail_stds)),
            "tail_update_norm_mean": float(np.mean(tail_update)),
            "success_rate": float(np.mean(final_losses <= threshold)),
            "threshold_cost_mean": mean_threshold_cost,
        }
    return summary


def summarize_baseline_comparison(grouped):
    summary = OrderedDict()
    for name, runs in grouped.items():
        final_losses = np.array([run["final_loss"] for run in runs], dtype=float)
        test_accs = np.array([run["final_test_acc"] for run in runs], dtype=float)
        wall_times = np.array([run["wall_time_total"] for run in runs], dtype=float)
        total_steps = np.array([run["total_steps"] for run in runs], dtype=float)
        summary[name] = {
            "n_runs": len(runs),
            "final_loss_mean": float(np.mean(final_losses)),
            "final_loss_std": float(np.std(final_losses)),
            "final_test_acc_mean": float(np.mean(test_accs)),
            "final_test_acc_std": float(np.std(test_accs)),
            "wall_time_mean": float(np.mean(wall_times)),
            "wall_time_std": float(np.std(wall_times)),
            "total_steps_mean": float(np.mean(total_steps)),
            "total_steps_std": float(np.std(total_steps)),
        }
    return summary


def latex_escape(text):
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def build_summary_table(summary, names):
    rows = []
    for name in names:
        if name not in summary:
            continue
        stats = summary[name]
        rows.append(
            f"{latex_escape(LABELS[name])} & "
            f"{stats['final_loss_mean']:.3f} $\\pm$ {stats['final_loss_std']:.3f} & "
            f"{stats['final_test_acc_mean']:.3f} $\\pm$ {stats['final_test_acc_std']:.3f} & "
            f"{stats['loss_step1_mean']:.3f} & "
            f"{stats['loss_step10_mean']:.3f} & "
            f"{stats['tail_loss_std_mean']:.3f} & "
            f"{stats['tail_update_norm_mean']:.3f} \\\\"
        )
    return "\n".join(rows)


def build_sweep_table(summary):
    rows = []
    for name in SWEEP_ORDER:
        if name not in summary or name == "krotov_online":
            continue
        stats = summary[name]
        rows.append(
            f"{latex_escape(LABELS[name])} & "
            f"{stats['final_loss_mean']:.3f} $\\pm$ {stats['final_loss_std']:.3f} & "
            f"{stats['success_rate']:.2f} & "
            f"{stats['threshold_cost_mean']:.0f} \\\\"
        )
    return "\n".join(rows)


def build_baseline_table(summary, names):
    rows = []
    for name in names:
        if name not in summary:
            continue
        stats = summary[name]
        rows.append(
            f"{latex_escape(LABELS[name])} & "
            f"{stats['final_loss_mean']:.3f} $\\pm$ {stats['final_loss_std']:.3f} & "
            f"{stats['final_test_acc_mean']:.3f} $\\pm$ {stats['final_test_acc_std']:.3f} & "
            f"{stats['wall_time_mean']:.1f} $\\pm$ {stats['wall_time_std']:.1f} & "
            f"{stats['total_steps_mean']:.1f} $\\pm$ {stats['total_steps_std']:.1f} \\\\"
        )
    return "\n".join(rows)


def build_report_tex(config, summary, baseline_summary):
    online = summary["krotov_online"]
    batch = summary["krotov_batch"]
    best_batch_name = min(
        (name for name in summary if name.startswith("krotov_batch")),
        key=lambda name: summary[name]["final_loss_mean"],
    )
    best_batch = summary[best_batch_name]

    threshold = config["loss_threshold"]
    summary_rows = build_summary_table(summary, BASE_ORDER)
    sweep_rows = build_sweep_table(summary)
    baseline_rows = build_baseline_table(baseline_summary, COMPARISON_ORDER)
    selected_batch = baseline_summary[COMPARISON_VARIANT]
    adam = baseline_summary["adam"]
    lbfgs = baseline_summary["lbfgs"]

    return rf"""\documentclass[11pt]{{article}}
\usepackage[a4paper,margin=1in]{{geometry}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{float}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}

\title{{Behavior of the New Krotov Variants in the Two-Moons QML Optimization Benchmark}}
\author{{Codex experiment report}}
\date{{\today}}

\begin{{document}}
\maketitle

\section*{{Experimental setup}}
The report analyzes the Krotov-family optimizers in the existing two-moons QML benchmark without changing the model or dataset pipeline. The setup uses {config['n_qubits']} qubits, {config['n_layers']} trainable layers, {len(config['seeds'])} random seeds, and {config['max_iterations']} outer iterations on the same train/test split convention as the original benchmark. The main comparison uses:
\begin{{itemize}}
\item \texttt{{krotov\_online}}: the preserved single-sample stale-adjoint update with constant step size $0.3$,
\item \texttt{{krotov\_batch}}: the new full-batch Krotov-inspired update with constant step size $0.3$.
\end{{itemize}}

The fair accounting axis is the propagation cost
\[
\text{{cost units}} = \text{{sample forward passes}} + \text{{sample backward passes}},
\]
which treats one single-sample forward propagation and one single-sample adjoint propagation as the primitive work units. Mean-loss evaluations and gradient-direction evaluations are logged separately in the raw JSON files, but the report uses cost units when interpreting optimization speed.

\section*{{Main empirical findings}}
\begin{{enumerate}}
\item The online Krotov baseline still shows the aggressive initial descent that motivated the original implementation. Averaged over seeds, its loss after the first optimization step is {online['loss_step1_mean']:.3f}, compared with {batch['loss_step1_mean']:.3f} for the full-batch variant.
\item The full-batch update is markedly smoother but substantially less aggressive at the baseline step size. Its mean final loss is {batch['final_loss_mean']:.3f} compared with {online['final_loss_mean']:.3f} for the online variant.
\item The online method pays for its fast initial progress with persistent late-stage oscillation. Its mean tail loss standard deviation over the last 20 logged points is {online['tail_loss_std_mean']:.3f}, versus {batch['tail_loss_std_mean']:.3f} for the batch variant.
\item The batch update is strongly step-size limited rather than obviously broken. Across the sweep, the best mean final loss comes from \texttt{{{latex_escape(LABELS[best_batch_name])}}}, which reaches {best_batch['final_loss_mean']:.3f}. This is materially better than the baseline batch setting and indicates that the default batch step size was too conservative for this problem.
\end{{enumerate}}

\section*{{Quantitative comparison of the main variants}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lcccccc}}
\toprule
Variant & Final loss & Final test acc. & Loss @ step 1 & Loss @ step 10 & Tail std. & Mean tail update norm \\
\midrule
{summary_rows}
\bottomrule
\end{{tabular}}
\caption{{Mean behavior of the main Krotov variants over the configured seeds. Tail statistics are computed from the last 20 logged optimization points.}}
\end{{table}}

The table makes the central contrast explicit. The online variant is a high-gain optimizer: it descends quickly, reaches the lower mean final loss in the baseline comparison, and keeps relatively large updates even late in the run. The batch variant is much more regularized by construction because it averages the gate-wise contributions over the whole dataset before applying an update. That removes sample-order noise and the stale-adjoint inconsistency across successive samples, but it also suppresses the large coordinated moves that made the online rule effective at the beginning of training.

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\linewidth]{{figures/krotov_loss_vs_cost.pdf}}
\caption{{Training loss versus the fair propagation-cost metric for the main Krotov variants.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\linewidth]{{figures/krotov_loss_vs_iteration.pdf}}
\caption{{Training loss versus iteration for the main Krotov variants.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\linewidth]{{figures/krotov_update_norm_vs_iteration.pdf}}
\caption{{Update norms reveal that the online rule keeps moving aggressively while the batch rule contracts to much smaller parameter changes.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\linewidth]{{figures/krotov_contribution_variance.pdf}}
\caption{{Per-parameter contribution variance. The online rule sees much noisier per-sample directions, which helps explain both the fast initial drop and the later oscillation.}}
\end{{figure}}

\section*{{Interpretation of the optimization dynamics}}
The new diagnostics support a fairly specific interpretation of the Krotov behavior.

First, the online optimizer's rapid initial improvement is real under fair accounting. The loss decreases sharply in both the loss-versus-cost and loss-versus-iteration views. This means that the earlier observation was not an artifact of the old function-evaluation counter. The online rule is genuinely capable of finding a useful direction very quickly on this task.

Second, the same mechanism that makes the online method aggressive also makes it unstable. Each update is built from one sample at a time, and the backward co-states are computed before the subsequent gate-wise parameter changes. As a result, the method keeps taking relatively large, sample-specific steps even after it reaches a reasonably good decision boundary. The elevated contribution variance and tail update norms are consistent with exactly this picture.

Third, the full-batch variant behaves more like deterministic gradient descent in the gate-local coordinates. It aligns the update with the true benchmark objective, because the gate-wise contributions are averaged over the full training loss before a single parameter update is applied. This removes stale-adjoint accumulation across samples and collapses the sample-order noise, but it also means the update magnitude must be tuned more carefully. At the baseline step size of $0.3$, the batch method is simply too conservative.

\section*{{Learning-rate sweep and what it implies}}
\begin{{table}}[H]
\centering
\begin{{tabular}}{{lccc}}
\toprule
Batch variant & Final loss & Success rate (final loss $\leq {threshold}$) & Mean cost to threshold \\
\midrule
{sweep_rows}
\bottomrule
\end{{tabular}}
\caption{{Krotov batch hyperparameter sweep. Success rate is measured against the configured loss threshold.}}
\end{{table}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.92\linewidth]{{figures/krotov_batch_sweep_boxplot.pdf}}
\caption{{Distribution of final training losses across batch-step-size variants.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.82\linewidth]{{figures/krotov_batch_selected_variants.pdf}}
\caption{{Selected batch variants. Larger initial step sizes recover much of the missing progress, while inverse decay trades speed for additional stability.}}
\end{{figure}}

The sweep shows that the batch optimizer is not failing because of an obvious implementation bug. When the step size is increased, the batch method becomes significantly more competitive. In particular, the best-performing batch setting in this study is \texttt{{{latex_escape(LABELS[best_batch_name])}}}, which reaches a mean final loss of {best_batch['final_loss_mean']:.3f}. This is strong evidence that a large part of the apparent batch underperformance at the default setting came from step-size mismatch rather than from the BCE adjoint derivation or the fair-cost accounting.

At the same time, the sweep does \emph{{not}} eliminate the qualitative difference between the online and batch dynamics. The online rule remains the more aggressive optimizer in the early phase, while the batch rule remains the smoother and more predictable one. The evidence therefore supports the following interpretation:
\begin{{itemize}}
\item unfair accounting was \emph{{not}} the reason for the observed fast early drop;
\item stale adjoints and single-sample noise are plausible explanations for the online oscillation;
\item the baseline batch underperformance was strongly amplified by step-size choice;
\item after retuning, the batch rule is viable, but its behavior is closer to stable first-order descent than to the highly expressive, sample-adaptive online Krotov trajectory.
\end{{itemize}}

\section*{{Selected batch variant versus Adam and L-BFGS-B}}
For the direct baseline check, the strongest tested batch setting in this sweep, \texttt{{{latex_escape(LABELS[COMPARISON_VARIANT])}}}, is compared against the stored Adam and L-BFGS-B runs on the same five seeds used in the variant study ({", ".join(str(seed) for seed in config["seeds"])}). The older baseline JSON logs do not contain the newer fair-cost trace, so the overlay below uses the common saved axes shared by all three optimizers: iteration and wall-clock time.

\begin{{table}}[H]
\centering
\begin{{tabular}}{{lcccc}}
\toprule
Optimizer & Final loss & Final test acc. & Wall time (s) & Total steps \\
\midrule
{baseline_rows}
\bottomrule
\end{{tabular}}
\caption{{Direct comparison on the aligned five-seed subset used in the Krotov-variant sweep.}}
\end{{table}}

The direct overlay shows that the retuned batch method is \emph{{close}} to the gradient baselines, but not fully matched in final loss. On the aligned seeds, \texttt{{{latex_escape(LABELS[COMPARISON_VARIANT])}}} reaches a mean final loss of {selected_batch['final_loss_mean']:.3f} $\pm$ {selected_batch['final_loss_std']:.3f}, versus {adam['final_loss_mean']:.3f} $\pm$ {adam['final_loss_std']:.3f} for Adam and {lbfgs['final_loss_mean']:.3f} $\pm$ {lbfgs['final_loss_std']:.3f} for L-BFGS-B. Its mean final test accuracy of {selected_batch['final_test_acc_mean']:.3f} is similar to Adam's {adam['final_test_acc_mean']:.3f} and slightly above L-BFGS-B's {lbfgs['final_test_acc_mean']:.3f}. The strongest practical advantage of the batch Krotov variant is wall-clock speed: it finishes in {selected_batch['wall_time_mean']:.1f}s on average, compared with {adam['wall_time_mean']:.1f}s for Adam and {lbfgs['wall_time_mean']:.1f}s for L-BFGS-B on this subset.

\begin{{figure}}[H]
\centering
\includegraphics[width=\linewidth]{{figures/selected_batch_vs_baselines.pdf}}
\caption{{Selected batch Krotov variant (\texttt{{lr=1.0}}) against Adam and L-BFGS-B. The batch method is competitive in wall-clock time and test accuracy, while Adam and L-BFGS-B still retain a small edge in final training loss.}}
\end{{figure}}

\section*{{Conclusion}}
The new Krotov variants separate two effects that were previously entangled. The preserved online optimizer confirms that the original method derives real benefit from sequential sample-wise updates, but it also exhibits the largest late-stage oscillations. The new full-batch variant removes the objective mismatch and the stale-adjoint sample ordering issue, yielding a cleaner optimization signal. However, it needs a substantially larger effective step size to compete on this benchmark. The overall picture is therefore not ``implementation bug versus no bug'', but rather ``aggressive noisy online dynamics versus smoother objective-aligned batch dynamics, with learning-rate scale playing a decisive role.''

\end{{document}}
"""


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    config = load_config()
    results = load_results(config)
    if not results:
        raise SystemExit("No Krotov-variant results found. Run run_krotov_variant_analysis.py first.")

    grouped_base = group_results(results, BASE_ORDER)
    grouped_sweep = group_results(results, SWEEP_ORDER)
    grouped_all = OrderedDict()
    grouped_all.update(grouped_base)
    for name, runs in grouped_sweep.items():
        grouped_all[name] = runs
    baseline_results = load_baseline_results(config["seeds"])
    grouped_baselines = group_results(baseline_results, ["adam", "lbfgs"])
    grouped_comparison = OrderedDict()
    for name in COMPARISON_ORDER:
        if name in grouped_all:
            grouped_comparison[name] = grouped_all[name]
        elif name in grouped_baselines:
            grouped_comparison[name] = grouped_baselines[name]

    plot_mean_band(
        grouped_base,
        "cost_units",
        "loss",
        BASE_ORDER,
        "Krotov variants: loss vs propagation cost",
        "Propagation cost = sample forwards + sample backwards",
        "Training loss (BCE)",
        "krotov_loss_vs_cost",
    )
    plot_mean_band(
        grouped_base,
        "step",
        "loss",
        BASE_ORDER,
        "Krotov variants: loss vs iteration",
        "Iteration",
        "Training loss (BCE)",
        "krotov_loss_vs_iteration",
    )
    plot_mean_band(
        grouped_base,
        "step",
        "update_norm",
        BASE_ORDER,
        "Krotov variants: update norm vs iteration",
        "Iteration",
        "Update norm",
        "krotov_update_norm_vs_iteration",
    )
    plot_mean_band(
        grouped_base,
        "step",
        "contribution_variance",
        BASE_ORDER,
        "Krotov variants: contribution variance vs iteration",
        "Iteration",
        "Mean per-parameter contribution variance",
        "krotov_contribution_variance",
    )
    plot_sweep_boxplot(grouped_all)
    plot_selected_batch_variants(grouped_all)
    plot_selected_batch_vs_baselines(grouped_comparison, COMPARISON_ORDER)

    summary = summarize_runs(grouped_all, config["loss_threshold"])
    baseline_summary = summarize_baseline_comparison(grouped_comparison)
    tex = build_report_tex(config, summary, baseline_summary)
    tex_path = os.path.join(REPORT_DIR, "krotov_variant_analysis.tex")
    with open(tex_path, "w") as f:
        f.write(tex)

    subprocess.run(
        [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "krotov_variant_analysis.tex",
        ],
        cwd=REPORT_DIR,
        check=True,
    )

    summary_path = os.path.join(REPORT_DIR, "krotov_variant_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    baseline_summary_path = os.path.join(REPORT_DIR, "baseline_comparison_summary.json")
    with open(baseline_summary_path, "w") as f:
        json.dump(baseline_summary, f, indent=2)

    print(f"Report written to {tex_path}")
    print(f"Compiled PDF: {os.path.join(REPORT_DIR, 'krotov_variant_analysis.pdf')}")


if __name__ == "__main__":
    main()
