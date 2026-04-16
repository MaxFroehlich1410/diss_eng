#!/usr/bin/env python3
"""Hyperparameter sweeps for Adam, L-BFGS-B, and QNG across three QML models.

Mirrors the structure of ``run_hybrid_krotov_sweep.py`` so every optimizer
gets a fair, systematic search.

Usage — run one optimizer at a time so they can be parallelised easily:

    python run_optimizer_sweeps.py --optimizer adam
    python run_optimizer_sweeps.py --optimizer lbfgs
    python run_optimizer_sweeps.py --optimizer qng

Or restrict to a single model:

    python run_optimizer_sweeps.py --optimizer adam  --models hea
    python run_optimizer_sweeps.py --optimizer qng   --models simonetti chen
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import OrderedDict
from dataclasses import replace

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from experiments.two_moons_common.config import ExperimentConfig
from datasets import generate_two_moons, load_iris_binary
from qml_models import VQCModel
from qml_models.variants import ChenSUNVQCModel, SimonettiHybridModel
from optimizers.runner import run_optimizer

sys.stdout.reconfigure(line_buffering=True)

RESULTS_BASE = os.path.join(SCRIPT_DIR, "results_optimizer_sweeps")
SEEDS = [0, 1, 2]

# ── Model specifications ─────────────────────────────────────────────────
MODEL_SPECS = OrderedDict([
    ("hea", {
        "label": "HEA (Iris)",
        "build_model": lambda: VQCModel(
            n_qubits=4, n_layers=3, entangler="ring",
            architecture="hea", observable="Z0Z1",
        ),
        "load_data": lambda seed: load_iris_binary(test_fraction=0.2, seed=seed),
        "base_config": {
            "max_iterations": 100,
            "lbfgs_maxiter": 100,
            "early_stopping_enabled": False,
        },
        "loss_threshold": 0.40,
    }),
    ("simonetti", {
        "label": "Simonetti (two-moons)",
        "build_model": lambda: SimonettiHybridModel(mode="hybrid"),
        "load_data": lambda seed: generate_two_moons(
            n_samples=1000, noise=0.05, test_fraction=0.2,
            seed=seed, encoding="linear_pm_pi",
        ),
        "base_config": {
            "max_iterations": 20,
            "lbfgs_maxiter": 20,
            "input_encoding": "linear_pm_pi",
            "early_stopping_enabled": False,
        },
        "loss_threshold": 0.45,
    }),
    ("chen", {
        "label": "Chen SUN-VQC (two-moons)",
        "build_model": lambda: ChenSUNVQCModel(
            n_macro_layers=2, encoding_axes=("y", "z"), readout="simple_z0",
        ),
        "load_data": lambda seed: generate_two_moons(
            n_samples=300, noise=0.07, test_fraction=0.2,
            seed=seed, encoding="linear_pm_pi",
        ),
        "base_config": {
            "max_iterations": 20,
            "lbfgs_maxiter": 20,
            "input_encoding": "linear_pm_pi",
            "early_stopping_enabled": False,
        },
        "loss_threshold": 0.45,
    }),
])

# ── Sweep grids per optimizer per model ───────────────────────────────────
#
# Adam: sweep learning rate
# L-BFGS-B: sweep maxcor (memory) and gtol (gradient tolerance)
# QNG: sweep learning rate and regularisation lambda
#
SWEEP_GRIDS = {
    "default": {
        "adam": {
            "hea":       {"adam_lr": [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]},
            "simonetti": {"adam_lr": [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]},
            "chen":      {"adam_lr": [0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2]},
        },
        "lbfgs": {
            "hea":       {"lbfgs_maxcor": [3, 5, 10, 20, 40], "lbfgs_gtol": [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]},
            "simonetti": {"lbfgs_maxcor": [3, 5, 10, 20, 40], "lbfgs_gtol": [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]},
            "chen":      {"lbfgs_maxcor": [3, 5, 10, 20, 40], "lbfgs_gtol": [1e-5, 1e-6, 1e-7, 1e-8, 1e-9]},
        },
        "qng": {
            "hea":       {"qng_lr": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0], "qng_lam": [0.001, 0.01, 0.1]},
            "simonetti": {"qng_lr": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0], "qng_lam": [0.001, 0.01, 0.1]},
            "chen":      {"qng_lr": [0.01, 0.05, 0.1, 0.5, 1.0, 2.0], "qng_lam": [0.001, 0.01, 0.1]},
        },
    },
    "extended": {
        "adam": {
            "hea":       {"adam_lr": [0.3, 0.7, 1.0]},
            "simonetti": {"adam_lr": [0.3, 0.5, 0.7, 1.0]},
            "chen":      {"adam_lr": [0.3, 0.5, 0.7, 1.0]},
        },
    },
}

# Baselines (current settings used in prior experiments)
BASELINES = {
    "adam": {
        "hea":       {"adam_lr": 0.05},
        "simonetti": {"adam_lr": 0.03},
        "chen":      {"adam_lr": 0.02},
    },
    "lbfgs": {
        "hea":       {"lbfgs_maxcor": 20, "lbfgs_gtol": 1e-7},
        "simonetti": {"lbfgs_maxcor": 20, "lbfgs_gtol": 1e-7},
        "chen":      {"lbfgs_maxcor": 20, "lbfgs_gtol": 1e-7},
    },
    "qng": {
        "hea":       {"qng_lr": 0.5, "qng_lam": 0.01},
        "simonetti": {"qng_lr": 0.5, "qng_lam": 0.01},
        "chen":      {"qng_lr": 0.5, "qng_lam": 0.01},
    },
}


# ── Grid expansion ────────────────────────────────────────────────────────
def expand_grid(grid_dict):
    """Expand a dict of {param_name: [values]} into a list of dicts."""
    names = list(grid_dict.keys())
    value_lists = [grid_dict[n] for n in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*value_lists)]


def hp_label(hp_dict):
    """Short string representation of a hyperparameter combo."""
    parts = []
    for k, v in hp_dict.items():
        short = k.split("_", 1)[-1]
        parts.append(f"{short}={v}")
    return " ".join(parts)


# ── Single run ────────────────────────────────────────────────────────────
def run_single(model_key, optimizer_name, hp_dict, seed):
    spec = MODEL_SPECS[model_key]
    model = spec["build_model"]()
    X_train, X_test, y_train, y_test = spec["load_data"](seed)[:4]
    init_params = model.init_params(seed=seed)

    config = replace(
        ExperimentConfig(),
        **spec["base_config"],
        **hp_dict,
    )

    t0 = time.time()
    final_params, trace = run_optimizer(
        optimizer_name, model, init_params.copy(),
        X_train, y_train, X_test, y_test, config,
    )
    wall = time.time() - t0

    losses = np.asarray(trace["loss"], dtype=float)
    walls = np.asarray(trace["wall_time"], dtype=float)
    threshold = spec["loss_threshold"]
    hits = np.where(losses <= threshold)[0]

    return {
        "model": model_key,
        "optimizer": optimizer_name,
        **hp_dict,
        "seed": seed,
        "final_loss": float(losses[-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "wall_time": float(wall),
        "total_cost_units": int(trace["cost_units"][-1]),
        "total_steps": int(trace["step"][-1]),
        "threshold_reached": len(hits) > 0,
        "time_to_threshold": float(walls[hits[0]]) if len(hits) else None,
        "tail_loss_std": float(np.std(losses[-10:])) if len(losses) > 10 else float(np.std(losses)),
    }


# ── Sweep runner ──────────────────────────────────────────────────────────
def _get_grid(optimizer_name, model_key, grid_level):
    grids = SWEEP_GRIDS.get(grid_level, {})
    opt_grids = grids.get(optimizer_name, {})
    return opt_grids.get(model_key)


def run_sweep(optimizer_name, model_key, results_dir, grid_level="default"):
    spec = MODEL_SPECS[model_key]
    grid = _get_grid(optimizer_name, model_key, grid_level)
    if grid is None:
        print(f"  No {grid_level} grid defined for {optimizer_name}/{model_key}, skipping.")
        return []
    combos = expand_grid(grid)
    total = len(combos) * len(SEEDS)

    print(f"\n{'#' * 72}")
    print(f"# Sweep: {optimizer_name.upper()} on {spec['label']}  "
          f"({len(combos)} configs × {len(SEEDS)} seeds = {total} runs)")
    print(f"{'#' * 72}")

    all_results = []
    for i, hp_dict in enumerate(combos):
        for j, seed in enumerate(SEEDS):
            idx = i * len(SEEDS) + j + 1
            print(f"  [{idx:3d}/{total}] {hp_label(hp_dict)} seed={seed}", end="", flush=True)
            result = run_single(model_key, optimizer_name, hp_dict, seed)
            all_results.append(result)
            print(f"  loss={result['final_loss']:.4f}  acc={result['final_test_acc']:.3f}  "
                  f"wall={result['wall_time']:.1f}s")

    suffix = f"_{grid_level}" if grid_level != "default" else ""
    out_path = os.path.join(results_dir, f"sweep_{optimizer_name}_{model_key}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  → Saved {len(all_results)} results to {out_path}")
    return all_results


# ── Analysis ──────────────────────────────────────────────────────────────
def _config_key(result, param_names):
    """Extract the grid-parameter tuple that identifies a config."""
    return tuple(result[p] for p in param_names)


def analyze_sweep(optimizer_name, model_key, results, results_dir, grid_level="default"):
    spec = MODEL_SPECS[model_key]
    grid = _get_grid(optimizer_name, model_key, grid_level)
    param_names = list(grid.keys())

    configs = OrderedDict()
    for r in results:
        key = _config_key(r, param_names)
        configs.setdefault(key, []).append(r)

    rows = []
    for key, runs in configs.items():
        hp = dict(zip(param_names, key))
        losses = np.array([r["final_loss"] for r in runs])
        test_accs = np.array([r["final_test_acc"] for r in runs])
        walls = np.array([r["wall_time"] for r in runs])
        tail_stds = np.array([r["tail_loss_std"] for r in runs])
        reached = [r for r in runs if r["threshold_reached"]]
        ttt = np.mean([r["time_to_threshold"] for r in reached]) if reached else None
        row = {
            **hp,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
            "wall_mean": float(np.mean(walls)),
            "success_rate": len(reached) / len(runs),
            "ttt_mean": float(ttt) if ttt is not None else None,
            "tail_std_mean": float(np.mean(tail_stds)),
        }
        rows.append(row)

    rows.sort(key=lambda r: r["loss_mean"])

    baseline_hp = BASELINES[optimizer_name][model_key]
    baseline_row = None
    for r in rows:
        if all(_close(r.get(k), baseline_hp[k]) for k in baseline_hp):
            baseline_row = r
            break

    summary = {
        "optimizer": optimizer_name,
        "model": model_key,
        "label": spec["label"],
        "threshold": spec["loss_threshold"],
        "param_names": param_names,
        "n_configs": len(rows),
        "n_seeds": len(SEEDS),
        "best": rows[0],
        "top10": rows[:10],
        "current_baseline": baseline_row,
        "all_configs": rows,
    }

    suffix = f"_{grid_level}" if grid_level != "default" else ""
    out_path = os.path.join(results_dir, f"analysis_{optimizer_name}_{model_key}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    _print_top10(summary, param_names)
    return summary


def _close(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) < max(1e-12, 1e-6 * abs(b))
    return a == b


def _print_top10(summary, param_names):
    label = summary["label"]
    opt = summary["optimizer"].upper()
    print(f"\n{'=' * 72}")
    print(f"  {opt} on {label} — Top 10 configurations (by mean final loss)")
    print(f"{'=' * 72}")

    hdr_params = "  ".join(f"{p.split('_',1)[-1]:>10}" for p in param_names)
    print(f"  {'Rank':>4} {hdr_params}        loss     test_acc     wall  succ      ttt     tail")
    print(f"  {'-' * (30 + 12 * len(param_names))}")

    for i, r in enumerate(summary["top10"]):
        vals = "  ".join(f"{r[p]:>10g}" for p in param_names)
        ttt = f"{r['ttt_mean']:.1f}" if r["ttt_mean"] is not None else "--"
        bl = summary["current_baseline"]
        is_bl = bl is not None and all(_close(r.get(k), bl.get(k)) for k in param_names)
        tag = " ◀ current" if is_bl else ""
        print(
            f"  {i+1:4d} {vals} "
            f"{r['loss_mean']:.4f}±{r['loss_std']:.3f} "
            f"{r['test_acc_mean']:.3f}±{r['test_acc_std']:.3f} "
            f"{r['wall_mean']:7.1f} {r['success_rate']:5.2f} {ttt:>8} {r['tail_std_mean']:.4f}{tag}"
        )


# ── Plotting ──────────────────────────────────────────────────────────────
def generate_plots(optimizer_name, model_key, summary, results_dir, grid_level="default"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = MODEL_SPECS[model_key]
    label = spec["label"]
    if grid_level != "default":
        label += f" ({grid_level})"
    param_names = summary["param_names"]
    all_configs = summary["all_configs"]
    tag = f"_{grid_level}" if grid_level != "default" else ""

    if len(param_names) == 1:
        _plot_1d(optimizer_name + tag, model_key, label, param_names[0], all_configs,
                 summary, results_dir, plt)
    elif len(param_names) == 2:
        _plot_2d(optimizer_name + tag, model_key, label, param_names, all_configs,
                 summary, results_dir, plt)


def _plot_1d(opt, model_key, label, param_name, all_configs, summary, results_dir, plt):
    """Bar chart for 1-D sweeps (Adam)."""
    fig, ax = plt.subplots(figsize=(max(7, len(all_configs) * 0.9), 4.5))
    xs = [r[param_name] for r in all_configs]
    means = [r["loss_mean"] for r in all_configs]
    stds = [r["loss_std"] for r in all_configs]

    bl = summary["current_baseline"]
    colors = []
    for r in all_configs:
        is_bl = bl is not None and _close(r[param_name], bl[param_name])
        colors.append("#d97706" if is_bl else "#2563eb")

    x_pos = np.arange(len(xs))
    ax.bar(x_pos, means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{v:g}" for v in xs], fontsize=9)
    ax.set_xlabel(param_name.split("_", 1)[-1])
    ax.set_ylabel("Mean final loss")
    ax.set_title(f"{opt.upper()} on {label}: loss vs {param_name.split('_',1)[-1]}")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(results_dir, f"sweep_{opt}_{model_key}_bar.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: sweep_{opt}_{model_key}_bar.pdf")


def _plot_2d(opt, model_key, label, param_names, all_configs, summary, results_dir, plt):
    """Heatmap for 2-D sweeps (L-BFGS-B, QNG)."""
    p0, p1 = param_names
    p0_vals = sorted(set(r[p0] for r in all_configs))
    p1_vals = sorted(set(r[p1] for r in all_configs))

    mat = np.full((len(p0_vals), len(p1_vals)), np.nan)
    for r in all_configs:
        i = p0_vals.index(r[p0])
        j = p1_vals.index(r[p1])
        mat[i, j] = r["loss_mean"]

    fig, ax = plt.subplots(figsize=(max(6, len(p1_vals) * 1.4), max(4, len(p0_vals) * 0.9)))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdYlGn_r", interpolation="nearest")
    ax.set_xticks(range(len(p1_vals)))
    ax.set_xticklabels([f"{v:g}" for v in p1_vals], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(p0_vals)))
    ax.set_yticklabels([f"{v:g}" for v in p0_vals], fontsize=8)
    ax.set_xlabel(p1.split("_", 1)[-1])
    ax.set_ylabel(p0.split("_", 1)[-1])
    ax.set_title(f"{opt.upper()} on {label}: mean final loss")

    for i in range(len(p0_vals)):
        for j in range(len(p1_vals)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(results_dir, f"sweep_{opt}_{model_key}_heatmap.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Also make a bar chart of top configs vs baseline
    top5 = summary["top10"][:5]
    bl = summary["current_baseline"]
    configs_to_plot = list(top5)
    if bl and not any(all(_close(c.get(k), bl.get(k)) for k in param_names) for c in configs_to_plot):
        configs_to_plot.append(bl)

    fig, ax = plt.subplots(figsize=(max(8, len(configs_to_plot) * 1.4), 4.5))
    labels_bar = []
    means = []
    stds = []
    colors = []
    for c in configs_to_plot:
        is_bl = bl is not None and all(_close(c.get(k), bl.get(k)) for k in param_names)
        parts = "\n".join(f"{k.split('_',1)[-1]}={c[k]:g}" for k in param_names)
        if is_bl:
            parts += "\n(current)"
        labels_bar.append(parts)
        means.append(c["loss_mean"])
        stds.append(c["loss_std"])
        colors.append("#d97706" if is_bl else "#2563eb")

    x = np.arange(len(labels_bar))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=8)
    ax.set_ylabel("Mean final loss")
    ax.set_title(f"{opt.upper()} on {label}: top configs vs baseline")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(results_dir, f"sweep_{opt}_{model_key}_bar.{ext}"),
                    dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plots saved: sweep_{opt}_{model_key}_heatmap.pdf, sweep_{opt}_{model_key}_bar.pdf")


# ── Report ────────────────────────────────────────────────────────────────
def write_report(optimizer_name, summaries, results_dir, grid_level="default"):
    param_docs = {
        "adam":  "`adam_lr` (learning rate)",
        "lbfgs": "`lbfgs_maxcor` (history/memory size) and `lbfgs_gtol` (gradient tolerance)",
        "qng":   "`qng_lr` (learning rate) and `qng_lam` (Tikhonov regularisation λ)",
    }
    grid_tag = f" ({grid_level})" if grid_level != "default" else ""
    lines = [
        f"# {optimizer_name.upper()} Hyperparameter Sweep Report{grid_tag}",
        "",
        "## Overview",
        "",
        f"Swept parameter(s): {param_docs[optimizer_name]}",
        "",
        f"Three models tested, each with {len(SEEDS)} random seeds per configuration.",
        "",
    ]
    for s in summaries:
        grid = _get_grid(optimizer_name, s["model"], grid_level)
        best = s["best"]
        bl = s["current_baseline"]
        param_names = s["param_names"]

        lines += [
            f"## {s['label']}",
            "",
            f"- Grid: " + ", ".join(f"{k}={grid[k]}" for k in grid),
            f"- Total configs: {s['n_configs']}",
            f"- Seeds per config: {s['n_seeds']}",
            f"- Loss threshold: {s['threshold']}",
            "",
            f"### Best configuration",
            "",
            "| Parameter | Value |",
            "|---|---|",
        ]
        for p in param_names:
            lines.append(f"| {p} | {best[p]} |")
        lines += [
            f"| **Mean final loss** | **{best['loss_mean']:.4f} ± {best['loss_std']:.3f}** |",
            f"| Mean test accuracy | {best['test_acc_mean']:.3f} ± {best['test_acc_std']:.3f} |",
            f"| Mean wall time | {best['wall_mean']:.1f}s |",
            f"| Success rate (≤{s['threshold']}) | {best['success_rate']:.2f} |",
            f"| Tail loss std | {best['tail_std_mean']:.4f} |",
            "",
        ]

        if bl:
            improvement = bl["loss_mean"] - best["loss_mean"]
            pct = 100 * improvement / bl["loss_mean"] if bl["loss_mean"] > 0 else 0
            bl_str = ", ".join(f"{k}={bl[k]}" for k in param_names)
            lines += [
                f"### Compared to current baseline ({bl_str})",
                "",
                "| Metric | Baseline | Best | Δ |",
                "|---|---|---|---|",
                f"| Final loss | {bl['loss_mean']:.4f} | {best['loss_mean']:.4f} | {improvement:+.4f} ({pct:+.1f}%) |",
                f"| Test accuracy | {bl['test_acc_mean']:.3f} | {best['test_acc_mean']:.3f} | {best['test_acc_mean'] - bl['test_acc_mean']:+.3f} |",
                f"| Wall time | {bl['wall_mean']:.1f}s | {best['wall_mean']:.1f}s | {best['wall_mean'] - bl['wall_mean']:+.1f}s |",
                "",
            ]

        lines += [
            "### Top 5 configurations",
            "",
            "| Rank | " + " | ".join(param_names) + " | loss (mean±std) | test acc | wall (s) |",
            "|---" * (3 + len(param_names)) + "|",
        ]
        for i, r in enumerate(s["top10"][:5]):
            vals = " | ".join(f"{r[p]:g}" for p in param_names)
            lines.append(
                f"| {i+1} | {vals} | "
                f"{r['loss_mean']:.4f}±{r['loss_std']:.3f} | {r['test_acc_mean']:.3f} | {r['wall_mean']:.1f} |"
            )
        lines += ["", ""]

    suffix = f"_{grid_level}" if grid_level != "default" else ""
    report_path = os.path.join(results_dir, f"sweep_{optimizer_name}_report{suffix}.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--optimizer", required=True, choices=["adam", "lbfgs", "qng"],
                        help="Which optimizer to sweep")
    parser.add_argument("--models", nargs="*", default=list(MODEL_SPECS.keys()),
                        choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--grid", default="default", choices=list(SWEEP_GRIDS.keys()),
                        help="Grid level: 'default' or 'extended'")
    parser.add_argument("--results-dir", default=RESULTS_BASE)
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = os.path.join(args.results_dir, args.optimizer)
    os.makedirs(results_dir, exist_ok=True)

    summaries = []
    for model_key in args.models:
        results = run_sweep(args.optimizer, model_key, results_dir, args.grid)
        if not results:
            continue
        summary = analyze_sweep(args.optimizer, model_key, results, results_dir, args.grid)
        generate_plots(args.optimizer, model_key, summary, results_dir, args.grid)
        summaries.append(summary)

    if summaries:
        write_report(args.optimizer, summaries, results_dir, args.grid)
    print(f"\nAll done. Results in {results_dir}/")


if __name__ == "__main__":
    main()
