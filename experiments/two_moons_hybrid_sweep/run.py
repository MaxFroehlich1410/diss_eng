#!/usr/bin/env python3
"""Hybrid Krotov hyperparameter sweep across three QML models.

Sweeps hybrid_switch_iteration, hybrid_online_step_size, and
hybrid_batch_step_size on:
  1. HEA 4-qubit on binary Iris (100 iterations, 3 seeds)
  2. Simonetti full hybrid on two-moons (20 iterations, 3 seeds)
  3. Chen SUN-VQC on two-moons (20 iterations, 3 seeds)

Usage:
    python run_hybrid_krotov_sweep.py
    python run_hybrid_krotov_sweep.py --models hea simonetti chen
    python run_hybrid_krotov_sweep.py --models hea
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

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results_hybrid_sweep")

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
            "early_stopping_enabled": False,
        },
        "sweep_grid": {
            "switch": [3, 5, 10, 20],
            "online_step": [0.1, 0.3, 0.5, 1.0],
            "batch_step": [0.5, 1.0, 2.0, 3.0],
        },
        "sweep_grid_extended": {
            "switch": [30, 50],
            "online_step": [1.5, 2.0],
            "batch_step": [1.0, 2.0, 3.0],
        },
        "seeds": [0, 1, 2],
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
            "input_encoding": "linear_pm_pi",
            "early_stopping_enabled": False,
        },
        "sweep_grid": {
            "switch": [3, 5, 10, 15],
            "online_step": [0.01, 0.03, 0.05, 0.1],
            "batch_step": [0.03, 0.05, 0.1, 0.3],
        },
        "sweep_grid_extended": {
            "switch": [3, 5, 10],
            "online_step": [0.1, 0.15, 0.2],
            "batch_step": [0.5, 0.7, 1.0],
        },
        "seeds": [0, 1, 2],
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
            "input_encoding": "linear_pm_pi",
            "early_stopping_enabled": False,
        },
        "sweep_grid": {
            "switch": [3, 5, 10, 15],
            "online_step": [0.005, 0.01, 0.03, 0.05],
            "batch_step": [0.01, 0.03, 0.05, 0.1],
        },
        "sweep_grid_extended": {
            "switch": [10, 15, 18],
            "online_step": [0.07, 0.1, 0.15],
            "batch_step": [0.15, 0.2, 0.3],
        },
        "seeds": [0, 1, 2],
        "loss_threshold": 0.45,
    }),
])

GRID_LEVELS = ["default", "extended"]

def _grid_key(grid_level):
    return "sweep_grid" if grid_level == "default" else "sweep_grid_extended"


# ── Single run ────────────────────────────────────────────────────────────
def run_single(model_key, switch, online_step, batch_step, seed):
    spec = MODEL_SPECS[model_key]
    model = spec["build_model"]()
    data = spec["load_data"](seed)
    X_train, X_test, y_train, y_test = data[:4]
    init_params = model.init_params(seed=seed)

    config = replace(
        ExperimentConfig(),
        hybrid_switch_iteration=switch,
        hybrid_online_step_size=online_step,
        hybrid_batch_step_size=batch_step,
        hybrid_online_schedule="constant",
        hybrid_batch_schedule="constant",
        **spec["base_config"],
    )

    t0 = time.time()
    final_params, trace = run_optimizer(
        "krotov_hybrid", model, init_params.copy(),
        X_train, y_train, X_test, y_test, config,
    )
    wall = time.time() - t0

    losses = np.asarray(trace["loss"], dtype=float)
    walls = np.asarray(trace["wall_time"], dtype=float)
    threshold = spec["loss_threshold"]
    hits = np.where(losses <= threshold)[0]

    return {
        "model": model_key,
        "switch": switch,
        "online_step": online_step,
        "batch_step": batch_step,
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
def run_sweep(model_key, results_dir, grid_level="default"):
    spec = MODEL_SPECS[model_key]
    grid = spec[_grid_key(grid_level)]
    seeds = spec["seeds"]
    combos = list(itertools.product(
        grid["switch"], grid["online_step"], grid["batch_step"],
    ))
    total = len(combos) * len(seeds)
    print(f"\n{'#' * 72}")
    print(f"# Sweep: {spec['label']}  ({len(combos)} configs × {len(seeds)} seeds = {total} runs)")
    print(f"{'#' * 72}")

    all_results = []
    for i, (sw, os_, bs) in enumerate(combos):
        for seed in seeds:
            idx = i * len(seeds) + seeds.index(seed) + 1
            print(f"  [{idx:3d}/{total}] sw={sw:2d} online={os_:.3f} batch={bs:.3f} seed={seed}", end="", flush=True)
            result = run_single(model_key, sw, os_, bs, seed)
            all_results.append(result)
            print(f"  loss={result['final_loss']:.4f}  acc={result['final_test_acc']:.3f}  wall={result['wall_time']:.1f}s")

    suffix = f"_{grid_level}" if grid_level != "default" else ""
    out_path = os.path.join(results_dir, f"sweep_{model_key}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  → Saved {len(all_results)} results to {out_path}")
    return all_results


# ── Analysis ──────────────────────────────────────────────────────────────
def analyze_sweep(model_key, results, results_dir, grid_level="default"):
    spec = MODEL_SPECS[model_key]
    threshold = spec["loss_threshold"]

    configs = OrderedDict()
    for r in results:
        key = (r["switch"], r["online_step"], r["batch_step"])
        configs.setdefault(key, []).append(r)

    rows = []
    for (sw, os_, bs), runs in configs.items():
        losses = np.array([r["final_loss"] for r in runs])
        test_accs = np.array([r["final_test_acc"] for r in runs])
        walls = np.array([r["wall_time"] for r in runs])
        tail_stds = np.array([r["tail_loss_std"] for r in runs])
        reached = [r for r in runs if r["threshold_reached"]]
        ttt = np.mean([r["time_to_threshold"] for r in reached]) if reached else None
        rows.append({
            "switch": sw,
            "online_step": os_,
            "batch_step": bs,
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
            "wall_mean": float(np.mean(walls)),
            "success_rate": len(reached) / len(runs),
            "ttt_mean": float(ttt) if ttt is not None else None,
            "tail_std_mean": float(np.mean(tail_stds)),
        })

    rows.sort(key=lambda r: r["loss_mean"])

    summary = {
        "model": model_key,
        "label": spec["label"],
        "threshold": threshold,
        "n_configs": len(rows),
        "n_seeds": len(spec["seeds"]),
        "best": rows[0],
        "top10": rows[:10],
        "current_baseline": _find_baseline(rows, model_key),
        "all_configs": rows,
    }

    suffix = f"_{grid_level}" if grid_level != "default" else ""
    out_path = os.path.join(results_dir, f"analysis_{model_key}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    grid_tag = f" ({grid_level})" if grid_level != "default" else ""
    print(f"\n{'=' * 72}")
    print(f"  {spec['label']}{grid_tag} — Top 10 configurations (by mean final loss)")
    print(f"{'=' * 72}")
    print(f"  {'Rank':>4} {'sw':>3} {'online':>8} {'batch':>8} {'loss':>12} {'test_acc':>12} {'wall':>8} {'succ':>5} {'ttt':>8} {'tail':>8}")
    print(f"  {'-'*85}")
    for i, r in enumerate(rows[:10]):
        ttt = f"{r['ttt_mean']:.1f}" if r["ttt_mean"] is not None else "--"
        tag = " ◀ current" if _is_baseline(r, model_key) else ""
        print(
            f"  {i+1:4d} {r['switch']:3d} {r['online_step']:8.4f} {r['batch_step']:8.4f} "
            f"{r['loss_mean']:.4f}±{r['loss_std']:.3f} "
            f"{r['test_acc_mean']:.3f}±{r['test_acc_std']:.3f} "
            f"{r['wall_mean']:7.1f} {r['success_rate']:5.2f} {ttt:>8} {r['tail_std_mean']:.4f}{tag}"
        )

    return summary


BASELINES = {
    "hea": (10, 0.3, 1.0),
    "simonetti": (10, 0.02, 0.05),
    "chen": (10, 0.01, 0.03),
}


def _is_baseline(row, model_key):
    bl = BASELINES[model_key]
    return (row["switch"] == bl[0] and
            abs(row["online_step"] - bl[1]) < 1e-6 and
            abs(row["batch_step"] - bl[2]) < 1e-6)


def _find_baseline(rows, model_key):
    for r in rows:
        if _is_baseline(r, model_key):
            return r
    return None


# ── Plotting ──────────────────────────────────────────────────────────────
def generate_plots(model_key, results, summary, results_dir, grid_level="default"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = MODEL_SPECS[model_key]
    label = spec["label"]
    if grid_level != "default":
        label += f" ({grid_level})"
    top5 = summary["top10"][:5]
    baseline = summary["current_baseline"]

    configs_to_plot = list(top5)
    if baseline and not any(_is_baseline(c, model_key) for c in configs_to_plot):
        configs_to_plot.append(baseline)

    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(configs_to_plot)))
    suffix = f"_{grid_level}" if grid_level != "default" else ""

    # ── Loss vs switch_iteration heatmap (best per switch) ────────────
    all_configs = summary["all_configs"]
    grid = spec[_grid_key(grid_level)]

    fig, axes = plt.subplots(1, len(grid["switch"]), figsize=(4.5 * len(grid["switch"]), 4),
                             sharey=True)
    if len(grid["switch"]) == 1:
        axes = [axes]

    for ax, sw in zip(axes, grid["switch"]):
        sub = [r for r in all_configs if r["switch"] == sw]
        online_vals = sorted(set(r["online_step"] for r in sub))
        batch_vals = sorted(set(r["batch_step"] for r in sub))
        mat = np.full((len(online_vals), len(batch_vals)), np.nan)
        for r in sub:
            i = online_vals.index(r["online_step"])
            j = batch_vals.index(r["batch_step"])
            mat[i, j] = r["loss_mean"]

        im = ax.imshow(mat, aspect="auto", origin="lower",
                       cmap="RdYlGn_r", interpolation="nearest")
        ax.set_xticks(range(len(batch_vals)))
        ax.set_xticklabels([f"{v:.3f}" for v in batch_vals], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(online_vals)))
        ax.set_yticklabels([f"{v:.3f}" for v in online_vals], fontsize=8)
        ax.set_xlabel("Batch step")
        if sw == grid["switch"][0]:
            ax.set_ylabel("Online step")
        ax.set_title(f"switch={sw}")
        for i in range(len(online_vals)):
            for j in range(len(batch_vals)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i,j]:.3f}", ha="center", va="center", fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"{label}: Mean final loss by hyperparameters", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, f"sweep_{model_key}{suffix}_heatmap.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(results_dir, f"sweep_{model_key}{suffix}_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Bar chart: top configs vs baseline ────────────────────────────
    fig, ax = plt.subplots(figsize=(max(8, len(configs_to_plot) * 1.2), 4.5))
    labels_bar = []
    means = []
    stds = []
    colors = []
    for i, c in enumerate(configs_to_plot):
        is_bl = _is_baseline(c, model_key)
        labels_bar.append(f"sw={c['switch']}\non={c['online_step']:.3f}\nbat={c['batch_step']:.3f}"
                          + ("\n(current)" if is_bl else ""))
        means.append(c["loss_mean"])
        stds.append(c["loss_std"])
        colors.append("#d97706" if is_bl else "#2563eb")

    x = np.arange(len(labels_bar))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_bar, fontsize=8)
    ax.set_ylabel("Mean final loss")
    ax.set_title(f"{label}: Top configs vs current baseline")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(results_dir, f"sweep_{model_key}{suffix}_bar.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(results_dir, f"sweep_{model_key}{suffix}_bar.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"  Plots saved: sweep_{model_key}{suffix}_heatmap.pdf, sweep_{model_key}{suffix}_bar.pdf")


# ── Report ────────────────────────────────────────────────────────────────
def write_report(summaries, results_dir, grid_level="default"):
    grid_tag = f" ({grid_level})" if grid_level != "default" else ""
    lines = [
        f"# Hybrid Krotov Hyperparameter Sweep Report{grid_tag}",
        "",
        "## Overview",
        "",
        "This sweep searches over three hyperparameters of the hybrid Krotov optimizer:",
        "- `hybrid_switch_iteration`: when to switch from online to batch phase",
        "- `hybrid_online_step_size`: step size during the online phase",
        "- `hybrid_batch_step_size`: step size during the batch phase",
        "",
        "Three models are tested, each with 3 random seeds per configuration.",
        "",
    ]
    for s in summaries:
        spec = MODEL_SPECS[s["model"]]
        grid = spec[_grid_key(grid_level)]
        best = s["best"]
        bl = s["current_baseline"]
        lines += [
            f"## {s['label']}",
            "",
            f"- Grid: switch={grid['switch']}, online={grid['online_step']}, batch={grid['batch_step']}",
            f"- Total configs: {s['n_configs']}",
            f"- Seeds per config: {s['n_seeds']}",
            f"- Loss threshold: {s['threshold']}",
            "",
            f"### Best configuration",
            "",
            f"| Parameter | Value |",
            f"|---|---|",
            f"| switch_iteration | {best['switch']} |",
            f"| online_step_size | {best['online_step']} |",
            f"| batch_step_size | {best['batch_step']} |",
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
            lines += [
                f"### Compared to current baseline (sw={bl['switch']}, on={bl['online_step']}, bat={bl['batch_step']})",
                "",
                f"| Metric | Baseline | Best | Δ |",
                f"|---|---|---|---|",
                f"| Final loss | {bl['loss_mean']:.4f} | {best['loss_mean']:.4f} | {improvement:+.4f} ({pct:+.1f}%) |",
                f"| Test accuracy | {bl['test_acc_mean']:.3f} | {best['test_acc_mean']:.3f} | {best['test_acc_mean'] - bl['test_acc_mean']:+.3f} |",
                f"| Wall time | {bl['wall_mean']:.1f}s | {best['wall_mean']:.1f}s | {best['wall_mean'] - bl['wall_mean']:+.1f}s |",
                "",
            ]

        lines += [
            "### Top 5 configurations",
            "",
            "| Rank | switch | online | batch | loss (mean±std) | test acc | wall (s) |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(s["top10"][:5]):
            lines.append(
                f"| {i+1} | {r['switch']} | {r['online_step']} | {r['batch_step']} | "
                f"{r['loss_mean']:.4f}±{r['loss_std']:.3f} | {r['test_acc_mean']:.3f} | {r['wall_mean']:.1f} |"
            )
        lines += ["", ""]

    suffix = f"_{grid_level}" if grid_level != "default" else ""
    report_path = os.path.join(results_dir, f"sweep_report{suffix}.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="*", default=list(MODEL_SPECS.keys()),
                        choices=list(MODEL_SPECS.keys()))
    parser.add_argument("--grid", default="default", choices=GRID_LEVELS,
                        help="Grid level: 'default' or 'extended'")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    summaries = []
    for model_key in args.models:
        results = run_sweep(model_key, args.results_dir, args.grid)
        summary = analyze_sweep(model_key, results, args.results_dir, args.grid)
        generate_plots(model_key, results, summary, args.results_dir, args.grid)
        summaries.append(summary)

    write_report(summaries, args.results_dir, args.grid)
    print(f"\nAll done. Results in {args.results_dir}/")


if __name__ == "__main__":
    main()
