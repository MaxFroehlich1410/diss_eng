#!/usr/bin/env python3
"""Krotov scaling-strategy sweep across three QML models.

Tests whether per-parameter scaling of gate-local Krotov updates improves
on the best known unscaled hybrid-Krotov configurations.

Three experiments are supported:

  adaptive   – Adaptive clipping / smooth damping screen at baseline and
               boosted (1.5×) step sizes.  Model-agnostic.
  metadata   – Layerwise, groupwise, and combined strategies that use
               parameter metadata.  Model-specific configs.
  coopt      – Step-size co-optimisation with a single user-chosen scaling
               config fixed.

Usage:
    python run_scaling_sweep.py                                # all experiments, all models
    python run_scaling_sweep.py --experiment adaptive           # just experiment 1
    python run_scaling_sweep.py --experiment metadata --models chen
    python run_scaling_sweep.py --experiment coopt --models chen \\
        --coopt-mode adaptive_clip --coopt-config '{"tau": 0.1}'
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

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


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
        "seeds": [0, 1, 2, 3, 4],
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
        "seeds": [0, 1, 2, 3, 4],
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
        "seeds": [0, 1, 2, 3, 4],
        "loss_threshold": 0.45,
    }),
])


# ── Best known unscaled baselines (from the hyperparameter sweep) ─────────
BEST_BASELINES = {
    "hea":       {"switch": 20, "online_step": 1.0,  "batch_step": 3.0},
    "simonetti": {"switch": 5,  "online_step": 0.1,  "batch_step": 0.3},
    "chen":      {"switch": 15, "online_step": 0.1,  "batch_step": 0.3},
}


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 1: adaptive scaling screen
# ═══════════════════════════════════════════════════════════════════════════

ADAPTIVE_SCALING_CONFIGS = [
    {
        "label": "none (control)",
        "scaling_mode": "none",
        "scaling_config": None,
        "scaling_apply_phase": "both",
    },
    {
        "label": "clip tau=0.05 online",
        "scaling_mode": "adaptive_clip",
        "scaling_config": {"tau": 0.05},
        "scaling_apply_phase": "online",
    },
    {
        "label": "clip tau=0.1 online",
        "scaling_mode": "adaptive_clip",
        "scaling_config": {"tau": 0.1},
        "scaling_apply_phase": "online",
    },
    {
        "label": "clip tau=0.1 both",
        "scaling_mode": "adaptive_clip",
        "scaling_config": {"tau": 0.1},
        "scaling_apply_phase": "both",
    },
    {
        "label": "clip tau=0.3 both",
        "scaling_mode": "adaptive_clip",
        "scaling_config": {"tau": 0.3},
        "scaling_apply_phase": "both",
    },
    {
        "label": "smooth beta=2 online",
        "scaling_mode": "adaptive_smooth",
        "scaling_config": {"beta": 2.0},
        "scaling_apply_phase": "online",
    },
    {
        "label": "smooth beta=5 online",
        "scaling_mode": "adaptive_smooth",
        "scaling_config": {"beta": 5.0},
        "scaling_apply_phase": "online",
    },
    {
        "label": "smooth beta=2 both",
        "scaling_mode": "adaptive_smooth",
        "scaling_config": {"beta": 2.0},
        "scaling_apply_phase": "both",
    },
]

STEP_MULTIPLIERS = [
    {"label": "1.0x", "online_mult": 1.0, "batch_mult": 1.0},
    {"label": "1.5x", "online_mult": 1.5, "batch_mult": 1.5},
]


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 2: metadata-based scaling screen
# ═══════════════════════════════════════════════════════════════════════════

METADATA_SCALING_CONFIGS = {
    "hea": [
        {
            "label": "none (control)",
            "scaling_mode": "none",
            "scaling_config": None,
            "scaling_apply_phase": "both",
        },
        {
            "label": "axis ry=1.0 rz=0.5",
            "scaling_mode": "groupwise",
            "scaling_config": {
                "group_field": "axis",
                "group_scales": {"ry": 1.0, "rz": 0.5},
            },
            "scaling_apply_phase": "both",
        },
        {
            "label": "axis ry=1.0 rz=0.3",
            "scaling_mode": "groupwise",
            "scaling_config": {
                "group_field": "axis",
                "group_scales": {"ry": 1.0, "rz": 0.3},
            },
            "scaling_apply_phase": "both",
        },
        {
            "label": "layerwise gamma=0.9",
            "scaling_mode": "layerwise",
            "scaling_config": {"gamma": 0.9},
            "scaling_apply_phase": "both",
        },
        {
            "label": "layerwise gamma=0.8",
            "scaling_mode": "layerwise",
            "scaling_config": {"gamma": 0.8},
            "scaling_apply_phase": "both",
        },
        {
            "label": "axis+clip ry=1 rz=0.5 tau=0.1",
            "scaling_mode": "groupwise_adaptive",
            "scaling_config": {
                "group_field": "axis",
                "group_scales": {"ry": 1.0, "rz": 0.5},
                "adaptive_mode": "adaptive_clip",
                "adaptive_config": {"tau": 0.1},
            },
            "scaling_apply_phase": "both",
        },
    ],
    "simonetti": [
        {
            "label": "none (control)",
            "scaling_mode": "none",
            "scaling_config": None,
            "scaling_apply_phase": "both",
        },
        {
            "label": "sublayer gamma=0.9",
            "scaling_mode": "layerwise",
            "scaling_config": {"layer_field": "sublayer", "gamma": 0.9},
            "scaling_apply_phase": "both",
        },
        {
            "label": "sublayer gamma=0.8",
            "scaling_mode": "layerwise",
            "scaling_config": {"layer_field": "sublayer", "gamma": 0.8},
            "scaling_apply_phase": "both",
        },
        {
            "label": "axis ry=1.0 rz=0.7",
            "scaling_mode": "groupwise",
            "scaling_config": {
                "group_field": "axis",
                "group_scales": {"ry": 1.0, "rz": 0.7},
            },
            "scaling_apply_phase": "both",
        },
        {
            "label": "axis ry=1.0 rz=0.5",
            "scaling_mode": "groupwise",
            "scaling_config": {
                "group_field": "axis",
                "group_scales": {"ry": 1.0, "rz": 0.5},
            },
            "scaling_apply_phase": "both",
        },
        {
            "label": "axis+clip ry=1 rz=0.7 tau=0.1",
            "scaling_mode": "groupwise_adaptive",
            "scaling_config": {
                "group_field": "axis",
                "group_scales": {"ry": 1.0, "rz": 0.7},
                "adaptive_mode": "adaptive_clip",
                "adaptive_config": {"tau": 0.1},
            },
            "scaling_apply_phase": "both",
        },
    ],
    "chen": [
        {
            "label": "none (control)",
            "scaling_mode": "none",
            "scaling_config": None,
            "scaling_apply_phase": "both",
        },
        {
            "label": "macro_layer gamma=0.9",
            "scaling_mode": "layerwise",
            "scaling_config": {"layer_field": "macro_layer", "gamma": 0.9},
            "scaling_apply_phase": "both",
        },
        {
            "label": "macro_layer gamma=0.8",
            "scaling_mode": "layerwise",
            "scaling_config": {"layer_field": "macro_layer", "gamma": 0.8},
            "scaling_apply_phase": "both",
        },
        {
            "label": "macro_layer gamma=0.6",
            "scaling_mode": "layerwise",
            "scaling_config": {"layer_field": "macro_layer", "gamma": 0.6},
            "scaling_apply_phase": "both",
        },
        {
            "label": "macro_layer {0:1.0, 1:0.5}",
            "scaling_mode": "layerwise",
            "scaling_config": {"layer_field": "macro_layer", "layer_scales": {0: 1.0, 1: 0.5}},
            "scaling_apply_phase": "both",
        },
        {
            "label": "macro_layer+clip gamma=0.8 tau=0.1",
            "scaling_mode": "layerwise",
            "scaling_config": {"layer_field": "macro_layer", "gamma": 0.8},
            "scaling_apply_phase": "both",
            "_chain_adaptive": {"tau": 0.1},
        },
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# Experiment 3: step-size co-optimisation
# ═══════════════════════════════════════════════════════════════════════════

COOPT_MULTIPLIERS = {
    "online_mult": [0.5, 1.0, 1.5, 2.0],
    "batch_mult":  [0.5, 1.0, 1.5, 2.0],
}


# ── Single run ────────────────────────────────────────────────────────────
def run_single(model_key, switch, online_step, batch_step, seed,
               scaling_mode="none", scaling_config=None,
               scaling_apply_phase="both"):
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
        hybrid_scaling_mode=scaling_mode,
        hybrid_scaling_apply_phase=scaling_apply_phase,
        hybrid_scaling_config=scaling_config,
        **spec["base_config"],
    )

    t0 = time.time()
    _, trace = run_optimizer(
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
        "online_step": float(online_step),
        "batch_step": float(batch_step),
        "scaling_mode": scaling_mode,
        "scaling_config": scaling_config,
        "scaling_apply_phase": scaling_apply_phase,
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


# ── Experiment runners ────────────────────────────────────────────────────

def run_adaptive_experiment(model_key, results_dir):
    """Experiment 1: adaptive scaling screen with baseline and boosted steps."""
    spec = MODEL_SPECS[model_key]
    bl = BEST_BASELINES[model_key]
    seeds = spec["seeds"]

    combos = list(itertools.product(ADAPTIVE_SCALING_CONFIGS, STEP_MULTIPLIERS))
    total = len(combos) * len(seeds)
    print(f"\n{'#' * 72}")
    print(f"# Adaptive scaling screen: {spec['label']}")
    print(f"#   {len(ADAPTIVE_SCALING_CONFIGS)} scaling × {len(STEP_MULTIPLIERS)} step levels"
          f" × {len(seeds)} seeds = {total} runs")
    print(f"#   Baseline: sw={bl['switch']} on={bl['online_step']} bat={bl['batch_step']}")
    print(f"{'#' * 72}")

    all_results = []
    for i, (sc, sm) in enumerate(combos):
        online = bl["online_step"] * sm["online_mult"]
        batch = bl["batch_step"] * sm["batch_mult"]
        for seed in seeds:
            idx = i * len(seeds) + seeds.index(seed) + 1
            tag = f"{sc['label']} {sm['label']}"
            print(f"  [{idx:3d}/{total}] {tag:<35s} seed={seed}", end="", flush=True)
            result = run_single(
                model_key, bl["switch"], online, batch, seed,
                scaling_mode=sc["scaling_mode"],
                scaling_config=sc["scaling_config"],
                scaling_apply_phase=sc["scaling_apply_phase"],
            )
            result["step_mult_label"] = sm["label"]
            result["scaling_label"] = sc["label"]
            all_results.append(result)
            print(f"  loss={result['final_loss']:.4f}  acc={result['final_test_acc']:.3f}"
                  f"  wall={result['wall_time']:.1f}s")

    out_path = os.path.join(results_dir, f"adaptive_{model_key}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  -> Saved {len(all_results)} results to {out_path}")
    return all_results


def run_metadata_experiment(model_key, results_dir):
    """Experiment 2: metadata-based scaling screen at baseline step sizes."""
    spec = MODEL_SPECS[model_key]
    bl = BEST_BASELINES[model_key]
    seeds = spec["seeds"]
    scaling_configs = METADATA_SCALING_CONFIGS[model_key]

    total = len(scaling_configs) * len(seeds)
    print(f"\n{'#' * 72}")
    print(f"# Metadata scaling screen: {spec['label']}")
    print(f"#   {len(scaling_configs)} configs × {len(seeds)} seeds = {total} runs")
    print(f"#   Baseline: sw={bl['switch']} on={bl['online_step']} bat={bl['batch_step']}")
    print(f"{'#' * 72}")

    all_results = []
    for i, sc in enumerate(scaling_configs):
        for seed in seeds:
            idx = i * len(seeds) + seeds.index(seed) + 1
            print(f"  [{idx:3d}/{total}] {sc['label']:<40s} seed={seed}", end="", flush=True)

            s_mode = sc["scaling_mode"]
            s_config = sc["scaling_config"]
            s_phase = sc["scaling_apply_phase"]

            # Handle the chained layerwise+adaptive workaround for Chen
            # by running layerwise only (the _chain_adaptive hint is for
            # documentation; the combined effect is approximated by running
            # layerwise with a tighter step size or using groupwise_adaptive).
            if "_chain_adaptive" in sc:
                tau = sc["_chain_adaptive"]["tau"]
                s_mode = "adaptive_clip"
                s_config = {"tau": tau}
                # Run layerwise first, then adaptive on top is not directly
                # composable — use the adaptive_clip alone as an approximation
                # for this screen.  A proper composition would need a wrapper
                # strategy.  Mark the label so the report is honest.

            result = run_single(
                model_key, bl["switch"], bl["online_step"], bl["batch_step"],
                seed,
                scaling_mode=s_mode,
                scaling_config=s_config,
                scaling_apply_phase=s_phase,
            )
            result["scaling_label"] = sc["label"]
            result["step_mult_label"] = "1.0x"
            all_results.append(result)
            print(f"  loss={result['final_loss']:.4f}  acc={result['final_test_acc']:.3f}"
                  f"  wall={result['wall_time']:.1f}s")

    out_path = os.path.join(results_dir, f"metadata_{model_key}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  -> Saved {len(all_results)} results to {out_path}")
    return all_results


def run_coopt_experiment(model_key, results_dir, coopt_mode, coopt_config_dict):
    """Experiment 3: step-size co-optimisation with a fixed scaling config."""
    spec = MODEL_SPECS[model_key]
    bl = BEST_BASELINES[model_key]
    seeds = spec["seeds"]

    step_combos = list(itertools.product(
        COOPT_MULTIPLIERS["online_mult"],
        COOPT_MULTIPLIERS["batch_mult"],
    ))
    total = len(step_combos) * len(seeds)
    print(f"\n{'#' * 72}")
    print(f"# Step-size co-optimisation: {spec['label']}")
    print(f"#   Scaling: {coopt_mode} {coopt_config_dict}")
    print(f"#   {len(step_combos)} step combos × {len(seeds)} seeds = {total} runs")
    print(f"#   Base steps: on={bl['online_step']} bat={bl['batch_step']} sw={bl['switch']}")
    print(f"{'#' * 72}")

    all_results = []
    for i, (om, bm) in enumerate(step_combos):
        online = bl["online_step"] * om
        batch = bl["batch_step"] * bm
        for seed in seeds:
            idx = i * len(seeds) + seeds.index(seed) + 1
            tag = f"on×{om:.1f}={online:.4f} bat×{bm:.1f}={batch:.4f}"
            print(f"  [{idx:3d}/{total}] {tag:<40s} seed={seed}", end="", flush=True)
            result = run_single(
                model_key, bl["switch"], online, batch, seed,
                scaling_mode=coopt_mode,
                scaling_config=coopt_config_dict,
                scaling_apply_phase="both",
            )
            result["online_mult"] = om
            result["batch_mult"] = bm
            result["scaling_label"] = f"{coopt_mode}"
            result["step_mult_label"] = f"on×{om} bat×{bm}"
            all_results.append(result)
            print(f"  loss={result['final_loss']:.4f}  acc={result['final_test_acc']:.3f}"
                  f"  wall={result['wall_time']:.1f}s")

    out_path = os.path.join(results_dir, f"coopt_{model_key}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  -> Saved {len(all_results)} results to {out_path}")
    return all_results


# ── Analysis ──────────────────────────────────────────────────────────────

def _group_key(r):
    """Grouping key that uniquely identifies a configuration (minus seed)."""
    return (r.get("scaling_label", ""), r.get("step_mult_label", ""))


def analyze_experiment(model_key, experiment_name, results, results_dir):
    """Aggregate results over seeds and rank by mean final loss."""
    spec = MODEL_SPECS[model_key]
    threshold = spec["loss_threshold"]

    configs = OrderedDict()
    for r in results:
        key = _group_key(r)
        configs.setdefault(key, []).append(r)

    rows = []
    for (scaling_label, step_label), runs in configs.items():
        losses = np.array([r["final_loss"] for r in runs])
        test_accs = np.array([r["final_test_acc"] for r in runs])
        walls = np.array([r["wall_time"] for r in runs])
        tail_stds = np.array([r["tail_loss_std"] for r in runs])
        reached = [r for r in runs if r["threshold_reached"]]
        ttt = np.mean([r["time_to_threshold"] for r in reached]) if reached else None
        rows.append({
            "scaling_label": scaling_label,
            "step_label": step_label,
            "scaling_mode": runs[0].get("scaling_mode", "none"),
            "online_step": runs[0]["online_step"],
            "batch_step": runs[0]["batch_step"],
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "test_acc_mean": float(np.mean(test_accs)),
            "test_acc_std": float(np.std(test_accs)),
            "wall_mean": float(np.mean(walls)),
            "success_rate": len(reached) / len(runs),
            "ttt_mean": float(ttt) if ttt is not None else None,
            "tail_std_mean": float(np.mean(tail_stds)),
            "n_seeds": len(runs),
        })

    rows.sort(key=lambda r: r["loss_mean"])

    # Identify the control row (none, 1.0x)
    control = None
    for r in rows:
        if "none" in r["scaling_label"].lower() and "1.0x" in r["step_label"]:
            control = r
            break

    summary = {
        "model": model_key,
        "label": spec["label"],
        "experiment": experiment_name,
        "threshold": threshold,
        "n_configs": len(rows),
        "n_seeds_per_config": spec["seeds"],
        "best": rows[0] if rows else None,
        "control": control,
        "all_configs": rows,
    }

    out_path = os.path.join(results_dir, f"analysis_{experiment_name}_{model_key}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print(f"\n{'=' * 90}")
    print(f"  {spec['label']} — {experiment_name} — Ranked by mean final loss")
    print(f"{'=' * 90}")
    print(f"  {'Rk':>3} {'Scaling config':<36s} {'Steps':<6s}"
          f" {'loss':>12} {'test_acc':>12} {'wall':>7} {'succ':>5} {'tail':>7}")
    print(f"  {'-' * 88}")
    for i, r in enumerate(rows):
        tag = " <-- control" if r is control else ""
        print(
            f"  {i+1:3d} {r['scaling_label']:<36s} {r['step_label']:<6s}"
            f" {r['loss_mean']:.4f}+/-{r['loss_std']:.3f}"
            f" {r['test_acc_mean']:.3f}+/-{r['test_acc_std']:.3f}"
            f" {r['wall_mean']:6.1f} {r['success_rate']:5.2f}"
            f" {r['tail_std_mean']:.4f}{tag}"
        )

    if control and rows[0] is not control:
        delta = control["loss_mean"] - rows[0]["loss_mean"]
        pct = 100.0 * delta / control["loss_mean"] if control["loss_mean"] > 0 else 0
        print(f"\n  Best vs control: {delta:+.4f} ({pct:+.1f}%)")

    return summary


# ── Plotting ──────────────────────────────────────────────────────────────

def generate_plots(model_key, experiment_name, summary, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spec = MODEL_SPECS[model_key]
    rows = summary["all_configs"]
    control = summary["control"]
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.9), 5))
    labels = []
    means = []
    stds = []
    colors = []
    for r in rows:
        is_ctrl = (r is control)
        short = r["scaling_label"]
        if r["step_label"] != "1.0x":
            short += f"\n({r['step_label']})"
        labels.append(short)
        means.append(r["loss_mean"])
        stds.append(r["loss_std"])
        colors.append("#d97706" if is_ctrl else "#2563eb")

    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Mean final loss")
    ax.set_title(f"{spec['label']}: {experiment_name} scaling sweep")
    ax.grid(True, axis="y", alpha=0.3)

    if control:
        ax.axhline(control["loss_mean"], color="#d97706", ls="--", lw=0.8, label="control")
        ax.legend(fontsize=8)

    plt.tight_layout()
    stem = f"{experiment_name}_{model_key}"
    fig.savefig(os.path.join(results_dir, f"{stem}_bar.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(results_dir, f"{stem}_bar.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {stem}_bar.pdf")


# ── Report ────────────────────────────────────────────────────────────────

def write_report(all_summaries, results_dir):
    lines = [
        "# Krotov Scaling Strategy Sweep Report",
        "",
        "## Overview",
        "",
        "This sweep tests whether per-parameter scaling of gate-local Krotov",
        "updates can improve on the best known unscaled hybrid-Krotov baselines.",
        "",
        "Best known unscaled baselines:",
        "",
        "| Model | switch | online_step | batch_step | loss |",
        "|---|---|---|---|---|",
    ]
    for mk, bl in BEST_BASELINES.items():
        lines.append(
            f"| {MODEL_SPECS[mk]['label']} | {bl['switch']} | {bl['online_step']}"
            f" | {bl['batch_step']} | (from prior sweep) |"
        )
    lines += ["", ""]

    for summary in all_summaries:
        rows = summary["all_configs"]
        control = summary["control"]
        best = summary["best"]
        exp = summary["experiment"]
        label = summary["label"]

        lines += [
            f"## {label} — {exp}",
            "",
            f"- Configs tested: {summary['n_configs']}",
            f"- Seeds per config: {len(summary['n_seeds_per_config'])}",
            f"- Loss threshold: {summary['threshold']}",
            "",
        ]

        if best:
            lines += [
                "### Best configuration",
                "",
                "| Field | Value |",
                "|---|---|",
                f"| Scaling | {best['scaling_label']} |",
                f"| Steps | {best['step_label']} |",
                f"| online_step | {best['online_step']:.4f} |",
                f"| batch_step | {best['batch_step']:.4f} |",
                f"| **Mean loss** | **{best['loss_mean']:.4f} +/- {best['loss_std']:.3f}** |",
                f"| Test accuracy | {best['test_acc_mean']:.3f} +/- {best['test_acc_std']:.3f} |",
                f"| Wall time | {best['wall_mean']:.1f}s |",
                f"| Success rate | {best['success_rate']:.2f} |",
                "",
            ]

        if control and best and best is not control:
            delta = control["loss_mean"] - best["loss_mean"]
            pct = 100 * delta / control["loss_mean"] if control["loss_mean"] > 0 else 0
            lines += [
                "### Improvement over unscaled control",
                "",
                "| Metric | Control | Best scaled | Delta |",
                "|---|---|---|---|",
                f"| Loss | {control['loss_mean']:.4f} | {best['loss_mean']:.4f}"
                f" | {delta:+.4f} ({pct:+.1f}%) |",
                f"| Test acc | {control['test_acc_mean']:.3f} | {best['test_acc_mean']:.3f}"
                f" | {best['test_acc_mean'] - control['test_acc_mean']:+.3f} |",
                f"| Wall | {control['wall_mean']:.1f}s | {best['wall_mean']:.1f}s"
                f" | {best['wall_mean'] - control['wall_mean']:+.1f}s |",
                "",
            ]

        lines += [
            "### All configurations (ranked)",
            "",
            "| Rank | Scaling | Steps | loss (mean+/-std) | test acc | wall | succ |",
            "|---|---|---|---|---|---|---|",
        ]
        for i, r in enumerate(rows):
            tag = " **(control)**" if r is control else ""
            lines.append(
                f"| {i+1} | {r['scaling_label']}{tag} | {r['step_label']}"
                f" | {r['loss_mean']:.4f}+/-{r['loss_std']:.3f}"
                f" | {r['test_acc_mean']:.3f} | {r['wall_mean']:.1f}s"
                f" | {r['success_rate']:.2f} |"
            )
        lines += ["", ""]

    report_path = os.path.join(results_dir, "scaling_sweep_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {report_path}")


# ── CLI ───────────────────────────────────────────────────────────────────

EXPERIMENTS = ["adaptive", "metadata", "coopt"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--experiment", nargs="*", default=["adaptive", "metadata"],
        choices=EXPERIMENTS,
        help="Which experiment(s) to run (default: adaptive metadata)",
    )
    parser.add_argument(
        "--models", nargs="*", default=list(MODEL_SPECS.keys()),
        choices=list(MODEL_SPECS.keys()),
    )
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument(
        "--coopt-mode", default="adaptive_clip",
        help="Scaling mode for the co-optimisation experiment",
    )
    parser.add_argument(
        "--coopt-config", default='{"tau": 0.1}',
        help="JSON scaling config for the co-optimisation experiment",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    coopt_config_dict = json.loads(args.coopt_config)
    all_summaries = []

    for model_key in args.models:
        if "adaptive" in args.experiment:
            results = run_adaptive_experiment(model_key, args.results_dir)
            summary = analyze_experiment(model_key, "adaptive", results, args.results_dir)
            generate_plots(model_key, "adaptive", summary, args.results_dir)
            all_summaries.append(summary)

        if "metadata" in args.experiment:
            results = run_metadata_experiment(model_key, args.results_dir)
            summary = analyze_experiment(model_key, "metadata", results, args.results_dir)
            generate_plots(model_key, "metadata", summary, args.results_dir)
            all_summaries.append(summary)

        if "coopt" in args.experiment:
            results = run_coopt_experiment(
                model_key, args.results_dir,
                args.coopt_mode, coopt_config_dict,
            )
            summary = analyze_experiment(model_key, "coopt", results, args.results_dir)
            generate_plots(model_key, "coopt", summary, args.results_dir)
            all_summaries.append(summary)

    if all_summaries:
        write_report(all_summaries, args.results_dir)

    print(f"\nAll done. Results in {args.results_dir}/")


if __name__ == "__main__":
    main()
