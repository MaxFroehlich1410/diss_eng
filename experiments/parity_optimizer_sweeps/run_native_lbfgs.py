#!/usr/bin/env python3
"""Small native L-BFGS sweep for the parity benchmark.

This fills the only missing native optimizer in the merged parity comparison so
the summary report and shaded best-candidate plot can include native L-BFGS.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
MPLCONFIGDIR = os.path.join(REPO_ROOT, ".mplconfig")

os.environ.setdefault("MPLCONFIGDIR", MPLCONFIGDIR)
os.makedirs(MPLCONFIGDIR, exist_ok=True)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets import generate_parity_4bit_unique_split
from optimizers.runner import run_optimizer
from qml_models.variants import ParityRotClassifierModel
from run_configurable_qml_test import RunnerConfig


RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")
SWEEP_GRID = OrderedDict(
    [
        ("lbfgs_maxcor", [10, 20]),
        ("lbfgs_gtol", [1e-6, 1e-7]),
    ]
)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--train-size", type=int, default=10)
    parser.add_argument("--test-size", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--run-name", default="native_lbfgs")
    return parser.parse_args()


def _timestamped_dir(results_root: str, run_name: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_name or "native_lbfgs"
    path = os.path.join(results_root, f"{timestamp}_{suffix}")
    os.makedirs(path, exist_ok=False)
    return path


def _save_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _serialize_trace(trace):
    payload = {}
    for key, values in trace.items():
        if key == "phase":
            payload[key] = [str(v) for v in values]
        else:
            payload[key] = [float(v) for v in values]
    return payload


def _expand_grid(grid_dict):
    names = list(grid_dict.keys())
    values = [grid_dict[name] for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def _hp_key(hp_dict):
    parts = []
    for key, value in hp_dict.items():
        if isinstance(value, float):
            value_str = f"{value:.12g}"
        else:
            value_str = str(value)
        parts.append(f"{key}={value_str}")
    return "|".join(parts)


def _hp_label(hp_dict):
    return ", ".join(f"{key}={value}" for key, value in hp_dict.items())


def _make_config(hp_dict, max_iterations: int, n_layers: int) -> RunnerConfig:
    return RunnerConfig(
        n_samples=0,
        test_fraction=0.0,
        model_architecture="parity_rot",
        n_qubits=4,
        n_layers=n_layers,
        observable="Z0",
        max_iterations=max_iterations,
        lbfgs_maxiter=max_iterations,
        lbfgs_maxcor=int(hp_dict["lbfgs_maxcor"]),
        lbfgs_gtol=float(hp_dict["lbfgs_gtol"]),
        early_stopping_enabled=False,
        results_dir="results",
        plots_dir="plots",
        optimizers=["lbfgs"],
        seeds=[0, 1, 2],
    )


def _run_single(seed: int, hp_dict, args):
    X_train, X_test, y_train, y_test = generate_parity_4bit_unique_split(
        train_size=args.train_size,
        test_size=args.test_size,
        seed=seed,
    )
    model = ParityRotClassifierModel(n_layers=args.n_layers)
    init_params = np.asarray(model.init_params(seed=seed), dtype=float)
    config = _make_config(hp_dict, args.max_iterations, args.n_layers)
    start = time.time()
    final_params, trace = run_optimizer(
        "lbfgs",
        model,
        init_params.copy(),
        X_train,
        y_train,
        X_test,
        y_test,
        config,
    )
    wall = time.time() - start
    return {
        "optimizer": "lbfgs_native",
        "seed": int(seed),
        "hp": dict(hp_dict),
        "hp_key": _hp_key(hp_dict),
        "hp_label": _hp_label(hp_dict),
        "train_size": int(args.train_size),
        "test_size": int(args.test_size),
        "n_layers": int(args.n_layers),
        "max_iterations": int(args.max_iterations),
        "wall_time_total": float(wall),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "initial_params": init_params.tolist(),
        "final_params": np.asarray(final_params, dtype=float).tolist(),
        "trace": _serialize_trace(trace),
    }


def _summarize(results):
    grouped = OrderedDict()
    for row in results:
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
                "final_test_acc_mean": float(np.mean(test_accs)),
                "wall_time_mean": float(np.mean(walls)),
            }
        )

    rows.sort(key=lambda row: (-row["final_test_acc_mean"], row["final_loss_mean"], row["wall_time_mean"]))
    return {
        "optimizer": "lbfgs_native",
        "n_configs": len(rows),
        "best": rows[0] if rows else None,
        "all_configs": rows,
    }


def _write_report(path: str, results, summary) -> None:
    lines = [
        "# Native L-BFGS Parity Sweep",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    best = summary["best"]
    if best:
        lines.extend(
            [
                f"| lbfgs_maxcor | {best['lbfgs_maxcor']} |",
                f"| lbfgs_gtol | {best['lbfgs_gtol']} |",
                f"| mean final test acc | {best['final_test_acc_mean']:.4f} |",
                f"| mean final loss | {best['final_loss_mean']:.4f} |",
                f"| mean wall time | {best['wall_time_mean']:.2f}s |",
            ]
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = _parse_args()
    os.makedirs(args.results_root, exist_ok=True)
    results_dir = _timestamped_dir(args.results_root, args.run_name)

    all_results = []
    combos = _expand_grid(SWEEP_GRID)
    total = len(combos) * len(args.seeds)
    completed = 0
    for hp_dict in combos:
        for seed in args.seeds:
            completed += 1
            print(f"[{completed:2d}/{total}] {_hp_label(hp_dict)} | seed={seed}", flush=True)
            result = _run_single(seed, hp_dict, args)
            all_results.append(result)
            print(
                f"       loss={result['final_loss']:.4f} train_acc={result['final_train_acc']:.3f} "
                f"test_acc={result['final_test_acc']:.3f} wall={result['wall_time_total']:.2f}s",
                flush=True,
            )

    summary = _summarize(all_results)
    _save_json(os.path.join(results_dir, "raw_results.json"), all_results)
    _save_json(os.path.join(results_dir, "analysis.json"), summary)
    _write_report(os.path.join(results_dir, "experiment.md"), all_results, summary)
    print(results_dir)


if __name__ == "__main__":
    main()
