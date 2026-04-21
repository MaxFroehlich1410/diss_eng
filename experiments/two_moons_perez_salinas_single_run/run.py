#!/usr/bin/env python3
"""Run one Perez-Salinas classification test and plot the learned boundary."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from datasets import generate_perez_salinas_dataset, perez_salinas_problem_num_classes
from optimizers.runner import run_optimizer
from qml_models.variants import PerezSalinasReuploadingModel
from run_configurable_qml_test import RunnerConfig


RESULTS_ROOT = os.path.join(SCRIPT_DIR, "results")

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


@dataclass(frozen=True)
class SingleRunConfig:
    problem: str
    seed: int
    n_qubits: int
    n_layers: int
    n_samples: int
    test_fraction: float
    use_entanglement: bool
    use_classical_head: bool
    optimizer: str
    max_iterations: int
    hybrid_switch_iteration: int
    hybrid_online_step_size: float
    hybrid_batch_step_size: float
    hybrid_online_schedule: str
    hybrid_batch_schedule: str
    hybrid_online_decay: float
    hybrid_batch_decay: float


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", default="crown")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-qubits", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-samples", type=int, default=240)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--no-entanglement", action="store_true")
    parser.add_argument("--no-affine-head", action="store_true")
    parser.add_argument("--optimizer", choices=["krotov_hybrid", "adam", "qng", "lbfgs"], default="krotov_hybrid")
    parser.add_argument("--max-iterations", type=int, default=12)
    parser.add_argument("--hybrid-switch-iteration", type=int, default=10)
    parser.add_argument("--hybrid-online-step-size", type=float, default=0.3)
    parser.add_argument("--hybrid-batch-step-size", type=float, default=0.5)
    parser.add_argument("--hybrid-online-schedule", choices=["constant", "inverse", "exp"], default="constant")
    parser.add_argument("--hybrid-batch-schedule", choices=["constant", "inverse", "exp"], default="constant")
    parser.add_argument("--hybrid-online-decay", type=float, default=0.05)
    parser.add_argument("--hybrid-batch-decay", type=float, default=0.05)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def _timestamped_dir(results_root: str, run_name: str | None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = run_name or "perez_salinas_single_run"
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


def _make_config(args) -> RunnerConfig:
    return RunnerConfig(
        n_samples=args.n_samples,
        test_fraction=args.test_fraction,
        max_iterations=args.max_iterations,
        optimizers=[args.optimizer],
        seeds=[args.seed],
        early_stopping_enabled=False,
        hybrid_switch_iteration=args.hybrid_switch_iteration,
        hybrid_online_step_size=args.hybrid_online_step_size,
        hybrid_batch_step_size=args.hybrid_batch_step_size,
        hybrid_online_schedule=args.hybrid_online_schedule,
        hybrid_batch_schedule=args.hybrid_batch_schedule,
        hybrid_online_decay=args.hybrid_online_decay,
        hybrid_batch_decay=args.hybrid_batch_decay,
        hybrid_scaling_mode="none",
        hybrid_scaling_apply_phase="both",
        hybrid_scaling_config=None,
        results_dir="results",
        plots_dir="plots",
    )


def _assert_no_overlap(X_train, X_test) -> None:
    train_points = {tuple(np.round(row, 12)) for row in np.asarray(X_train, dtype=float)}
    test_points = {tuple(np.round(row, 12)) for row in np.asarray(X_test, dtype=float)}
    overlap = train_points & test_points
    if overlap:
        raise RuntimeError(
            f"Train/test split contains {len(overlap)} duplicate points; aborting to avoid leakage."
        )


def _score_margin(model, params, x):
    scores = model.class_scores(params, x)
    return float(scores[1] - scores[0])


def _plot_boundary(model, final_params, X_train, y_train, X_test, y_test, result, results_dir):
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    cmap_bg = ListedColormap(["#f7d6d6", "#dbe8ff"])

    x_grid = np.linspace(-1.0, 1.0, 150)
    y_grid = np.linspace(-1.0, 1.0, 150)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    margins = np.array([_score_margin(model, final_params, point) for point in grid_points], dtype=float)
    preds = (margins >= 0.0).astype(float).reshape(xx.shape)
    margin_grid = margins.reshape(xx.shape)

    ax.contourf(xx, yy, preds, levels=[-0.5, 0.5, 1.5], cmap=cmap_bg, alpha=0.78)
    ax.contour(xx, yy, margin_grid, levels=[0.0], colors="black", linewidths=1.1)

    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    y_train = np.asarray(y_train, dtype=int)
    y_test = np.asarray(y_test, dtype=int)

    ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], c="#c44e52", s=14, alpha=0.55, label="train class 0")
    ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], c="#4c72b0", s=14, alpha=0.55, label="train class 1")
    ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], c="#7f1d1d", s=24, marker="x", alpha=0.85, label="test class 0")
    ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], c="#1d4ed8", s=24, marker="x", alpha=0.85, label="test class 1")

    ax.set_title(
        "Perez-Salinas decision boundary\n"
        f"loss={result['final_loss']:.3f}, test acc={result['final_test_acc']:.3f}"
    )
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.legend(frameon=False, loc="upper right")

    fig.savefig(os.path.join(results_dir, "decision_boundary.png"))
    fig.savefig(os.path.join(results_dir, "decision_boundary.pdf"))
    plt.close(fig)


def main():
    args = _parse_args()
    os.makedirs(args.results_root, exist_ok=True)
    results_dir = _timestamped_dir(args.results_root, args.run_name)

    X_train, X_test, y_train, y_test = generate_perez_salinas_dataset(
        problem=args.problem,
        n_samples=args.n_samples,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )
    _assert_no_overlap(X_train, X_test)

    model = PerezSalinasReuploadingModel(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        n_classes=perez_salinas_problem_num_classes(args.problem),
        use_entanglement=not args.no_entanglement,
        use_classical_head=not args.no_affine_head,
        loss_mode="weighted_fidelity",
    )
    init_params = np.asarray(model.init_params(seed=args.seed), dtype=float)
    config = _make_config(args)

    start = time.time()
    final_params, trace = run_optimizer(
        args.optimizer,
        model,
        init_params.copy(),
        X_train,
        y_train,
        X_test,
        y_test,
        config,
    )
    wall = time.time() - start

    result = {
        "problem": args.problem,
        "optimizer": args.optimizer,
        "seed": int(args.seed),
        "n_qubits": int(args.n_qubits),
        "n_layers": int(args.n_layers),
        "use_entanglement": bool(not args.no_entanglement),
        "use_classical_head": bool(not args.no_affine_head),
        "n_samples_total": int(args.n_samples),
        "test_fraction": float(args.test_fraction),
        "wall_time_total": float(wall),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "trace": _serialize_trace(trace),
        "initial_params": init_params.tolist(),
        "final_params": np.asarray(final_params, dtype=float).tolist(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "train_test_overlap": 0,
    }

    _save_json(
        os.path.join(results_dir, "config.json"),
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": asdict(
                SingleRunConfig(
                    problem=args.problem,
                    seed=args.seed,
                    n_qubits=args.n_qubits,
                    n_layers=args.n_layers,
                    n_samples=args.n_samples,
                    test_fraction=args.test_fraction,
                    use_entanglement=not args.no_entanglement,
                    use_classical_head=not args.no_affine_head,
                    optimizer=args.optimizer,
                    max_iterations=args.max_iterations,
                    hybrid_switch_iteration=args.hybrid_switch_iteration,
                    hybrid_online_step_size=args.hybrid_online_step_size,
                    hybrid_batch_step_size=args.hybrid_batch_step_size,
                    hybrid_online_schedule=args.hybrid_online_schedule,
                    hybrid_batch_schedule=args.hybrid_batch_schedule,
                    hybrid_online_decay=args.hybrid_online_decay,
                    hybrid_batch_decay=args.hybrid_batch_decay,
                )
            ),
        },
    )
    _save_json(os.path.join(results_dir, "result.json"), result)
    _save_json(
        os.path.join(results_dir, "dataset_summary.json"),
        {
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "train_unique": int(len({tuple(np.round(row, 12)) for row in X_train})),
            "test_unique": int(len({tuple(np.round(row, 12)) for row in X_test})),
            "overlap": 0,
        },
    )

    _plot_boundary(model, np.asarray(final_params, dtype=float), X_train, y_train, X_test, y_test, result, results_dir)

    print(f"Saved single-run Perez-Salinas benchmark to {results_dir}/")


if __name__ == "__main__":
    main()
