#!/usr/bin/env python3
"""Hyperparameter sweeps for exact-statevector HV-VQE on 1x2 Fermi-Hubbard.

The benchmark uses the 4-qubit, 5-layer Hamiltonian-variational ansatz from
`qml_models.vqe` and compares four optimizer families on the same exact loss
function and the same initial parameters.

Typical usage:

    python -m experiments.vqe_hubbard_1x2.run --optimizer adam
    python -m experiments.vqe_hubbard_1x2.run --optimizer bfgs
    python -m experiments.vqe_hubbard_1x2.run --optimizer qng
    python -m experiments.vqe_hubbard_1x2.run --optimizer krotov_hybrid
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np

from optimizers._vqe_impl import (
    BASELINES,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_SEEDS,
    DEFAULT_TOLERANCE,
    INSTANCE_SPECS,
    SWEEP_GRIDS,
    build_initial_params,
    expand_grid,
    hp_label,
    run_optimizer,
)
from qml_models.vqe import Hubbard1x2HVVQEProblem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(line_buffering=True)

RESULTS_BASE = os.path.join(SCRIPT_DIR, "results", "optimizer_sweeps")

def run_single(
    optimizer_name: str,
    instance_key: str,
    hp_dict: dict[str, float | int],
    seed: int,
    *,
    max_iterations: int,
    tolerance: float,
) -> dict[str, object]:
    instance = INSTANCE_SPECS[instance_key]
    problem = Hubbard1x2HVVQEProblem(U=instance.U)
    theta0 = build_initial_params(seed, problem.n_params)
    initial_energy = problem.energy(theta0)
    exact_ground_energy = problem.exact_ground_energy()

    t0 = time.time()
    final_theta, trace = run_optimizer(
        optimizer_name,
        problem,
        theta0,
        hp_dict,
        max_iterations=max_iterations,
    )
    wall_time = time.time() - t0

    energy_trace = np.asarray(trace["energy"], dtype=float)
    error_trace = np.asarray(trace["energy_error"], dtype=float)
    wall_trace = np.asarray(trace["wall_time"], dtype=float)
    hits = np.where(error_trace <= tolerance)[0]

    return {
        "instance": instance_key,
        "U": instance.U,
        "optimizer": optimizer_name,
        **hp_dict,
        "seed": seed,
        "initial_energy": float(initial_energy),
        "exact_ground_energy": float(exact_ground_energy),
        "final_energy": float(energy_trace[-1]),
        "final_energy_error": float(error_trace[-1]),
        "wall_time": float(wall_time),
        "total_cost_units": int(trace["cost_units"][-1]),
        "total_steps": int(trace["step"][-1]),
        "threshold_reached": len(hits) > 0,
        "time_to_tolerance": float(wall_trace[hits[0]]) if len(hits) else None,
        "tail_energy_std": float(np.std(energy_trace[-10:])) if len(energy_trace) > 10 else float(np.std(energy_trace)),
        "final_theta": np.asarray(final_theta, dtype=float).tolist(),
        "trace": {
            key: [str(value) for value in values] if key == "phase" else [float(value) for value in values]
            for key, values in trace.items()
        },
    }


def run_sweep(
    optimizer_name: str,
    instance_keys: list[str],
    seeds: list[int],
    *,
    max_iterations: int,
    tolerance: float,
    results_dir: str,
) -> list[dict[str, object]]:
    grid = expand_grid(SWEEP_GRIDS[optimizer_name])
    total = len(instance_keys) * len(grid) * len(seeds)
    print(f"\n{'#' * 72}")
    print(
        f"# Sweep: {optimizer_name.upper()} "
        f"({len(instance_keys)} instances × {len(grid)} configs × {len(seeds)} seeds = {total} runs)"
    )
    print(f"{'#' * 72}")

    results: list[dict[str, object]] = []
    run_idx = 0
    for instance_key in instance_keys:
        instance = INSTANCE_SPECS[instance_key]
        for hp_dict in grid:
            for seed in seeds:
                run_idx += 1
                print(
                    f"  [{run_idx:3d}/{total}] {instance.label} {hp_label(hp_dict)} seed={seed}",
                    end="",
                    flush=True,
                )
                result = run_single(
                    optimizer_name,
                    instance_key,
                    hp_dict,
                    seed,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                )
                results.append(result)
                print(
                    f"  E={result['final_energy']:.6f}  "
                    f"err={result['final_energy_error']:.3e}  "
                    f"wall={result['wall_time']:.2f}s"
                )

    out_path = os.path.join(results_dir, f"sweep_{optimizer_name}.json")
    with open(out_path, "w") as handle:
        json.dump(results, handle, indent=2)
    print(f"  → Saved {len(results)} raw runs to {out_path}")
    return results


def _config_key(result: dict[str, object], param_names: list[str]) -> tuple[object, ...]:
    return tuple(result[name] for name in param_names)


def _close(a: object, b: object) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        return abs(float(a) - float(b)) < max(1e-12, 1e-8 * max(abs(float(a)), abs(float(b)), 1.0))
    return a == b


def analyze_sweep(
    optimizer_name: str,
    results: list[dict[str, object]],
    *,
    tolerance: float,
    results_dir: str,
) -> dict[str, object]:
    param_names = list(SWEEP_GRIDS[optimizer_name].keys())
    baseline_hp = BASELINES[optimizer_name]
    per_instance = OrderedDict()

    for instance_key in sorted({result["instance"] for result in results}):
        grouped: OrderedDict[tuple[object, ...], list[dict[str, object]]] = OrderedDict()
        instance_runs = [result for result in results if result["instance"] == instance_key]
        for result in instance_runs:
            key = _config_key(result, param_names)
            grouped.setdefault(key, []).append(result)

        rows = []
        for key, runs in grouped.items():
            hp_dict = dict(zip(param_names, key))
            errors = np.array([run["final_energy_error"] for run in runs], dtype=float)
            energies = np.array([run["final_energy"] for run in runs], dtype=float)
            walls = np.array([run["wall_time"] for run in runs], dtype=float)
            tail_stds = np.array([run["tail_energy_std"] for run in runs], dtype=float)
            reached = [run for run in runs if run["threshold_reached"]]
            times = [run["time_to_tolerance"] for run in reached]
            rows.append(
                {
                    **hp_dict,
                    "energy_mean": float(np.mean(energies)),
                    "energy_std": float(np.std(energies)),
                    "error_mean": float(np.mean(errors)),
                    "error_std": float(np.std(errors)),
                    "wall_mean": float(np.mean(walls)),
                    "success_rate": len(reached) / len(runs),
                    "time_to_tolerance_mean": float(np.mean(times)) if times else None,
                    "tail_std_mean": float(np.mean(tail_stds)),
                }
            )

        rows.sort(key=lambda row: row["error_mean"])
        baseline_row = None
        for row in rows:
            if all(_close(row[name], baseline_hp[name]) for name in param_names):
                baseline_row = row
                break

        per_instance[instance_key] = {
            "instance": instance_key,
            "label": INSTANCE_SPECS[instance_key].label,
            "U": INSTANCE_SPECS[instance_key].U,
            "tolerance": tolerance,
            "param_names": param_names,
            "n_configs": len(rows),
            "n_runs": len(instance_runs),
            "best": rows[0],
            "top10": rows[:10],
            "current_baseline": baseline_row,
            "all_configs": rows,
        }

    summary = {
        "optimizer": optimizer_name,
        "param_names": param_names,
        "instances": per_instance,
    }
    out_path = os.path.join(results_dir, f"analysis_{optimizer_name}.json")
    with open(out_path, "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def print_summary(summary: dict[str, object]) -> None:
    optimizer_name = str(summary["optimizer"]).upper()
    for instance_summary in summary["instances"].values():
        param_names = instance_summary["param_names"]
        print(f"\n{'=' * 72}")
        print(
            f"  {optimizer_name} on {instance_summary['label']} — "
            "Top configurations (by mean final energy error)"
        )
        print(f"{'=' * 72}")
        hdr_params = "  ".join(f"{name:>12}" for name in param_names)
        print(
            f"  {'Rank':>4} {hdr_params}    {'err':>16} {'energy':>16} "
            f"{'wall':>8} {'succ':>5} {'ttt':>8} {'tail':>8}"
        )
        for rank, row in enumerate(instance_summary["top10"][:10], start=1):
            param_values = "  ".join(f"{row[name]:>12g}" for name in param_names)
            time_to_tol = row["time_to_tolerance_mean"]
            ttt = f"{time_to_tol:.2f}" if time_to_tol is not None else "--"
            baseline = instance_summary["current_baseline"]
            is_baseline = baseline is not None and all(
                _close(row[name], baseline[name]) for name in param_names
            )
            tag = " ◀ current" if is_baseline else ""
            print(
                f"  {rank:4d} {param_values} "
                f"{row['error_mean']:.3e}±{row['error_std']:.1e} "
                f"{row['energy_mean']:.6f}±{row['energy_std']:.1e} "
                f"{row['wall_mean']:7.2f} {row['success_rate']:5.2f} {ttt:>8} "
                f"{row['tail_std_mean']:.3e}{tag}"
            )


def write_report(summary: dict[str, object], results_dir: str) -> str:
    optimizer_name = str(summary["optimizer"])
    param_names = summary["param_names"]
    lines = [
        f"# {optimizer_name.upper()} VQE Sweep Report",
        "",
        "Exact-statevector HV-VQE sweep on the 1x2 Fermi-Hubbard instance.",
        "",
        "Swept parameters:",
        "",
    ]
    for name, values in SWEEP_GRIDS[optimizer_name].items():
        lines.append(f"- `{name}`: {values}")
    lines += ["", f"Baseline: `{BASELINES[optimizer_name]}`", ""]

    for instance_summary in summary["instances"].values():
        best = instance_summary["best"]
        baseline = instance_summary["current_baseline"]
        lines += [
            f"## {instance_summary['label']}",
            "",
            f"- U: {instance_summary['U']}",
            f"- Tolerance: {instance_summary['tolerance']}",
            f"- Configurations: {instance_summary['n_configs']}",
            f"- Runs: {instance_summary['n_runs']}",
            "",
            "### Best configuration",
            "",
            "| Parameter | Value |",
            "|---|---|",
        ]
        for name in param_names:
            lines.append(f"| {name} | {best[name]} |")
        lines += [
            f"| Mean final energy | {best['energy_mean']:.6f} ± {best['energy_std']:.3e} |",
            f"| Mean final energy error | {best['error_mean']:.3e} ± {best['error_std']:.1e} |",
            f"| Mean wall time | {best['wall_mean']:.2f}s |",
            f"| Success rate | {best['success_rate']:.2f} |",
            "",
        ]

        if baseline is not None:
            improvement = baseline["error_mean"] - best["error_mean"]
            baseline_desc = ", ".join(f"{name}={baseline[name]}" for name in param_names)
            lines += [
                f"### Compared to baseline ({baseline_desc})",
                "",
                "| Metric | Baseline | Best | Δ |",
                "|---|---|---|---|",
                f"| Final energy error | {baseline['error_mean']:.3e} | {best['error_mean']:.3e} | {improvement:+.3e} |",
                f"| Mean wall time | {baseline['wall_mean']:.2f}s | {best['wall_mean']:.2f}s | {best['wall_mean'] - baseline['wall_mean']:+.2f}s |",
                "",
            ]

        lines += [
            "### Top 5 configurations",
            "",
            "| Rank | " + " | ".join(param_names) + " | error (mean±std) | energy | wall (s) |",
            "|---" * (4 + len(param_names)) + "|",
        ]
        for rank, row in enumerate(instance_summary["top10"][:5], start=1):
            params = " | ".join(f"{row[name]:g}" for name in param_names)
            lines.append(
                f"| {rank} | {params} | {row['error_mean']:.3e}±{row['error_std']:.1e} | "
                f"{row['energy_mean']:.6f} | {row['wall_mean']:.2f} |"
            )
        lines += ["", ""]

    report_path = os.path.join(results_dir, f"sweep_{optimizer_name}_report.md")
    with open(report_path, "w") as handle:
        handle.write("\n".join(lines))
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--optimizer",
        required=True,
        choices=list(SWEEP_GRIDS.keys()),
        help="Optimizer family to sweep.",
    )
    parser.add_argument(
        "--instances",
        nargs="*",
        default=list(INSTANCE_SPECS.keys()),
        choices=list(INSTANCE_SPECS.keys()),
        help="Subset of interaction strengths to run.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds for the shared initial parameters.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help="Maximum outer iterations for Adam, QNG, and hybrid Krotov; passed to BFGS as maxiter.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help="Success threshold on absolute energy error.",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_BASE,
        help="Directory where raw JSON and reports are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optimizer_results_dir = os.path.join(args.results_dir, args.optimizer)
    os.makedirs(optimizer_results_dir, exist_ok=True)

    results = run_sweep(
        args.optimizer,
        list(args.instances),
        list(args.seeds),
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        results_dir=optimizer_results_dir,
    )
    summary = analyze_sweep(
        args.optimizer,
        results,
        tolerance=args.tolerance,
        results_dir=optimizer_results_dir,
    )
    print_summary(summary)
    report_path = write_report(summary, optimizer_results_dir)
    print(f"\nReport written to {report_path}")
    print(f"All done. Results in {optimizer_results_dir}/")


if __name__ == "__main__":
    main()
