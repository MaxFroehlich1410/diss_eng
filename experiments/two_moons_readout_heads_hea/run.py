#!/usr/bin/env python3
"""Compare HEA on binary Iris with and without a trainable affine readout head.

The script uses the best hybrid-Krotov hyperparameters found for the HEA Iris
benchmark in the repository's prior sweep:

- ``hybrid_switch_iteration = 20``
- ``hybrid_online_step_size = 1.0``
- ``hybrid_batch_step_size = 3.0``

To isolate the effect of training ``W`` and ``b``, the affine head uses the
same two quantum features already implicit in the benchmark readout:

    m = [<Z0>, <Z1>]

The baseline keeps the original fixed readout

    p = clip((0.5 * (<Z0> + <Z1>) + 1) / 2)

while the head variant uses

    s = w^T m + b
    p = sigmoid(s)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import OrderedDict
from dataclasses import asdict, replace

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if __package__ in {None, ""}:
    from experiments.two_moons_common.config import DEFAULT_CONFIG
    from datasets import load_iris_binary
    from qml_models import VQCModel
    from qml_models.common import BaseQMLModel, clip_probability, expectation, sigmoid, z_observable
    from optimizers.runner import run_optimizer
else:
    from experiments.two_moons_common.config import DEFAULT_CONFIG
    from datasets import load_iris_binary
    from qml_models import VQCModel
    from qml_models.common import (
        BaseQMLModel,
        clip_probability,
        expectation,
        sigmoid,
        z_observable,
    )
    from optimizers.runner import run_optimizer


RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
BEST_HYBRID_KROTOV_CONFIG = {
    "max_iterations": 100,
    "hybrid_switch_iteration": 20,
    "hybrid_online_step_size": 1.0,
    "hybrid_batch_step_size": 3.0,
    "hybrid_online_schedule": "constant",
    "hybrid_batch_schedule": "constant",
    "early_stopping_enabled": False,
}

class HEAReadoutComparisonModel(BaseQMLModel):
    """HEA benchmark circuit with optional trainable affine readout head."""

    def __init__(self, readout="simple_z0z1"):
        if readout not in {"simple_z0z1", "hybrid_linear"}:
            raise ValueError(f"Unknown HEA readout mode: {readout}")

        self.readout = readout
        self.base_model = VQCModel(
            n_qubits=4,
            n_layers=3,
            entangler="ring",
            architecture="hea",
            observable="Z0Z1",
        )
        self.n_qubits = self.base_model.n_qubits
        self.n_layers = self.base_model.n_layers
        self.dim = self.base_model.dim
        self.n_quantum_params = self.base_model.n_params
        self.measurement_observables = [z_observable(0, self.n_qubits), z_observable(1, self.n_qubits)]
        self.obs = 0.5 * (self.measurement_observables[0] + self.measurement_observables[1])
        self.n_output_params = 0 if self.readout == "simple_z0z1" else 3
        self.n_params = self.n_quantum_params + self.n_output_params
        self._metadata = self._build_parameter_metadata()

    def _quantum_params(self, params):
        return np.asarray(params[: self.n_quantum_params], dtype=float)

    def _output_weight_slice(self):
        return slice(self.n_quantum_params, self.n_quantum_params + 2)

    def _output_bias_index(self):
        return self._output_weight_slice().stop

    def _measurement_vector(self, final_state):
        return np.array(
            [expectation(final_state, observable) for observable in self.measurement_observables],
            dtype=float,
        )

    def _sample_forward_details(self, params, x):
        quantum_params = self._quantum_params(np.asarray(params, dtype=float))
        _, states = self.base_model.get_gate_sequence_and_states(quantum_params, x)
        final_state = states[-1]

        if self.readout == "simple_z0z1":
            prob = float(clip_probability(0.5 * (expectation(final_state, self.obs) + 1.0)))
            return {
                "state": final_state,
                "probability": prob,
            }

        measurement_vector = self._measurement_vector(final_state)
        output_weights = np.asarray(params[self._output_weight_slice()], dtype=float)
        output_bias = float(params[self._output_bias_index()])
        logit = float(output_weights @ measurement_vector + output_bias)
        prob = float(clip_probability(sigmoid(logit)))
        return {
            "state": final_state,
            "measurement_vector": measurement_vector,
            "output_weights": output_weights,
            "output_bias": output_bias,
            "probability": prob,
        }

    def init_params(self, seed=0):
        quantum_params = np.asarray(self.base_model.init_params(seed=seed), dtype=float)
        if self.readout == "simple_z0z1":
            return quantum_params

        rng = np.random.RandomState(seed)
        params = np.zeros(self.n_params, dtype=float)
        params[: self.n_quantum_params] = quantum_params
        params[self._output_weight_slice()] = rng.normal(scale=0.25, size=2)
        params[self._output_bias_index()] = rng.normal(scale=0.1)
        return params

    def parameter_metadata(self):
        return list(self._metadata)

    def gate_parameter_indices(self):
        return list(range(self.n_quantum_params))

    def nongate_parameter_indices(self):
        if self.readout == "simple_z0z1":
            return []
        return list(range(self.n_quantum_params, self.n_params))

    def forward(self, params, x):
        return self._sample_forward_details(np.asarray(params, dtype=float), x)["probability"]

    def terminal_costate(self, params, x, y, final_state):
        if self.readout == "simple_z0z1":
            z = expectation(final_state, self.obs)
            p = clip_probability(0.5 * (z + 1.0))
            dloss_dp = -float(y) / p + (1.0 - float(y)) / (1.0 - p)
            return 0.5 * dloss_dp * (self.obs @ final_state)

        sample = self._sample_forward_details(params, x)
        delta = sample["probability"] - float(y)
        operator = sum(
            weight * observable
            for weight, observable in zip(sample["output_weights"], self.measurement_observables)
        )
        return delta * (operator @ final_state)

    def loss_gradient(self, params, X, y):
        params_arr = np.asarray(params, dtype=float)
        quantum_params = self._quantum_params(params_arr)
        grad = np.zeros_like(params_arr)

        for x_i, y_i in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=float)):
            gates, states = self.base_model.get_gate_sequence_and_states(quantum_params, x_i)
            final_state = states[-1]

            if self.readout == "simple_z0z1":
                z = expectation(final_state, self.obs)
                p = clip_probability(0.5 * (z + 1.0))
                dloss_dp = -y_i / p + (1.0 - y_i) / (1.0 - p)
                chi_states = [None] * len(states)
                chi_states[-1] = self.obs @ final_state
                prefactor = 0.5 * dloss_dp
            else:
                sample = self._sample_forward_details(params_arr, x_i)
                delta = sample["probability"] - y_i
                grad[self._output_weight_slice()] += delta * sample["measurement_vector"]
                grad[self._output_bias_index()] += delta
                terminal_operator = sum(
                    weight * observable
                    for weight, observable in zip(sample["output_weights"], self.measurement_observables)
                )
                chi_states = [None] * len(states)
                chi_states[-1] = terminal_operator @ final_state
                prefactor = delta

            for gate_idx in range(len(gates) - 1, -1, -1):
                chi_states[gate_idx] = gates[gate_idx][0].conj().T @ chi_states[gate_idx + 1]

            for gate_idx, (_, pidx) in enumerate(gates):
                if pidx is None:
                    continue
                gen = self.base_model.gate_derivative_generator(pidx, x_i)
                grad_vec = gen @ states[gate_idx + 1]
                grad[pidx] += prefactor * 2.0 * np.real(chi_states[gate_idx + 1].conj() @ grad_vec)

        grad /= len(X)
        return grad, {
            "sample_forward_passes": len(X),
            "sample_backward_passes": len(X),
            "full_loss_evaluations": 0,
        }

    def nongate_loss_gradient(self, params, X, y):
        params_arr = np.asarray(params, dtype=float)
        grad = np.zeros_like(params_arr)
        if self.readout == "simple_z0z1":
            return grad, {
                "sample_forward_passes": 0,
                "sample_backward_passes": 0,
                "full_loss_evaluations": 0,
            }

        for x_i, y_i in zip(np.asarray(X, dtype=float), np.asarray(y, dtype=float)):
            sample = self._sample_forward_details(params_arr, x_i)
            delta = sample["probability"] - y_i
            grad[self._output_weight_slice()] += delta * sample["measurement_vector"]
            grad[self._output_bias_index()] += delta

        grad /= len(X)
        return grad, {
            "sample_forward_passes": len(X),
            "sample_backward_passes": 0,
            "full_loss_evaluations": 0,
        }

    def rebuild_param_gate(self, param_idx, params, x):
        if param_idx >= self.n_quantum_params:
            raise ValueError("Classical HEA readout parameters do not correspond to circuit gates.")
        return self.base_model.rebuild_param_gate(param_idx, self._quantum_params(params), x)

    def gate_derivative_generator(self, param_idx, x=None):
        if param_idx >= self.n_quantum_params:
            raise ValueError("Classical HEA readout parameters do not have gate generators.")
        return self.base_model.gate_derivative_generator(param_idx, x)

    def get_gate_sequence_and_states(self, params, x):
        return self.base_model.get_gate_sequence_and_states(self._quantum_params(params), x)

    def _build_parameter_metadata(self):
        metadata = list(self.base_model.parameter_metadata())
        if self.readout == "hybrid_linear":
            for offset, pidx in enumerate(
                range(self._output_weight_slice().start, self._output_weight_slice().stop)
            ):
                metadata.append(
                    {
                        "index": pidx,
                        "name": f"output_weight[{offset}]",
                        "group": "classical_output",
                        "kind": "classical",
                        "supports_gate_derivative": False,
                        "layer": None,
                        "qubit": offset,
                        "axis": None,
                    }
                )
            metadata.append(
                {
                    "index": self._output_bias_index(),
                    "name": "output_bias",
                    "group": "classical_output",
                    "kind": "classical",
                    "supports_gate_derivative": False,
                    "layer": None,
                    "qubit": None,
                    "axis": None,
                }
            )
        return metadata


VARIANTS = OrderedDict(
    [
        (
            "simple_z0z1",
            {
                "label": "HEA without classical head",
                "builder": lambda: HEAReadoutComparisonModel(readout="simple_z0z1"),
            },
        ),
        (
            "hybrid_linear",
            {
                "label": "HEA with classical affine head",
                "builder": lambda: HEAReadoutComparisonModel(readout="hybrid_linear"),
            },
        ),
    ]
)


def build_config(seeds):
    return replace(
        DEFAULT_CONFIG,
        optimizers=["krotov_hybrid"],
        run_krotov_batch_sweep=False,
        run_krotov_hybrid_sweep=False,
        results_dir="results",
        seeds=list(seeds),
        **BEST_HYBRID_KROTOV_CONFIG,
    )


def jsonify_trace(trace):
    out = {}
    for key, values in trace.items():
        if key == "phase":
            out[key] = [str(v) for v in values]
        else:
            out[key] = [float(v) for v in values]
    return out


def run_single(variant_name, seed, config):
    variant = VARIANTS[variant_name]
    model = variant["builder"]()
    X_train, X_test, y_train, y_test, _, _ = load_iris_binary(
        test_fraction=0.2,
        seed=seed,
    )
    init_params = np.asarray(model.init_params(seed=seed), dtype=float)

    print(f"\n{'=' * 80}")
    print(f"Variant: {variant['label']} | Seed: {seed}")
    print(f"{'=' * 80}")

    start_time = time.time()
    final_params, trace = run_optimizer(
        "krotov_hybrid",
        model,
        init_params.copy(),
        X_train,
        y_train,
        X_test,
        y_test,
        config,
    )
    wall_total = time.time() - start_time

    result = {
        "variant": variant_name,
        "variant_label": variant["label"],
        "seed": seed,
        "n_params": int(model.n_params),
        "n_quantum_params": int(model.n_quantum_params),
        "n_classical_params": int(model.n_output_params),
        "final_loss": float(trace["loss"][-1]),
        "final_train_acc": float(trace["train_acc"][-1]),
        "final_test_acc": float(trace["test_acc"][-1]),
        "wall_time_total": float(wall_total),
        "total_cost_units": int(trace["cost_units"][-1]),
        "total_steps": int(trace["step"][-1]),
        "initial_params": init_params.tolist(),
        "final_params": np.asarray(final_params, dtype=float).tolist(),
        "trace": jsonify_trace(trace),
    }

    print(
        f"  Done: loss={result['final_loss']:.4f} "
        f"train_acc={result['final_train_acc']:.3f} "
        f"test_acc={result['final_test_acc']:.3f} "
        f"cost={result['total_cost_units']} wall={wall_total:.2f}s",
        flush=True,
    )
    return result


def summarise_results(results):
    summary = OrderedDict()
    for variant_name in VARIANTS:
        runs = [result for result in results if result["variant"] == variant_name]
        losses = np.asarray([run["final_loss"] for run in runs], dtype=float)
        train_accs = np.asarray([run["final_train_acc"] for run in runs], dtype=float)
        test_accs = np.asarray([run["final_test_acc"] for run in runs], dtype=float)
        wall_times = np.asarray([run["wall_time_total"] for run in runs], dtype=float)
        costs = np.asarray([run["total_cost_units"] for run in runs], dtype=float)
        summary[variant_name] = {
            "label": VARIANTS[variant_name]["label"],
            "n_runs": len(runs),
            "n_params": int(runs[0]["n_params"]),
            "n_quantum_params": int(runs[0]["n_quantum_params"]),
            "n_classical_params": int(runs[0]["n_classical_params"]),
            "final_loss_mean": float(np.mean(losses)),
            "final_loss_std": float(np.std(losses)),
            "final_train_acc_mean": float(np.mean(train_accs)),
            "final_train_acc_std": float(np.std(train_accs)),
            "final_test_acc_mean": float(np.mean(test_accs)),
            "final_test_acc_std": float(np.std(test_accs)),
            "wall_time_mean": float(np.mean(wall_times)),
            "wall_time_std": float(np.std(wall_times)),
            "cost_mean": float(np.mean(costs)),
            "cost_std": float(np.std(costs)),
        }
    return summary


def compare_variants(summary):
    base = summary["simple_z0z1"]
    head = summary["hybrid_linear"]
    return {
        "loss_delta": head["final_loss_mean"] - base["final_loss_mean"],
        "test_acc_delta": head["final_test_acc_mean"] - base["final_test_acc_mean"],
        "train_acc_delta": head["final_train_acc_mean"] - base["final_train_acc_mean"],
        "wall_time_delta": head["wall_time_mean"] - base["wall_time_mean"],
        "cost_delta": head["cost_mean"] - base["cost_mean"],
    }


def write_report(summary, comparison, config, results_dir):
    base = summary["simple_z0z1"]
    head = summary["hybrid_linear"]
    lines = [
        "# HEA Readout Head Comparison",
        "",
        "## Setup",
        "",
        "Hybrid Krotov configuration taken from the best prior HEA sweep:",
        "",
        f"- `hybrid_switch_iteration = {config.hybrid_switch_iteration}`",
        f"- `hybrid_online_step_size = {config.hybrid_online_step_size}`",
        f"- `hybrid_batch_step_size = {config.hybrid_batch_step_size}`",
        f"- `max_iterations = {config.max_iterations}`",
        f"- `seeds = {config.seeds}`",
        "",
        "## Results",
        "",
        "| Variant | Params | Classical head params | Final loss | Final test acc | Wall time (s) |",
        "|---|---:|---:|---|---|---|",
        (
            f"| {base['label']} | {base['n_params']} | {base['n_classical_params']} | "
            f"{base['final_loss_mean']:.4f} ± {base['final_loss_std']:.4f} | "
            f"{base['final_test_acc_mean']:.4f} ± {base['final_test_acc_std']:.4f} | "
            f"{base['wall_time_mean']:.2f} ± {base['wall_time_std']:.2f} |"
        ),
        (
            f"| {head['label']} | {head['n_params']} | {head['n_classical_params']} | "
            f"{head['final_loss_mean']:.4f} ± {head['final_loss_std']:.4f} | "
            f"{head['final_test_acc_mean']:.4f} ± {head['final_test_acc_std']:.4f} | "
            f"{head['wall_time_mean']:.2f} ± {head['wall_time_std']:.2f} |"
        ),
        "",
        "## Delta: classical head minus no-head",
        "",
        f"- Final loss delta: `{comparison['loss_delta']:+.4f}`",
        f"- Final test accuracy delta: `{comparison['test_acc_delta']:+.4f}`",
        f"- Final train accuracy delta: `{comparison['train_acc_delta']:+.4f}`",
        f"- Wall time delta: `{comparison['wall_time_delta']:+.2f}s`",
        f"- Cost units delta: `{comparison['cost_delta']:+.1f}`",
        "",
    ]

    if comparison["loss_delta"] < 0.0 and comparison["test_acc_delta"] >= 0.0:
        verdict = (
            "Under this matched HEA configuration, the classical affine head helps: "
            "it lowers loss without hurting test accuracy."
        )
    elif comparison["loss_delta"] > 0.0 and comparison["test_acc_delta"] <= 0.0:
        verdict = (
            "Under this matched HEA configuration, the classical affine head hurts: "
            "it increases loss and does not improve accuracy."
        )
    else:
        verdict = (
            "Under this matched HEA configuration, the effect of the classical affine head is mixed: "
            "it changes loss and accuracy in different directions or only weakly."
        )
    lines.extend(["## Verdict", "", verdict, ""])

    report_path = os.path.join(results_dir, "comparison_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return report_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    config = build_config(args.seeds)

    all_results = []
    for variant_name in VARIANTS:
        for seed in config.seeds:
            result = run_single(variant_name, seed, config)
            all_results.append(result)
            out_path = os.path.join(args.results_dir, f"result_{variant_name}_seed{seed}.json")
            with open(out_path, "w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)

    summary = summarise_results(all_results)
    comparison = compare_variants(summary)
    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "config": asdict(config),
                "summary": summary,
                "comparison": comparison,
            },
            handle,
            indent=2,
        )
    report_path = write_report(summary, comparison, config, args.results_dir)

    print(f"\nComparison report written to {report_path}")
    print(json.dumps({"summary": summary, "comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
