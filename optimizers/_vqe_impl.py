from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import minimize

from qml_models.vqe import Hubbard1x2HVVQEProblem, build_parametrized_block_unitary

DEFAULT_U_VALUES = [2.0, 4.0, 8.0]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_MAX_ITERATIONS = 80
DEFAULT_TOLERANCE = 1e-3

TRACE_FIELDS = (
    "energy",
    "energy_error",
    "step",
    "phase",
    "wall_time",
    "cost_units",
    "state_forward_passes",
    "state_backward_passes",
    "full_energy_evaluations",
    "gradient_evaluations",
    "metric_tensor_evaluations",
    "step_size",
    "update_norm",
    "gradient_norm",
)

SWEEP_GRIDS = {
    "adam": {
        "adam_lr": [0.01, 0.03, 0.1, 0.3, 1.0],
    },
    "bfgs": {
        "bfgs_gtol": [1e-5, 1e-7, 1e-9],
    },
    "qng": {
        "qng_lr": [0.01, 0.03, 0.1, 0.3, 1.0],
        "qng_lam": [1e-4, 1e-3, 1e-2],
    },
    "krotov_hybrid": {
        "switch": [5, 10, 20],
        "online_step": [0.03, 0.1, 0.3],
        "batch_step": [0.1, 0.3, 1.0],
    },
}

BASELINES = {
    "adam": {"adam_lr": 0.1},
    "bfgs": {"bfgs_gtol": 1e-7},
    "qng": {"qng_lr": 0.1, "qng_lam": 1e-3},
    "krotov_hybrid": {"switch": 10, "online_step": 0.1, "batch_step": 0.3},
}


@dataclass(frozen=True)
class InstanceSpec:
    key: str
    U: float
    label: str


INSTANCE_SPECS = OrderedDict(
    (
        ("u2", InstanceSpec(key="u2", U=2.0, label="FH 1x2 HV (U=2)")),
        ("u4", InstanceSpec(key="u4", U=4.0, label="FH 1x2 HV (U=4)")),
        ("u8", InstanceSpec(key="u8", U=8.0, label="FH 1x2 HV (U=8)")),
    )
)


def _init_trace() -> dict[str, list[float | int | str]]:
    return {key: [] for key in TRACE_FIELDS}


def _init_counters() -> dict[str, int]:
    return {
        "state_forward_passes": 0,
        "state_backward_passes": 0,
        "full_energy_evaluations": 0,
        "gradient_evaluations": 0,
        "metric_tensor_evaluations": 0,
    }


def _apply_counter_delta(counters: dict[str, int], delta: dict[str, int]) -> None:
    for key in counters:
        counters[key] += int(delta.get(key, 0))


def _current_cost_units(counters: dict[str, int]) -> int:
    return counters["state_forward_passes"] + counters["state_backward_passes"]


def _append_trace(
    trace: dict[str, list[float | int | str]],
    counters: dict[str, int],
    *,
    step: int,
    phase: str,
    wall_time: float,
    energy: float,
    exact_ground_energy: float,
    step_size: float,
    update_norm: float,
    gradient_norm: float,
) -> None:
    trace["energy"].append(float(energy))
    trace["energy_error"].append(float(energy - exact_ground_energy))
    trace["step"].append(int(step))
    trace["phase"].append(str(phase))
    trace["wall_time"].append(float(wall_time))
    trace["cost_units"].append(int(_current_cost_units(counters)))
    trace["state_forward_passes"].append(int(counters["state_forward_passes"]))
    trace["state_backward_passes"].append(int(counters["state_backward_passes"]))
    trace["full_energy_evaluations"].append(int(counters["full_energy_evaluations"]))
    trace["gradient_evaluations"].append(int(counters["gradient_evaluations"]))
    trace["metric_tensor_evaluations"].append(int(counters["metric_tensor_evaluations"]))
    trace["step_size"].append(float(step_size))
    trace["update_norm"].append(float(update_norm))
    trace["gradient_norm"].append(float(gradient_norm))


def expand_grid(grid_dict: dict[str, list[float | int]]) -> list[dict[str, float | int]]:
    names = list(grid_dict.keys())
    values = [grid_dict[name] for name in names]
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def hp_label(hp_dict: dict[str, float | int]) -> str:
    return " ".join(f"{name}={value}" for name, value in hp_dict.items())


def build_initial_params(seed: int, n_params: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-np.pi, np.pi, size=n_params)


def exact_energy(problem: Hubbard1x2HVVQEProblem, theta: np.ndarray) -> tuple[float, dict[str, int]]:
    return problem.energy(theta), {
        "state_forward_passes": 1,
        "state_backward_passes": 0,
        "full_energy_evaluations": 1,
        "gradient_evaluations": 0,
        "metric_tensor_evaluations": 0,
    }


def exact_energy_and_gradient(
    problem: Hubbard1x2HVVQEProblem,
    theta: np.ndarray,
) -> tuple[float, np.ndarray, dict[str, int]]:
    gates, forward_states = problem.get_gate_sequence_and_states(theta)
    final_state = forward_states[-1]
    terminal_costate = problem.terminal_costate(final_state)
    energy = float(np.real(np.vdot(final_state, terminal_costate)))

    costates: list[np.ndarray | None] = [None] * len(forward_states)
    costates[-1] = terminal_costate
    for gate_idx in range(len(gates) - 1, -1, -1):
        gate = gates[gate_idx][0]
        costates[gate_idx] = gate.conj().T @ costates[gate_idx + 1]

    grad = np.zeros_like(theta, dtype=float)
    for gate_idx, (_, param_idx) in enumerate(gates):
        if param_idx is None:
            continue
        left_generator = problem.gate_derivative_generator(param_idx)
        grad[param_idx] = 2.0 * np.real(
            np.vdot(costates[gate_idx + 1], left_generator @ forward_states[gate_idx + 1])
        )

    stats = {
        "state_forward_passes": 1,
        "state_backward_passes": 1,
        "full_energy_evaluations": 0,
        "gradient_evaluations": 1,
        "metric_tensor_evaluations": 0,
    }
    return energy, grad, stats


def compute_state_derivatives(
    problem: Hubbard1x2HVVQEProblem,
    theta: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Return exact derivatives of the final state with respect to each angle."""
    gates, forward_states = problem.get_gate_sequence_and_states(theta)
    final_state = forward_states[-1]
    n_gates = len(gates)
    if n_gates == 0:
        return {}, final_state

    suffix_after = [np.eye(problem.dim, dtype=complex) for _ in range(n_gates)]
    for gate_idx in range(n_gates - 2, -1, -1):
        suffix_after[gate_idx] = suffix_after[gate_idx + 1] @ gates[gate_idx + 1][0]

    dpsi: dict[int, np.ndarray] = {}
    for gate_idx, (_, param_idx) in enumerate(gates):
        if param_idx is None:
            continue
        left_generator = problem.gate_derivative_generator(param_idx)
        dpsi[param_idx] = suffix_after[gate_idx] @ (left_generator @ forward_states[gate_idx + 1])

    return dpsi, final_state


def compute_metric_tensor(
    problem: Hubbard1x2HVVQEProblem,
    theta: np.ndarray,
    *,
    approx: str | None = None,
    lam: float = 0.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute the exact Fubini-Study metric tensor for the HV ansatz."""
    gate_indices = problem.gate_parameter_indices()
    dpsi, final_state = compute_state_derivatives(problem, theta)
    n_gate = len(gate_indices)
    metric = np.zeros((n_gate, n_gate), dtype=float)

    overlaps = np.zeros(n_gate, dtype=complex)
    dpsi_vecs: list[np.ndarray] = []
    for i, param_idx in enumerate(gate_indices):
        vec = dpsi[param_idx]
        dpsi_vecs.append(vec)
        overlaps[i] = np.vdot(final_state, vec)

    if approx == "diag":
        for i in range(n_gate):
            inner = np.real(np.vdot(dpsi_vecs[i], dpsi_vecs[i]))
            correction = np.real(np.abs(overlaps[i]) ** 2)
            metric[i, i] = inner - correction
    else:
        for i in range(n_gate):
            for j in range(i, n_gate):
                inner = np.real(np.vdot(dpsi_vecs[i], dpsi_vecs[j]))
                correction = np.real(overlaps[i].conj() * overlaps[j])
                value = inner - correction
                metric[i, j] = value
                metric[j, i] = value

    metric += float(lam) * np.eye(n_gate)
    stats = {
        "state_forward_passes": 1,
        "state_backward_passes": 1,
        "full_energy_evaluations": 0,
        "gradient_evaluations": 0,
        "metric_tensor_evaluations": 1,
    }
    return metric, stats


def train_adam(
    problem: Hubbard1x2HVVQEProblem,
    theta0: np.ndarray,
    *,
    max_iterations: int,
    lr: float,
) -> tuple[np.ndarray, dict[str, list[float | int | str]]]:
    trace = _init_trace()
    counters = _init_counters()
    theta = np.asarray(theta0, dtype=float).copy()
    exact_ground_energy = problem.exact_ground_energy()
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    t0 = time.time()

    energy0, stats0 = exact_energy(problem, theta)
    _apply_counter_delta(counters, stats0)
    _append_trace(
        trace,
        counters,
        step=0,
        phase="init",
        wall_time=0.0,
        energy=energy0,
        exact_ground_energy=exact_ground_energy,
        step_size=0.0,
        update_norm=0.0,
        gradient_norm=0.0,
    )

    for iteration in range(1, max_iterations + 1):
        _, grad, grad_stats = exact_energy_and_gradient(problem, theta)
        _apply_counter_delta(counters, grad_stats)

        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad**2
        m_hat = m / (1.0 - beta1**iteration)
        v_hat = v / (1.0 - beta2**iteration)
        update = lr * m_hat / (np.sqrt(v_hat) + eps)
        theta = theta - update

        energy, energy_stats = exact_energy(problem, theta)
        _apply_counter_delta(counters, energy_stats)
        _append_trace(
            trace,
            counters,
            step=iteration,
            phase="adam",
            wall_time=time.time() - t0,
            energy=energy,
            exact_ground_energy=exact_ground_energy,
            step_size=lr,
            update_norm=float(np.linalg.norm(update)),
            gradient_norm=float(np.linalg.norm(grad)),
        )

    return theta, trace


def train_qng(
    problem: Hubbard1x2HVVQEProblem,
    theta0: np.ndarray,
    *,
    max_iterations: int,
    lr: float,
    lam: float,
    approx: str | None = None,
) -> tuple[np.ndarray, dict[str, list[float | int | str]]]:
    trace = _init_trace()
    counters = _init_counters()
    theta = np.asarray(theta0, dtype=float).copy()
    exact_ground_energy = problem.exact_ground_energy()
    t0 = time.time()

    energy0, stats0 = exact_energy(problem, theta)
    _apply_counter_delta(counters, stats0)
    _append_trace(
        trace,
        counters,
        step=0,
        phase="init",
        wall_time=0.0,
        energy=energy0,
        exact_ground_energy=exact_ground_energy,
        step_size=0.0,
        update_norm=0.0,
        gradient_norm=0.0,
    )

    for iteration in range(1, max_iterations + 1):
        _, grad, grad_stats = exact_energy_and_gradient(problem, theta)
        _apply_counter_delta(counters, grad_stats)

        metric_tensor, mt_stats = compute_metric_tensor(problem, theta, approx=approx, lam=lam)
        _apply_counter_delta(counters, mt_stats)

        update = lr * (np.linalg.pinv(metric_tensor) @ grad)
        theta = theta - update

        energy, energy_stats = exact_energy(problem, theta)
        _apply_counter_delta(counters, energy_stats)
        _append_trace(
            trace,
            counters,
            step=iteration,
            phase="qng" if approx is None else f"qng_{approx}",
            wall_time=time.time() - t0,
            energy=energy,
            exact_ground_energy=exact_ground_energy,
            step_size=lr,
            update_norm=float(np.linalg.norm(update)),
            gradient_norm=float(np.linalg.norm(grad)),
        )

    return theta, trace


def train_bfgs(
    problem: Hubbard1x2HVVQEProblem,
    theta0: np.ndarray,
    *,
    max_iterations: int,
    gtol: float,
) -> tuple[np.ndarray, dict[str, list[float | int | str]]]:
    trace = _init_trace()
    counters = _init_counters()
    exact_ground_energy = problem.exact_ground_energy()
    step_counter = [0]
    previous_point = [np.asarray(theta0, dtype=float).copy()]
    t0 = time.time()

    energy0, stats0 = exact_energy(problem, theta0)
    _apply_counter_delta(counters, stats0)
    _append_trace(
        trace,
        counters,
        step=0,
        phase="init",
        wall_time=0.0,
        energy=energy0,
        exact_ground_energy=exact_ground_energy,
        step_size=0.0,
        update_norm=0.0,
        gradient_norm=0.0,
    )

    def objective_and_grad(theta: np.ndarray) -> tuple[float, np.ndarray]:
        energy, grad, stats = exact_energy_and_gradient(problem, theta)
        _apply_counter_delta(counters, stats)
        step_counter[0] += 1

        update_norm = float(np.linalg.norm(theta - previous_point[0]))
        previous_point[0] = np.asarray(theta, dtype=float).copy()
        _append_trace(
            trace,
            counters,
            step=step_counter[0],
            phase="bfgs",
            wall_time=time.time() - t0,
            energy=energy,
            exact_ground_energy=exact_ground_energy,
            step_size=np.nan,
            update_norm=update_norm,
            gradient_norm=float(np.linalg.norm(grad)),
        )
        return energy, grad

    result = minimize(
        objective_and_grad,
        np.asarray(theta0, dtype=float),
        method="BFGS",
        jac=True,
        options={
            "maxiter": max_iterations,
            "gtol": gtol,
            "disp": False,
        },
    )
    return np.asarray(result.x, dtype=float), trace


def _build_costates(
    gates: list[tuple[np.ndarray, int | None]],
    terminal_costate: np.ndarray,
) -> list[np.ndarray]:
    costates = [np.zeros_like(terminal_costate) for _ in range(len(gates) + 1)]
    costates[-1] = terminal_costate
    for gate_idx in range(len(gates) - 1, -1, -1):
        costates[gate_idx] = gates[gate_idx][0].conj().T @ costates[gate_idx + 1]
    return costates


def krotov_online_step(
    problem: Hubbard1x2HVVQEProblem,
    theta: np.ndarray,
    *,
    step_size: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, int]]:
    gates, forward_states = problem.get_gate_sequence_and_states(theta)
    terminal_costate = problem.terminal_costate(forward_states[-1])
    costates = _build_costates(gates, terminal_costate)

    new_theta = np.asarray(theta, dtype=float).copy()
    current_state = problem.psi_ref.copy()
    raw_gradient = np.zeros_like(new_theta)

    for gate_idx, (gate, param_idx) in enumerate(gates):
        if param_idx is None:
            current_state = gate @ current_state
            continue

        left_generator = problem.gate_derivative_generator(param_idx)
        gate_output = gate @ current_state
        gradient = 2.0 * np.real(np.vdot(costates[gate_idx + 1], left_generator @ gate_output))
        raw_gradient[param_idx] = gradient

        new_theta[param_idx] -= step_size * gradient
        rebuilt_gate = build_parametrized_block_unitary(
            new_theta[param_idx],
            problem.block_generator(param_idx),
        )
        current_state = rebuilt_gate @ current_state

    diagnostics = {
        "phase": "online",
        "step_size": float(step_size),
        "update_norm": float(np.linalg.norm(new_theta - theta)),
        "gradient_norm": float(np.linalg.norm(raw_gradient)),
    }
    stats = {
        "state_forward_passes": 1,
        "state_backward_passes": 1,
        "full_energy_evaluations": 0,
        "gradient_evaluations": 1,
        "metric_tensor_evaluations": 0,
    }
    return new_theta, diagnostics, stats


def krotov_batch_step(
    problem: Hubbard1x2HVVQEProblem,
    theta: np.ndarray,
    *,
    step_size: float,
) -> tuple[np.ndarray, dict[str, float], dict[str, int]]:
    _, grad, stats = exact_energy_and_gradient(problem, theta)
    update = step_size * grad
    new_theta = np.asarray(theta, dtype=float) - update
    diagnostics = {
        "phase": "batch",
        "step_size": float(step_size),
        "update_norm": float(np.linalg.norm(update)),
        "gradient_norm": float(np.linalg.norm(grad)),
    }
    return new_theta, diagnostics, stats


def train_krotov_hybrid(
    problem: Hubbard1x2HVVQEProblem,
    theta0: np.ndarray,
    *,
    max_iterations: int,
    switch_iteration: int,
    online_step_size: float,
    batch_step_size: float,
) -> tuple[np.ndarray, dict[str, list[float | int | str]]]:
    trace = _init_trace()
    counters = _init_counters()
    theta = np.asarray(theta0, dtype=float).copy()
    exact_ground_energy = problem.exact_ground_energy()
    t0 = time.time()

    energy0, stats0 = exact_energy(problem, theta)
    _apply_counter_delta(counters, stats0)
    _append_trace(
        trace,
        counters,
        step=0,
        phase="init",
        wall_time=0.0,
        energy=energy0,
        exact_ground_energy=exact_ground_energy,
        step_size=0.0,
        update_norm=0.0,
        gradient_norm=0.0,
    )

    for iteration in range(1, max_iterations + 1):
        if iteration <= switch_iteration:
            theta, diagnostics, stats = krotov_online_step(
                problem,
                theta,
                step_size=online_step_size,
            )
        else:
            theta, diagnostics, stats = krotov_batch_step(
                problem,
                theta,
                step_size=batch_step_size,
            )
        _apply_counter_delta(counters, stats)

        energy, energy_stats = exact_energy(problem, theta)
        _apply_counter_delta(counters, energy_stats)
        _append_trace(
            trace,
            counters,
            step=iteration,
            phase=str(diagnostics["phase"]),
            wall_time=time.time() - t0,
            energy=energy,
            exact_ground_energy=exact_ground_energy,
            step_size=float(diagnostics["step_size"]),
            update_norm=float(diagnostics["update_norm"]),
            gradient_norm=float(diagnostics["gradient_norm"]),
        )

    return theta, trace


def run_optimizer(
    optimizer_name: str,
    problem: Hubbard1x2HVVQEProblem,
    theta0: np.ndarray,
    hp_dict: dict[str, float | int],
    *,
    max_iterations: int,
) -> tuple[np.ndarray, dict[str, list[float | int | str]]]:
    if optimizer_name == "adam":
        return train_adam(problem, theta0, max_iterations=max_iterations, lr=float(hp_dict["adam_lr"]))
    if optimizer_name == "bfgs":
        return train_bfgs(problem, theta0, max_iterations=max_iterations, gtol=float(hp_dict["bfgs_gtol"]))
    if optimizer_name == "qng":
        return train_qng(
            problem,
            theta0,
            max_iterations=max_iterations,
            lr=float(hp_dict["qng_lr"]),
            lam=float(hp_dict["qng_lam"]),
        )
    if optimizer_name == "krotov_hybrid":
        return train_krotov_hybrid(
            problem,
            theta0,
            max_iterations=max_iterations,
            switch_iteration=int(hp_dict["switch"]),
            online_step_size=float(hp_dict["online_step"]),
            batch_step_size=float(hp_dict["batch_step"]),
        )
    raise ValueError(f"Unknown optimizer '{optimizer_name}'.")


