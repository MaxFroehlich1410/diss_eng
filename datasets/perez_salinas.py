"""Dataset generators for the Perez-Salinas data re-uploading benchmarks."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


PROBLEM_DEFAULT_SAMPLES = {
    "circle": 4200,
    "3_circles": 4200,
    "non_convex": 4200,
    "crown": 4200,
    "squares": 4200,
    "wavy_lines": 4200,
}


def available_perez_salinas_problems():
    return tuple(PROBLEM_DEFAULT_SAMPLES)


def perez_salinas_problem_num_classes(problem):
    problem_key = _normalise_problem_name(problem)
    if problem_key in {"circle", "non_convex", "crown"}:
        return 2
    if problem_key == "3_circles":
        return 4
    if problem_key == "squares":
        return 4
    if problem_key == "wavy_lines":
        return 4
    raise ValueError(f"Unknown Perez-Salinas problem: {problem}")


def perez_salinas_benchmark_preset(
    problem="non_convex",
    n_qubits=4,
    n_layers=8,
    use_entanglement=True,
):
    """Return a reusable benchmark preset for the paper-style model."""
    problem_key = _normalise_problem_name(problem)
    return {
        "problem": problem_key,
        "n_qubits": int(n_qubits),
        "n_layers": int(n_layers),
        "use_entanglement": bool(use_entanglement),
        "loss_mode": "weighted_fidelity",
        "n_classes": perez_salinas_problem_num_classes(problem_key),
    }


def perez_salinas_4q8l_preset(problem="non_convex"):
    """Return a practical default benchmark preset for the paper model."""
    return perez_salinas_benchmark_preset(
        problem=problem,
        n_qubits=4,
        n_layers=8,
        use_entanglement=True,
    )


def generate_perez_salinas_dataset(
    problem="non_convex",
    n_samples=None,
    test_fraction=0.3,
    seed=42,
):
    """Generate one of the paper's synthetic classification tasks.

    The geometry and label assignment follow the companion reference code.
    Samples are drawn uniformly from ``[-1, 1]^2`` and then split into train
    and test partitions with stratification.
    """

    problem_key = _normalise_problem_name(problem)
    total_samples = int(PROBLEM_DEFAULT_SAMPLES[problem_key] if n_samples is None else n_samples)
    if total_samples <= 1:
        raise ValueError("n_samples must be greater than 1.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must lie strictly between 0 and 1.")

    rng = np.random.RandomState(seed)
    X, y = _generate_problem_samples(problem_key, total_samples, rng)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def _normalise_problem_name(problem):
    key = str(problem).strip().lower().replace("-", "_").replace(" ", "_")
    if key not in PROBLEM_DEFAULT_SAMPLES:
        raise ValueError(
            f"Unknown Perez-Salinas problem '{problem}'. "
            f"Available: {sorted(PROBLEM_DEFAULT_SAMPLES)}."
        )
    return key


def _sample_square(rng):
    return 2.0 * rng.rand(2) - 1.0


def _generate_problem_samples(problem, n_samples, rng):
    X = np.empty((n_samples, 2), dtype=float)
    y = np.empty(n_samples, dtype=int)

    if problem == "circle":
        radius = np.sqrt(2.0 / np.pi)
        for idx in range(n_samples):
            x = _sample_square(rng)
            X[idx] = x
            y[idx] = int(np.linalg.norm(x) < radius)
        return X, y

    if problem == "3_circles":
        centers = np.array([[-1.0, 1.0], [1.0, 0.0], [-0.5, -0.5]], dtype=float)
        radii = np.array([1.0, np.sqrt(6.0 / np.pi - 1.0), 0.5], dtype=float)
        for idx in range(n_samples):
            x = _sample_square(rng)
            label = 0
            for circle_idx, (center, radius) in enumerate(zip(centers, radii)):
                if np.linalg.norm(x - center) < radius:
                    label = circle_idx + 1
            X[idx] = x
            y[idx] = label
        return X, y

    if problem == "non_convex":
        for idx in range(n_samples):
            x = _sample_square(rng)
            boundary = -2.0 * x[0] + 1.5 * np.sin(np.pi * x[0])
            X[idx] = x
            y[idx] = 0 if x[1] < boundary else 1
        return X, y

    if problem == "crown":
        outer = np.sqrt(0.8)
        inner = np.sqrt(0.8 - 2.0 / np.pi)
        for idx in range(n_samples):
            x = _sample_square(rng)
            radius = np.linalg.norm(x)
            X[idx] = x
            y[idx] = int(radius < outer and radius > inner)
        return X, y

    if problem == "squares":
        for idx in range(n_samples):
            x = _sample_square(rng)
            if x[0] < 0.0 and x[1] < 0.0:
                label = 0
            elif x[0] < 0.0 and x[1] > 0.0:
                label = 1
            elif x[0] > 0.0 and x[1] < 0.0:
                label = 2
            else:
                label = 3
            X[idx] = x
            y[idx] = label
        return X, y

    if problem == "wavy_lines":
        for idx in range(n_samples):
            x = _sample_square(rng)
            upper = x[0] + np.sin(np.pi * x[0])
            lower = -x[0] + np.sin(np.pi * x[0])
            if x[1] < upper and x[1] < lower:
                label = 0
            elif x[1] < upper and x[1] > lower:
                label = 1
            elif x[1] > upper and x[1] < lower:
                label = 2
            else:
                label = 3
            X[idx] = x
            y[idx] = label
        return X, y

    raise ValueError(f"Unhandled Perez-Salinas problem: {problem}")


__all__ = [
    "PROBLEM_DEFAULT_SAMPLES",
    "available_perez_salinas_problems",
    "generate_perez_salinas_dataset",
    "perez_salinas_4q8l_preset",
    "perez_salinas_benchmark_preset",
    "perez_salinas_problem_num_classes",
]
