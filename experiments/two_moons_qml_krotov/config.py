"""Configuration for the two-moons QML Krotov benchmark."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ExperimentConfig:
    # Dataset
    n_samples: int = 500
    moon_noise: float = 0.15
    test_fraction: float = 0.3

    # Model
    n_qubits: int = 4
    n_layers: int = 3
    entangler: str = "ring"  # "ring" or "chain"
    observable: str = "Z0Z1"  # "Z0" or "Z0Z1" (average of Z0, Z1)

    # Training
    max_iterations: int = 100
    adam_lr: float = 0.05
    lbfgs_maxiter: int = 100
    krotov_step_size: float = 0.3

    # Experiment
    seeds: List[int] = field(default_factory=lambda: list(range(10)))
    optimizers: List[str] = field(
        default_factory=lambda: ["krotov", "adam", "lbfgs"]
    )
    loss_threshold: float = 0.4  # for success-rate metric

    # Output
    results_dir: str = "results"
    plots_dir: str = "plots"


DEFAULT_CONFIG = ExperimentConfig()
