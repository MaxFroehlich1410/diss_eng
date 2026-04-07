"""Configuration for the two-moons QML Krotov benchmark."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentConfig:
    # Dataset
    n_samples: int = 500
    moon_noise: float = 0.15
    test_fraction: float = 0.3
    input_encoding: str = "tanh_0_pi"  # "tanh_0_pi" or "linear_pm_pi"

    # Model
    model_architecture: str = "hea"  # "hea" or "data_reuploading"
    n_qubits: int = 4
    n_layers: int = 3
    entangler: str = "ring"  # "ring", "chain", or "none"
    observable: str = "Z0Z1"  # "Z0" or "Z0Z1" (average of Z0, Z1)

    # Training
    max_iterations: int = 100
    adam_lr: float = 0.05
    lbfgs_maxiter: int = 100
    qng_lr: float = 0.5
    qng_lam: float = 0.01  # Fubini-Study metric tensor regularisation
    qng_approx: Optional[str] = None  # None (full) or "diag"
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 12
    early_stopping_min_delta: float = 1e-4
    early_stopping_warmup: int = 20

    # Legacy shared Krotov knobs retained for backward compatibility with
    # older variant scripts that override the generic fields directly.
    krotov_step_size: float = 0.3
    krotov_lr_schedule: str = "constant"  # "constant", "inverse", "exp"
    krotov_decay: float = 0.05
    krotov_batch_size: Optional[int] = None  # None => full batch
    krotov_target_loss: float = 0.4

    # Per-optimizer Krotov defaults used by the main benchmark.
    krotov_online_step_size: float = 0.3
    krotov_online_schedule: str = "constant"
    krotov_online_decay: float = 0.05
    krotov_batch_step_size: float = 1.0
    krotov_batch_schedule: str = "constant"
    krotov_batch_decay: float = 0.05

    # Hybrid online -> batch schedule.
    hybrid_switch_iteration: int = 20
    hybrid_online_step_size: float = 0.3
    hybrid_batch_step_size: float = 1.0
    hybrid_online_schedule: str = "constant"
    hybrid_batch_schedule: str = "constant"
    hybrid_online_decay: float = 0.05
    hybrid_batch_decay: float = 0.05

    # Experiment
    seeds: List[int] = field(default_factory=lambda: list(range(5)))
    optimizers: List[str] = field(
        default_factory=lambda: [
            "krotov_online",
            "krotov_batch",
            "krotov_hybrid",
            "adam",
            "lbfgs",
        ]
    )
    run_krotov_batch_sweep: bool = False
    krotov_batch_sweep_step_sizes: List[float] = field(
        default_factory=lambda: [1.0, 0.3, 0.1, 0.05, 0.02]
    )
    krotov_batch_sweep_schedules: List[str] = field(
        default_factory=lambda: ["constant", "inverse"]
    )
    run_krotov_hybrid_sweep: bool = True
    hybrid_switch_iterations: List[int] = field(
        default_factory=lambda: [5, 10, 20, 30, 50]
    )
    loss_threshold: float = 0.4  # for success-rate metric
    loss_thresholds: List[float] = field(
        default_factory=lambda: [0.45, 0.40, 0.38, 0.36]
    )

    # Output
    results_dir: str = "results"
    plots_dir: str = "plots"


DEFAULT_CONFIG = ExperimentConfig()
