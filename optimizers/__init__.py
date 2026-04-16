"""Reusable optimizer implementations for the QML benchmarks."""

from .adam import train_adam, train_vqe_adam
from .krotov import (
    AdaptiveClipScaling,
    AdaptiveSmoothScaling,
    GroupwiseAdaptiveScaling,
    GroupwiseScaling,
    KrotovScalingStrategy,
    LayerwiseScaling,
    NoScaling,
    build_scaling_strategy,
    get_gate_metadata,
    train_krotov_batch,
    train_krotov_hybrid,
    train_krotov_minibatch,
    train_krotov_online,
    train_vqe_krotov_hybrid,
)
from .lbfgs import train_lbfgs, train_vqe_bfgs
from .qng import (
    _compute_metric_tensor,
    _compute_state_derivatives,
    compute_metric_tensor,
    compute_state_derivatives,
    train_qng,
    train_vqe_qng,
)
from .runner import build_initial_params, run_optimizer, run_vqe_optimizer

__all__ = [
    "AdaptiveClipScaling",
    "AdaptiveSmoothScaling",
    "GroupwiseAdaptiveScaling",
    "GroupwiseScaling",
    "KrotovScalingStrategy",
    "LayerwiseScaling",
    "NoScaling",
    "_compute_metric_tensor",
    "_compute_state_derivatives",
    "build_initial_params",
    "build_scaling_strategy",
    "compute_metric_tensor",
    "compute_state_derivatives",
    "get_gate_metadata",
    "run_optimizer",
    "run_vqe_optimizer",
    "train_adam",
    "train_krotov_batch",
    "train_krotov_hybrid",
    "train_krotov_minibatch",
    "train_krotov_online",
    "train_lbfgs",
    "train_qng",
    "train_vqe_adam",
    "train_vqe_bfgs",
    "train_vqe_krotov_hybrid",
    "train_vqe_qng",
]
