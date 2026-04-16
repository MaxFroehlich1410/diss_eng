"""Krotov-family optimizers and scaling strategies."""

from ._classification_impl import (
    train_krotov_batch,
    train_krotov_hybrid,
    train_krotov_minibatch,
    train_krotov_online,
)
from ._krotov_scaling import (
    AdaptiveClipScaling,
    AdaptiveSmoothScaling,
    GroupwiseAdaptiveScaling,
    GroupwiseScaling,
    KrotovScalingStrategy,
    LayerwiseScaling,
    NoScaling,
    SCALING_APPLY_PHASES,
    build_scaling_strategy,
    get_gate_metadata,
)
from ._vqe_impl import (
    krotov_batch_step as vqe_krotov_batch_step,
    krotov_online_step as vqe_krotov_online_step,
    train_krotov_hybrid as train_vqe_krotov_hybrid,
)

__all__ = [
    "AdaptiveClipScaling",
    "AdaptiveSmoothScaling",
    "GroupwiseAdaptiveScaling",
    "GroupwiseScaling",
    "KrotovScalingStrategy",
    "LayerwiseScaling",
    "NoScaling",
    "SCALING_APPLY_PHASES",
    "build_scaling_strategy",
    "get_gate_metadata",
    "train_krotov_online",
    "train_krotov_batch",
    "train_krotov_minibatch",
    "train_krotov_hybrid",
    "vqe_krotov_online_step",
    "vqe_krotov_batch_step",
    "train_vqe_krotov_hybrid",
]
