"""Loss helpers shared across QML models."""

from __future__ import annotations

import numpy as np

EPS = 1e-7

def sigmoid(x):
    """Numerically stable logistic function."""
    x = np.asarray(x, dtype=float)
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))

def clip_probability(prob):
    """Clip probabilities away from the BCE singularities."""
    return np.clip(prob, EPS, 1.0 - EPS)

def binary_cross_entropy(probs, targets):
    """Mean binary cross-entropy for probability outputs."""
    p = clip_probability(np.asarray(probs, dtype=float))
    y = np.asarray(targets, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

__all__ = ["EPS", "sigmoid", "clip_probability", "binary_cross_entropy"]
