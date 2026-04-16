"""Iris dataset loaders and preprocessing helpers for classification benchmarks."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_iris_binary(test_fraction=0.2, seed=42):
    """Load Iris (setosa vs. versicolor), first 2 features, scaled to [0, pi].

    Returns X_train, X_test, y_train, y_test, and the scaling parameters
    (x_min, span) needed to map raw features to the encoded space.
    """
    iris = load_iris()
    mask = iris.target < 2
    X = iris.data[mask, :2].astype(float)
    y = iris.target[mask].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y,
    )

    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    span = np.where(x_max > x_min, x_max - x_min, 1.0)
    X_train_enc = np.pi * (X_train - x_min) / span
    X_test_enc = np.pi * (X_test - x_min) / span

    return X_train_enc, X_test_enc, y_train, y_test, x_min, span


__all__ = ["load_iris_binary"]
