"""4-bit parity dataset helpers."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def generate_parity_4bit(test_fraction=0.25, seed=42, repeats=1):
    """Return a train/test split for 4-bit parity classification.

    Samples are bitstrings in ``{0,1}^4`` and labels follow the PennyLane demo
    convention: odd parity maps to ``+1`` and even parity maps to ``-1``.
    """

    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    bitstrings = np.array(
        [[(value >> shift) & 1 for shift in range(3, -1, -1)] for value in range(16)],
        dtype=int,
    )
    labels = np.where(np.sum(bitstrings, axis=1) % 2 == 1, 1, -1).astype(int)

    X = np.repeat(bitstrings, repeats, axis=0)
    y = np.repeat(labels, repeats, axis=0)

    return train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )


__all__ = ["generate_parity_4bit"]
