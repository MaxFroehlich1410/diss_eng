"""4-bit parity dataset helpers."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def _parity_truth_table():
    bitstrings = np.array(
        [[(value >> shift) & 1 for shift in range(3, -1, -1)] for value in range(16)],
        dtype=int,
    )
    labels = np.where(np.sum(bitstrings, axis=1) % 2 == 1, 1, -1).astype(int)
    return bitstrings, labels


def generate_parity_4bit(test_fraction=0.25, seed=42, repeats=1):
    """Return a train/test split for 4-bit parity classification.

    Samples are bitstrings in ``{0,1}^4`` and labels follow the PennyLane demo
    convention: odd parity maps to ``+1`` and even parity maps to ``-1``.
    """

    if repeats <= 0:
        raise ValueError("repeats must be positive.")

    bitstrings, labels = _parity_truth_table()

    X = np.repeat(bitstrings, repeats, axis=0)
    y = np.repeat(labels, repeats, axis=0)

    return train_test_split(
        X,
        y,
        test_size=test_fraction,
        random_state=seed,
        stratify=y,
    )


def generate_parity_4bit_unique_split(train_size=10, test_size=6, seed=42):
    """Return a leakage-free split over unique 4-bit inputs.

    The 16 unique bitstrings are split directly, so no bitstring appears in both
    train and test. Labels follow odd parity -> ``+1`` and even parity -> ``-1``.
    """

    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive.")
    if train_size + test_size != 16:
        raise ValueError("4-bit parity has exactly 16 unique bitstrings, so train_size + test_size must equal 16.")

    bitstrings, labels = _parity_truth_table()
    rng = np.random.RandomState(seed)
    permutation = rng.permutation(len(bitstrings))
    train_idx = permutation[:train_size]
    test_idx = permutation[train_size : train_size + test_size]
    return (
        bitstrings[train_idx].copy(),
        bitstrings[test_idx].copy(),
        labels[train_idx].copy(),
        labels[test_idx].copy(),
    )


__all__ = ["generate_parity_4bit", "generate_parity_4bit_unique_split"]
