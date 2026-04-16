"""Two-moons dataset generation and preprocessing."""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_two_moons(
    n_samples=500,
    noise=0.15,
    test_fraction=0.3,
    seed=42,
    encoding="tanh_0_pi",
):
    """Generate two-moons dataset with configurable angle preprocessing.

    Returns
    -------
    X_train, X_test : ndarray of shape (n, 2)
    y_train, y_test : ndarray of shape (n,) with values {0, 1}
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )

    if encoding == "tanh_0_pi":
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        X_train_enc = np.pi * (np.tanh(X_train_s) + 1) / 2
        X_test_enc = np.pi * (np.tanh(X_test_s) + 1) / 2
    elif encoding == "linear_pm_pi":
        mins = X_train.min(axis=0)
        maxs = X_train.max(axis=0)
        span = np.where(maxs > mins, maxs - mins, 1.0)
        X_train_enc = (2 * X_train - maxs - mins) * np.pi / span
        X_test_enc = (2 * X_test - maxs - mins) * np.pi / span
    else:
        raise ValueError(f"Unknown input encoding: {encoding}")

    return X_train_enc, X_test_enc, y_train, y_test


__all__ = ["generate_two_moons"]
