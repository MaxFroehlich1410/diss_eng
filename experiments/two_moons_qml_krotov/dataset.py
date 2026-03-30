"""Two-moons dataset generation and preprocessing."""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_two_moons(n_samples=500, noise=0.15, test_fraction=0.3, seed=42):
    """Generate two-moons dataset, standardize, and map to [0, pi].

    Returns
    -------
    X_train, X_test : ndarray of shape (n, 2) in [0, pi]
    y_train, y_test : ndarray of shape (n,) with values {0, 1}
    """
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Map standardized features to [0, pi] via sigmoid-like rescaling
    # tanh maps to (-1, 1), then shift to (0, pi)
    X_train_enc = np.pi * (np.tanh(X_train_s) + 1) / 2
    X_test_enc = np.pi * (np.tanh(X_test_s) + 1) / 2

    return X_train_enc, X_test_enc, y_train, y_test
