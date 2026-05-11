"""Functions for injecting synthetic corruptions into time-series data."""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(data: np.ndarray, sigma: float) -> np.ndarray:
    """Return a copy of *data* with i.i.d. Gaussian noise added.

    Args:
        data: Input array of any shape.
        sigma: Standard deviation of the noise distribution.

    Returns:
        Noisy array with the same dtype and shape as *data*.
    """
    noise = np.random.normal(loc=0.0, scale=sigma, size=data.shape).astype(data.dtype)
    return data + noise


def inject_point_anomalies(data: np.ndarray, ratio: float) -> np.ndarray:
    """Randomly replace a fraction of values with large spike anomalies.

    Each selected element is replaced by ``mean + 5 * std`` of the entire
    array, simulating sensor spikes or outliers.

    Args:
        data: Input array of any shape.  A copy is made internally.
        ratio: Fraction of total elements to corrupt, in ``[0, 1)``.

    Returns:
        Array of the same shape as *data* with ``ratio`` fraction of values
        replaced by spike values.

    Raises:
        ValueError: If *ratio* is not in ``[0, 1)``.
    """
    if not (0.0 <= ratio < 1.0):
        raise ValueError(f"ratio must be in [0, 1), got {ratio}")

    corrupted = data.copy()
    spike_value = float(data.mean() + 5 * data.std())

    flat = corrupted.ravel()
    n_corrupt = int(len(flat) * ratio)
    indices = np.random.choice(len(flat), size=n_corrupt, replace=False)
    flat[indices] = spike_value
    # ravel() may return a view, but reshape ensures the original is updated
    corrupted = flat.reshape(data.shape).astype(data.dtype)
    return corrupted
