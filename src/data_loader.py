"""Load and preprocess the Jena Climate dataset for time-series forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


_DEFAULT_CSV = Path(__file__).parent.parent / "data" / "raw" / "jena_climate_2009_2016.csv"


class DataSplit(NamedTuple):
    """Container for train / validation / test numpy arrays.

    Each array has shape ``(n_samples, window_size, n_features)`` for ``X``
    and ``(n_samples,)`` for ``y``.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]


def load_and_preprocess(
    csv_path: Path | str = _DEFAULT_CSV,
    window: int = 720,
    horizon: int = 1,
    target_col: str = "T (degC)",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> DataSplit:
    """Load the Jena Climate CSV, clean it, normalise it and build windows.

    Steps:
    1. Parse the CSV and drop the ``Date Time`` column.
    2. Remove rows where ``wv (m/s)`` equals -9999 (sensor error sentinel).
    3. Z-score-normalise every feature using *training-set* statistics only to
       avoid data leakage.
    4. Build sliding-window sequences of length *window* predicting *horizon*
       steps ahead.
    5. Split chronologically into train / val / test (70 / 15 / 15 by default).

    Args:
        csv_path: Path to ``jena_climate_2009_2016.csv``.
        window: Number of time-steps in each input sequence.
        horizon: Number of steps ahead to forecast (target offset from window end).
        target_col: Column name to use as the prediction target.
        train_frac: Fraction of rows used for training.
        val_frac: Fraction of rows used for validation.

    Returns:
        A :class:`DataSplit` named-tuple with ``X_train``, ``y_train``,
        ``X_val``, ``y_val``, ``X_test``, ``y_test``, and ``feature_names``.
    """
    df = pd.read_csv(csv_path)

    # Drop timestamp column
    if "Date Time" in df.columns:
        df = df.drop(columns=["Date Time"])

    # Remove bad wind-speed rows
    if "wv (m/s)" in df.columns:
        df = df[df["wv (m/s)"] != -9999].reset_index(drop=True)

    feature_names: list[str] = df.columns.tolist()
    data = df.values.astype(np.float32)

    # Chronological split indices
    n = len(data)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    # Fit scaler on training data only
    train_mean = data[:train_end].mean(axis=0)
    train_std = data[:train_end].std(axis=0)
    train_std[train_std == 0] = 1.0  # avoid division by zero

    data = (data - train_mean) / train_std

    target_idx = feature_names.index(target_col)

    def _make_windows(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Slice *arr* into overlapping (X, y) window pairs."""
        xs, ys = [], []
        limit = len(arr) - window - horizon + 1
        for i in range(limit):
            xs.append(arr[i : i + window])
            ys.append(arr[i + window + horizon - 1, target_idx])
        return np.stack(xs), np.array(ys, dtype=np.float32)

    X_train, y_train = _make_windows(data[:train_end])
    X_val, y_val = _make_windows(data[train_end:val_end])
    X_test, y_test = _make_windows(data[val_end:])

    return DataSplit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
    )
