"""Load and preprocess the Jena Climate dataset for time-series forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd


_DEFAULT_CSV = Path(__file__).parent.parent / "data" / "raw" / "jena_climate_2009_2016.csv"


class DataSplit(NamedTuple):
    """Container for train / validation / test splits.

    Attributes:
        X_train: Shape ``(n, window, features)`` — training input windows.
        y_train: Shape ``(n,)`` — training targets.
        X_val:   Shape ``(n, window, features)`` — validation input windows.
        y_val:   Shape ``(n,)`` — validation targets.
        X_test:  Shape ``(n, window, features)`` — test input windows.
        y_test:  Shape ``(n,)`` — test targets.
        feature_names: Column names of the feature matrix.
        raw_train_series: Normalized univariate target column for the training
            period (used by ARIMA which requires a plain time-series, not
            windowed arrays).
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    raw_train_series: np.ndarray  # 1-D, normalised target column


def load_and_preprocess(
    csv_path: Path | str = _DEFAULT_CSV,
    window: int = 120,
    horizon: int = 1,
    target_col: str = "T (degC)",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    subsample: int = 6,
) -> DataSplit:
    """Load the Jena Climate CSV, clean it, normalise it and build windows.

    Pipeline:
    1. Parse CSV and drop the ``Date Time`` column.
    2. Remove rows where ``wv (m/s)`` equals -9999 (sensor error sentinel).
    3. Z-score-normalise every feature using *training-set* statistics only
       to avoid data leakage into validation / test sets.
    4. Store the normalised target column for the training split as
       ``raw_train_series`` (for ARIMA).
    5. Build overlapping sliding-window sequences of length ``window``
       predicting ``horizon`` steps ahead.
    6. Split chronologically: train / val / test (70 / 15 / 15 by default).

    Args:
        csv_path: Path to ``jena_climate_2009_2016.csv``.
        window: Number of time-steps per input sequence.
        horizon: Steps ahead to forecast (offset from end of window).
        target_col: Feature column to use as the prediction target.
        train_frac: Fraction of rows assigned to training.
        val_frac: Fraction of rows assigned to validation.
        subsample: Keep every *subsample*-th row (default 6 = hourly from
            10-min data).  Reduces memory from ~12 GB to ~300 MB while
            preserving all temporal patterns at hourly resolution.

    Returns:
        A :class:`DataSplit` named-tuple.
    """
    df = pd.read_csv(csv_path)

    # Drop timestamp column
    if "Date Time" in df.columns:
        df = df.drop(columns=["Date Time"])

    # Subsample to reduce memory footprint (default: 10-min → hourly)
    if subsample > 1:
        df = df.iloc[::subsample].reset_index(drop=True)

    # Remove bad wind-speed sentinel values
    if "wv (m/s)" in df.columns:
        df = df[df["wv (m/s)"] != -9999].reset_index(drop=True)

    feature_names: list[str] = df.columns.tolist()
    data = df.values.astype(np.float32)

    n = len(data)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    # Fit scaler on training data only — no leakage
    train_mean = data[:train_end].mean(axis=0)
    train_std = data[:train_end].std(axis=0)
    train_std[train_std == 0] = 1.0  # guard against constant columns

    data = (data - train_mean) / train_std

    target_idx = feature_names.index(target_col)

    # Raw (non-windowed) normalised training series for ARIMA
    raw_train_series = data[:train_end, target_idx].copy()

    def _make_windows(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build overlapping (X, y) window pairs using stride tricks (fast).

        Uses ``numpy.lib.stride_tricks.sliding_window_view`` which creates a
        *view* of the data — no per-window memory copies — making it orders of
        magnitude faster than a Python for-loop.
        """
        from numpy.lib.stride_tricks import sliding_window_view
        n_samples = len(arr) - window - horizon + 1
        if n_samples <= 0:
            empty_x = np.empty((0, window, arr.shape[1]), dtype=np.float32)
            empty_y = np.empty((0,), dtype=np.float32)
            return empty_x, empty_y
        # sliding_window_view: (n_rows, n_features, window_size)
        view = sliding_window_view(arr, window_shape=window, axis=0)  # (N, feats, win)
        # Take only n_samples rows (exclude last horizon-1 rows)
        X = view[:n_samples].transpose(0, 2, 1).astype(np.float32)   # (n, win, feats)
        y_indices = np.arange(window + horizon - 1, window + horizon - 1 + n_samples)
        y = arr[y_indices, target_idx].astype(np.float32)
        return X, y

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
        raw_train_series=raw_train_series,
    )
