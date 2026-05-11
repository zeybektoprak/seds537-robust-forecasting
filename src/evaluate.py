"""Evaluation metrics for time-series forecasting."""

from __future__ import annotations

import numpy as np


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    verbose: bool = True,
) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE between *y_true* and *y_pred*.

    Args:
        y_true: Ground-truth target values, shape ``(n,)``.
        y_pred: Model predictions, shape ``(n,)``.
        verbose: If ``True``, print the metrics to stdout.

    Returns:
        Dictionary with keys ``"MAE"``, ``"RMSE"``, and ``"MAPE"``.

    Raises:
        ValueError: If *y_true* and *y_pred* have different shapes.
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # Avoid division by zero in MAPE; mask near-zero true values
    nonzero = np.abs(y_true) > 1e-8
    if nonzero.any():
        mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)
    else:
        mape = float("nan")

    results = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    if verbose:
        print(f"  MAE  : {mae:.4f}")
        print(f"  RMSE : {rmse:.4f}")
        print(f"  MAPE : {mape:.2f}%")

    return results
