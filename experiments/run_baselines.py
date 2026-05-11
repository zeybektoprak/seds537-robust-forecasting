"""Baseline experiment: train ARIMA, RNN, and LSTM on clean Jena Climate data."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Allow importing from src/ when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import load_and_preprocess
from src.evaluate import compute_metrics
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import LSTMModel
from src.models.rnn_model import RNNModel


# ── Hyper-parameters ──────────────────────────────────────────────────────────
WINDOW = 720
HORIZON = 1
BATCH_SIZE = 256
EPOCHS = 5
ARIMA_ORDER = (5, 1, 0)
# Number of rows used for ARIMA fitting (full training set is very large)
ARIMA_TRAIN_ROWS = 5000


def build_dataloader(X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
    """Wrap numpy arrays in a PyTorch DataLoader.

    Args:
        X: Input windows, shape ``(n, window, features)``.
        y: Targets, shape ``(n,)``.
        shuffle: Whether to shuffle batches.

    Returns:
        A :class:`~torch.utils.data.DataLoader` yielding ``(X_batch, y_batch)``
        float-32 tensors.
    """
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def print_table(results: dict[str, dict[str, float]]) -> None:
    """Print a formatted results table to stdout.

    Args:
        results: Mapping of model-name → metric-dict (MAE, RMSE, MAPE).
    """
    header = f"{'Model':<12} {'MAE':>10} {'RMSE':>10} {'MAPE (%)':>12}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for name, metrics in results.items():
        print(
            f"{name:<12} {metrics['MAE']:>10.4f} {metrics['RMSE']:>10.4f}"
            f" {metrics['MAPE']:>12.2f}"
        )
    print("=" * len(header) + "\n")


def main() -> None:
    """Load data, train all baselines, evaluate on the clean test set."""
    print("Loading and preprocessing data …")
    splits = load_and_preprocess(window=WINDOW, horizon=HORIZON)
    n_features = splits.X_train.shape[2]

    train_loader = build_dataloader(splits.X_train, splits.y_train, shuffle=True)
    results: dict[str, dict[str, float]] = {}

    # ── ARIMA ──────────────────────────────────────────────────────────────────
    print("\n[ARIMA] Fitting …")
    target_idx = splits.feature_names.index("T (degC)")
    arima_train = splits.X_train[:ARIMA_TRAIN_ROWS, -1, target_idx]  # last step of each window
    arima = ARIMAModel(order=ARIMA_ORDER)
    arima.fit(arima_train)

    n_test = len(splits.y_test)
    arima_preds = arima.predict(steps=n_test)
    # Pad or trim to match test length
    arima_preds = np.resize(arima_preds, n_test)
    print("[ARIMA] Metrics on test set:")
    results["ARIMA"] = compute_metrics(splits.y_test, arima_preds)

    # ── Vanilla RNN ────────────────────────────────────────────────────────────
    print("\n[RNN] Training …")
    rnn = RNNModel(input_size=n_features)
    rnn.fit(train_loader, epochs=EPOCHS)
    rnn_preds = rnn.predict(splits.X_test)
    print("[RNN] Metrics on test set:")
    results["RNN"] = compute_metrics(splits.y_test, rnn_preds)

    # ── Stacked LSTM ───────────────────────────────────────────────────────────
    print("\n[LSTM] Training …")
    lstm = LSTMModel(input_size=n_features)
    lstm.fit(train_loader, epochs=EPOCHS)
    lstm_preds = lstm.predict(splits.X_test)
    print("[LSTM] Metrics on test set:")
    results["LSTM"] = compute_metrics(splits.y_test, lstm_preds)

    # ── MC-Dropout uncertainty example ────────────────────────────────────────
    print("\n[LSTM-MC] Running MC-Dropout on first 100 test samples …")
    mean_preds, std_preds = lstm.mc_dropout_predict(splits.X_test[:100], n_samples=50)
    print(f"  Mean predictive std: {std_preds.mean():.4f}")

    # ── Summary table ──────────────────────────────────────────────────────────
    print_table(results)


if __name__ == "__main__":
    main()
