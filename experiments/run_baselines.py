"""Baseline + proposed-method experiment on clean Jena Climate data.

Models
------
Baselines  : ARIMA · Vanilla RNN · Stacked LSTM
Proposed   : Temporal Transformer (self-attention over the input window)

Trained PyTorch models are saved to ``results/checkpoints/`` so that
``run_corruption.py`` and ``error_analysis.py`` can reload them without
re-training.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_and_preprocess
from src.evaluate import compute_metrics
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import LSTMModel
from src.models.rnn_model import RNNModel
from src.models.transformer_model import TransformerModel

# ── Directories ───────────────────────────────────────────────────────────────
RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
for d in (METRICS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Hyper-parameters ──────────────────────────────────────────────────────────
WINDOW           = 120
HORIZON          = 1
BATCH_SIZE       = 256
EPOCHS           = 5
ARIMA_ORDER      = (5, 1, 0)
ARIMA_TRAIN_ROWS = 2000


def build_dataloader(X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
    """Wrap numpy arrays in a PyTorch DataLoader.

    Args:
        X: Windows ``(n, window, features)``.
        y: Targets ``(n,)``.
        shuffle: Shuffle each epoch.

    Returns:
        DataLoader yielding float-32 ``(X_batch, y_batch)`` pairs.
    """
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def print_table(results: dict[str, dict[str, float]]) -> None:
    """Print a formatted results table.

    Args:
        results: ``{model_name: {metric: value}}``.
    """
    header = f"{'Model':<14} {'MAE':>10} {'RMSE':>10} {'MAPE (%)':>12}"
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-' * len(header)}")
    for name, m in results.items():
        print(f"{name:<14} {m['MAE']:>10.4f} {m['RMSE']:>10.4f} {m['MAPE']:>12.2f}")
    print(f"{sep}\n")


def main() -> None:
    """Load clean data, train all models, evaluate on clean test set."""
    print("=" * 60)
    print("  SEDS 537 — Robust Forecasting  |  Baseline Run")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[DATA] Loading and preprocessing …")
    splits = load_and_preprocess(window=WINDOW, horizon=HORIZON)
    n_features = splits.X_train.shape[2]
    n_test = len(splits.X_test)
    print(f"       Train: {len(splits.X_train):,}  Val: {len(splits.X_val):,}"
          f"  Test: {n_test:,}  Features: {n_features}")

    train_loader = build_dataloader(splits.X_train, splits.y_train, shuffle=True)
    results: dict[str, dict[str, float]] = {}

    # ── ARIMA (baseline 1) ────────────────────────────────────────────────────
    # Fit on the raw normalised temperature training series (non-windowed).
    # ARIMA does not consume X_test at inference — it forecasts from its own
    # parameters.  This is an important distinction in the robustness study.
    print(f"\n[ARIMA] Fitting on last {ARIMA_TRAIN_ROWS:,} training observations …")
    arima = ARIMAModel(order=ARIMA_ORDER)
    arima.fit(splits.raw_train_series[-ARIMA_TRAIN_ROWS:])
    arima_preds = arima.predict(steps=n_test)
    print("[ARIMA] Metrics on clean test set:")
    results["ARIMA"] = compute_metrics(splits.y_test, arima_preds)

    # ── Vanilla RNN (baseline 2) ──────────────────────────────────────────────
    print(f"\n[RNN] Training for {EPOCHS} epochs …")
    rnn = RNNModel(input_size=n_features)
    rnn.fit(train_loader, epochs=EPOCHS)
    rnn_preds = rnn.predict(splits.X_test)
    print("[RNN] Metrics on clean test set:")
    results["RNN"] = compute_metrics(splits.y_test, rnn_preds)
    torch.save(rnn.model.state_dict(), CKPT_DIR / "rnn.pt")
    print(f"      → Checkpoint saved: {CKPT_DIR / 'rnn.pt'}")

    # ── Stacked LSTM (baseline 3) ─────────────────────────────────────────────
    print(f"\n[LSTM] Training for {EPOCHS} epochs …")
    lstm = LSTMModel(input_size=n_features)
    lstm.fit(train_loader, epochs=EPOCHS)
    lstm_preds = lstm.predict(splits.X_test)
    print("[LSTM] Metrics on clean test set:")
    results["LSTM"] = compute_metrics(splits.y_test, lstm_preds)
    torch.save(lstm.model.state_dict(), CKPT_DIR / "lstm.pt")
    print(f"       → Checkpoint saved: {CKPT_DIR / 'lstm.pt'}")

    # ── LSTM MC-Dropout uncertainty (clean) ───────────────────────────────────
    print("\n[LSTM-MC] MC-Dropout on 200 test samples (n_samples=50) …")
    mc_mean, mc_std = lstm.mc_dropout_predict(splits.X_test[:200], n_samples=50)
    print(f"          Mean predictive std (clean): {mc_std.mean():.4f}")

    # ── Temporal Transformer (proposed method) ────────────────────────────────
    print(f"\n[Transformer] Training for {EPOCHS} epochs …")
    transformer = TransformerModel(input_size=n_features)
    transformer.fit(train_loader, epochs=EPOCHS)
    transformer_preds = transformer.predict(splits.X_test)
    print("[Transformer] Metrics on clean test set:")
    results["Transformer"] = compute_metrics(splits.y_test, transformer_preds)
    torch.save(transformer.model.state_dict(), CKPT_DIR / "transformer.pt")
    print(f"              → Checkpoint saved: {CKPT_DIR / 'transformer.pt'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_table(results)

    out_path = METRICS_DIR / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Metrics saved → {out_path}")
    print("\nNext step: python experiments/run_corruption.py")


if __name__ == "__main__":
    main()
