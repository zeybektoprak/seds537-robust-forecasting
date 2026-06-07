"""Robustness experiment: evaluate all models under corrupted test inputs.

Experimental design
-------------------
All models are **trained on clean data** and tested on **corrupted inputs**,
measuring how accuracy degrades as data quality worsens.

Corruption types
~~~~~~~~~~~~~~~~
* **Gaussian noise**  — σ ∈ {0.00, 0.25, 0.50, 1.00, 2.00}
* **Point anomalies** — ratio ∈ {0.00, 0.01, 0.05, 0.10, 0.20}

For each (model, corruption_type, level):
  1. Corrupt X_test.
  2. Run inference → compute MAE / RMSE / MAPE vs. clean y_test.
  3. For LSTM: record mean MC-Dropout predictive std (uncertainty signal).

ARIMA note
~~~~~~~~~~
ARIMA does not consume X_test windows at inference — predictions come entirely
from its training-phase parameters.  It therefore appears as a *flat baseline*
across corruption levels, which is itself informative: generative models are
trivially input-robust but cannot exploit rich multivariate context.

Output
~~~~~~
``results/metrics/corruption_results.json``
``results/metrics/corruption_results.csv``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.corruption import add_gaussian_noise, inject_point_anomalies
from src.data_loader import load_and_preprocess
from src.evaluate import compute_metrics
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import LSTMModel
from src.models.rnn_model import RNNModel
from src.models.transformer_model import TransformerModel

RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
for d in (METRICS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

NOISE_SIGMAS:   list[float] = [0.0, 0.25, 0.5, 1.0, 2.0]
ANOMALY_RATIOS: list[float] = [0.0, 0.01, 0.05, 0.10, 0.20]

WINDOW           = 120
HORIZON          = 1
BATCH_SIZE       = 256
EPOCHS           = 5
ARIMA_ORDER      = (5, 1, 0)
ARIMA_TRAIN_ROWS = 2000
MC_SAMPLES       = 50
MC_TEST_SIZE     = 200


def build_dataloader(X: np.ndarray, y: np.ndarray) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)


def _load_or_train_neural(cls, ckpt_name: str, input_size: int,
                           train_loader: DataLoader) -> object:
    """Load model from checkpoint, or train and save if checkpoint is missing.

    Args:
        cls: Model class (RNNModel, LSTMModel, TransformerModel).
        ckpt_name: Filename inside ``CKPT_DIR``.
        input_size: Number of input features.
        train_loader: DataLoader for training.

    Returns:
        Trained model instance.
    """
    model = cls(input_size=input_size)
    ckpt = CKPT_DIR / ckpt_name
    if ckpt.exists():
        model.model.load_state_dict(torch.load(ckpt, map_location=model.device))
        print(f"  Loaded {cls.__name__} from {ckpt}")
    else:
        print(f"  No checkpoint — training {cls.__name__} for {EPOCHS} epochs …")
        model.fit(train_loader, epochs=EPOCHS)
        torch.save(model.model.state_dict(), ckpt)
        print(f"  Saved → {ckpt}")
    return model


def main() -> None:
    """Run full corruption sweep and save results."""
    print("=" * 60)
    print("  SEDS 537 — Robust Forecasting  |  Corruption Study")
    print("=" * 60)

    print("\n[DATA] Loading …")
    splits = load_and_preprocess(window=WINDOW, horizon=HORIZON)
    n_features = splits.X_train.shape[2]
    n_test = len(splits.X_test)
    print(f"       Test: {n_test:,}  Features: {n_features}")

    train_loader = build_dataloader(splits.X_train, splits.y_train)

    # ── ARIMA ─────────────────────────────────────────────────────────────────
    print("\n[ARIMA] Fitting …")
    arima = ARIMAModel(order=ARIMA_ORDER)
    arima.fit(splits.raw_train_series[-ARIMA_TRAIN_ROWS:])
    arima_clean_preds = arima.predict(steps=n_test)

    # ── Neural models ─────────────────────────────────────────────────────────
    rnn         = _load_or_train_neural(RNNModel,         "rnn.pt",         n_features, train_loader)
    lstm        = _load_or_train_neural(LSTMModel,        "lstm.pt",        n_features, train_loader)
    transformer = _load_or_train_neural(TransformerModel, "transformer.pt", n_features, train_loader)

    results: dict = {"gaussian": {}, "point_anomaly": {}}

    def _eval_all(X_corr: np.ndarray) -> dict:
        """Return metrics for all models on corrupted X_corr."""
        arima_m = compute_metrics(splits.y_test, arima_clean_preds, verbose=False)
        rnn_m   = compute_metrics(splits.y_test, rnn.predict(X_corr),         verbose=False)
        lstm_m  = compute_metrics(splits.y_test, lstm.predict(X_corr),        verbose=False)
        tf_m    = compute_metrics(splits.y_test, transformer.predict(X_corr), verbose=False)
        mc_mean, mc_std = lstm.mc_dropout_predict(X_corr[:MC_TEST_SIZE], MC_SAMPLES)
        return {
            "ARIMA":       arima_m,
            "RNN":         rnn_m,
            "LSTM":        lstm_m,
            "Transformer": tf_m,
            "LSTM_mc_std": float(mc_std.mean()),
        }

    # ── Gaussian noise sweep ──────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Gaussian Noise Sweep")
    print("─" * 60)
    for sigma in NOISE_SIGMAS:
        X_c = splits.X_test if sigma == 0.0 else add_gaussian_noise(splits.X_test, sigma)
        res = _eval_all(X_c)
        results["gaussian"][sigma] = res
        print(f"  σ={sigma:<5}  ARIMA={res['ARIMA']['MAE']:.4f}  "
              f"RNN={res['RNN']['MAE']:.4f}  LSTM={res['LSTM']['MAE']:.4f}  "
              f"Transformer={res['Transformer']['MAE']:.4f}  "
              f"MC-std={res['LSTM_mc_std']:.4f}")

    # ── Point anomaly sweep ───────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("  Point Anomaly Sweep")
    print("─" * 60)
    for ratio in ANOMALY_RATIOS:
        X_c = splits.X_test if ratio == 0.0 else inject_point_anomalies(splits.X_test, ratio)
        res = _eval_all(X_c)
        results["point_anomaly"][ratio] = res
        print(f"  ratio={ratio:<5}  ARIMA={res['ARIMA']['MAE']:.4f}  "
              f"RNN={res['RNN']['MAE']:.4f}  LSTM={res['LSTM']['MAE']:.4f}  "
              f"Transformer={res['Transformer']['MAE']:.4f}  "
              f"MC-std={res['LSTM_mc_std']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    json_path = METRICS_DIR / "corruption_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    rows = []
    for corr_type, levels in results.items():
        for level, models in levels.items():
            for model_name in ("ARIMA", "RNN", "LSTM", "Transformer"):
                m = models[model_name]
                row = {
                    "corruption_type": corr_type,
                    "level":           float(level),
                    "model":           model_name,
                    "MAE":             m["MAE"],
                    "RMSE":            m["RMSE"],
                    "MAPE":            m["MAPE"],
                }
                if model_name == "LSTM":
                    row["MC_std"] = models["LSTM_mc_std"]
                rows.append(row)

    csv_path = METRICS_DIR / "corruption_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"CSV saved    → {csv_path}")
    print("\nNext step: python experiments/plot_results.py")


if __name__ == "__main__":
    main()
