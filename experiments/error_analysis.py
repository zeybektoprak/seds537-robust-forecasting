"""Error analysis for the robustness study.

This script investigates *failure cases* — where and why predictions are worst.

Analyses performed
------------------
1. **Error distribution** — histogram of absolute errors for each model on the
   clean test set.
2. **Worst predictions** — top-50 samples with highest absolute error plotted
   against ground truth.
3. **Error vs. input variance** — scatter plot of |error| vs. the variance of
   the input window (high-variance windows correspond to more volatile weather).
4. **Temporal error pattern** — mean absolute error binned by position in the
   test set (reveals drift / non-stationarity effects).
5. **Error under corruption** — bar chart comparing mean absolute error on
   clean vs. heavily corrupted (σ=2.0) inputs, side-by-side for all models.

All figures are saved to ``results/figures/``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.corruption import add_gaussian_noise
from src.data_loader import load_and_preprocess
from src.models.lstm_model import LSTMModel
from src.models.rnn_model import RNNModel
from src.models.transformer_model import TransformerModel

FIGURES_DIR = ROOT / "results" / "figures"
CKPT_DIR    = ROOT / "results" / "checkpoints"
METRICS_DIR = ROOT / "results" / "metrics"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE = {
    "ARIMA":       "#4878d0",
    "RNN":         "#ee854a",
    "LSTM":        "#6acc65",
    "Transformer": "#d65f5f",
}

WINDOW          = 720
HORIZON         = 1
ARIMA_ORDER     = (5, 1, 0)
ARIMA_TRAIN_ROWS = 500
TOP_N_WORST     = 50    # number of worst samples to plot
N_BINS          = 20    # temporal bins


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _load_model(cls, ckpt_name: str, input_size: int):
    """Instantiate *cls* and load weights from checkpoint if available."""
    m = cls(input_size=input_size)
    ckpt = CKPT_DIR / ckpt_name
    if ckpt.exists():
        m.model.load_state_dict(torch.load(ckpt, map_location=m.device))
    else:
        print(f"  WARNING: checkpoint {ckpt} not found — model is untrained.")
    return m


# ── Analysis functions ────────────────────────────────────────────────────────

def plot_error_distributions(
    errors: dict[str, np.ndarray],
) -> None:
    """Histogram of absolute errors for each model.

    Args:
        errors: ``{model_name: abs_error_array}`` on clean test set.
    """
    fig, axes = plt.subplots(1, len(errors), figsize=(4 * len(errors), 4), sharey=False)
    if len(errors) == 1:
        axes = [axes]

    for ax, (name, err) in zip(axes, errors.items()):
        ax.hist(err, bins=60, color=PALETTE[name], edgecolor="white", alpha=0.85)
        ax.axvline(err.mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"mean={err.mean():.3f}")
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("|error|")
        ax.set_ylabel("Count")
        ax.legend(fontsize=10)

    fig.suptitle("Absolute Error Distribution (Clean Test Set)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "error_distributions")


def plot_worst_predictions(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    n: int = TOP_N_WORST,
) -> None:
    """Scatter: ground truth vs. predicted for the *n* worst LSTM samples.

    Args:
        y_true: Clean test targets.
        preds: ``{model_name: predictions}`` dict.
        n: Number of worst-case samples to highlight.
    """
    lstm_err = np.abs(y_true - preds["LSTM"])
    worst_idx = np.argsort(lstm_err)[-n:]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(
        y_true[worst_idx],
        preds["LSTM"][worst_idx],
        color=PALETTE["LSTM"], alpha=0.7, s=40, label="LSTM (worst 50)",
    )
    ax.scatter(
        y_true[worst_idx],
        preds["Transformer"][worst_idx],
        color=PALETTE["Transformer"], alpha=0.7, s=40, marker="^",
        label="Transformer (worst 50)",
    )
    lo = min(y_true[worst_idx].min(), preds["LSTM"][worst_idx].min()) - 0.2
    hi = max(y_true[worst_idx].max(), preds["LSTM"][worst_idx].max()) + 0.2
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1.2, label="Perfect prediction")
    ax.set_xlabel("True value (normalised T)", fontsize=13)
    ax.set_ylabel("Predicted value", fontsize=13)
    ax.set_title("Worst-50 Predictions: True vs. Predicted", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, "worst_predictions")


def plot_error_vs_input_variance(
    X_test: np.ndarray,
    preds: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> None:
    """Scatter: |error| vs. variance of the input window.

    High-variance windows correspond to rapid weather changes.

    Args:
        X_test: Test input windows ``(n, window, features)``.
        preds: ``{model_name: predictions}`` dict.
        y_true: True test targets.
    """
    # Use variance of the target channel across the window
    target_channel = 1  # T (degC) is typically column 1 after Date Time drop
    window_var = X_test[:, :, target_channel].var(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for ax, (name, pred) in zip(axes, preds.items()):
        err = np.abs(y_true - pred)
        ax.scatter(window_var, err, alpha=0.15, s=8, color=PALETTE[name])
        # Bin and plot mean
        bins = np.percentile(window_var, np.linspace(0, 100, 15))
        bin_means, bin_centers = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (window_var >= lo) & (window_var < hi)
            if mask.sum() > 5:
                bin_means.append(err[mask].mean())
                bin_centers.append((lo + hi) / 2)
        ax.plot(bin_centers, bin_means, color="black", linewidth=2, marker="o", markersize=5)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Input window variance", fontsize=11)
        ax.set_ylabel("|error|", fontsize=11)

    fig.suptitle("|Error| vs. Input Window Variance", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "error_vs_input_variance")


def plot_temporal_error_pattern(
    y_true: np.ndarray,
    preds: dict[str, np.ndarray],
    n_bins: int = N_BINS,
) -> None:
    """Line plot: mean absolute error binned by position in the test set.

    Reveals temporal drift or non-stationarity.

    Args:
        y_true: True test targets.
        preds: ``{model_name: predictions}`` dict.
        n_bins: Number of temporal bins.
    """
    n = len(y_true)
    bin_size = n // n_bins
    xs = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(10, 5))
    for name, pred in preds.items():
        err = np.abs(y_true - pred)
        binned = [err[i * bin_size: (i + 1) * bin_size].mean() for i in range(n_bins)]
        ax.plot(xs, binned, label=name, color=PALETTE[name], linewidth=2.0, marker="o",
                markersize=5)

    ax.set_xlabel("Test set position (binned)", fontsize=13)
    ax.set_ylabel("Mean Absolute Error", fontsize=13)
    ax.set_title("Temporal Error Pattern Across Test Set", fontsize=14, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    _save(fig, "temporal_error_pattern")


def plot_clean_vs_corrupted_mae(
    preds_clean: dict[str, np.ndarray],
    preds_corrupt: dict[str, np.ndarray],
    y_true: np.ndarray,
) -> None:
    """Grouped bar chart: MAE on clean vs. σ=2.0 Gaussian corrupted inputs.

    Args:
        preds_clean: Clean predictions per model.
        preds_corrupt: Predictions on σ=2.0 corrupted input per model.
        y_true: True test targets.
    """
    models = list(preds_clean.keys())
    mae_clean   = [np.abs(y_true - preds_clean[m]).mean()   for m in models]
    mae_corrupt = [np.abs(y_true - preds_corrupt[m]).mean() for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, mae_clean,   width, label="Clean",
                   color=[PALETTE[m] for m in models], alpha=0.9)
    bars2 = ax.bar(x + width / 2, mae_corrupt, width, label="Corrupted (σ=2.0)",
                   color=[PALETTE[m] for m in models], alpha=0.45,
                   edgecolor=[PALETTE[m] for m in models], linewidth=1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13)
    ax.set_ylabel("MAE", fontsize=13)
    ax.set_title("MAE: Clean vs. Heavily Corrupted (σ=2.0)", fontsize=14, fontweight="bold")
    ax.legend()

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    _save(fig, "clean_vs_corrupted_mae")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load models and run all error analysis plots."""
    import gc
    print("=" * 55)
    print("  SEDS 537 — Robust Forecasting  |  Error Analysis")
    print("=" * 55)

    print("\n[DATA] Loading test split only …")
    splits = load_and_preprocess(window=WINDOW, horizon=HORIZON)
    # Use only first 2000 test samples to save RAM
    X_test  = splits.X_test[:2000].copy()
    y_test  = splits.y_test[:2000].copy()
    n_features = X_test.shape[2]
    n_test = len(X_test)
    del splits
    gc.collect()
    print(f"       Using {n_test} test samples (RAM-safe mode)")

    # ── Load models ───────────────────────────────────────────────────────────
    print("\n[MODELS] Loading checkpoints …")
    rnn         = _load_model(RNNModel,         "rnn.pt",         n_features)
    lstm        = _load_model(LSTMModel,        "lstm.pt",        n_features)
    transformer = _load_model(TransformerModel, "transformer.pt", n_features)

    # ── Clean predictions ─────────────────────────────────────────────────────
    preds_clean = {
        "RNN":         rnn.predict(X_test),
        "LSTM":        lstm.predict(X_test),
        "Transformer": transformer.predict(X_test),
    }
    del rnn, lstm, transformer
    gc.collect()
    errors_clean = {k: np.abs(y_test - v) for k, v in preds_clean.items()}

    # ── Corrupted predictions (σ=2.0) ─────────────────────────────────────────
    X_corrupt = add_gaussian_noise(X_test, sigma=2.0)
    # Reload models for corrupt predictions
    rnn         = _load_model(RNNModel,         "rnn.pt",         n_features)
    lstm        = _load_model(LSTMModel,        "lstm.pt",        n_features)
    transformer = _load_model(TransformerModel, "transformer.pt", n_features)
    preds_corrupt = {
        "RNN":         rnn.predict(X_corrupt),
        "LSTM":        lstm.predict(X_corrupt),
        "Transformer": transformer.predict(X_corrupt),
    }
    del X_corrupt, rnn, lstm, transformer
    gc.collect()

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating error analysis figures …")

    plot_error_distributions(errors_clean)
    plot_worst_predictions(y_test, preds_clean)
    plot_error_vs_input_variance(X_test, preds_clean, y_test)
    plot_temporal_error_pattern(y_test, preds_clean)
    plot_clean_vs_corrupted_mae(preds_clean, preds_corrupt, y_test)

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n── Error Summary (Clean Test Set) ──")
    print(f"{'Model':<14} {'Mean |err|':>12} {'Median |err|':>14} {'90th pct':>10}")
    print("-" * 54)
    for name, err in errors_clean.items():
        print(f"{name:<14} {err.mean():>12.4f} {np.median(err):>14.4f} "
              f"{np.percentile(err, 90):>10.4f}")

    print(f"\nAll error analysis figures saved to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
