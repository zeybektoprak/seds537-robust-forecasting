"""Generate all publication-quality figures for the robustness study.

Figures produced
----------------
1.  mae_vs_gaussian_noise.png      — MAE vs σ, all models
2.  rmse_vs_gaussian_noise.png     — RMSE vs σ, all models
3.  mae_vs_anomaly_ratio.png       — MAE vs anomaly ratio, all models
4.  rmse_vs_anomaly_ratio.png      — RMSE vs anomaly ratio, all models
5.  mc_uncertainty_gaussian.png    — LSTM MC-Dropout std vs σ
6.  mc_uncertainty_anomaly.png     — LSTM MC-Dropout std vs anomaly ratio
7.  predictions_clean.png          — True vs predicted, first 200 test samples
8.  mc_dropout_bands.png           — LSTM ±2σ uncertainty bands
9.  robustness_degradation.png     — Relative MAE increase (%) vs corruption

Run AFTER ``run_corruption.py`` has produced ``corruption_results.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_and_preprocess
from src.models.arima_model import ARIMAModel
from src.models.lstm_model import LSTMModel
from src.models.rnn_model import RNNModel
from src.models.transformer_model import TransformerModel

FIGURES_DIR = ROOT / "results" / "figures"
METRICS_DIR = ROOT / "results" / "metrics"
CKPT_DIR    = ROOT / "results" / "checkpoints"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.15)
PALETTE  = {"ARIMA": "#4878d0", "RNN": "#ee854a", "LSTM": "#6acc65", "Transformer": "#d65f5f"}
MARKERS  = {"ARIMA": "o",       "RNN": "s",        "LSTM": "^",       "Transformer": "D"}
MODELS   = ["ARIMA", "RNN", "LSTM", "Transformer"]
LW, MS   = 2.2, 8
FIGSIZE  = (9, 5)

WINDOW           = 120
HORIZON          = 1
ARIMA_ORDER      = (5, 1, 0)
ARIMA_TRAIN_ROWS = 2000
NOISE_SIGMAS     = [0.0, 0.25, 0.5, 1.0, 2.0]
ANOMALY_RATIOS   = [0.0, 0.01, 0.05, 0.10, 0.20]
PRED_N           = 200


def _save(fig: plt.Figure, name: str) -> None:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def plot_metric_vs_corruption(
    results: dict, corr_key: str, levels: list[float],
    metric: str, xlabel: str, title: str, filename: str,
) -> None:
    """Line plot of *metric* vs. corruption level for all models."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for model in MODELS:
        ys = [results[corr_key][str(lvl)][model][metric] for lvl in levels]
        ax.plot(levels, ys, label=model, color=PALETTE[model],
                marker=MARKERS[model], linewidth=LW, markersize=MS)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(metric, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    _save(fig, filename)


def plot_mc_uncertainty(
    results: dict, corr_key: str, levels: list[float],
    xlabel: str, title: str, filename: str,
) -> None:
    """Bar chart of LSTM MC-Dropout predictive std vs. corruption level."""
    stds = [results[corr_key][str(lvl)]["LSTM_mc_std"] for lvl in levels]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar([str(lvl) for lvl in levels], stds,
           color=PALETTE["LSTM"], edgecolor="white", width=0.55)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Mean MC-Dropout Std", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, filename)


def plot_robustness_degradation(
    results: dict, corr_key: str, levels: list[float],
    xlabel: str, title: str, filename: str,
) -> None:
    """Line plot of *relative* MAE increase (%) vs. clean baseline."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for model in MODELS:
        base = results[corr_key][str(levels[0])][model]["MAE"]
        rel  = [
            (results[corr_key][str(lvl)][model]["MAE"] - base) / (base + 1e-9) * 100
            for lvl in levels
        ]
        ax.plot(levels, rel, label=model, color=PALETTE[model],
                marker=MARKERS[model], linewidth=LW, markersize=MS)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("MAE increase over clean (%)", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    fig.tight_layout()
    _save(fig, filename)


def plot_predictions(y_true, preds, n=PRED_N) -> None:
    """True vs. predicted line plot for the first *n* test samples."""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(np.arange(n), y_true[:n], label="True", color="black",
            linewidth=1.8, zorder=5)
    for name, pred in preds.items():
        ax.plot(np.arange(n), pred[:n], label=name, color=PALETTE[name],
                linewidth=1.4, alpha=0.85)
    ax.set_xlabel("Time step", fontsize=13)
    ax.set_ylabel("Normalised T (degC)", fontsize=13)
    ax.set_title("True vs. Predicted — Clean Test Set (first 200 samples)",
                 fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    _save(fig, "predictions_clean")


def plot_mc_bands(y_true, mc_mean, mc_std, n=PRED_N) -> None:
    """LSTM MC-Dropout ±2σ uncertainty bands."""
    xs = np.arange(n)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(xs, y_true[:n], label="True", color="black", linewidth=1.8, zorder=5)
    ax.plot(xs, mc_mean[:n], label="LSTM mean", color=PALETTE["LSTM"], linewidth=1.6)
    ax.fill_between(xs, mc_mean[:n] - 2*mc_std[:n], mc_mean[:n] + 2*mc_std[:n],
                    color=PALETTE["LSTM"], alpha=0.25, label="±2σ")
    ax.set_xlabel("Time step", fontsize=13)
    ax.set_ylabel("Normalised T (degC)", fontsize=13)
    ax.set_title("LSTM MC-Dropout Uncertainty — Clean Test Set",
                 fontsize=14, fontweight="bold")
    ax.legend(framealpha=0.9)
    fig.tight_layout()
    _save(fig, "mc_dropout_bands")


def main() -> None:
    """Load results + models, then generate all figures."""
    print("=" * 55)
    print("  SEDS 537 — Robust Forecasting  |  Plotting")
    print("=" * 55)

    json_path = METRICS_DIR / "corruption_results.json"
    if not json_path.exists():
        print(f"\nERROR: {json_path} not found.")
        print("Run `python experiments/run_corruption.py` first.")
        sys.exit(1)

    with open(json_path) as f:
        results = json.load(f)
    print(f"\nLoaded results from {json_path}")

    # ── Figures 1–4: metric vs. corruption ───────────────────────────────────
    print("\nGenerating metric vs corruption figures …")
    for metric in ("MAE", "RMSE"):
        plot_metric_vs_corruption(
            results, "gaussian", NOISE_SIGMAS, metric,
            "Gaussian Noise σ", f"{metric} vs Gaussian Noise Level",
            f"{metric.lower()}_vs_gaussian_noise",
        )
        plot_metric_vs_corruption(
            results, "point_anomaly", ANOMALY_RATIOS, metric,
            "Anomaly Injection Ratio", f"{metric} vs Point Anomaly Ratio",
            f"{metric.lower()}_vs_anomaly_ratio",
        )

    # ── Figures 5–6: MC uncertainty ───────────────────────────────────────────
    print("Generating MC-Dropout uncertainty figures …")
    plot_mc_uncertainty(results, "gaussian",      NOISE_SIGMAS,   "Gaussian Noise σ",
                        "LSTM Uncertainty vs Gaussian Noise",  "mc_uncertainty_gaussian")
    plot_mc_uncertainty(results, "point_anomaly", ANOMALY_RATIOS, "Anomaly Injection Ratio",
                        "LSTM Uncertainty vs Point Anomaly",   "mc_uncertainty_anomaly")

    # ── Figure 9: Relative degradation ────────────────────────────────────────
    print("Generating robustness degradation figures …")
    plot_robustness_degradation(
        results, "gaussian", NOISE_SIGMAS, "Gaussian Noise σ",
        "Relative MAE Degradation vs Gaussian Noise",
        "robustness_degradation_gaussian",
    )
    plot_robustness_degradation(
        results, "point_anomaly", ANOMALY_RATIOS, "Anomaly Injection Ratio",
        "Relative MAE Degradation vs Point Anomaly Ratio",
        "robustness_degradation_anomaly",
    )

    # ── Figures 7–8: prediction plots (need checkpoints) ─────────────────────
    print("Generating prediction and uncertainty-band figures …")
    splits = load_and_preprocess(window=WINDOW, horizon=HORIZON)
    n_features = splits.X_train.shape[2]
    n_test = len(splits.X_test)

    def _load(cls, fname):
        m = cls(input_size=n_features)
        ckpt = CKPT_DIR / fname
        if ckpt.exists():
            m.model.load_state_dict(torch.load(ckpt, map_location=m.device))
        else:
            print(f"  WARNING: {ckpt} not found — predictions will be random.")
        return m

    rnn         = _load(RNNModel,         "rnn.pt")
    lstm        = _load(LSTMModel,        "lstm.pt")
    transformer = _load(TransformerModel, "transformer.pt")

    arima = ARIMAModel(order=ARIMA_ORDER)
    arima.fit(splits.raw_train_series[-ARIMA_TRAIN_ROWS:])
    arima_preds = arima.predict(steps=n_test)

    preds = {
        "ARIMA":       arima_preds,
        "RNN":         rnn.predict(splits.X_test),
        "LSTM":        lstm.predict(splits.X_test),
        "Transformer": transformer.predict(splits.X_test),
    }
    plot_predictions(splits.y_test, preds)

    mc_mean, mc_std = lstm.mc_dropout_predict(splits.X_test[:PRED_N], n_samples=50)
    plot_mc_bands(splits.y_test, mc_mean, mc_std)

    print(f"\n✓ All figures saved to {FIGURES_DIR}")
    print("Next step: python experiments/error_analysis.py")


if __name__ == "__main__":
    main()
