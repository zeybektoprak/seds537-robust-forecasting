"""Ablation study for the Temporal Transformer.

We systematically remove or shrink components of the proposed Transformer
to identify which design choices contribute to both clean-data accuracy
*and* corruption robustness.

Ablation variants
-----------------
1. **Full model** (baseline — our proposed method):
   d_model=64, 2 encoder layers, 4 heads, sinusoidal PE, dropout=0.1

2. **No positional encoding** (PE disabled — all zeros instead of sinusoidal):
   Removes the time-ordering signal. Tests whether PE helps robustness.

3. **Single encoder layer** (num_layers=1 instead of 2):
   Reduces depth. Tests whether stacking helps.

4. **Reduced d_model** (d_model=32 instead of 64):
   Half the model capacity. Tests whether width matters.

5. **No dropout** (dropout=0.0):
   Removes regularisation. Tests whether dropout helps clean accuracy / robustness.

Each variant is trained for 5 epochs on clean data, then evaluated on:
  - Clean test set  (σ = 0.0)
  - Moderate noise  (σ = 0.5)
  - Heavy noise     (σ = 1.0)
  - Point anomalies (ratio = 0.05)

Results saved to ``results/metrics/ablation_results.json`` and ``.csv``.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.corruption import add_gaussian_noise, inject_point_anomalies
from src.data_loader import load_and_preprocess
from src.evaluate import compute_metrics

RESULTS_DIR = ROOT / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
for d in (METRICS_DIR, CKPT_DIR):
    d.mkdir(parents=True, exist_ok=True)

WINDOW     = 120
HORIZON    = 1
BATCH_SIZE = 256
EPOCHS     = 5

EVAL_CONDITIONS = {
    "clean":          ("gaussian",      0.0),
    "gaussian_0.5":   ("gaussian",      0.5),
    "gaussian_1.0":   ("gaussian",      1.0),
    "anomaly_0.05":   ("point_anomaly", 0.05),
}


# ── Positional Encoding (copied here to allow disabling) ──────────────────────

class _PEFull(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _PENone(nn.Module):
    """Disabled positional encoding (identity + dropout)."""
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        return self.dropout(x)


class _AblationTransformer(nn.Module):
    """Configurable Transformer for ablation."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        use_pe: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        if use_pe:
            self.pos_enc = _PEFull(d_model, dropout=dropout)
        else:
            self.pos_enc = _PENone(dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        return self.fc(x[:, -1, :]).squeeze(-1)


class AblationModel:
    """Thin wrapper around _AblationTransformer — same fit/predict interface."""

    def __init__(self, input_size: int, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.1, use_pe=True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = _AblationTransformer(
            input_size, d_model, nhead, num_layers, dim_feedforward, dropout, use_pe
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=2, factor=0.5
        )
        self.criterion = nn.MSELoss()

    def fit(self, loader: DataLoader, epochs: int) -> "AblationModel":
        self.model.train()
        for epoch in range(epochs):
            total = 0.0
            for Xb, yb in loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(Xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total += loss.item()
            avg = total / len(loader)
            self.scheduler.step(avg)
            print(f"    Epoch {epoch+1}/{epochs}  loss={avg:.4f}")
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        t = torch.tensor(x, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            return self.model(t).cpu().numpy()


# ── Ablation variant definitions ──────────────────────────────────────────────

def get_variants(input_size: int) -> dict[str, AblationModel]:
    """Return a fresh dict of all ablation variants."""
    return {
        "Full (proposed)": AblationModel(
            input_size, d_model=64, nhead=4, num_layers=2,
            dim_feedforward=256, dropout=0.1, use_pe=True,
        ),
        "No positional enc.": AblationModel(
            input_size, d_model=64, nhead=4, num_layers=2,
            dim_feedforward=256, dropout=0.1, use_pe=False,
        ),
        "1 encoder layer": AblationModel(
            input_size, d_model=64, nhead=4, num_layers=1,
            dim_feedforward=256, dropout=0.1, use_pe=True,
        ),
        "d_model=32": AblationModel(
            input_size, d_model=32, nhead=4, num_layers=2,
            dim_feedforward=128, dropout=0.1, use_pe=True,
        ),
        "No dropout": AblationModel(
            input_size, d_model=64, nhead=4, num_layers=2,
            dim_feedforward=256, dropout=0.0, use_pe=True,
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  SEDS 537 — Robust Forecasting  |  Ablation Study")
    print("=" * 60)

    print("\n[DATA] Loading …")
    splits = load_and_preprocess(window=WINDOW, horizon=HORIZON)
    n_features = splits.X_train.shape[2]

    loader = DataLoader(
        TensorDataset(
            torch.tensor(splits.X_train, dtype=torch.float32),
            torch.tensor(splits.y_train, dtype=torch.float32),
        ),
        batch_size=BATCH_SIZE, shuffle=True,
    )

    # Pre-compute corrupted test sets
    X_test   = splits.X_test
    y_test   = splits.y_test
    test_sets = {
        "clean":         X_test,
        "gaussian_0.5":  add_gaussian_noise(X_test, sigma=0.5),
        "gaussian_1.0":  add_gaussian_noise(X_test, sigma=1.0),
        "anomaly_0.05":  inject_point_anomalies(X_test, ratio=0.05),
    }

    results: dict[str, dict[str, dict]] = {}

    variants = get_variants(n_features)
    for name, model in variants.items():
        print(f"\n[ABLATION] Training variant: {name}")
        model.fit(loader, epochs=EPOCHS)

        variant_results: dict[str, dict] = {}
        for cond_name, X_c in test_sets.items():
            preds = model.predict(X_c)
            m = compute_metrics(y_test, preds, verbose=False)
            variant_results[cond_name] = m
            print(f"  {cond_name:<20}  MAE={m['MAE']:.4f}  RMSE={m['RMSE']:.4f}")
        results[name] = variant_results

    # ── Save JSON ──────────────────────────────────────────────────────────────
    json_path = METRICS_DIR / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {json_path}")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    rows = []
    for variant, conditions in results.items():
        for cond, metrics in conditions.items():
            rows.append({"variant": variant, "condition": cond, **metrics})
    csv_path = METRICS_DIR / "ablation_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    # ── Pretty table ───────────────────────────────────────────────────────────
    print("\n── Ablation Summary (MAE) ──────────────────────────────────────────")
    header = f"{'Variant':<24} {'clean':>8} {'σ=0.5':>8} {'σ=1.0':>8} {'ano5%':>8}"
    print(header)
    print("-" * len(header))
    for variant, conditions in results.items():
        row = (
            f"{variant:<24}"
            f"  {conditions['clean']['MAE']:>6.4f}"
            f"  {conditions['gaussian_0.5']['MAE']:>6.4f}"
            f"  {conditions['gaussian_1.0']['MAE']:>6.4f}"
            f"  {conditions['anomaly_0.05']['MAE']:>6.4f}"
        )
        print(row)


if __name__ == "__main__":
    main()
