"""Temporal Transformer model for time-series forecasting.

This is the *proposed method* of the robustness study.  While ARIMA, RNN, and
LSTM serve as baselines, the Transformer is motivated by its global
self-attention mechanism: rather than summarising history through a fixed-size
hidden state (as RNN / LSTM do), it can attend to any position in the input
window simultaneously.  The hypothesis is that global attention makes the
model more robust to *localised* corruptions (point anomalies) because a
single corrupted time-step can be "outvoted" by all other clean positions,
whereas RNN / LSTM propagate corrupted hidden states forward.

Architecture
------------
* Sinusoidal positional encoding (no learned position parameters).
* Two stacked Transformer encoder layers (4 attention heads, FFN dim=256).
* The representation of the *last* time-step is projected to a scalar via a
  linear head.
* Dropout = 0.1 on the encoder for light regularisation.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding as in Vaswani et al. (2017).

    Args:
        d_model: Embedding / model dimension.
        max_len: Maximum sequence length to pre-compute.
        dropout: Dropout probability applied after adding the encoding.
    """

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
        # Shape: (1, max_len, d_model) for broadcasting over batch
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x*.

        Args:
            x: Tensor of shape ``(batch, seq_len, d_model)``.

        Returns:
            Tensor of same shape with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class _TransformerModule(nn.Module):
    """Internal PyTorch module: projection → positional encoding → Transformer
    encoder → linear output head.

    Args:
        input_size: Number of input features per time step.
        d_model: Internal model dimension (projected from input_size).
        nhead: Number of attention heads.
        num_layers: Number of stacked encoder layers.
        dim_feedforward: Hidden size of the point-wise FFN.
        dropout: Dropout probability inside the encoder.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Project raw features → d_model
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # (batch, seq, features)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor ``(batch, seq_len, input_size)``.

        Returns:
            Tensor ``(batch,)`` — one prediction per sample.
        """
        x = self.input_proj(x)          # (B, T, d_model)
        x = self.pos_enc(x)              # add positional encoding
        x = self.encoder(x)              # (B, T, d_model)
        return self.fc(x[:, -1, :]).squeeze(-1)   # use last token


class TransformerModel:
    """Temporal Transformer forecaster — the *proposed method*.

    Motivation: self-attention attends globally to all time-steps, so a single
    corrupted position has less influence on the final representation than in
    RNN / LSTM where corruptions propagate through the hidden state.

    Args:
        input_size: Number of input features per time step.
        d_model: Internal embedding dimension (default 64).
        nhead: Number of attention heads (default 4).
        num_layers: Number of stacked encoder layers (default 2).
        dim_feedforward: Point-wise FFN hidden size (default 256).
        dropout: Dropout probability (default 0.1).
        lr: Learning rate for the Adam optimiser.
        device: ``"cuda"`` or ``"cpu"``.  Auto-detected if ``None``.
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _TransformerModule(
            input_size, d_model, nhead, num_layers, dim_feedforward, dropout
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=2, factor=0.5
        )
        self.criterion = nn.MSELoss()

    def fit(self, dataloader: DataLoader, epochs: int = 10) -> "TransformerModel":
        """Train the model on batches from *dataloader*.

        Args:
            dataloader: Yields ``(X_batch, y_batch)`` float-32 tensor pairs.
            epochs: Number of full passes over the dataset.

        Returns:
            Self, to allow method chaining.
        """
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = self.criterion(preds, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(dataloader)
            self.scheduler.step(avg)
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
        return self

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Generate point predictions for input *x*.

        Args:
            x: Array or tensor ``(n_samples, seq_len, features)``.

        Returns:
            1-D numpy array of predictions, shape ``(n_samples,)``.
        """
        self.model.eval()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)
        with torch.no_grad():
            preds = self.model(x)
        return preds.cpu().numpy()
