"""Two-layer stacked LSTM model in PyTorch with MC-Dropout uncertainty estimation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class _LSTMModule(nn.Module):
    """Internal PyTorch module: 2-layer stacked LSTM with dropout and linear head."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Separate dropout applied before the output head (used for MC-Dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Tensor of shape ``(batch,)`` with one prediction per sample.
        """
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze(-1)


class LSTMModel:
    """Stacked LSTM forecaster (2 layers, 128 hidden units, dropout=0.2).

    Supports standard point prediction and Monte-Carlo Dropout uncertainty
    estimation via :meth:`mc_dropout_predict`.

    Args:
        input_size: Number of input features per time step.
        hidden_size: Number of hidden units in each LSTM layer.
        num_layers: Number of stacked LSTM layers.
        dropout: Dropout probability applied between layers and before output.
        lr: Learning rate for the Adam optimiser.
        device: ``"cuda"`` or ``"cpu"``.  Auto-detected if ``None``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _LSTMModule(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def fit(self, dataloader: DataLoader, epochs: int = 10) -> "LSTMModel":
        """Train the model on batches from *dataloader*.

        Args:
            dataloader: Yields ``(X_batch, y_batch)`` tensor pairs where
                ``X_batch`` has shape ``(batch, seq_len, features)``.
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
                self.optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(dataloader)
            print(f"  Epoch {epoch + 1}/{epochs}  loss={avg:.4f}")
        return self

    def predict(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Generate point predictions for input *x*.

        Args:
            x: Array or tensor of shape ``(n_samples, seq_len, features)``.

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

    def mc_dropout_predict(
        self,
        x: np.ndarray | torch.Tensor,
        n_samples: int = 50,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate predictive uncertainty via Monte-Carlo Dropout.

        Dropout is left active at inference time and the model is called
        *n_samples* times to build an empirical predictive distribution.

        Args:
            x: Array or tensor of shape ``(n_samples_data, seq_len, features)``.
            n_samples: Number of stochastic forward passes.

        Returns:
            Tuple ``(mean, std)`` where both arrays have shape ``(n_data,)``.
            *mean* is the predictive mean and *std* is the predictive standard
            deviation across the MC samples.
        """
        # Keep dropout active during inference
        self.model.train()
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)

        samples: list[np.ndarray] = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds = self.model(x)
                samples.append(preds.cpu().numpy())

        stacked = np.stack(samples, axis=0)  # (n_samples, n_data)
        return stacked.mean(axis=0), stacked.std(axis=0)
