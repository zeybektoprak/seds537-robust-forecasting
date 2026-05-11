"""Vanilla RNN model implemented in PyTorch."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class _RNNModule(nn.Module):
    """Internal PyTorch module: single-layer vanilla RNN with linear head."""

    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape ``(batch, seq_len, input_size)``.

        Returns:
            Tensor of shape ``(batch,)`` with one prediction per sample.
        """
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


class RNNModel:
    """Vanilla RNN forecaster with a single recurrent layer (64 hidden units).

    Args:
        input_size: Number of input features per time step.
        hidden_size: Number of hidden units in the RNN layer.
        lr: Learning rate for the Adam optimiser.
        device: ``"cuda"`` or ``"cpu"``.  Auto-detected if ``None``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        lr: float = 1e-3,
        device: str | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _RNNModule(input_size, hidden_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def fit(self, dataloader: DataLoader, epochs: int = 10) -> "RNNModel":
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
        """Generate predictions for input *x*.

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
