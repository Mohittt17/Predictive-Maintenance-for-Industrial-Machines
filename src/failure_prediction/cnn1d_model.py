"""
1D-Convolutional Neural Network for failure prediction.

CNNs excel at detecting local patterns (short-duration fault impulses) in
time-series data with very low computational cost vs. LSTM.

Architecture
------------
Input (T×F)  →  Conv1D → BN → ReLU → Pool  (×N blocks)
              →  GlobalAvgPool  →  FC  →  Output

Advantages over LSTM for industrial signals:
  - Faster training (parallel convolutions vs sequential LSTM)
  - Better at detecting sharp fault impulses in vibration signals
  - More interpretable (visualise which timesteps activate each filter)

Both classification and regression modes supported (matching LSTMPredictor API).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


def _make_cnn_model(
    input_size: int,
    seq_len: int,
    n_filters: int,
    kernel_size: int,
    n_blocks: int,
    dropout: float,
    task: str,
) -> "nn.Module":
    class CNNModel(nn.Module):
        def __init__(self):
            super().__init__()
            blocks = []
            in_ch = 1   # treat features as channels? No: treat time as channels
            # We treat the sequence as (batch, features, time)
            # i.e. input is (B, F, T)
            in_ch = input_size
            out_ch = n_filters
            for _ in range(n_blocks):
                blocks += [
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(out_ch),
                    nn.ReLU(),
                    nn.MaxPool1d(2, stride=2, ceil_mode=True),
                ]
                in_ch = out_ch
            self.conv = nn.Sequential(*blocks)
            self.gap   = nn.AdaptiveAvgPool1d(1)
            self.drop  = nn.Dropout(dropout)
            self.fc    = nn.Linear(out_ch, 1)
            self.task  = task

        def forward(self, x):
            # x: (B, T, F)  →  permute to (B, F, T)
            x = x.permute(0, 2, 1)
            x = self.conv(x)
            x = self.gap(x).squeeze(-1)
            x = self.drop(x)
            logit = self.fc(x).squeeze(-1)
            if self.task == "classification":
                return torch.sigmoid(logit)
            return logit

    return CNNModel()


class CNN1DPredictor:
    """
    1D-CNN failure predictor (classification or regression).

    Args:
        n_filters:   Number of convolutional filters per block.
        kernel_size: Convolution kernel size.
        n_blocks:    Number of Conv→BN→ReLU→Pool blocks.
        seq_len:     Temporal window length (number of consecutive feature-vectors).
        dropout:     Dropout rate before final FC.
        epochs:      Training epochs.
        lr:          Adam learning rate.
        batch_size:  Mini-batch size.
        task:        "classification" or "regression".
        device:      "cpu", "cuda", or None (auto).
    """

    MODEL_NAME = "CNN-1D"

    def __init__(
        self,
        n_filters: int = 64,
        kernel_size: int = 3,
        n_blocks: int = 3,
        seq_len: int = 10,
        dropout: float = 0.3,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 128,
        task: str = "classification",
        device: Optional[str] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")
        self.n_filters   = n_filters
        self.kernel_size = kernel_size
        self.n_blocks    = n_blocks
        self.seq_len     = seq_len
        self.dropout     = dropout
        self.epochs      = epochs
        self.lr          = lr
        self.batch_size  = batch_size
        self.task        = task
        self._device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: Optional[nn.Module] = None
        self._input_mean: Optional[np.ndarray] = None
        self._input_std:  Optional[np.ndarray] = None
        self._fitted: bool = False

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._input_mean) / (self._input_std + 1e-8)

    def _make_sequences(self, X, y):
        seqs, targets = [], []
        n = len(X)
        for i in range(n - self.seq_len + 1):
            seqs.append(X[i : i + self.seq_len])
            targets.append(y[i + self.seq_len - 1])
        X_seq = torch.tensor(np.stack(seqs), dtype=torch.float32).to(self._device)
        y_seq = torch.tensor(targets, dtype=torch.float32).to(self._device)
        return X_seq, y_seq

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CNN1DPredictor":
        self._input_mean = X_train.mean(axis=0)
        self._input_std  = X_train.std(axis=0)
        X_norm = self._normalise(X_train)
        X_seq, y_seq = self._make_sequences(X_norm, y_train)

        self._model = _make_cnn_model(
            input_size=X_train.shape[1],
            seq_len=self.seq_len,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
            n_blocks=self.n_blocks,
            dropout=self.dropout,
            task=self.task,
        ).to(self._device)

        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.BCELoss() if self.task == "classification" else nn.HuberLoss(delta=10.0)

        loader = DataLoader(TensorDataset(X_seq, y_seq), batch_size=self.batch_size, shuffle=True)
        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_b, y_b in loader:
                optimiser.zero_grad()
                out  = self._model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug(f"CNN1D epoch {epoch+1}/{self.epochs} loss={epoch_loss/len(loader):.4f}")

        self._fitted = True
        logger.info(f"CNN1DPredictor ({self.task}) fitted: {X_train.shape}")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        X_norm = self._normalise(X)
        n = len(X_norm)
        seqs = [X_norm[i : i + self.seq_len] for i in range(n - self.seq_len + 1)]
        if not seqs:
            return np.array([])
        X_seq = torch.tensor(np.stack(seqs), dtype=torch.float32).to(self._device)
        self._model.eval()
        with torch.no_grad():
            out = self._model(X_seq).cpu().numpy()
        pad = np.full(self.seq_len - 1, out[0])
        return np.concatenate([pad, out])

    def predict(self, X: np.ndarray) -> np.ndarray:
        raw = self.predict_proba(X)
        if self.task == "classification":
            return (raw >= 0.5).astype(int)
        return raw

    def get_params(self) -> dict:
        return {
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "n_blocks": self.n_blocks,
            "seq_len": self.seq_len,
            "task": self.task,
        }
