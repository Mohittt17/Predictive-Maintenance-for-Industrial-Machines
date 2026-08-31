"""
LSTM-based failure prediction and RUL estimation.

Architecture
------------
Input sequence (T timesteps × F features)
    │
    ▼  LSTM stack (configurable depth and hidden size)
    ▼  Dropout (regularisation)
    ▼  Fully Connected
    ▼  Output (binary failure probability OR continuous RUL)

The LSTM is the canonical choice for bearing/engine health sequences because:
  1. It maintains a hidden state across time — memory without explicit lag features
  2. The gate mechanism lets it learn WHEN degradation starts (not just that it did)
  3. It generalises across machines of different cycle lengths

Both tasks share the same backbone:
  - task="classification" → sigmoid output → binary cross-entropy
  - task="regression"     → linear output → MSE / Huber loss (for RUL)
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


# ─── LSTM Model Definition ────────────────────────────────────────────────────

def _make_lstm_model(
    input_size: int,
    hidden_size: int,
    num_layers: int,
    dropout: float,
    task: str,
) -> "nn.Module":
    """Build LSTM + FC head."""
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.dropout = nn.Dropout(dropout)
            self.fc = nn.Linear(hidden_size, 1)
            self.task = task

        def forward(self, x):
            # x: (batch, seq_len, features)
            out, _ = self.lstm(x)
            last = out[:, -1, :]   # take final timestep
            last = self.dropout(last)
            logit = self.fc(last).squeeze(-1)
            if self.task == "classification":
                return torch.sigmoid(logit)
            return logit   # raw output for regression

    return LSTMModel()


# ─── Main class ───────────────────────────────────────────────────────────────

class LSTMPredictor:
    """
    LSTM-based temporal predictor for both failure classification and RUL regression.

    Sequences are formed by grouping consecutive windows into fixed-length time steps.

    Args:
        hidden_size:   LSTM hidden dimension.
        num_layers:    Number of stacked LSTM layers.
        seq_len:       Sequence length (number of consecutive windows per sample).
        dropout:       Dropout rate.
        epochs:        Training epochs.
        lr:            Learning rate.
        batch_size:    Mini-batch size.
        task:          "classification" (binary) or "regression" (RUL).
        device:        "cpu", "cuda", or None (auto).
    """

    MODEL_NAME = "LSTM"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
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
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
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

    def _make_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        """Reshape flat (N, F) → (N-seq_len+1, seq_len, F) sequences."""
        n, f = X.shape
        if n < self.seq_len:
            raise ValueError(f"Need at least {self.seq_len} samples, got {n}")
        seqs, targets = [], []
        for i in range(n - self.seq_len + 1):
            seqs.append(X[i : i + self.seq_len])
            targets.append(y[i + self.seq_len - 1])
        X_seq = torch.tensor(np.stack(seqs), dtype=torch.float32).to(self._device)
        if self.task == "classification":
            y_seq = torch.tensor(targets, dtype=torch.float32).to(self._device)
        else:
            y_seq = torch.tensor(targets, dtype=torch.float32).to(self._device)
        return X_seq, y_seq

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "LSTMPredictor":
        """
        Train the LSTM on sequence data.

        Args:
            X_train: (N, F) feature matrix — windows in temporal order for ONE machine.
            y_train: (N,) binary labels or RUL values.
        """
        self._input_mean = X_train.mean(axis=0)
        self._input_std  = X_train.std(axis=0)
        X_norm = self._normalise(X_train)

        X_seq, y_seq = self._make_sequences(X_norm, y_train)

        input_size = X_train.shape[1]
        self._model = _make_lstm_model(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            task=self.task,
        ).to(self._device)

        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        if self.task == "classification":
            # Compute positive class weight for imbalanced data
            n_pos = float((y_train == 1).sum())
            n_neg = float((y_train == 0).sum())
            pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(self._device)
            criterion = nn.BCELoss()   # sigmoid applied in model
        else:
            criterion = nn.HuberLoss(delta=10.0)   # robust to RUL outliers

        dataset = TensorDataset(X_seq, y_seq)
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for X_b, y_b in loader:
                optimiser.zero_grad()
                out  = self._model(X_b)
                loss = criterion(out, y_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimiser.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug(f"LSTM epoch {epoch+1}/{self.epochs} loss={epoch_loss/len(loader):.4f}")

        self._fitted = True
        logger.info(f"LSTMPredictor ({self.task}) fitted: {X_train.shape}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions (class labels or RUL values)."""
        proba = self._predict_raw(X)
        if self.task == "classification":
            return (proba >= 0.5).astype(int)
        return proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw model output (probabilities for classification, RUL for regression)."""
        return self._predict_raw(X)

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        X_norm = self._normalise(X)
        n, f = X_norm.shape
        # Build sequences
        seqs = []
        for i in range(n - self.seq_len + 1):
            seqs.append(X_norm[i : i + self.seq_len])
        if not seqs:
            return np.array([])
        X_seq = torch.tensor(np.stack(seqs), dtype=torch.float32).to(self._device)
        self._model.eval()
        with torch.no_grad():
            out = self._model(X_seq).cpu().numpy()
        # Pad beginning with first prediction (no sequence yet)
        pad = np.full(self.seq_len - 1, out[0])
        return np.concatenate([pad, out])

    def get_params(self) -> dict:
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "seq_len": self.seq_len,
            "dropout": self.dropout,
            "task": self.task,
        }
