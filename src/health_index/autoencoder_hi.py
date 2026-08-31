"""
Autoencoder-based Machine Health Index and Anomaly Detector.

Principle
---------
A shallow autoencoder is trained exclusively on healthy-state sensor windows.
It learns to compress and reconstruct the normal operating pattern.  When
the machine degrades, the input distribution shifts — the autoencoder struggles
to reconstruct abnormal patterns, so reconstruction error rises.

Architecture
------------
Input → [hidden_dims] → Bottleneck → [hidden_dims reversed] → Reconstruction

The reconstruction error (MSE between input and output) forms the Health Index:
  HI = 100 × (1 - error / error_max)

This is equivalent to an anomaly score: larger error → lower health.

Note on PyTorch availability
-----------------------------
PyTorch is listed in requirements.txt but may not be installed for early
milestones.  This module degrades gracefully with an ImportError message
rather than crashing the whole pipeline.
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


# ─── Model definition ─────────────────────────────────────────────────────────

def _build_autoencoder(
    input_dim: int,
    hidden_dims: list[int],
) -> "nn.Module":
    """Build a symmetric encoder–decoder network."""
    if not _TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for AutoencoderHealthIndex.")

    encoder_layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        encoder_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
        prev_dim = h

    decoder_layers = []
    for h in reversed(hidden_dims[:-1]):
        decoder_layers += [nn.Linear(prev_dim, h), nn.ReLU()]
        prev_dim = h
    decoder_layers.append(nn.Linear(prev_dim, input_dim))

    return nn.Sequential(*encoder_layers, *decoder_layers)


# ─── Main class ───────────────────────────────────────────────────────────────

class AutoencoderHealthIndex:
    """
    Autoencoder-based unsupervised Health Index.

    Args:
        hidden_dims:    Encoder hidden layer sizes (decoder is symmetric).
                        E.g. [64, 32, 16] → bottleneck at 16 neurons.
        epochs:         Training epochs.
        lr:             Learning rate.
        batch_size:     Mini-batch size.
        error_percentile: Percentile of training reconstruction error used as
                          ceiling for HI normalisation.
        device:         "cpu", "cuda", or None (auto-detect).
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        error_percentile: float = 99.0,
        device: Optional[str] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for AutoencoderHealthIndex.\n"
                "Install with: pip install torch"
            )
        self.hidden_dims = hidden_dims or [64, 32, 16]
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.error_percentile = error_percentile

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._model: Optional[nn.Module] = None
        self._error_max: float = 1.0
        self._input_mean: Optional[np.ndarray] = None
        self._input_std: Optional[np.ndarray] = None
        self._fitted: bool = False

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalise(self, X: np.ndarray) -> np.ndarray:
        return (X - self._input_mean) / (self._input_std + 1e-8)

    def _to_tensor(self, X: np.ndarray) -> "torch.Tensor":
        return torch.tensor(X, dtype=torch.float32).to(self._device)

    def _reconstruction_error(self, X_norm: np.ndarray) -> np.ndarray:
        self._model.eval()
        with torch.no_grad():
            t = self._to_tensor(X_norm)
            recon = self._model(t).cpu().numpy()
        return np.mean((X_norm - recon) ** 2, axis=1)

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X_healthy: np.ndarray) -> "AutoencoderHealthIndex":
        """
        Train the autoencoder on healthy-state windows.

        Args:
            X_healthy: 2-D array (n_windows, n_features). NaN-free.
        """
        if X_healthy.ndim != 2 or X_healthy.shape[0] == 0:
            raise ValueError(f"X_healthy must be non-empty 2-D array; got {X_healthy.shape}")

        input_dim = X_healthy.shape[1]

        # Normalise inputs (z-score, fit on training data)
        self._input_mean = X_healthy.mean(axis=0)
        self._input_std  = X_healthy.std(axis=0)
        X_norm = self._normalise(X_healthy)

        # Build model
        self._model = _build_autoencoder(input_dim, self.hidden_dims).to(self._device)
        optimiser = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        dataset = TensorDataset(self._to_tensor(X_norm))
        loader  = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for (batch,) in loader:
                optimiser.zero_grad()
                recon = self._model(batch)
                loss  = criterion(recon, batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Autoencoder epoch {epoch+1}/{self.epochs} loss={epoch_loss/len(loader):.6f}")

        # Calibrate error ceiling
        errors = self._reconstruction_error(X_norm)
        self._error_max = float(np.percentile(errors, self.error_percentile))
        if self._error_max == 0:
            self._error_max = 1e-6

        self._fitted = True
        logger.info(
            f"AutoencoderHealthIndex fitted: input_dim={input_dim}, "
            f"hidden={self.hidden_dims}, device={self._device}, "
            f"error_max(p{self.error_percentile:.0f})={self._error_max:.6f}"
        )
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Return per-window MSE reconstruction error (higher = more anomalous)."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        X_norm = self._normalise(X)
        return self._reconstruction_error(X_norm)

    def health_index(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Health Index in [0, 100].
        100 = healthy (low reconstruction error), 0 = failure.
        """
        errors = self.reconstruction_error(X)
        hi = 100.0 * (1.0 - np.clip(errors / self._error_max, 0.0, 1.0))
        return hi.clip(0.0, 100.0)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Alias for reconstruction_error — higher = more anomalous."""
        return self.reconstruction_error(X)

    def get_params(self) -> dict:
        return {
            "hidden_dims": self.hidden_dims,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "error_percentile": self.error_percentile,
            "error_max": self._error_max,
        }
