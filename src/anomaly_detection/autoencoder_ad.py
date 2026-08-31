"""
Autoencoder-based anomaly detector.

Shares the same reconstruction-error principle as AutoencoderHealthIndex
but provides the AnomalyResult interface expected by the pipeline.

The autoencoder is trained on healthy windows only.  At inference time,
reconstruction MSE for each window is compared against a calibrated threshold.

AnomalyScore = ||x - x̂||²  (per-window MSE)

Windows where AnomalyScore > threshold are flagged as anomalies.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from src.ingestion.schema import AnomalyResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AutoencoderAnomalyDetector:
    """
    Autoencoder-based anomaly detector.

    Wraps AutoencoderHealthIndex and adds:
      - Calibrated anomaly threshold (percentile of training reconstruction error)
      - predict() → bool array
      - to_results() → list[AnomalyResult]

    Args:
        hidden_dims:       Encoder hidden layer sizes.
        epochs:            Training epochs.
        lr:                Adam learning rate.
        batch_size:        Mini-batch size.
        score_percentile:  Training score percentile used as threshold.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        score_percentile: float = 95.0,
    ) -> None:
        try:
            from src.health_index.autoencoder_hi import AutoencoderHealthIndex
            self._ae = AutoencoderHealthIndex(
                hidden_dims=hidden_dims or [64, 32, 16],
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
            )
            self._torch_available = True
        except ImportError:
            self._torch_available = False
            self._ae = None

        self.score_percentile = score_percentile
        self._threshold: float = 0.0
        self._fitted: bool = False

    def fit(self, X: np.ndarray) -> "AutoencoderAnomalyDetector":
        if not self._torch_available:
            raise ImportError("PyTorch required for AutoencoderAnomalyDetector.")
        self._ae.fit(X)
        scores = self._ae.anomaly_score(X)
        self._threshold = float(np.percentile(scores, self.score_percentile))
        self._fitted = True
        logger.info(
            f"AutoencoderAnomalyDetector fitted. "
            f"Threshold (p{self.score_percentile:.0f}) = {self._threshold:.6f}"
        )
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return self._ae.anomaly_score(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.anomaly_score(X) > self._threshold).astype(int)

    def to_results(
        self,
        X: np.ndarray,
        machine_id: str,
        timestamps: list[datetime],
    ) -> list[AnomalyResult]:
        """Convert predictions to a list of AnomalyResult objects."""
        scores   = self.anomaly_score(X)
        is_anom  = self.predict(X).astype(bool)
        recon_errors = scores   # MSE is the reconstruction error

        results = []
        for i, ts in enumerate(timestamps):
            results.append(AnomalyResult(
                machine_id=machine_id,
                window_end=ts,
                anomaly_score=float(scores[i]),
                is_anomaly=bool(is_anom[i]),
                detector="Autoencoder",
                reconstruction_error=float(recon_errors[i]),
            ))
        return results

    def get_params(self) -> dict:
        return {
            "score_percentile": self.score_percentile,
            "threshold": self._threshold,
            **(self._ae.get_params() if self._ae else {}),
        }
