"""
Isolation Forest anomaly detector.

A simple, effective baseline for unsupervised anomaly detection.
The Isolation Forest isolates anomalies by randomly partitioning the feature
space — anomalous observations require fewer splits to isolate (shorter path
length), yielding a high anomaly score.

This wrapper provides a consistent interface matching the other detectors
in this package, and adds threshold calibration on the training set.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import IsolationForest as _IsolationForest
from sklearn.preprocessing import StandardScaler

from src.ingestion.schema import AnomalyResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector with calibrated threshold.

    Args:
        contamination:     Expected fraction of anomalies in training data.
        n_estimators:      Number of isolation trees.
        score_percentile:  Percentile of training scores used as anomaly threshold.
        random_state:      Random seed.
    """

    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 200,
        score_percentile: float = 95.0,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.score_percentile = score_percentile
        self.random_state  = random_state
        self._scaler = StandardScaler()
        self._model  = _IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self._threshold: float = 0.0
        self._fitted: bool = False

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Fit on (ideally healthy-state) training data."""
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)
        # Calibrate threshold: use percentile of training anomaly scores
        scores = self._anomaly_scores(X_scaled)
        self._threshold = float(np.percentile(scores, self.score_percentile))
        self._fitted = True
        logger.info(
            f"IsolationForestDetector fitted on {X.shape[0]} samples. "
            f"Threshold (p{self.score_percentile:.0f}) = {self._threshold:.4f}"
        )
        return self

    def _anomaly_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        """Return per-sample anomaly scores (higher = more anomalous)."""
        # IsolationForest.score_samples returns negative anomaly scores;
        # negate so that higher = more anomalous (consistent with other detectors)
        return -self._model.score_samples(X_scaled)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for new data. Higher = more anomalous."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return self._anomaly_scores(self._scaler.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary labels: 1 = anomaly, 0 = normal."""
        scores = self.anomaly_score(X)
        return (scores > self._threshold).astype(int)

    def get_params(self) -> dict:
        return {
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "score_percentile": self.score_percentile,
            "threshold": self._threshold,
        }
