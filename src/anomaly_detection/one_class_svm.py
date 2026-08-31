"""
One-Class SVM anomaly detector.

One-Class SVM learns a tight boundary around the healthy operating region.
Points outside this boundary are flagged as anomalies.  It is more sensitive
to the decision boundary shape than Isolation Forest but is slower on large
datasets.

Useful as a comparison baseline in the research experiments (Exp 5).
"""
from __future__ import annotations

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class OneClassSVMDetector:
    """
    One-Class SVM anomaly detector.

    Args:
        kernel:        SVM kernel type ("rbf" recommended for vibration data).
        nu:            Upper bound on the fraction of training errors (0 < nu ≤ 1).
        gamma:         Kernel coefficient ("scale" = 1/(n_features * X.var())).
        score_percentile: Percentile of training scores used as threshold.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.05,
        gamma: str = "scale",
        score_percentile: float = 95.0,
    ) -> None:
        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.score_percentile = score_percentile
        self._scaler = StandardScaler()
        self._model  = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)
        self._threshold: float = 0.0
        self._fitted: bool = False

    def fit(self, X: np.ndarray) -> "OneClassSVMDetector":
        """Fit on healthy-state training data."""
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled)
        scores = self._anomaly_scores(X_scaled)
        self._threshold = float(np.percentile(scores, self.score_percentile))
        self._fitted = True
        logger.info(
            f"OneClassSVMDetector fitted on {X.shape[0]} samples (nu={self.nu}). "
            f"Threshold = {self._threshold:.4f}"
        )
        return self

    def _anomaly_scores(self, X_scaled: np.ndarray) -> np.ndarray:
        # score_samples: higher = closer to centre = normal
        # Negate so higher = more anomalous
        return -self._model.score_samples(X_scaled)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return self._anomaly_scores(self._scaler.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self.anomaly_score(X)
        return (scores > self._threshold).astype(int)

    def get_params(self) -> dict:
        return {
            "kernel": self.kernel,
            "nu": self.nu,
            "gamma": self.gamma,
            "score_percentile": self.score_percentile,
            "threshold": self._threshold,
        }
