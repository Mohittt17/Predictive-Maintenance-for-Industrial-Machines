"""
PCA-based Machine Health Index.

Principle
---------
During healthy operation, the principal components of multimodal sensor
features capture the dominant variance structure.  As a machine degrades,
new modes of variance emerge that are not well-explained by the healthy-state
PCA subspace.  The reconstruction error in the PCA subspace therefore serves
as a sensitive, unsupervised health indicator.

Health Index construction
--------------------------
1. Fit PCA on healthy-state windows (health_label == 0 or early time).
2. For each new window: project → reconstruct → compute SPE (Squared
   Prediction Error = reconstruction error).
3. Map SPE monotonically to [0, 100]:
     HI = 100 × (1 - SPE / SPE_max)
   where SPE_max is the 99th-percentile SPE on the training set.

References
----------
Qiu, H. et al. (2003). Robust performance degradation assessment methods for
enhanced rolling element bearing prognostics. Advanced Engineering Informatics.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


class PCAHealthIndex:
    """
    Unsupervised PCA-based Health Index.

    Args:
        n_components: Number of principal components to retain.
        spe_percentile: Percentile of training SPE used as normalisation ceiling.
    """

    def __init__(self, n_components: int = 3, spe_percentile: float = 99.0) -> None:
        self.n_components = n_components
        self.spe_percentile = spe_percentile
        self._scaler = StandardScaler()
        self._pca    = PCA(n_components=n_components)
        self._spe_max: float = 1.0
        self._fitted: bool = False

    # ── Fit ───────────────────────────────────────────────────────────────────

    def fit(self, X_healthy: np.ndarray) -> "PCAHealthIndex":
        """
        Fit the PCA model on healthy-state feature matrix.

        Args:
            X_healthy: 2-D array (n_windows, n_features) from healthy machine periods.
                       Should be NaN-free.

        Returns:
            self
        """
        if X_healthy.ndim != 2 or X_healthy.shape[0] < self.n_components:
            raise ValueError(
                f"X_healthy must be 2-D with at least {self.n_components} rows; "
                f"got shape {X_healthy.shape}"
            )

        # Standardise so all features contribute equally
        X_scaled = self._scaler.fit_transform(X_healthy)

        # Clamp components to available features
        n_comp = min(self.n_components, X_scaled.shape[1])
        if n_comp != self.n_components:
            logger.warning(
                f"Requested {self.n_components} PCA components but only "
                f"{X_scaled.shape[1]} features available. Using {n_comp}."
            )
            self._pca = PCA(n_components=n_comp)

        self._pca.fit(X_scaled)
        spe = self._compute_spe(X_scaled)
        self._spe_max = float(np.percentile(spe, self.spe_percentile))
        if self._spe_max == 0:
            self._spe_max = 1.0   # avoid division by zero on constant signals
        self._fitted = True

        logger.info(
            f"PCAHealthIndex fitted: {n_comp} components, "
            f"explained variance = {self._pca.explained_variance_ratio_.sum():.3f}, "
            f"SPE_max (p{self.spe_percentile:.0f}) = {self._spe_max:.4f}"
        )
        return self

    # ── Predict ───────────────────────────────────────────────────────────────

    def _compute_spe(self, X_scaled: np.ndarray) -> np.ndarray:
        """Squared Prediction Error in the PCA residual subspace."""
        # Project to PC space then back
        scores     = self._pca.transform(X_scaled)
        X_reconstructed = self._pca.inverse_transform(scores)
        residuals  = X_scaled - X_reconstructed
        return np.sum(residuals ** 2, axis=1)

    def health_index(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Health Index scores for new windows.

        Args:
            X: 2-D feature matrix (n_windows, n_features).  NaN-free.

        Returns:
            1-D array of Health Index values in [0, 100].
            100 = healthy (low SPE), 0 = failed (SPE ≥ SPE_max).
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .health_index().")
        X_scaled = self._scaler.transform(X)
        spe = self._compute_spe(X_scaled)
        hi = 100.0 * (1.0 - np.clip(spe / self._spe_max, 0.0, 1.0))
        return hi.clip(0.0, 100.0)

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return raw SPE scores (higher = more anomalous)."""
        if not self._fitted:
            raise RuntimeError("Call .fit() before .anomaly_score().")
        X_scaled = self._scaler.transform(X)
        return self._compute_spe(X_scaled)

    # ── Persistence ───────────────────────────────────────────────────────────

    def get_params(self) -> dict:
        """Return serialisable parameters (for MLflow logging)."""
        return {
            "n_components": self.n_components,
            "spe_percentile": self.spe_percentile,
            "spe_max": self._spe_max,
            "explained_variance_ratio": self._pca.explained_variance_ratio_.tolist()
            if self._fitted else [],
        }
