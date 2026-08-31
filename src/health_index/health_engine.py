"""
Unified Health Engine — combines multiple health indicators into a single
machine health score on a [0, 100] scale.

Design
------
The Health Engine maintains up to three sub-indicators:
  1. PCA Health Index (lightweight, always-on)
  2. Autoencoder Health Index (deep, optional — requires PyTorch)
  3. Isolation Forest anomaly score (inverted to health scale)

The final Health Index is a configurable weighted average of whichever
sub-indicators are fitted.

Score semantics
---------------
  100        Perfectly healthy
   80–100    Good / normal operation
   60–80     Early degradation (watch closely)
   40–60     Moderate degradation (plan maintenance)
   20–40     Severe degradation (schedule soon)
    0–20     Critical — imminent failure
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.health_index.pca_hi import PCAHealthIndex
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Health label thresholds (HI → label)
_HI_LABEL_MAP = [
    (80.0, 0),   # Healthy
    (60.0, 1),   # Minor
    (40.0, 2),   # Moderate
    (20.0, 3),   # Severe
    (0.0,  4),   # Critical
]


def hi_to_health_label(hi: float) -> int:
    """Map a Health Index (0–100) to a coarse health label (0–4)."""
    for threshold, label in _HI_LABEL_MAP:
        if hi >= threshold:
            return label
    return 4


@dataclass
class HealthEngineConfig:
    """Weights for combining sub-indicator Health Indices."""
    pca_weight:         float = 0.5
    autoencoder_weight: float = 0.3
    isolation_weight:   float = 0.2
    min_hi:             float = 0.0
    max_hi:             float = 100.0


class HealthEngine:
    """
    Unified machine Health Index combiner.

    Usage
    -----
    engine = HealthEngine()
    engine.fit_pca(X_healthy)
    hi = engine.compute(X_new)   # returns float array [0, 100]
    """

    def __init__(self, config: HealthEngineConfig | None = None) -> None:
        self.config = config or HealthEngineConfig()
        self._pca_hi:   Optional[PCAHealthIndex] = None
        self._ae_hi:    Optional[object] = None   # AutoencoderHealthIndex
        self._iso_max:  float = 1.0
        self._fitted_pca: bool = False
        self._fitted_ae:  bool = False
        self._fitted_iso: bool = False

    # ── Fit sub-indicators ────────────────────────────────────────────────────

    def fit_pca(self, X_healthy: np.ndarray, n_components: int = 3) -> "HealthEngine":
        """Fit the PCA-based health indicator on healthy-state windows."""
        self._pca_hi = PCAHealthIndex(n_components=n_components)
        self._pca_hi.fit(X_healthy)
        self._fitted_pca = True
        logger.info("HealthEngine: PCA indicator fitted.")
        return self

    def fit_autoencoder(
        self,
        X_healthy: np.ndarray,
        hidden_dims: list[int] | None = None,
        epochs: int = 50,
    ) -> "HealthEngine":
        """Fit the Autoencoder-based health indicator (requires PyTorch)."""
        try:
            from src.health_index.autoencoder_hi import AutoencoderHealthIndex
            self._ae_hi = AutoencoderHealthIndex(
                hidden_dims=hidden_dims or [64, 32, 16],
                epochs=epochs,
            )
            self._ae_hi.fit(X_healthy)
            self._fitted_ae = True
            logger.info("HealthEngine: Autoencoder indicator fitted.")
        except ImportError as e:
            logger.warning(f"Autoencoder indicator skipped: {e}")
        return self

    def fit_isolation_forest(
        self,
        X_healthy: np.ndarray,
        contamination: float = 0.05,
    ) -> "HealthEngine":
        """Fit an Isolation Forest on healthy-state data for anomaly scoring."""
        from sklearn.ensemble import IsolationForest
        iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
        iso.fit(X_healthy)
        # Store the model; calibrate score range on training data
        scores = -iso.score_samples(X_healthy)   # higher = more anomalous
        self._iso_model = iso
        self._iso_max = float(np.percentile(scores, 99))
        if self._iso_max == 0:
            self._iso_max = 1e-6
        self._fitted_iso = True
        logger.info("HealthEngine: IsolationForest indicator fitted.")
        return self

    def fit_all(
        self,
        X_healthy: np.ndarray,
        fit_autoencoder: bool = False,
        **kwargs,
    ) -> "HealthEngine":
        """Convenience: fit PCA + IsolationForest (+ Autoencoder if requested)."""
        self.fit_pca(X_healthy)
        self.fit_isolation_forest(X_healthy)
        if fit_autoencoder:
            self.fit_autoencoder(X_healthy, **kwargs)
        return self

    # ── Compute ───────────────────────────────────────────────────────────────

    def compute(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the combined Health Index for new windows.

        Args:
            X: 2-D feature matrix (n_windows, n_features). NaN-free.

        Returns:
            1-D array of Health Index values in [0, 100].
        """
        if not (self._fitted_pca or self._fitted_ae or self._fitted_iso):
            raise RuntimeError("HealthEngine has no fitted sub-indicators. Call fit_*() first.")

        components: list[tuple[np.ndarray, float]] = []
        total_weight = 0.0

        if self._fitted_pca:
            hi_pca = self._pca_hi.health_index(X)
            components.append((hi_pca, self.config.pca_weight))
            total_weight += self.config.pca_weight

        if self._fitted_ae and self._ae_hi is not None:
            hi_ae = self._ae_hi.health_index(X)
            components.append((hi_ae, self.config.autoencoder_weight))
            total_weight += self.config.autoencoder_weight

        if self._fitted_iso:
            scores = -self._iso_model.score_samples(X)
            hi_iso = 100.0 * (1.0 - np.clip(scores / self._iso_max, 0.0, 1.0))
            components.append((hi_iso, self.config.isolation_weight))
            total_weight += self.config.isolation_weight

        if total_weight == 0:
            return np.full(X.shape[0], 50.0)

        combined = sum(hi * (w / total_weight) for hi, w in components)
        return np.clip(combined, self.config.min_hi, self.config.max_hi)

    def health_labels(self, X: np.ndarray) -> np.ndarray:
        """Return coarse health labels (0–4) from the combined HI."""
        hi_scores = self.compute(X)
        return np.array([hi_to_health_label(float(h)) for h in hi_scores])

    def is_fitted(self) -> bool:
        return self._fitted_pca or self._fitted_ae or self._fitted_iso

    def get_params(self) -> dict:
        params: dict = {"config": vars(self.config)}
        if self._fitted_pca:
            params["pca"] = self._pca_hi.get_params()
        if self._fitted_ae and self._ae_hi is not None:
            params["autoencoder"] = self._ae_hi.get_params()
        return params
