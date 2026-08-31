"""
Probabilistic RUL — prediction intervals and confidence-aware estimates.

Classical point RUL prediction gives a single number.  A maintenance engineer
needs to know: "Is that 56-hour prediction tight or uncertain?"

This module provides:
  1. Bootstrap confidence intervals on Random Forest / XGBoost predictions
  2. Quantile regression with Gradient Boosting (GBR) for native intervals
  3. RUL interval to natural-language description mapping

The output format matches the required project spec:
  "Estimated RUL: 56–78 hours (95% confidence)"
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RULInterval:
    """A probabilistic RUL prediction with confidence interval."""
    point_estimate:   float
    lower_bound:      float
    upper_bound:      float
    confidence_level: float     # e.g. 0.90 = 90% CI
    method:           str       # "bootstrap" or "quantile"

    def to_string(self) -> str:
        """Format as the canonical pipeline output."""
        return (
            f"Estimated RUL: {self.lower_bound:.0f}–{self.upper_bound:.0f} hours "
            f"({int(self.confidence_level*100)}% confidence)"
        )

    def width(self) -> float:
        """Width of the confidence interval — proxy for model uncertainty."""
        return self.upper_bound - self.lower_bound


class BootstrapRUL:
    """
    Bootstrap confidence intervals by training multiple sub-sampled RF models.

    Each bootstrap replicate fits on a random 80% subsample of training data
    and makes independent predictions.  The CI is the empirical percentile.

    Args:
        n_estimators:     Trees per RF model.
        n_bootstrap:      Number of bootstrap replicates.
        confidence_level: Width of the confidence interval (e.g. 0.90).
        random_state:     Seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        n_bootstrap: int = 30,
        confidence_level: float = 0.90,
        random_state: int = 42,
    ) -> None:
        self.n_estimators    = n_estimators
        self.n_bootstrap     = n_bootstrap
        self.confidence_level = confidence_level
        self.random_state    = random_state
        self._models: list[RandomForestRegressor] = []
        self._fitted: bool = False

    def fit(self, X: np.ndarray, y_rul: np.ndarray) -> "BootstrapRUL":
        rng = np.random.default_rng(self.random_state)
        n = len(X)
        self._models = []
        for i in range(self.n_bootstrap):
            idx = rng.choice(n, size=int(n * 0.8), replace=True)
            m = RandomForestRegressor(
                n_estimators=self.n_estimators,
                random_state=self.random_state + i,
                n_jobs=-1,
            )
            m.fit(X[idx], y_rul[idx])
            self._models.append(m)
        self._fitted = True
        logger.info(f"BootstrapRUL fitted: {self.n_bootstrap} replicates on {n} samples")
        return self

    def predict_interval(self, X: np.ndarray) -> list[RULInterval]:
        """Return one :class:`RULInterval` per row of X."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        alpha = (1 - self.confidence_level) / 2
        all_preds = np.stack([m.predict(X) for m in self._models], axis=0)   # (B, N)
        lower = np.percentile(all_preds, alpha * 100, axis=0)
        upper = np.percentile(all_preds, (1 - alpha) * 100, axis=0)
        point = all_preds.mean(axis=0)
        return [
            RULInterval(
                point_estimate=max(0.0, float(point[i])),
                lower_bound=max(0.0, float(lower[i])),
                upper_bound=max(0.0, float(upper[i])),
                confidence_level=self.confidence_level,
                method="bootstrap",
            )
            for i in range(len(X))
        ]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point estimate (mean across replicates)."""
        intervals = self.predict_interval(X)
        return np.array([r.point_estimate for r in intervals])


class QuantileRUL:
    """
    Quantile regression using Gradient Boosting Regressor.

    Trains three separate quantile models:
      - Lower bound (alpha/2 quantile)
      - Median (0.50 quantile)
      - Upper bound (1 - alpha/2 quantile)

    Args:
        n_estimators:     Boosting rounds.
        max_depth:        Tree depth.
        confidence_level: Width of the prediction interval.
        learning_rate:    Shrinkage rate.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 5,
        confidence_level: float = 0.90,
        learning_rate: float = 0.05,
    ) -> None:
        self.n_estimators    = n_estimators
        self.max_depth       = max_depth
        self.confidence_level = confidence_level
        self.learning_rate   = learning_rate
        self._scaler = StandardScaler()
        alpha = (1 - confidence_level) / 2
        self._models: dict[str, GradientBoostingRegressor] = {
            "lower":  GradientBoostingRegressor(
                loss="quantile", alpha=alpha,
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate,
            ),
            "median": GradientBoostingRegressor(
                loss="quantile", alpha=0.5,
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate,
            ),
            "upper":  GradientBoostingRegressor(
                loss="quantile", alpha=1 - alpha,
                n_estimators=n_estimators, max_depth=max_depth,
                learning_rate=learning_rate,
            ),
        }
        self._fitted = False

    def fit(self, X: np.ndarray, y_rul: np.ndarray) -> "QuantileRUL":
        X_s = self._scaler.fit_transform(X)
        for name, m in self._models.items():
            m.fit(X_s, y_rul)
            logger.debug(f"QuantileRUL '{name}' fitted")
        self._fitted = True
        logger.info(f"QuantileRUL fitted ({int(self.confidence_level*100)}% CI)")
        return self

    def predict_interval(self, X: np.ndarray) -> list[RULInterval]:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        X_s = self._scaler.transform(X)
        lower  = np.maximum(0, self._models["lower"].predict(X_s))
        median = np.maximum(0, self._models["median"].predict(X_s))
        upper  = np.maximum(0, self._models["upper"].predict(X_s))
        return [
            RULInterval(
                point_estimate=float(median[i]),
                lower_bound=float(lower[i]),
                upper_bound=float(upper[i]),
                confidence_level=self.confidence_level,
                method="quantile",
            )
            for i in range(len(X))
        ]

    def predict(self, X: np.ndarray) -> np.ndarray:
        intervals = self.predict_interval(X)
        return np.array([r.point_estimate for r in intervals])
