"""
Random Forest failure predictor.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from src.failure_prediction.evaluator import evaluate_classifier, EvaluationReport
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestPredictor:
    """
    Random Forest failure predictor.

    Args:
        n_estimators: Number of trees.
        max_depth:    Maximum tree depth (None = fully grown).
        class_weight: "balanced" or "balanced_subsample" for imbalance.
        random_state: Random seed.
    """

    MODEL_NAME = "Random Forest"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        class_weight: str = "balanced_subsample",
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.class_weight = class_weight
        self.random_state = random_state
        # RF doesn't require scaling, but we keep StandardScaler for consistency
        self._model  = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._feature_names: list[str] = []
        self._fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "RandomForestPredictor":
        self._model.fit(X_train, y_train)
        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self._fitted = True
        logger.info(
            f"RandomForestPredictor fitted: "
            f"{X_train.shape[0]} samples, {X_train.shape[1]} features, "
            f"{self.n_estimators} trees"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        proba = self._model.predict_proba(X)
        # Guard: if only one class seen during training, return zeros or ones
        if proba.shape[1] == 1:
            # Model only saw class 0 — return zero failure probability
            return np.zeros(len(X), dtype=np.float64)
        return proba[:, 1]

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> EvaluationReport:
        y_pred = self.predict(X_test)
        y_prob = self.predict_proba(X_test)
        return evaluate_classifier(
            self.MODEL_NAME, y_test, y_pred, y_prob, timestamps
        )

    def feature_importance(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Return top-N features by mean decrease in impurity (MDI)."""
        if not self._fitted:
            return []
        importances = self._model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        return [(self._feature_names[i], float(importances[i])) for i in idx]

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "class_weight": self.class_weight,
        }
