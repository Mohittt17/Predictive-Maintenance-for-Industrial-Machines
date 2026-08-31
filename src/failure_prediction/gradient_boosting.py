"""
Gradient Boosting failure predictor.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from src.failure_prediction.evaluator import evaluate_classifier, EvaluationReport
from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradientBoostingPredictor:
    """
    Gradient Boosting failure predictor.

    Args:
        n_estimators:  Number of boosting stages.
        learning_rate: Shrinks the contribution of each tree.
        max_depth:     Maximum depth of individual regression estimators.
        random_state:  Random seed.
    """

    MODEL_NAME = "Gradient Boosting"

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        random_state: int = 42,
    ) -> None:
        self.n_estimators  = n_estimators
        self.learning_rate = learning_rate
        self.max_depth     = max_depth
        self.random_state  = random_state
        self._model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )
        self._feature_names: list[str] = []
        self._fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "GradientBoostingPredictor":
        self._model.fit(X_train, y_train)
        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self._fitted = True
        logger.info(
            f"GradientBoostingPredictor fitted: "
            f"{X_train.shape[0]} samples, {X_train.shape[1]} features"
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
        if proba.shape[1] == 1:
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
        """Return top-N features by mean decrease in impurity."""
        if not self._fitted:
            return []
        importances = self._model.feature_importances_
        idx = np.argsort(importances)[::-1][:top_n]
        return [(self._feature_names[i], float(importances[i])) for i in idx]

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
        }
