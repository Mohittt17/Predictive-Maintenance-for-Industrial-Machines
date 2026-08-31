"""
Logistic Regression failure predictor — the simplest possible baseline.


"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.failure_prediction.evaluator import evaluate_classifier, EvaluationReport
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LogisticRegressionPredictor:
    """
    Logistic Regression failure predictor.

    Args:
        C:             Regularisation strength (smaller = stronger regularisation).
        max_iter:      Maximum solver iterations.
        class_weight:  "balanced" to handle class imbalance automatically.
        random_state:  Random seed.
    """

    MODEL_NAME = "Logistic Regression"

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        class_weight: str = "balanced",
        random_state: int = 42,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.random_state = random_state
        self._scaler = StandardScaler()
        self._model  = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            random_state=random_state,
            solver="lbfgs",
        )
        self._feature_names: list[str] = []
        self._fitted: bool = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[list[str]] = None,
    ) -> "LogisticRegressionPredictor":
        X_scaled = self._scaler.fit_transform(X_train)
        self._model.fit(X_scaled, y_train)
        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]
        self._fitted = True
        logger.info(
            f"LogisticRegressionPredictor fitted: "
            f"{X_train.shape[0]} samples, {X_train.shape[1]} features"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return self._model.predict(self._scaler.transform(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return self._model.predict_proba(self._scaler.transform(X))[:, 1]

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
        """Return top-N features by absolute coefficient magnitude."""
        if not self._fitted:
            return []
        coefs = np.abs(self._model.coef_[0])
        idx   = np.argsort(coefs)[::-1][:top_n]
        return [(self._feature_names[i], float(coefs[i])) for i in idx]

    def get_params(self) -> dict:
        return {"C": self.C, "max_iter": self.max_iter, "class_weight": self.class_weight}
