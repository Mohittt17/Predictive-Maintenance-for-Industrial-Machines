"""
XGBoost failure predictor.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

from src.failure_prediction.evaluator import evaluate_classifier, EvaluationReport
from src.utils.logger import get_logger

logger = get_logger(__name__)


class XGBoostPredictor:
    """
    XGBoost failure predictor.

    Args:
        n_estimators:     Number of boosting rounds.
        max_depth:        Maximum tree depth.
        learning_rate:    Step size shrinkage.
        scale_pos_weight: Weight for positive class = n_neg / n_pos.
                          Pass None to auto-compute from training labels.
        subsample:        Fraction of samples for tree fitting.
        colsample_bytree: Fraction of features per tree.
        random_state:     Random seed.
    """

    MODEL_NAME = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        scale_pos_weight: Optional[float] = None,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        if not _XGB_AVAILABLE:
            raise ImportError(
                "XGBoost is required. Install with: pip install xgboost"
            )
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self._scale_pos_weight = scale_pos_weight   # None = auto
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state     = random_state
        self._model: Optional[xgb.XGBClassifier] = None
        self._feature_names: list[str] = []
        self._fitted: bool = False

    def _build_model(self, scale_pos_weight: float) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            scale_pos_weight=scale_pos_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            eval_metric="aucpr",    # PR-AUC is most appropriate for imbalanced problems
            use_label_encoder=False,
            verbosity=0,
            n_jobs=-1,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[list[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 50,
    ) -> "XGBoostPredictor":
        """
        Fit XGBoost with optional validation set for early stopping.

        Args:
            X_train, y_train:     Training data.
            feature_names:        Column names for SHAP explainability.
            X_val, y_val:         Optional validation set for early stopping.
            early_stopping_rounds: Stop if val metric doesn't improve.
        """
        # Auto-compute scale_pos_weight from label distribution
        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        spw = self._scale_pos_weight if self._scale_pos_weight is not None else (
            n_neg / max(n_pos, 1)
        )
        logger.info(f"XGBoost scale_pos_weight = {spw:.1f} ({n_neg} neg / {n_pos} pos)")

        self._model = self._build_model(scale_pos_weight=spw)
        self._feature_names = feature_names or [f"f{i}" for i in range(X_train.shape[1])]

        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
            # Early stopping via callbacks
            try:
                fit_kwargs["callbacks"] = [
                    xgb.callback.EarlyStopping(rounds=early_stopping_rounds, metric_name="aucpr")
                ]
            except AttributeError:
                pass   # older XGBoost versions

        self._model.fit(X_train, y_train, **fit_kwargs)
        self._fitted = True
        logger.info(
            f"XGBoostPredictor fitted: {X_train.shape[0]} samples, "
            f"{X_train.shape[1]} features"
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return self._model.predict_proba(X)[:, 1]

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

    def feature_importance(self, top_n: int = 10, importance_type: str = "gain") -> list[tuple[str, float]]:
        """Return top-N features by XGBoost feature importance."""
        if not self._fitted:
            return []
        scores = self._model.get_booster().get_score(importance_type=importance_type)
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [(name, float(val)) for name, val in sorted_scores]

    def get_booster(self):
        """Return the underlying XGBoost booster (needed for SHAP)."""
        return self._model.get_booster() if self._fitted else None

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }
