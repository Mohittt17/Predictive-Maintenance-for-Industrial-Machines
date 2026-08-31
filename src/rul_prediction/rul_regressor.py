"""
RUL (Remaining Useful Life) regression — classical baseline models.

Task framing
------------
Given a feature vector at time t, predict how many hours/cycles remain
before failure.  This is a regression problem with a heavy right-skewed
label distribution (most training samples are far from failure).

The NASA scoring function penalises under-prediction (predicting too much
life remaining) more heavily than over-prediction — a safety-first design.
See `src/utils/metrics.py` for the implementation.

Models
------
Three baselines are provided:
  1. SVR (Support Vector Regression) — good for small–medium datasets
  2. Random Forest Regressor — robust, handles non-linearities
  3. XGBoost Regressor — typically best classical baseline for this task

All three are wrapped with the same API for easy swapping.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from src.rul_prediction.rul_evaluator import evaluate_rul, RULEvaluationReport
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Base class ───────────────────────────────────────────────────────────────

class BaseRULRegressor:
    """Shared interface for all RUL regressors."""

    MODEL_NAME = "BaseRUL"

    def fit(self, X: np.ndarray, y_rul: np.ndarray) -> "BaseRULRegressor":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        timestamps: Optional[np.ndarray] = None,
    ) -> "RULEvaluationReport":
        y_pred = self.predict(X_test)
        return evaluate_rul(self.MODEL_NAME, y_test, y_pred)

    def get_params(self) -> dict:
        return {}


# ─── SVR ──────────────────────────────────────────────────────────────────────

class SVRRULRegressor(BaseRULRegressor):
    """
    Support Vector Regression for RUL.

    Args:
        C:       Regularisation strength.
        kernel:  SVR kernel ("rbf" recommended).
        epsilon: Epsilon-tube width (insensitive zone).
    """

    MODEL_NAME = "SVR-RUL"

    def __init__(
        self,
        C: float = 10.0,
        kernel: str = "rbf",
        epsilon: float = 5.0,
        gamma: str = "scale",
    ) -> None:
        self.C = C
        self.kernel = kernel
        self.epsilon = epsilon
        self.gamma = gamma
        self._scaler = StandardScaler()
        self._model  = SVR(C=C, kernel=kernel, epsilon=epsilon, gamma=gamma)
        self._fitted = False

    def fit(self, X: np.ndarray, y_rul: np.ndarray) -> "SVRRULRegressor":
        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y_rul)
        self._fitted = True
        logger.info(f"SVRRULRegressor fitted: {X.shape}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return np.maximum(0.0, self._model.predict(self._scaler.transform(X)))

    def get_params(self) -> dict:
        return {"C": self.C, "kernel": self.kernel, "epsilon": self.epsilon}


# ─── Random Forest Regressor ──────────────────────────────────────────────────

class RFRULRegressor(BaseRULRegressor):
    """Random Forest RUL regressor."""

    MODEL_NAME = "RF-RUL"

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth    = max_depth
        self.random_state = random_state
        self._model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self._fitted = False

    def fit(self, X: np.ndarray, y_rul: np.ndarray) -> "RFRULRegressor":
        self._model.fit(X, y_rul)
        self._fitted = True
        logger.info(f"RFRULRegressor fitted: {X.shape}, {self.n_estimators} trees")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return np.maximum(0.0, self._model.predict(X))

    def feature_importance(self, top_n: int = 10, feature_names: Optional[list] = None) -> list:
        if not self._fitted:
            return []
        imp = self._model.feature_importances_
        idx = np.argsort(imp)[::-1][:top_n]
        names = feature_names or [f"f{i}" for i in range(len(imp))]
        return [(names[i], float(imp[i])) for i in idx]

    def get_params(self) -> dict:
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth}


# ─── XGBoost Regressor ────────────────────────────────────────────────────────

class XGBoostRULRegressor(BaseRULRegressor):
    """XGBoost RUL regressor."""

    MODEL_NAME = "XGBoost-RUL"

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ) -> None:
        try:
            import xgboost as xgb
            self._xgb = xgb
        except ImportError:
            raise ImportError("XGBoost required: pip install xgboost")

        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state     = random_state
        self._model = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y_rul: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBoostRULRegressor":
        self._model = self._xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            objective="reg:squarederror",
            verbosity=0,
            n_jobs=-1,
        )
        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]
            fit_kwargs["verbose"] = False
        self._model.fit(X, y_rul, **fit_kwargs)
        self._fitted = True
        logger.info(f"XGBoostRULRegressor fitted: {X.shape}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        return np.maximum(0.0, self._model.predict(X))

    def get_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
        }
