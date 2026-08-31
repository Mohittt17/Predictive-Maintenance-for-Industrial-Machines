
from __future__ import annotations

import numpy as np
import pytest

from src.failure_prediction.evaluator import (
    evaluate_classifier,
    compare_models,
    EvaluationReport,
)
from src.failure_prediction.logistic_regression import LogisticRegressionPredictor
from src.failure_prediction.random_forest import RandomForestPredictor


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def imbalanced_dataset():
    """950 normal, 50 failure — 19:1 imbalance. Shuffled so splits have both classes."""
    rng = np.random.default_rng(42)
    X_neg = rng.standard_normal((950, 15))
    X_pos = rng.standard_normal((50, 15)) * 2 + 3.0   # separable cluster
    X = np.vstack([X_neg, X_pos])
    y = np.array([0] * 950 + [1] * 50)
    # Shuffle so both classes appear in any contiguous slice
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


@pytest.fixture
def perfect_preds():
    """Perfect prediction scenario for metric sanity checks."""
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.2, 0.9, 0.8])
    return y_true, y_pred, y_prob


@pytest.fixture
def worst_preds():
    """All-zero prediction (naive baseline) — typical class imbalance trap."""
    y_true = np.array([0] * 950 + [1] * 50)
    y_pred = np.zeros(1000, dtype=int)
    return y_true, y_pred


# ─── Evaluator Tests ──────────────────────────────────────────────────────────

class TestEvaluator:

    def test_perfect_prediction_metrics(self, perfect_preds):
        y_true, y_pred, y_prob = perfect_preds
        report = evaluate_classifier("Test", y_true, y_pred, y_prob)
        assert report.precision == pytest.approx(1.0)
        assert report.recall    == pytest.approx(1.0)
        assert report.f1        == pytest.approx(1.0)
        assert report.tp == 2
        assert report.fp == 0
        assert report.fn == 0

    def test_naive_baseline_zero_recall(self, worst_preds):
        """Predicting all-normal → 99.9% accuracy but F1=0."""
        y_true, y_pred = worst_preds
        report = evaluate_classifier("Naive", y_true, y_pred)
        assert report.recall == pytest.approx(0.0)
        assert report.f1     == pytest.approx(0.0)
        assert report.precision == pytest.approx(0.0)

    def test_confusion_matrix_elements(self, perfect_preds):
        y_true, y_pred, _ = perfect_preds
        report = evaluate_classifier("T", y_true, y_pred)
        assert report.tn + report.fp + report.fn + report.tp == len(y_true)

    def test_class_imbalance_ratio(self, worst_preds):
        y_true, y_pred = worst_preds
        report = evaluate_classifier("T", y_true, y_pred)
        assert report.class_imbalance_ratio == pytest.approx(950 / 50)

    def test_no_proba_gives_nan_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        report = evaluate_classifier("T", y_true, y_pred, y_prob=None)
        assert np.isnan(report.roc_auc)
        assert np.isnan(report.pr_auc)

    def test_detection_lead_time(self):
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 0, 1, 1])  # detected 2 steps early
        ts     = np.array([0, 1, 2, 3, 4, 5], dtype=float)
        report = evaluate_classifier("T", y_true, y_pred, timestamps=ts)
        assert report.detection_lead_time == pytest.approx(2.0)

    def test_summary_string(self, perfect_preds):
        y_true, y_pred, y_prob = perfect_preds
        report = evaluate_classifier("MyModel", y_true, y_pred, y_prob)
        summary = report.summary()
        assert "MyModel" in summary
        assert "F1" in summary
        assert "PR-AUC" in summary

    def test_compare_models_markdown(self, perfect_preds):
        y_true, y_pred, y_prob = perfect_preds
        r1 = evaluate_classifier("M1", y_true, y_pred, y_prob)
        r2 = evaluate_classifier("M2", y_true, y_pred, y_prob)
        table = compare_models([r1, r2])
        assert "M1" in table
        assert "M2" in table
        assert "|" in table


# ─── Logistic Regression Tests ────────────────────────────────────────────────

class TestLogisticRegression:

    def test_fit_predict_shape(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        lr = LogisticRegressionPredictor()
        lr.fit(X[:800], y[:800])
        preds = lr.predict(X[800:])
        proba = lr.predict_proba(X[800:])
        assert preds.shape == (200,)
        assert proba.shape == (200,)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_balanced_weights_improve_recall(self, imbalanced_dataset):
        """Balanced class weights should produce non-zero recall."""
        X, y = imbalanced_dataset
        lr = LogisticRegressionPredictor(class_weight="balanced")
        lr.fit(X[:800], y[:800])
        report = lr.evaluate(X[800:], y[800:])
        assert report.recall > 0.0, "Balanced LR should have non-zero recall"

    def test_evaluate_returns_report(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        lr = LogisticRegressionPredictor()
        lr.fit(X[:800], y[:800])
        report = lr.evaluate(X[800:], y[800:])
        assert isinstance(report, EvaluationReport)
        assert report.model_name == "Logistic Regression"

    def test_feature_importance_length(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        names = [f"sensor_{i}" for i in range(15)]
        lr = LogisticRegressionPredictor()
        lr.fit(X[:800], y[:800], feature_names=names)
        fi = lr.feature_importance(top_n=5)
        assert len(fi) == 5
        assert all(isinstance(name, str) and isinstance(val, float) for name, val in fi)

    def test_unfitted_predict_raises(self):
        lr = LogisticRegressionPredictor()
        with pytest.raises(RuntimeError):
            lr.predict(np.random.randn(10, 5))

    def test_get_params(self):
        lr = LogisticRegressionPredictor(C=0.1)
        params = lr.get_params()
        assert params["C"] == 0.1


# ─── Random Forest Tests ──────────────────────────────────────────────────────

class TestRandomForest:

    def test_fit_predict_shape(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        rf = RandomForestPredictor(n_estimators=20)
        rf.fit(X[:800], y[:800])
        preds = rf.predict(X[800:])
        proba = rf.predict_proba(X[800:])
        assert preds.shape == (200,)
        assert proba.shape == (200,)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_non_zero_recall(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        rf = RandomForestPredictor(n_estimators=50, class_weight="balanced_subsample")
        rf.fit(X[:800], y[:800])
        report = rf.evaluate(X[800:], y[800:])
        assert report.recall > 0.0

    def test_feature_importance(self, imbalanced_dataset):
        X, y = imbalanced_dataset
        names = [f"f{i}" for i in range(15)]
        rf = RandomForestPredictor(n_estimators=20)
        rf.fit(X[:800], y[:800], feature_names=names)
        fi = rf.feature_importance(top_n=3)
        assert len(fi) == 3
        # Importances should be in descending order
        vals = [v for _, v in fi]
        assert vals == sorted(vals, reverse=True)

    def test_unfitted_predict_raises(self):
        rf = RandomForestPredictor()
        with pytest.raises(RuntimeError):
            rf.predict(np.random.randn(5, 10))


# ─── XGBoost Tests ────────────────────────────────────────────────────────────

class TestXGBoost:

    def test_xgboost_available(self):
        try:
            import xgboost
        except ImportError:
            pytest.skip("XGBoost not installed")

    def test_fit_predict(self, imbalanced_dataset):
        try:
            import xgboost
        except ImportError:
            pytest.skip("XGBoost not installed")
        from src.failure_prediction.xgboost_model import XGBoostPredictor
        X, y = imbalanced_dataset
        xgb = XGBoostPredictor(n_estimators=50)
        xgb.fit(X[:800], y[:800])
        preds = xgb.predict(X[800:])
        proba = xgb.predict_proba(X[800:])
        assert preds.shape == (200,)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_auto_scale_pos_weight(self, imbalanced_dataset):
        """scale_pos_weight=None should auto-compute from training labels."""
        try:
            import xgboost
        except ImportError:
            pytest.skip("XGBoost not installed")
        from src.failure_prediction.xgboost_model import XGBoostPredictor
        X, y = imbalanced_dataset
        xgb = XGBoostPredictor(n_estimators=20, scale_pos_weight=None)
        xgb.fit(X[:800], y[:800])   # should not raise
        assert xgb._fitted

    def test_non_zero_recall(self, imbalanced_dataset):
        try:
            import xgboost
        except ImportError:
            pytest.skip("XGBoost not installed")
        from src.failure_prediction.xgboost_model import XGBoostPredictor
        X, y = imbalanced_dataset
        xgb = XGBoostPredictor(n_estimators=50)
        xgb.fit(X[:800], y[:800])
        report = xgb.evaluate(X[800:], y[800:])
        assert report.recall > 0.0
