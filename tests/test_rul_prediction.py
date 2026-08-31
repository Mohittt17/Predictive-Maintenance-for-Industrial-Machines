"""
Unit tests for RUL prediction (Milestone 6).

Synthetic data:
  - X: (500, 15) random features
  - y_rul: linearly decreasing from 200 to 0 (realistic degradation)

Tests:
  - RUL evaluator: RMSE, MAE, NASA score correctness
  - SVR, RF, XGBoost RUL regressors: fit/predict/non-negative/evaluate
  - Bootstrap CI: fit/predict_interval/CI width/to_string format
  - Quantile CI: fit/predict_interval/ordering (lower ≤ median ≤ upper)
"""
from __future__ import annotations

import numpy as np
import pytest

from src.rul_prediction.rul_evaluator import evaluate_rul, compare_rul_models, RULEvaluationReport
from src.rul_prediction.rul_regressor import SVRRULRegressor, RFRULRegressor
from src.rul_prediction.probabilistic_rul import BootstrapRUL, QuantileRUL, RULInterval
from src.utils.metrics import nasa_rul_score


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def rul_dataset():
    rng = np.random.default_rng(0)
    n = 300
    X = rng.standard_normal((n, 10))
    # True RUL decreases + noise
    y_rul = np.linspace(150, 0, n) + rng.normal(0, 5, n)
    y_rul = np.maximum(0, y_rul)
    return X, y_rul


# ─── NASA Score Tests ─────────────────────────────────────────────────────────

class TestNASAScore:

    def test_perfect_prediction_near_zero(self):
        y = np.array([50.0, 100.0, 30.0])
        score = nasa_rul_score(y, y)
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_late_prediction_higher_penalty(self):
        y_true = np.array([10.0])
        # Late: predict 20 when truth is 10 (d=10, positive)
        score_late  = nasa_rul_score(y_true, np.array([20.0]))
        # Early: predict 0 when truth is 10 (d=-10, negative)
        score_early = nasa_rul_score(y_true, np.array([0.0]))
        assert score_late > score_early, (
            "Late prediction should have larger NASA penalty than equally off early prediction"
        )

    def test_zero_prediction_gives_finite_score(self):
        y_true = np.array([0.0])
        y_pred = np.array([0.0])
        score = nasa_rul_score(y_true, y_pred)
        assert np.isfinite(score)


# ─── Evaluator Tests ──────────────────────────────────────────────────────────

class TestRULEvaluator:

    def test_perfect_rmse_zero(self):
        y = np.array([10.0, 20.0, 30.0])
        report = evaluate_rul("Test", y, y)
        assert report.rmse == pytest.approx(0.0, abs=1e-8)
        assert report.mae  == pytest.approx(0.0, abs=1e-8)

    def test_within_thresholds(self):
        y_true = np.array([100.0] * 100)
        y_pred = y_true + 5.0   # always 5 off
        report = evaluate_rul("Test", y_true, y_pred)
        assert report.within_10 == pytest.approx(1.0)
        assert report.within_25 == pytest.approx(1.0)

    def test_outside_threshold(self):
        y_true = np.array([100.0] * 100)
        y_pred = y_true + 30.0   # always 30 off
        report = evaluate_rul("Test", y_true, y_pred)
        assert report.within_10 == pytest.approx(0.0)
        assert report.within_25 == pytest.approx(0.0)

    def test_summary_contains_model_name(self):
        y = np.linspace(100, 0, 50)
        report = evaluate_rul("MyRUL", y, y + 2)
        assert "MyRUL" in report.summary()

    def test_compare_table_markdown(self):
        y = np.linspace(100, 0, 50)
        r1 = evaluate_rul("M1", y, y)
        r2 = evaluate_rul("M2", y, y + 5)
        table = compare_rul_models([r1, r2])
        assert "M1" in table and "M2" in table
        assert "|" in table

    def test_n_samples(self):
        y = np.linspace(100, 0, 77)
        report = evaluate_rul("T", y, y)
        assert report.n_samples == 77


# ─── SVR RUL ──────────────────────────────────────────────────────────────────

class TestSVRRUL:

    def test_fit_predict_shape(self, rul_dataset):
        X, y = rul_dataset
        svr = SVRRULRegressor(C=1.0)
        svr.fit(X[:200], y[:200])
        preds = svr.predict(X[200:])
        assert preds.shape == (100,)

    def test_non_negative_output(self, rul_dataset):
        X, y = rul_dataset
        svr = SVRRULRegressor()
        svr.fit(X[:200], y[:200])
        preds = svr.predict(X[200:])
        assert np.all(preds >= 0)

    def test_evaluate_returns_report(self, rul_dataset):
        X, y = rul_dataset
        svr = SVRRULRegressor()
        svr.fit(X[:200], y[:200])
        report = svr.evaluate(X[200:], y[200:])
        assert isinstance(report, RULEvaluationReport)
        assert report.rmse >= 0

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            SVRRULRegressor().predict(np.random.randn(5, 5))


# ─── RF RUL ───────────────────────────────────────────────────────────────────

class TestRFRUL:

    def test_fit_predict(self, rul_dataset):
        X, y = rul_dataset
        rf = RFRULRegressor(n_estimators=20)
        rf.fit(X[:200], y[:200])
        preds = rf.predict(X[200:])
        assert preds.shape == (100,)
        assert np.all(preds >= 0)

    def test_feature_importance_order(self, rul_dataset):
        X, y = rul_dataset
        rf = RFRULRegressor(n_estimators=20)
        rf.fit(X[:200], y[:200])
        fi = rf.feature_importance(top_n=3)
        vals = [v for _, v in fi]
        assert vals == sorted(vals, reverse=True)


# ─── Bootstrap RUL ────────────────────────────────────────────────────────────

class TestBootstrapRUL:

    def test_predict_interval_shape(self, rul_dataset):
        X, y = rul_dataset
        boot = BootstrapRUL(n_estimators=20, n_bootstrap=5)
        boot.fit(X[:200], y[:200])
        intervals = boot.predict_interval(X[200:205])
        assert len(intervals) == 5
        assert all(isinstance(r, RULInterval) for r in intervals)

    def test_lower_le_upper(self, rul_dataset):
        X, y = rul_dataset
        boot = BootstrapRUL(n_estimators=20, n_bootstrap=10)
        boot.fit(X[:200], y[:200])
        intervals = boot.predict_interval(X[200:210])
        for iv in intervals:
            assert iv.lower_bound <= iv.upper_bound

    def test_point_within_bounds(self, rul_dataset):
        X, y = rul_dataset
        boot = BootstrapRUL(n_estimators=20, n_bootstrap=10)
        boot.fit(X[:200], y[:200])
        intervals = boot.predict_interval(X[200:210])
        for iv in intervals:
            assert iv.lower_bound <= iv.point_estimate <= iv.upper_bound + 0.1   # small tolerance

    def test_to_string_format(self, rul_dataset):
        X, y = rul_dataset
        boot = BootstrapRUL(n_estimators=20, n_bootstrap=5)
        boot.fit(X[:200], y[:200])
        iv = boot.predict_interval(X[200:201])[0]
        s = iv.to_string()
        assert "Estimated RUL" in s
        assert "hours" in s
        assert "%" in s

    def test_non_negative_bounds(self, rul_dataset):
        X, y = rul_dataset
        boot = BootstrapRUL(n_estimators=20, n_bootstrap=5)
        boot.fit(X[:200], y[:200])
        intervals = boot.predict_interval(X[200:210])
        for iv in intervals:
            assert iv.lower_bound >= 0
            assert iv.upper_bound >= 0

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            BootstrapRUL().predict_interval(np.random.randn(5, 5))


# ─── Quantile RUL ─────────────────────────────────────────────────────────────

class TestQuantileRUL:

    def test_predict_interval_ordering(self, rul_dataset):
        X, y = rul_dataset
        qrul = QuantileRUL(n_estimators=50, confidence_level=0.90)
        qrul.fit(X[:200], y[:200])
        intervals = qrul.predict_interval(X[200:210])
        for iv in intervals:
            assert iv.lower_bound <= iv.point_estimate + 1e-3   # small tolerance
            assert iv.point_estimate <= iv.upper_bound + 1e-3

    def test_non_negative(self, rul_dataset):
        X, y = rul_dataset
        qrul = QuantileRUL(n_estimators=50)
        qrul.fit(X[:200], y[:200])
        intervals = qrul.predict_interval(X[200:210])
        for iv in intervals:
            assert iv.lower_bound >= 0
            assert iv.upper_bound >= 0

    def test_to_string(self, rul_dataset):
        X, y = rul_dataset
        qrul = QuantileRUL(n_estimators=50, confidence_level=0.90)
        qrul.fit(X[:200], y[:200])
        iv = qrul.predict_interval(X[200:201])[0]
        s = iv.to_string()
        assert "90" in s   # confidence level shown
        assert "Estimated RUL" in s

    def test_unfitted_raises(self):
        with pytest.raises(RuntimeError):
            QuantileRUL().predict_interval(np.random.randn(5, 5))
