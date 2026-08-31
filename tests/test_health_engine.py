"""
Unit tests for anomaly detection and health index (Milestone 3).

Tests use synthetic data (no dataset downloads required):
  - Healthy data: Gaussian noise around zero
  - Degraded data: Gaussian noise with larger variance + drift

Covers:
  - PCAHealthIndex: fit, HI range, HI ordering (healthy > degraded)
  - HealthEngine: fit_all, compute, health_labels, weighting
  - IsolationForestDetector: fit, anomaly_score, predict
  - OneClassSVMDetector: fit, anomaly_score, predict
  - hi_to_health_label: boundary correctness
"""
from __future__ import annotations

import numpy as np
import pytest

from src.health_index.pca_hi import PCAHealthIndex
from src.health_index.health_engine import HealthEngine, hi_to_health_label
from src.anomaly_detection.isolation_forest import IsolationForestDetector
from src.anomaly_detection.one_class_svm import OneClassSVMDetector


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def healthy_data():
    rng = np.random.default_rng(0)
    return rng.standard_normal((200, 15))

@pytest.fixture
def degraded_data():
    rng = np.random.default_rng(1)
    return rng.standard_normal((50, 15)) * 4.0 + 3.0   # larger variance + drift


# ─── hi_to_health_label ───────────────────────────────────────────────────────

class TestHIToHealthLabel:

    def test_fully_healthy(self):
        assert hi_to_health_label(100.0) == 0

    def test_boundary_80(self):
        assert hi_to_health_label(80.0) == 0
        assert hi_to_health_label(79.9) == 1

    def test_boundary_60(self):
        assert hi_to_health_label(60.0) == 1
        assert hi_to_health_label(59.9) == 2

    def test_boundary_40(self):
        assert hi_to_health_label(40.0) == 2
        assert hi_to_health_label(39.9) == 3

    def test_boundary_20(self):
        assert hi_to_health_label(20.0) == 3
        assert hi_to_health_label(19.9) == 4

    def test_zero(self):
        assert hi_to_health_label(0.0) == 4


# ─── PCA Health Index ─────────────────────────────────────────────────────────

class TestPCAHealthIndex:

    def test_fit_and_range(self, healthy_data):
        phi = PCAHealthIndex(n_components=3)
        phi.fit(healthy_data)
        hi = phi.health_index(healthy_data)
        assert hi.shape == (200,)
        assert np.all(hi >= 0) and np.all(hi <= 100)

    def test_healthy_hi_higher_than_degraded(self, healthy_data, degraded_data):
        phi = PCAHealthIndex(n_components=3)
        phi.fit(healthy_data)
        hi_healthy  = phi.health_index(healthy_data).mean()
        hi_degraded = phi.health_index(degraded_data).mean()
        assert hi_healthy > hi_degraded, (
            f"Expected healthy HI > degraded HI, got {hi_healthy:.1f} vs {hi_degraded:.1f}"
        )

    def test_anomaly_score_order(self, healthy_data, degraded_data):
        phi = PCAHealthIndex(n_components=3)
        phi.fit(healthy_data)
        score_h = phi.anomaly_score(healthy_data).mean()
        score_d = phi.anomaly_score(degraded_data).mean()
        assert score_d > score_h

    def test_unfitted_raises(self):
        phi = PCAHealthIndex()
        with pytest.raises(RuntimeError):
            phi.health_index(np.random.randn(10, 5))

    def test_too_few_rows_raises(self):
        phi = PCAHealthIndex(n_components=5)
        with pytest.raises(ValueError):
            phi.fit(np.random.randn(3, 10))

    def test_get_params(self, healthy_data):
        phi = PCAHealthIndex(n_components=2)
        phi.fit(healthy_data)
        params = phi.get_params()
        assert "n_components" in params
        assert "spe_max" in params
        assert len(params["explained_variance_ratio"]) > 0

    def test_component_clamping_no_error(self):
        """Should not crash when n_components > n_features."""
        X = np.random.randn(50, 4)
        phi = PCAHealthIndex(n_components=10)   # more than 4 features
        phi.fit(X)
        hi = phi.health_index(X)
        assert hi.shape == (50,)


# ─── Health Engine ────────────────────────────────────────────────────────────

class TestHealthEngine:

    def test_fit_pca_only(self, healthy_data, degraded_data):
        engine = HealthEngine()
        engine.fit_pca(healthy_data)
        assert engine.is_fitted()
        hi = engine.compute(healthy_data[:5])
        assert hi.shape == (5,)
        assert np.all(hi >= 0) and np.all(hi <= 100)

    def test_fit_all(self, healthy_data, degraded_data):
        engine = HealthEngine()
        engine.fit_all(healthy_data)
        hi_h = engine.compute(healthy_data).mean()
        hi_d = engine.compute(degraded_data).mean()
        assert hi_h > hi_d

    def test_health_labels_range(self, healthy_data):
        engine = HealthEngine()
        engine.fit_pca(healthy_data)
        labels = engine.health_labels(healthy_data)
        assert np.all((labels >= 0) & (labels <= 4))

    def test_unfitted_raises(self):
        engine = HealthEngine()
        with pytest.raises(RuntimeError):
            engine.compute(np.random.randn(5, 10))

    def test_get_params(self, healthy_data):
        engine = HealthEngine()
        engine.fit_pca(healthy_data)
        params = engine.get_params()
        assert "pca" in params


# ─── Isolation Forest ─────────────────────────────────────────────────────────

class TestIsolationForest:

    def test_fit_predict_shape(self, healthy_data):
        det = IsolationForestDetector(contamination=0.05, n_estimators=50)
        det.fit(healthy_data)
        scores = det.anomaly_score(healthy_data)
        preds  = det.predict(healthy_data)
        assert scores.shape == (200,)
        assert preds.shape == (200,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_degraded_higher_scores(self, healthy_data, degraded_data):
        det = IsolationForestDetector(contamination=0.05, n_estimators=100)
        det.fit(healthy_data)
        score_h = det.anomaly_score(healthy_data).mean()
        score_d = det.anomaly_score(degraded_data).mean()
        assert score_d > score_h

    def test_contamination_rate(self, healthy_data):
        # With contamination=0.05, ~5% of healthy data should be flagged
        det = IsolationForestDetector(contamination=0.05, score_percentile=95.0)
        det.fit(healthy_data)
        preds = det.predict(healthy_data)
        flagged_rate = preds.mean()
        assert flagged_rate <= 0.1   # should be roughly 5%

    def test_unfitted_raises(self):
        det = IsolationForestDetector()
        with pytest.raises(RuntimeError):
            det.anomaly_score(np.random.randn(5, 10))

    def test_get_params(self, healthy_data):
        det = IsolationForestDetector()
        det.fit(healthy_data)
        params = det.get_params()
        assert "threshold" in params
        assert params["threshold"] > 0


# ─── One-Class SVM ────────────────────────────────────────────────────────────

class TestOneClassSVM:

    def test_fit_predict_shape(self, healthy_data):
        det = OneClassSVMDetector(nu=0.05, gamma="scale")
        det.fit(healthy_data)
        scores = det.anomaly_score(healthy_data)
        preds  = det.predict(healthy_data)
        assert scores.shape == (200,)
        assert preds.shape == (200,)

    def test_degraded_higher_scores(self, healthy_data, degraded_data):
        det = OneClassSVMDetector(nu=0.05)
        det.fit(healthy_data)
        score_h = det.anomaly_score(healthy_data).mean()
        score_d = det.anomaly_score(degraded_data).mean()
        assert score_d > score_h

    def test_unfitted_raises(self):
        det = OneClassSVMDetector()
        with pytest.raises(RuntimeError):
            det.anomaly_score(np.random.randn(5, 10))

    def test_get_params(self, healthy_data):
        det = OneClassSVMDetector()
        det.fit(healthy_data)
        params = det.get_params()
        assert "threshold" in params
        assert "kernel" in params
