"""
Unit tests for Explainability (Milestone 8).
"""
import numpy as np
import pytest
from datetime import datetime

from src.explainability.shap_explainer import (
    SHAPExplainer,
    infer_degradation_mode,
)
from src.ingestion.schema import ExplainabilityResult


class MockModel:
    def __init__(self):
        self.feature_importances_ = np.array([0.5, 0.3, 0.1, 0.1])

    def predict_proba(self, X):
        return np.array([[0.2, 0.8]])


def test_infer_degradation_mode():
    features_vib = [("vib_rms", 0.45), ("temp_max", 0.15)]
    mode = infer_degradation_mode(features_vib)
    assert "Bearing" in mode or "rotational" in mode.lower() or "thermal" in mode.lower()

    features_curr = [("curr_rms", 0.60)]
    mode_curr = infer_degradation_mode(features_curr)
    assert "motor" in mode_curr.lower() or "electrical" in mode_curr.lower() or "winding" in mode_curr.lower()


def test_shap_explainer_fallback():
    model = MockModel()
    feature_names = ["vib_rms", "acou_rms", "temp_mean", "curr_mean"]
    explainer = SHAPExplainer(model=model, feature_names=feature_names)
    
    x = np.array([1.2, 0.8, 0.2, 0.1])
    res = explainer.explain_instance(x, machine_id="M-TEST", top_k=2)

    assert isinstance(res, ExplainabilityResult)
    assert res.machine_id == "M-TEST"
    assert len(res.top_features) == 2
    assert isinstance(res.degradation_mode, str)
