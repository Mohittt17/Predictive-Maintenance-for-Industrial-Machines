"""
API Unit Tests for Predictive Maintenance FastAPI Backend.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "Predictive Maintenance API"


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "active_model" in data
    assert len(data["supported_models"]) >= 4


def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "XGBoost" in data["models"] or len(data["models"]) > 0


def test_predict_single_valid():
    payload = {
        "telemetry": {
            "machine_id": "M-07",
            "vibration_x": 3.8,
            "vibration_y": 3.1,
            "vibration_z": 2.9,
            "temperature": 88.5,
            "current": 18.4,
            "pressure": 3.1,
            "acoustic": 68.2,
            "rpm": 1720.0,
            "load": 92.0
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["machine_id"] == "M-07"
    assert 0.0 <= data["health_score"] <= 100.0
    assert data["health_status"] in ["Healthy", "Warning", "At Risk", "Critical"]
    assert 0.0 <= data["failure_probability_72h"] <= 1.0
    assert "shap_contributions" in data
    assert len(data["shap_contributions"]) > 0
    assert "recommended_action" in data


def test_predict_single_invalid_missing_field():
    payload = {
        "telemetry": {
            "machine_id": "M-07",
            "vibration_x": 3.8,
            # missing required temperature, current, etc.
        }
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity validation error


def test_predict_batch_valid():
    payload = {
        "readings": [
            {
                "machine_id": "M-01",
                "vibration_x": 0.4, "vibration_y": 0.3, "vibration_z": 0.3,
                "temperature": 62.0, "current": 11.5, "pressure": 4.2,
                "acoustic": 41.0, "rpm": 1780.0, "load": 80.0
            },
            {
                "machine_id": "M-07",
                "vibration_x": 4.2, "vibration_y": 3.8, "vibration_z": 3.5,
                "temperature": 92.0, "current": 19.5, "pressure": 2.8,
                "acoustic": 74.0, "rpm": 1700.0, "load": 95.0
            }
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_machines"] == 2
    assert len(data["predictions"]) == 2
