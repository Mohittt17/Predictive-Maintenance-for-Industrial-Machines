"""
Pydantic API Schemas for Predictive Maintenance API.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class SensorTelemetryInput(BaseModel):
    machine_id: str = Field(..., description="Unique machine identifier", json_schema_extra={"example": "M-07"})
    timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc), description="Observation timestamp")
    vibration_x: float = Field(..., description="X-axis vibration RMS (m/s²)", json_schema_extra={"example": 1.45})
    vibration_y: float = Field(..., description="Y-axis vibration RMS (m/s²)", json_schema_extra={"example": 1.12})
    vibration_z: float = Field(..., description="Z-axis vibration RMS (m/s²)", json_schema_extra={"example": 0.98})
    temperature: float = Field(..., description="Bearing/Motor temperature (°C)", json_schema_extra={"example": 78.5})
    current: float = Field(..., description="Motor current draw (A)", json_schema_extra={"example": 14.2})
    pressure: float = Field(..., description="System operating pressure (bar)", json_schema_extra={"example": 3.8})
    acoustic: float = Field(..., description="Acoustic emission (dB)", json_schema_extra={"example": 58.4})
    rpm: float = Field(..., description="Rotational speed (RPM)", json_schema_extra={"example": 1750.0})
    load: float = Field(..., description="Operating load percentage (%)", json_schema_extra={"example": 85.0})


class SinglePredictionRequest(BaseModel):
    telemetry: SensorTelemetryInput


class BatchPredictionRequest(BaseModel):
    readings: list[SensorTelemetryInput] = Field(..., min_length=1, description="List of telemetry readings")


class FeatureContribution(BaseModel):
    feature_name: str
    contribution_score: float


class PredictionOutput(BaseModel):
    machine_id: str
    evaluated_at: datetime
    health_score: float = Field(..., ge=0.0, le=100.0, description="Machine Health Score 0-100")
    health_status: str = Field(..., description="Healthy | Warning | At Risk | Critical")
    failure_probability_72h: float = Field(..., ge=0.0, le=1.0, description="Failure risk within 72 hours")
    predicted_failure: bool = Field(..., description="Binary failure flag based on threshold")
    rul_median_hours: float = Field(..., description="Median RUL estimate in hours")
    rul_p10_hours: float = Field(..., description="10th percentile RUL (conservative)")
    rul_p90_hours: float = Field(..., description="90th percentile RUL (optimistic)")
    degradation_mode: str = Field(..., description="Inferred physical root cause")
    shap_contributions: list[FeatureContribution] = Field(..., description="Top risk contributing features")
    recommended_action: str = Field(..., description="Prescriptive maintenance recommendation")
    optimal_intervention_window_hours: float = Field(..., description="Cost-optimal maintenance window")
    expected_unplanned_cost: float = Field(..., description="Estimated cost if left unmaintained")
    expected_planned_cost: float = Field(..., description="Planned preventive maintenance cost")


class BatchPredictionOutput(BaseModel):
    total_machines: int
    critical_count: int
    at_risk_count: int
    warning_count: int
    healthy_count: int
    predictions: list[PredictionOutput]


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "Predictive Maintenance API"
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ModelInfoResponse(BaseModel):
    active_model: str = "XGBoost Classifier"
    version: str = "1.0.0"
    num_features: int = 15
    supported_models: list[str] = [
        "Logistic Regression",
        "Random Forest",
        "XGBoost",
        "Gradient Boosting"
    ]
    supported_explainers: list[str] = ["SHAP TreeExplainer", "Marginal Contribution Fallback"]


class MetricDetail(BaseModel):
    precision: float
    recall: float
    f1: float
    pr_auc: float
    roc_auc: float
    false_alarm_rate: float
    missed_failure_rate: float


class ModelMetricsResponse(BaseModel):
    evaluated_at: str
    dataset_summary: str
    models: dict[str, MetricDetail]
