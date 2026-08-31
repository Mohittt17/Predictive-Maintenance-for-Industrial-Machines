"""
Metrics API Router — Returns Empirical Model Evaluation Benchmark Results.
"""
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter
from backend.app.schemas.schemas import ModelMetricsResponse, MetricDetail
from backend.app.services.predict_service import prediction_service

router = APIRouter(tags=["Metrics"])


@router.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    summary="Get Empirical Model Benchmark Evaluation Metrics"
)
async def get_metrics():
    """
    Returns actual measured metrics (Precision, Recall, F1, PR-AUC, ROC-AUC,
    False Alarm Rate, Missed Failure Rate) evaluated across candidate models.
    """
    raw_metrics = prediction_service.get_benchmark_metrics()
    models_dict = {}
    
    metrics_source = raw_metrics if "models" in raw_metrics else raw_metrics
    for model_name, m in metrics_source.get("models", {}).items():
        models_dict[model_name] = MetricDetail(
            precision=m["precision"],
            recall=m["recall"],
            f1=m["f1"],
            pr_auc=m["pr_auc"],
            roc_auc=m["roc_auc"],
            false_alarm_rate=m["false_alarm_rate"],
            missed_failure_rate=m["missed_failure_rate"]
        )

    return ModelMetricsResponse(
        evaluated_at=raw_metrics.get("evaluated_at", datetime.now(timezone.utc).isoformat()),
        dataset_summary=raw_metrics.get(
            "dataset_summary", "Empirical Evaluation on Industrial Telemetry"
        ),
        models=models_dict
    )
