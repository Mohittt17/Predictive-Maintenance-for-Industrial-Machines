"""
Prediction API Router — Single and Batch Machine Telemetry Failure Risk Inference.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from backend.app.schemas.schemas import (
    SinglePredictionRequest,
    BatchPredictionRequest,
    PredictionOutput,
    BatchPredictionOutput,
    ModelInfoResponse,
)
from backend.app.services.predict_service import prediction_service
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Predictions"])


@router.post(
    "/predict",
    response_model=PredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Predict Single Machine Failure Risk, Health Index & SHAP Attribution"
)
async def predict_single_machine(request: SinglePredictionRequest):
    """
    Evaluates raw machine sensor telemetry and returns:
    - **Health Score (0-100)** & Risk Level
    - **Failure Probability (72h Horizon)**
    - **Probabilistic RUL (p10, p50, p90)**
    - **SHAP Feature Importance & Physical Degradation Diagnosis**
    - **Cost-Optimized Maintenance Recommendation**
    """
    try:
        return prediction_service.predict_single(request.telemetry)
    except Exception as e:
        logger.error(f"Prediction error for machine {request.telemetry.machine_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failure: {str(e)}"
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionOutput,
    status_code=status.HTTP_200_OK,
    summary="Batch Machine Monitoring Telemetry Evaluation"
)
async def predict_batch_machines(request: BatchPredictionRequest):
    """Evaluates multiple machine telemetry streams in a single batch request."""
    try:
        return prediction_service.predict_batch(request.readings)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failure: {str(e)}"
        )


@router.get(
    "/model/info",
    response_model=ModelInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Active ML Model Information and Architecture Metadata"
)
async def get_model_info():
    """Return metadata about active prediction model, feature schema, and explainer."""
    return ModelInfoResponse()
