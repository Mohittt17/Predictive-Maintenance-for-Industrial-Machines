"""
Health Check API Router.
"""
from __future__ import annotations

from fastapi import APIRouter
from backend.app.schemas.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness & Readiness Health Probe")
async def health_check():
    """Return 200 OK with API operational status."""
    return HealthResponse()
