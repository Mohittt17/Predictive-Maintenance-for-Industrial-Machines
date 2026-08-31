"""
FastAPI Enterprise Application Entrypoint.
Predictive Maintenance & RUL Optimization Platform API.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.config import settings
from backend.app.routers import health, predict, metrics
from backend.app.services.predict_service import prediction_service
from src.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up ML models and services upon application startup."""
    logger.info("Initializing Predictive Maintenance API services...")
    prediction_service.initialize()
    yield
    logger.info("Shutting down Predictive Maintenance API services...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-grade REST API for Machine Health Monitoring, Anomaly Detection, "
        "72-Hour Failure Prediction, Quantile RUL Estimation, SHAP Root-Cause Attribution, "
        "and Cost-Aware Maintenance Optimization."
    ),
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"message": "An internal server error occurred.", "details": str(exc)},
    )

# Include Routers
app.include_router(health.router, prefix=settings.API_V1_STR)
app.include_router(predict.router, prefix=settings.API_V1_STR)
app.include_router(metrics.router, prefix=settings.API_V1_STR)

# Also expose /health, /predict, /predict/batch, /model/info, /metrics at root level for quick access
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(metrics.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Predictive Maintenance AI Platform API",
        "documentation": "/docs",
        "health": "/health",
        "version": settings.VERSION,
    }
