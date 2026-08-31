"""
FastAPI Backend Configuration.
"""
from __future__ import annotations

import os
from pathlib import Path
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]

class AppSettings(BaseModel):
    PROJECT_NAME: str = "Predictive Maintenance AI Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    CORS_ORIGINS: list[str] = ["*"]
    MODEL_DIR: Path = PROJECT_ROOT / "models"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"

settings = AppSettings()
