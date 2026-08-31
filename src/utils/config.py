"""
Central configuration for the Predictive Maintenance system.
All tuneable parameters live here; never hard-code paths or hyperparameters
in individual modules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ─── Root paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
SYNTHETIC_DATA_DIR = DATA_ROOT / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"
MLFLOW_TRACKING_URI = f"sqlite:///{PROJECT_ROOT / 'mlflow' / 'mlflow.db'}"

# ─── Dataset sub-directories ─────────────────────────────────────────────────
CMAPSS_DIR = RAW_DATA_DIR / "cmapss"
IMS_DIR = RAW_DATA_DIR / "ims"
XJTU_DIR = RAW_DATA_DIR / "xjtu"

# ─── Sensor column names (canonical) ─────────────────────────────────────────
SENSOR_COLS = [
    "vibration_x", "vibration_y", "vibration_z",
    "temperature", "current", "pressure", "acoustic",
]
OPERATIONAL_COLS = ["rpm", "load"]
ALL_SIGNAL_COLS = SENSOR_COLS + OPERATIONAL_COLS


@dataclass
class WindowConfig:
    """Sliding-window parameters for feature extraction."""
    size: int = 30           # window length (samples / time-steps)
    stride: int = 1          # hop size
    min_samples: int = 10    # minimum samples to form a window


@dataclass
class SignalProcessingConfig:
    """FFT and wavelet settings."""
    sampling_rate_hz: float = 25_600.0   # typical accelerometer rate
    fft_n: int = 1024                    # FFT points
    wavelet: str = "db4"                 # Daubechies-4
    wavelet_levels: int = 5              # decomposition depth
    bearing_freq_bands: list[tuple[float, float]] = field(default_factory=lambda: [
        (100.0, 200.0),   # BPFO / BPFI band (example; tune per machine)
        (200.0, 500.0),   # cage + roller band
    ])


@dataclass
class HealthIndexConfig:
    """Health index construction."""
    n_pca_components: int = 3
    autoencoder_hidden_dims: list[int] = field(default_factory=lambda: [64, 32, 16])
    autoencoder_epochs: int = 50
    autoencoder_lr: float = 1e-3
    autoencoder_batch_size: int = 256
    hi_scale_min: float = 0.0
    hi_scale_max: float = 100.0


@dataclass
class FailurePredictionConfig:
    """Failure / anomaly detection settings."""
    failure_label: int = 1
    early_rul_threshold: float = 30.0   # cycles/hours — label as 'near failure'
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5
    # Class-weight strategy for imbalanced data
    class_weight: str = "balanced"
    # XGBoost
    xgb_n_estimators: int = 500
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.05
    # Random Forest
    rf_n_estimators: int = 300


@dataclass
class RULConfig:
    """RUL prediction settings."""
    max_rul: float = 125.0           # clip RUL at this value (C-MAPSS convention)
    quantiles: list[float] = field(default_factory=lambda: [0.1, 0.5, 0.9])
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_epochs: int = 100
    lstm_lr: float = 1e-3
    lstm_batch_size: int = 64


@dataclass
class CostConfig:
    """Maintenance cost parameters (configurable per machine class)."""
    c_pm: float = 20_000.0       # preventive maintenance cost (₹)
    c_cm: float = 200_000.0      # corrective maintenance / failure cost (₹)
    c_d: float = 50_000.0        # downtime / production-loss cost (₹)
    p_max: float = 0.90          # maximum tolerated failure probability constraint
    rul_safety_hours: float = 12.0  # minimum acceptable RUL safety margin


@dataclass
class SystemConfig:
    window: WindowConfig = field(default_factory=WindowConfig)
    signal: SignalProcessingConfig = field(default_factory=SignalProcessingConfig)
    health_index: HealthIndexConfig = field(default_factory=HealthIndexConfig)
    failure: FailurePredictionConfig = field(default_factory=FailurePredictionConfig)
    rul: RULConfig = field(default_factory=RULConfig)
    cost: CostConfig = field(default_factory=CostConfig)


# ─── Default singleton ────────────────────────────────────────────────────────
CONFIG = SystemConfig()
