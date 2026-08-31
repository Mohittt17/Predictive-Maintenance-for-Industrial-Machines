"""
Canonical data schemas for the Predictive Maintenance system.

All loaders convert their native format into these dataclasses before any
downstream processing.  This ensures a single source of truth for column
names, units, and optional fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ─── Raw sensor reading ───────────────────────────────────────────────────────

@dataclass
class SensorReading:
    """
    One time-stamped, multimodal sensor observation from a single machine.

    Units
    -----
    vibration_x/y/z : m/s²  (or g for IMS/XJTU raw data; normalised downstream)
    temperature      : °C
    current          : A
    pressure         : bar
    acoustic         : dB (RMS envelope)
    rpm              : rev/min
    load             : % of rated load
    rul_hours        : remaining useful life in hours (NaN when unknown)
    health_label     : 0=Healthy … 4=Failure  (None when unlabelled)
    """
    machine_id:   str
    timestamp:    datetime
    vibration_x:  float
    vibration_y:  float
    vibration_z:  float
    temperature:  float
    current:      float
    pressure:     float
    acoustic:     float
    rpm:          float
    load:         float
    rul_hours:    Optional[float] = None
    health_label: Optional[int] = None   # 0=Healthy, 1=Minor, 2=Moderate, 3=Severe, 4=Failure
    dataset:      str = "unknown"        # source dataset tag


# ─── Per-window feature vector ────────────────────────────────────────────────

@dataclass
class FeatureVector:
    """
    Engineered feature vector computed over a sliding window of SensorReadings.
    Combines time-domain, frequency-domain, and wavelet-domain statistics.
    """
    machine_id:   str
    window_start: datetime
    window_end:   datetime
    n_samples:    int

    # ── Time-domain ───────────────────────────────────────────────────────────
    vib_rms:           float = float("nan")
    vib_kurtosis:      float = float("nan")
    vib_crest_factor:  float = float("nan")
    vib_skewness:      float = float("nan")
    vib_peak_to_peak:  float = float("nan")
    vib_variance:      float = float("nan")
    vib_std:           float = float("nan")

    temp_mean:  float = float("nan")
    temp_std:   float = float("nan")
    temp_max:   float = float("nan")

    curr_mean:  float = float("nan")
    curr_std:   float = float("nan")
    curr_rms:   float = float("nan")

    pres_mean:  float = float("nan")
    pres_std:   float = float("nan")

    acou_rms:      float = float("nan")
    acou_kurtosis: float = float("nan")
    acou_peak:     float = float("nan")

    rpm_mean:   float = float("nan")
    load_mean:  float = float("nan")

    # ── Frequency-domain ──────────────────────────────────────────────────────
    vib_dominant_freq:      float = float("nan")
    vib_spectral_entropy:   float = float("nan")
    vib_spectral_centroid:  float = float("nan")
    bearing_band1_energy:   float = float("nan")   # BPFO/BPFI band
    bearing_band2_energy:   float = float("nan")   # cage + roller band
    imbalance_1x_energy:    float = float("nan")   # 1× RPM harmonic
    imbalance_2x_energy:    float = float("nan")   # 2× RPM harmonic

    # ── Wavelet (DWT detail levels D1–D5) ─────────────────────────────────────
    vib_wavelet_d1_energy: float = float("nan")
    vib_wavelet_d2_energy: float = float("nan")
    vib_wavelet_d3_energy: float = float("nan")
    vib_wavelet_d4_energy: float = float("nan")
    vib_wavelet_d5_energy: float = float("nan")
    vib_wavelet_approx_energy: float = float("nan")

    # ── Derived / health ──────────────────────────────────────────────────────
    health_index: float = float("nan")   # 0 (failure) – 100 (healthy)

    # ── Labels (propagated from raw data) ────────────────────────────────────
    rul_hours:    Optional[float] = None
    health_label: Optional[int] = None
    dataset:      str = "unknown"


# ─── Anomaly detection result ─────────────────────────────────────────────────

@dataclass
class AnomalyResult:
    """Output from the anomaly detection layer for a single window."""
    machine_id:       str
    window_end:       datetime
    anomaly_score:    float          # higher → more anomalous
    is_anomaly:       bool
    detector:         str            # "IsolationForest" | "OneClassSVM" | "Autoencoder"
    reconstruction_error: Optional[float] = None   # Autoencoder only


# ─── Failure prediction result ────────────────────────────────────────────────

@dataclass
class FailurePrediction:
    """Binary failure-within-horizon prediction for a single window."""
    machine_id:       str
    window_end:       datetime
    failure_prob:     float          # P(failure within horizon)
    is_failure:       bool
    horizon_hours:    float          # prediction horizon (e.g. 72)
    model:            str
    confidence:       float          # model confidence (e.g. probability of predicted class)


# ─── RUL prediction result ────────────────────────────────────────────────────

@dataclass
class RULPrediction:
    """
    Remaining Useful Life prediction with uncertainty bounds.

    Attributes
    ----------
    rul_p10  : 10th-percentile RUL (pessimistic)
    rul_p50  : median RUL (point estimate)
    rul_p90  : 90th-percentile RUL (optimistic)
    """
    machine_id:  str
    window_end:  datetime
    rul_p10:     float
    rul_p50:     float
    rul_p90:     float
    model:       str
    units:       str = "hours"


# ─── SHAP / explainability result ─────────────────────────────────────────────

@dataclass
class ExplainabilityResult:
    """Top SHAP feature contributions for a single prediction."""
    machine_id:        str
    window_end:        datetime
    top_features:      list[tuple[str, float]]   # (feature_name, shap_value)
    degradation_mode:  str                        # inferred from top features
    base_value:        float                      # SHAP base value


# ─── Cost optimization result ─────────────────────────────────────────────────

@dataclass
class CostOptimizationResult:
    """Output of the maintenance cost optimizer."""
    machine_id:            str
    evaluated_at:          datetime
    optimal_window_hours:  float        # hours from now
    expected_cost_now:     float        # ₹
    expected_cost_optimal: float        # ₹ at optimal window
    cost_schedule:         list[dict]   # [{hours, failure_risk, expected_cost}, …]
    constraint_violated:   bool         # True if P_max or RUL_safety triggered


# ─── Final recommendation ─────────────────────────────────────────────────────

@dataclass
class MaintenanceRecommendation:
    """
    The end-to-end output shown to the maintenance engineer.

    Example
    -------
    Machine M-07 | Health Score: 38/100
    Failure probability in next 72h: 81%
    Estimated RUL: 56–78 hours
    Primary suspected degradation: Bearing degradation | Confidence: 87%
    Recommended: Schedule intervention within 24–36 hours.
    Reason: Waiting beyond 48h has higher expected cost than planned maintenance.
    """
    machine_id:              str
    evaluated_at:            datetime
    health_score:            float          # 0–100
    failure_prob_72h:        float          # 0–1
    rul_median_hours:        float
    rul_p10_hours:           float
    rul_p90_hours:           float
    degradation_mode:        str
    confidence:              float          # 0–1
    top_shap_features:       list[tuple[str, float]]
    recommended_action:      str
    action_reasoning:        str
    optimal_window_hours:    float
    expected_cost_now:       float
    expected_cost_optimal:   float
    anomaly_score:           float
    is_anomaly:              bool


# ─── Dataset container ────────────────────────────────────────────────────────

@dataclass
class RawDataset:
    """Container returned by every dataset loader."""
    name:         str
    readings:     list[SensorReading]
    machine_ids:  list[str]
    metadata:     dict = field(default_factory=dict)

    @property
    def n_readings(self) -> int:
        return len(self.readings)

    @property
    def n_machines(self) -> int:
        return len(self.machine_ids)
