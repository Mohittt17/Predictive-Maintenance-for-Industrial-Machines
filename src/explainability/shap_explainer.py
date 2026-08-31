"""
Explainability layer — SHAP (SHapley Additive exPlanations) & degradation mode inference.

Translates black-box ML predictions into actionable engineering insights:
  1. Computes local SHAP values for individual machine windows.
  2. Identifies top positive risk-contributing sensor features.
  3. Maps feature importance signatures to physical degradation modes:
     - High vibration RMS / Crest factor / Peak-to-Peak -> Bearing degradation
     - High acoustic emission + high temp -> Gear wear / boundary lubrication loss
     - High current + high temp -> Motor stator / winding overheating
     - Pressure drop / fluctuation -> Hydraulic leakage / seal failure
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, Sequence
import numpy as np

try:
    import shap
    _SHAP_AVAILABLE = True
except ImportError:
    _SHAP_AVAILABLE = False

from src.ingestion.schema import ExplainabilityResult
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ─── Domain-specific degradation mapping rules ────────────────────────────────

_FEATURE_DEGRADATION_MAP = {
    "vib_rms": "Bearing race degradation & rotational unbalance",
    "vib_kurtosis": "Impulsive bearing impact fault (BPFO/BPFI)",
    "vib_crest_factor": "Severe surface spalling & shock transients",
    "vib_peak_to_peak": "Mechanical looseness & structural misalignment",
    "acou_rms": "Friction-induced micro-cracking & lubrication breakdown",
    "acou_kurtosis": "Acoustic emission transient bursts (early fatigue)",
    "temp_max": "Thermal overload & winding friction heating",
    "temp_mean": "Steady-state motor thermal stress",
    "curr_rms": "Electrical motor phase overload & load resistance",
    "pres_mean": "Hydraulic line pressure loss & pneumatic leak",
    "bearing_band1_energy": "Inner race defect harmonic resonance (BPFI)",
    "bearing_band2_energy": "Outer race defect harmonic resonance (BPFO)",
    "imbalance_1x_energy": "Rotor mass imbalance (1X shaft speed)",
    "imbalance_2x_energy": "Shaft angular misalignment (2X harmonic)",
}


def infer_degradation_mode(top_features: list[tuple[str, float]]) -> str:
    """
    Map top SHAP feature contributions to a physical machine degradation diagnosis.

    Args:
        top_features: List of (feature_name, shap_value) sorted by magnitude.

    Returns:
        Natural-language diagnosis string.
    """
    if not top_features:
        return "Normal operating condition — no critical anomaly detected."

    top_feat_name, _ = top_features[0]
    
    # Check direct map
    for key, desc in _FEATURE_DEGRADATION_MAP.items():
        if key in top_feat_name.lower():
            # If second feature also reinforces diagnosis, combine them
            if len(top_features) > 1 and "temp" in top_features[1][0].lower():
                return f"{desc} with secondary thermal escalation"
            return desc

    if "vib" in top_feat_name.lower():
        return "Mechanical vibration anomaly (suspected rotating component fault)"
    elif "temp" in top_feat_name.lower():
        return "Thermal overheating & abnormal friction"
    elif "curr" in top_feat_name.lower():
        return "Electrical subsystem overload & winding stress"
    elif "pres" in top_feat_name.lower():
        return "Fluid dynamic & seal degradation"

    return f"Multi-sensor degradation driven primarily by {top_feat_name}"


class SHAPExplainer:
    """
    SHAP-based model explainer for failure prediction & RUL.

    Uses TreeExplainer for tree models (RandomForest, XGBoost) and falls back
    to exact marginal contributions or background baselines when shap package
    is unavailable.
    """

    def __init__(
        self,
        model: Any,
        feature_names: Optional[list[str]] = None,
        background_data: Optional[np.ndarray] = None,
    ) -> None:
        self.model = model
        self.feature_names = feature_names or []
        self.background_data = background_data
        self._explainer = None
        self._fitted = False

        if _SHAP_AVAILABLE:
            try:
                # TreeExplainer for sklearn tree models
                if hasattr(model, "_model") and hasattr(model._model, "estimators_"):
                    self._explainer = shap.TreeExplainer(model._model)
                elif hasattr(model, "estimators_"):
                    self._explainer = shap.TreeExplainer(model)
                elif background_data is not None:
                    self._explainer = shap.Explainer(model.predict if hasattr(model, "predict") else model, background_data)
                self._fitted = True
            except Exception as e:
                logger.warning(f"SHAP TreeExplainer init failed, fallback mode enabled: {e}")

    def explain_instance(
        self,
        x: np.ndarray,
        machine_id: str = "Machine",
        window_end: Optional[datetime] = None,
        top_k: int = 5,
    ) -> ExplainabilityResult:
        """
        Explain a single feature vector instance.

        Args:
            x: 1-D array of features for a single window.
            machine_id: Machine identifier.
            window_end: Timestamp for the window.
            top_k: Number of top contributing features to return.

        Returns:
            :class:`ExplainabilityResult` dataclass.
        """
        x_2d = np.atleast_2d(x)
        names = self.feature_names or [f"feature_{i}" for i in range(x_2d.shape[1])]
        ts = window_end or datetime.now()

        if self._explainer is not None and _SHAP_AVAILABLE:
            try:
                shap_values = self._explainer.shap_values(x_2d)
                # For binary classification TreeExplainer outputs list [shap_neg, shap_pos]
                if isinstance(shap_values, list):
                    vals = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                elif hasattr(shap_values, "values"):
                    vals = shap_values.values[0]
                else:
                    vals = shap_values[0]
                base_val = float(getattr(self._explainer, "expected_value", 0.0))
            except Exception as e:
                logger.warning(f"SHAP calculation error: {e}, using normalized deviation")
                vals = self._fallback_contributions(x_2d[0])
                base_val = 0.0
        else:
            vals = self._fallback_contributions(x_2d[0])
            base_val = 0.0

        # Sort features by absolute contribution to positive risk
        ranked_indices = np.argsort(np.abs(vals))[::-1][:top_k]
        top_features = [(names[i], float(vals[i])) for i in ranked_indices]

        mode = infer_degradation_mode(top_features)

        return ExplainabilityResult(
            machine_id=machine_id,
            window_end=ts,
            top_features=top_features,
            degradation_mode=mode,
            base_value=base_val,
        )

    def _fallback_contributions(self, x: np.ndarray) -> np.ndarray:
        """Fallback feature attribution based on standard deviations or model importances."""
        if hasattr(self.model, "feature_importance"):
            try:
                fi = dict(self.model.feature_importance(top_n=len(x)))
                return np.array([fi.get(name, 0.05) for name in self.feature_names])
            except Exception:
                pass
        if hasattr(self.model, "_model") and hasattr(self.model._model, "feature_importances_"):
            return self.model._model.feature_importances_ * np.sign(x)
        # Default: normalized deviation from 0
        return np.abs(x) / (np.sum(np.abs(x)) + 1e-8)
