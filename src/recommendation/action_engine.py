"""
Action Recommendation Engine.

Fuses outputs from:
  1. Health Index Engine (0–100 score + health category)
  2. Failure Prediction Classifier (72-hour failure probability)
  3. Probabilistic RUL Regressor (p10, p50, p90 confidence bounds)
  4. SHAP Explainability Engine (Root-cause degradation mode)
  5. Cost & Schedule Optimizer (Optimal intervention window + ROI)

Produces the full canonical :class:`MaintenanceRecommendation` object.
"""
from __future__ import annotations

from datetime import datetime, timezone
import numpy as np

from src.ingestion.schema import (
    MaintenanceRecommendation,
    ExplainabilityResult,
    CostOptimizationResult,
    RULPrediction,
    FailurePrediction,
)
from src.optimization.optimizer import MaintenanceOptimizer
from src.explainability.shap_explainer import SHAPExplainer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ActionEngine:
    """
    Assembles comprehensive predictive maintenance recommendations.

    Args:
        optimizer: :class:`MaintenanceOptimizer` instance.
        explainer: :class:`SHAPExplainer` instance.
    """

    def __init__(
        self,
        optimizer: MaintenanceOptimizer | None = None,
        explainer: SHAPExplainer | None = None,
    ) -> None:
        self.optimizer = optimizer or MaintenanceOptimizer()
        self.explainer = explainer

    def formulate_recommendation(
        self,
        machine_id: str,
        feature_vector: np.ndarray,
        health_score: float,
        failure_prob_72h: float,
        rul_p10_hours: float,
        rul_median_hours: float,
        rul_p90_hours: float,
        anomaly_score: float = 0.0,
        is_anomaly: bool = False,
        evaluated_at: datetime | None = None,
    ) -> MaintenanceRecommendation:
        """
        Synthesize all model outputs into an actionable engineer-facing recommendation.

        Args:
            machine_id: Machine identifier.
            feature_vector: 1-D or 2-D numpy feature vector for current window.
            health_score: Machine health index (0–100).
            failure_prob_72h: Probability of failure within 72h (0–1).
            rul_p10_hours: Pessimistic RUL.
            rul_median_hours: Expected median RUL.
            rul_p90_hours: Optimistic RUL.
            anomaly_score: Unsupervised anomaly score.
            is_anomaly: Binary anomaly flag.
            evaluated_at: Timestamp.

        Returns:
            :class:`MaintenanceRecommendation`
        """
        ts = evaluated_at or datetime.now(timezone.utc)

        # 1. Cost & Schedule Optimization
        opt_res: CostOptimizationResult = self.optimizer.optimize_intervention_window(
            machine_id=machine_id,
            rul_p10_hours=rul_p10_hours,
            rul_median_hours=rul_median_hours,
            current_failure_prob_72h=failure_prob_72h,
            evaluated_at=ts,
        )

        # 2. Explainability & Root-Cause Attribution
        if self.explainer is not None:
            exp_res: ExplainabilityResult = self.explainer.explain_instance(
                feature_vector, machine_id=machine_id, window_end=ts
            )
            degradation_mode = exp_res.degradation_mode
            top_shap = exp_res.top_features
        else:
            degradation_mode = "Mechanical wear & structural vibration degradation"
            top_shap = [("vibration_energy", 0.45), ("acoustic_emission", 0.30)]

        # 3. Action Prescriptions & Operational Reasonings
        if health_score <= 40 or failure_prob_72h >= 0.75:
            rec_action = (
                f"Schedule emergency intervention within next {int(opt_res.optimal_window_hours)}–"
                f"{int(opt_res.optimal_window_hours + 12)} hours. "
                "Throttle operating load by 20% to mitigate failure risk."
            )
            reason = (
                f"High failure risk ({int(failure_prob_72h*100)}% in 72h) with RUL bounded at "
                f"{int(rul_p10_hours)}–{int(rul_p90_hours)}h. "
                "Delaying beyond optimal window increases expected downtime cost significantly."
            )
            conf = 0.92
        elif health_score <= 70 or failure_prob_72h >= 0.35:
            rec_action = (
                f"Plan maintenance overhaul during upcoming scheduled maintenance window "
                f"(within {int(opt_res.optimal_window_hours)} hours)."
            )
            reason = (
                f"Early degradation detected in {degradation_mode}. "
                "Asset is operational but exhibits accelerated wear."
            )
            conf = 0.85
        else:
            rec_action = "Continue normal operations. Routine inspection at standard interval."
            reason = "Machine health index is nominal. Vibration and thermal signatures within baseline."
            conf = 0.95

        return MaintenanceRecommendation(
            machine_id=machine_id,
            evaluated_at=ts,
            health_score=float(health_score),
            failure_prob_72h=float(failure_prob_72h),
            rul_median_hours=float(rul_median_hours),
            rul_p10_hours=float(rul_p10_hours),
            rul_p90_hours=float(rul_p90_hours),
            degradation_mode=degradation_mode,
            confidence=float(conf),
            top_shap_features=top_shap,
            recommended_action=rec_action,
            action_reasoning=reason,
            optimal_window_hours=float(opt_res.optimal_window_hours),
            expected_cost_now=float(opt_res.expected_cost_now),
            expected_cost_optimal=float(opt_res.expected_cost_optimal),
            anomaly_score=float(anomaly_score),
            is_anomaly=bool(is_anomaly),
        )
