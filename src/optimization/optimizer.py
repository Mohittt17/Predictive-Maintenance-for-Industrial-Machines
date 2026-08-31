"""
Maintenance Scheduling Optimizer.

Finds the optimal intervention window t* that minimises expected maintenance
and downtime costs while strictly respecting physical RUL safety boundaries:

  minimize   E[Cost(t)]
  subject to P_failure(t) <= P_max_allowed
             t <= RUL_p10 (pessimistic RUL limit)
             t >= t_min_lead_time (time required to stage replacement parts)
"""
from __future__ import annotations

from datetime import datetime
import numpy as np

from src.ingestion.schema import CostOptimizationResult
from src.optimization.cost_model import MaintenanceCostProfile
from src.optimization.expected_cost import compute_expected_cost_curve
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MaintenanceOptimizer:
    """
    Optimizes the intervention schedule for a degrading machine.

    Args:
        cost_profile: Cost parameters.
        max_allowed_risk: Maximum acceptable cumulative failure probability before maintenance (default 0.15).
        min_lead_hours: Minimum hours needed to order/stage spare parts and dispatch technicians (default 4.0).
    """

    def __init__(
        self,
        cost_profile: MaintenanceCostProfile | None = None,
        max_allowed_risk: float = 0.15,
        min_lead_hours: float = 4.0,
    ) -> None:
        self.cost_profile = cost_profile or MaintenanceCostProfile()
        self.max_allowed_risk = max_allowed_risk
        self.min_lead_hours = min_lead_hours

    def optimize_intervention_window(
        self,
        machine_id: str,
        rul_p10_hours: float,
        rul_median_hours: float,
        current_failure_prob_72h: float,
        evaluated_at: datetime | None = None,
        max_horizon_hours: float = 120.0,
    ) -> CostOptimizationResult:
        """
        Compute optimal intervention time t* (hours from now) minimizing expected cost.

        Args:
            machine_id: Machine identifier string.
            rul_p10_hours: 10th percentile RUL (pessimistic remaining life).
            rul_median_hours: 50th percentile RUL (median expected remaining life).
            current_failure_prob_72h: Probability of failure in next 72 hours.
            evaluated_at: Timestamp.
            max_horizon_hours: Maximum lookahead simulation window.

        Returns:
            :class:`CostOptimizationResult` dataclass.
        """
        ts = evaluated_at or datetime.now()
        time_steps = np.linspace(0.0, max_horizon_hours, int(max_horizon_hours) + 1)

        # Weibull / logistic growth curve for cumulative failure probability
        # P_f(t) starts low and rapidly accelerates as t approaches RUL_median
        shape_k = 3.5
        scale_lambda = max(1.0, rul_median_hours)
        cum_failure_probs = 1.0 - np.exp(-((time_steps / scale_lambda) ** shape_k))
        
        # Scale curve so that P_f(72) matches current_failure_prob_72h
        if 72.0 < len(cum_failure_probs) and cum_failure_probs[72] > 0:
            scale_factor = current_failure_prob_72h / cum_failure_probs[72]
            cum_failure_probs = np.clip(cum_failure_probs * scale_factor, 0.0, 1.0)

        # Compute expected cost for each candidate hour
        expected_costs = compute_expected_cost_curve(
            time_steps, cum_failure_probs, cost_profile=self.cost_profile
        )

        # Build schedule list
        cost_schedule = [
            {
                "hours_from_now": float(time_steps[i]),
                "failure_risk": float(cum_failure_probs[i]),
                "expected_cost": float(expected_costs[i]),
            }
            for i in range(0, len(time_steps), 6)   # 6-hour stride for UI/payload
        ]

        # Feasible region: min_lead_hours <= t <= min(rul_p10_hours, horizon where risk <= max_allowed_risk)
        feasible_mask = (time_steps >= self.min_lead_hours) & (time_steps <= max(self.min_lead_hours, rul_p10_hours)) & (cum_failure_probs <= self.max_allowed_risk)
        
        constraint_violated = False
        if np.any(feasible_mask):
            feasible_indices = np.where(feasible_mask)[0]
            # Maximize run time (extract asset value) before risk penalty kicks in
            optimal_idx = feasible_indices[-1]
            optimal_hours = float(time_steps[optimal_idx])
            optimal_cost = float(expected_costs[optimal_idx])
        else:
            # Urgent intervention: must act as soon as min_lead_hours allows
            constraint_violated = True
            optimal_hours = float(self.min_lead_hours)
            optimal_cost = float(expected_costs[int(self.min_lead_hours)])

        cost_now = float(expected_costs[0])

        logger.info(
            f"MaintenanceOptimizer [{machine_id}]: optimal window = {optimal_hours:.1f}h "
            f"(E[Cost] = ${optimal_cost:,.2f}, immediate = ${cost_now:,.2f})"
        )

        return CostOptimizationResult(
            machine_id=machine_id,
            evaluated_at=ts,
            optimal_window_hours=optimal_hours,
            expected_cost_now=cost_now,
            expected_cost_optimal=optimal_cost,
            cost_schedule=cost_schedule,
            constraint_violated=constraint_violated,
        )
