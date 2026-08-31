"""
Expected Cost Modeling for Maintenance Optimization.

Given a cumulative failure probability curve P_f(t) over future time t,
the Expected Cost of deferring maintenance until time t is:

  E[Cost(t)] = (1 - P_f(t)) * C_planned(t) + P_f(t) * C_unplanned

where:
  - (1 - P_f(t)) is the probability the machine survives until scheduled intervention at time t
  - P_f(t) is the cumulative probability of failure occurring before time t
  - C_planned(t) includes slight production loss if deferred unnecessarily
  - C_unplanned includes full emergency downtime, scrap loss, and hazard penalties.
"""
from __future__ import annotations

import numpy as np
from src.optimization.cost_model import MaintenanceCostProfile


def compute_expected_cost_curve(
    time_horizon_hours: np.ndarray,
    cumulative_failure_probs: np.ndarray,
    cost_profile: MaintenanceCostProfile | None = None,
) -> np.ndarray:
    """
    Compute the expected cost at each future intervention timestamp.

    Args:
        time_horizon_hours: Array of future hours [0, 1, 2, ..., H].
        cumulative_failure_probs: Cumulative failure probability P_f(t) in [0, 1].
        cost_profile: :class:`MaintenanceCostProfile` instance.

    Returns:
        1-D numpy array of expected costs ($) corresponding to each time horizon.
    """
    profile = cost_profile or MaintenanceCostProfile()
    t = np.asarray(time_horizon_hours, dtype=float)
    p_f = np.clip(np.asarray(cumulative_failure_probs, dtype=float), 0.0, 1.0)

    c_plan = profile.total_planned_cost
    c_unplan = profile.total_unplanned_cost

    # Expected Cost: E[C(t)] = (1 - P_f(t)) * C_planned + P_f(t) * C_unplanned
    expected_costs = (1.0 - p_f) * c_plan + p_f * c_unplan

    return expected_costs
