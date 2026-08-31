"""
Unit tests for Maintenance Cost Modeling, Optimization, and Action Engine (Milestones 9 & 10).
"""
import numpy as np
import pytest
from datetime import datetime

from src.optimization.cost_model import MaintenanceCostProfile
from src.optimization.expected_cost import compute_expected_cost_curve
from src.optimization.optimizer import MaintenanceOptimizer
from src.recommendation.action_engine import ActionEngine
from src.ingestion.schema import MaintenanceRecommendation, CostOptimizationResult


def test_cost_profile_totals():
    profile = MaintenanceCostProfile(
        planned_maintenance_cost=3000.0,
        unplanned_failure_cost=15000.0,
        downtime_hourly_rate=1000.0,
        planned_downtime_hours=2.0,
        unplanned_downtime_hours=10.0,
        safety_penalty_cost=2000.0,
    )
    assert profile.total_planned_cost == 5000.0  # 3000 + 2*1000
    assert profile.total_unplanned_cost == 27000.0  # 15000 + 10*1000 + 2000


def test_expected_cost_curve_monotonicity():
    profile = MaintenanceCostProfile()
    hours = np.linspace(0, 100, 101)
    # Monotonically increasing failure probability
    probs = 1.0 / (1.0 + np.exp(-(hours - 50) / 10.0))
    
    costs = compute_expected_cost_curve(hours, probs, cost_profile=profile)
    assert len(costs) == 101
    assert costs[0] < costs[-1]
    assert np.all(costs >= profile.total_planned_cost)


def test_maintenance_optimizer():
    optimizer = MaintenanceOptimizer(min_lead_hours=4.0, max_allowed_risk=0.20)
    res = optimizer.optimize_intervention_window(
        machine_id="M-07",
        rul_p10_hours=48.0,
        rul_median_hours=72.0,
        current_failure_prob_72h=0.80,
    )

    assert isinstance(res, CostOptimizationResult)
    assert res.machine_id == "M-07"
    assert res.optimal_window_hours >= 4.0
    assert res.optimal_window_hours <= 72.0
    assert len(res.cost_schedule) > 0


def test_action_engine_recommendation():
    engine = ActionEngine()
    x = np.array([1.5, 3.2, 0.4, 1.1])
    
    rec = engine.formulate_recommendation(
        machine_id="M-07",
        feature_vector=x,
        health_score=35.0,
        failure_prob_72h=0.85,
        rul_p10_hours=45.0,
        rul_median_hours=65.0,
        rul_p90_hours=85.0,
        anomaly_score=0.9,
        is_anomaly=True,
    )

    assert isinstance(rec, MaintenanceRecommendation)
    assert rec.machine_id == "M-07"
    assert rec.health_score == 35.0
    assert "Schedule emergency intervention" in rec.recommended_action
    assert rec.optimal_window_hours > 0
