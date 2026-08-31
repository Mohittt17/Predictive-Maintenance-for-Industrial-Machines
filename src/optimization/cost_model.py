"""
Maintenance Cost Model & Economics.

Industrial machines involve severe economic trade-offs:
  1. Unplanned Failure:
     - Catastrophic secondary equipment damage (e.g. sheared shaft)
     - Emergency technician overtime & expedited component freight
     - Unscheduled assembly line stoppage ($5,000–$25,000 / hour)
  2. Planned Intervention:
     - Scheduled during off-peak shifts or planned plant downtime
     - Standard replacement component cost
     - Minimal disruption to master production schedule

This module formalises these costs into parameterised economic dataclasses.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaintenanceCostProfile:
    """
    Cost profile parameters for a specific industrial asset / machine class.

    Attributes
    ----------
    planned_maintenance_cost : Fixed cost of scheduled maintenance ($/intervention)
    unplanned_failure_cost   : Direct cost of catastrophic failure repair ($)
    downtime_hourly_rate     : Lost production output value ($/hour)
    planned_downtime_hours   : Duration of scheduled maintenance (hours)
    unplanned_downtime_hours : Duration of emergency repair & recovery (hours)
    safety_penalty_cost      : Environmental / personnel hazard penalty ($)
    """
    planned_maintenance_cost: float = 3200.0
    unplanned_failure_cost:   float = 18500.0
    downtime_hourly_rate:     float = 1200.0
    planned_downtime_hours:   float = 2.0
    unplanned_downtime_hours: float = 12.0
    safety_penalty_cost:      float = 2500.0

    @property
    def total_planned_cost(self) -> float:
        """Total economic cost of executing planned maintenance."""
        return self.planned_maintenance_cost + (self.downtime_hourly_rate * self.planned_downtime_hours)

    @property
    def total_unplanned_cost(self) -> float:
        """Total economic cost of suffering an unplanned catastrophic failure."""
        return (
            self.unplanned_failure_cost
            + (self.downtime_hourly_rate * self.unplanned_downtime_hours)
            + self.safety_penalty_cost
        )
