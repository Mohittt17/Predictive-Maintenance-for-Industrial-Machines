"""
RUL Evaluation — metrics specifically designed for remaining useful life regression.

Key metrics for RUL
-------------------
1.  RMSE — Root Mean Squared Error (overall accuracy)
2.  MAE  — Mean Absolute Error (robust, less sensitive to outliers)
3.  MAPE — Mean Absolute Percentage Error (scale-independent)
4.  NASA Score — the official C-MAPSS scoring function:
      s = Σ exp(-d/13) - 1  if d < 0   (late prediction — dangerous)
          Σ exp( d/10) - 1  if d ≥ 0   (early prediction — acceptable)
    where d = y_pred - y_true (positive = predicted more life than actual)
5.  Within-N-cycles accuracy — fraction of predictions within ±N cycles of truth
6.  RUL Monotonicity Penalty — measures if the predicted RUL decreases over time
    (physically required: remaining life should always decrease or be flat)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.metrics import nasa_rul_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RULEvaluationReport:
    """Complete RUL evaluation metrics."""
    model_name:       str
    rmse:             float
    mae:              float
    mape:             float
    nasa_score:       float
    within_10:        float   # fraction predicted within ±10 cycles/hours
    within_25:        float   # fraction predicted within ±25 cycles/hours
    n_samples:        int
    mean_true_rul:    float
    mean_pred_rul:    float
    std_pred_rul:     float

    def summary(self) -> str:
        return (
            f"{'='*60}\n"
            f"Model: {self.model_name}\n"
            f"{'='*60}\n"
            f"  RMSE              : {self.rmse:.2f}\n"
            f"  MAE               : {self.mae:.2f}\n"
            f"  MAPE              : {self.mape:.2f}%\n"
            f"  NASA Score        : {self.nasa_score:.2f}  (lower is better)\n"
            f"  Within ±10        : {self.within_10:.4f} ({self.within_10*100:.1f}%)\n"
            f"  Within ±25        : {self.within_25:.4f} ({self.within_25*100:.1f}%)\n"
            f"  Mean true RUL     : {self.mean_true_rul:.1f}\n"
            f"  Mean pred RUL     : {self.mean_pred_rul:.1f} (σ={self.std_pred_rul:.1f})\n"
            f"  Samples           : {self.n_samples}\n"
        )


def evaluate_rul(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    within_thresholds: tuple[float, float] = (10.0, 25.0),
) -> RULEvaluationReport:
    """
    Compute comprehensive RUL evaluation metrics.

    Args:
        model_name:         Human-readable identifier.
        y_true:             Ground-truth RUL values.
        y_pred:             Predicted RUL values.
        within_thresholds:  Two thresholds for within-N accuracy.

    Returns:
        :class:`RULEvaluationReport`.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    errors = y_pred - y_true

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae  = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors) / (np.abs(y_true) + 1e-8)) * 100)

    # NASA score
    score = float(nasa_rul_score(y_true, y_pred))

    # Within-N accuracy
    t1, t2 = within_thresholds
    within_10 = float(np.mean(np.abs(errors) <= t1))
    within_25 = float(np.mean(np.abs(errors) <= t2))

    report = RULEvaluationReport(
        model_name=model_name,
        rmse=rmse,
        mae=mae,
        mape=mape,
        nasa_score=score,
        within_10=within_10,
        within_25=within_25,
        n_samples=len(y_true),
        mean_true_rul=float(y_true.mean()),
        mean_pred_rul=float(y_pred.mean()),
        std_pred_rul=float(y_pred.std()),
    )

    logger.info(f"\n{report.summary()}")
    return report


def compare_rul_models(reports: list[RULEvaluationReport]) -> str:
    """Format a comparison table of RUL model results."""
    header = (
        "| Model | RMSE | MAE | NASA Score | Within ±10 | Within ±25 |\n"
        "|---|---|---|---|---|---|\n"
    )
    rows = [
        f"| {r.model_name} "
        f"| {r.rmse:.2f} "
        f"| {r.mae:.2f} "
        f"| {r.nasa_score:.2f} "
        f"| {r.within_10:.4f} "
        f"| {r.within_25:.4f} |"
        for r in reports
    ]
    return header + "\n".join(rows)
