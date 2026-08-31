"""
Comprehensive failure prediction evaluator.

Imbalanced classification is the core challenge in predictive maintenance:
  - 1,000,000 sensor records
  - 999,000 → normal
  - 1,000   → near-failure

A naive "always predict normal" model gets 99.9% accuracy but is completely
useless.  This evaluator computes the correct metrics for this setting:

  Precision  — of all failure alarms, how many were real?
  Recall     — of all actual failures, how many did we catch?
  F1         — harmonic mean of precision and recall
  PR-AUC     — area under precision-recall curve (better than ROC for skewed data)
  ROC-AUC    — area under ROC curve
  Detection Lead Time — how far before failure did we first predict it?

Detection lead time is the most operationally relevant metric:
  "Model detected failure 31 hours before actual failure."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.utils.metrics import detection_lead_time, nasa_rul_score
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    """
    Complete evaluation metrics for a failure prediction model.

    Attributes
    ----------
    model_name     : Identifier string
    precision      : Precision at default threshold (0.5)
    recall         : Recall at default threshold
    f1             : F1 score
    pr_auc         : Area under precision-recall curve
    roc_auc        : Area under ROC curve
    tn, fp, fn, tp : Confusion matrix elements
    detection_lead_time : Median lead time across all runs (hours / cycles)
    false_alarm_rate    : FP / (FP + TN)
    missed_failure_rate : FN / (FN + TP)
    n_samples      : Total evaluation samples
    n_positive     : Positive samples (near-failure)
    class_imbalance_ratio : n_negative / n_positive
    """
    model_name:          str
    precision:           float
    recall:              float
    f1:                  float
    pr_auc:              float
    roc_auc:             float
    tn:                  int
    fp:                  int
    fn:                  int
    tp:                  int
    detection_lead_time: float
    false_alarm_rate:    float
    missed_failure_rate: float
    n_samples:           int
    n_positive:          int
    class_imbalance_ratio: float
    threshold:           float = 0.5

    def summary(self) -> str:
        """Return a human-readable summary string."""
        return (
            f"{'='*60}\n"
            f"Model: {self.model_name}\n"
            f"{'='*60}\n"
            f"  Precision        : {self.precision:.4f}\n"
            f"  Recall           : {self.recall:.4f}\n"
            f"  F1 Score         : {self.f1:.4f}\n"
            f"  PR-AUC           : {self.pr_auc:.4f}\n"
            f"  ROC-AUC          : {self.roc_auc:.4f}\n"
            f"  Detection Lead T : {self.detection_lead_time:.1f} units\n"
            f"  False Alarm Rate : {self.false_alarm_rate:.4f}\n"
            f"  Missed Failures  : {self.missed_failure_rate:.4f}\n"
            f"  Confusion Matrix : TP={self.tp} FP={self.fp} TN={self.tn} FN={self.fn}\n"
            f"  Samples          : {self.n_samples} "
            f"(+{self.n_positive} | imbalance {self.class_imbalance_ratio:.0f}:1)\n"
        )


def evaluate_classifier(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    timestamps: Optional[np.ndarray] = None,
    threshold: float = 0.5,
    failure_label: int = 1,
) -> EvaluationReport:
    """
    Compute the full failure prediction evaluation report.

    Args:
        model_name:    Human-readable model identifier.
        y_true:        Ground-truth binary labels (0=normal, 1=failure).
        y_pred:        Predicted binary labels.
        y_prob:        Predicted failure probabilities (for AUC metrics).
        timestamps:    Time indices for detection lead time calculation.
        threshold:     Decision threshold used to produce y_pred from y_prob.
        failure_label: Label indicating failure (default: 1).

    Returns:
        :class:`EvaluationReport` with all metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n_pos = int((y_true == failure_label).sum())
    n_neg = int((y_true != failure_label).sum())
    imbalance = n_neg / max(n_pos, 1)

    # ── Core metrics ──────────────────────────────────────────────────────────
    prec   = float(precision_score(y_true, y_pred, zero_division=0))
    rec    = float(recall_score(y_true, y_pred, zero_division=0))
    f1     = float(f1_score(y_true, y_pred, zero_division=0))

    # ── AUC metrics (require probability scores) ──────────────────────────────
    if y_prob is not None and len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc  = float(average_precision_score(y_true, y_prob))
    else:
        roc_auc = float("nan")
        pr_auc  = float("nan")
        if y_prob is None:
            logger.warning(f"{model_name}: y_prob not provided — ROC-AUC and PR-AUC will be NaN")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / max(fp + tn, 1)
    mfr = fn / max(fn + tp, 1)

    # ── Detection lead time ───────────────────────────────────────────────────
    if timestamps is not None:
        dlt = detection_lead_time(y_true, y_pred, timestamps, failure_label=failure_label)
    else:
        dlt = float("nan")

    report = EvaluationReport(
        model_name=model_name,
        precision=prec,
        recall=rec,
        f1=f1,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        detection_lead_time=dlt,
        false_alarm_rate=far,
        missed_failure_rate=mfr,
        n_samples=len(y_true),
        n_positive=n_pos,
        class_imbalance_ratio=imbalance,
        threshold=threshold,
    )

    logger.info(f"\n{report.summary()}")
    return report


def compare_models(reports: list[EvaluationReport]) -> str:
    """
    Format a comparison table of multiple model evaluation reports.

    Returns a markdown-formatted table string.
    """
    header = (
        "| Model | F1 | PR-AUC | ROC-AUC | Lead Time | FAR | MFR |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for r in reports:
        rows.append(
            f"| {r.model_name} "
            f"| {r.f1:.4f} "
            f"| {r.pr_auc:.4f} "
            f"| {r.roc_auc:.4f} "
            f"| {r.detection_lead_time:.1f} "
            f"| {r.false_alarm_rate:.4f} "
            f"| {r.missed_failure_rate:.4f} |"
        )
    return header + "\n".join(rows)
