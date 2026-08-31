"""
src.utils.metrics — shared evaluation utilities used across multiple milestones.

Centralising metric computation ensures consistency between failure prediction,
RUL prediction, and anomaly detection evaluators.
"""
from __future__ import annotations

import numpy as np
from typing import Sequence


def rms(arr: np.ndarray) -> float:
    """Root mean square of a 1-D array."""
    return float(np.sqrt(np.mean(arr ** 2)))


def kurtosis(arr: np.ndarray) -> float:
    """Excess kurtosis (Fisher's definition)."""
    mu = np.mean(arr)
    sigma = np.std(arr)
    if sigma == 0:
        return 0.0
    return float(np.mean(((arr - mu) / sigma) ** 4)) - 3.0


def skewness(arr: np.ndarray) -> float:
    """Sample skewness."""
    mu = np.mean(arr)
    sigma = np.std(arr)
    if sigma == 0:
        return 0.0
    return float(np.mean(((arr - mu) / sigma) ** 3))


def crest_factor(arr: np.ndarray) -> float:
    """Peak / RMS ratio."""
    rms_val = rms(arr)
    if rms_val == 0:
        return 0.0
    return float(np.max(np.abs(arr)) / rms_val)


def detection_lead_time(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    timestamps: Sequence[float],
    failure_label: int = 1,
) -> float:
    """
    Compute detection lead time in the same unit as timestamps.

    Lead time = (first true-positive prediction timestamp)
                - (first actual failure timestamp)

    Returns -inf if no true positive was found, or +inf if no failure occurred.
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    ts_arr = np.asarray(timestamps, dtype=float)

    failure_idxs = np.where(y_true_arr == failure_label)[0]
    if len(failure_idxs) == 0:
        return float("inf")

    first_failure_ts = ts_arr[failure_idxs[0]]

    # True positives before or at first failure
    tp_idxs = np.where((y_pred_arr == failure_label) & (ts_arr <= first_failure_ts))[0]
    if len(tp_idxs) == 0:
        return float("-inf")

    first_tp_ts = ts_arr[tp_idxs[0]]
    return float(first_failure_ts - first_tp_ts)


def nasa_rul_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    NASA asymmetric scoring function for RUL prediction.

    Penalises late predictions (under-estimation of RUL) more heavily than
    early predictions.

    S = Σ exp(d/13 - 1) if d < 0  (early, less penalty)
        Σ exp(d/10 - 1) if d ≥ 0  (late,  more penalty)

    where d = y_pred - y_true.
    """
    d = y_pred - y_true
    scores = np.where(d < 0, np.exp(-d / 13.0) - 1.0, np.exp(d / 10.0) - 1.0)
    return float(np.sum(scores))


def coverage(y_lower: np.ndarray, y_upper: np.ndarray, y_true: np.ndarray) -> float:
    """Empirical coverage of a prediction interval."""
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))


def sharpness(y_lower: np.ndarray, y_upper: np.ndarray) -> float:
    """Mean interval width (smaller = sharper prediction intervals)."""
    return float(np.mean(y_upper - y_lower))
