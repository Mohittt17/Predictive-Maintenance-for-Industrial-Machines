"""
Time-domain signal feature extraction.

Computes the standard statistical descriptors used in condition monitoring:
RMS, kurtosis, skewness, crest factor, peak-to-peak, variance, standard
deviation, and shape factor.

These features alone can reveal significant machine degradation — kurtosis in
particular is a sensitive early indicator of bearing faults.

References
----------
Randall, R.B. (2011). Vibration-based Condition Monitoring. Wiley.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TimeDomainFeatures:
    """
    All time-domain statistical features for a single signal window.

    Attributes
    ----------
    rms           : Root Mean Square — overall energy level
    kurtosis      : Fourth standardised moment — impulsiveness indicator
    skewness      : Third standardised moment — signal asymmetry
    crest_factor  : Peak / RMS — sensitivity to impulsive faults
    peak_to_peak  : max(x) - min(x) — overall amplitude swing
    variance      : σ²
    std           : σ
    shape_factor  : RMS / mean(|x|) — waveform shape indicator
    impulse_factor: peak / mean(|x|)
    margin_factor : peak / (mean(√|x|))²
    mean_abs      : Mean of |x|
    energy        : Σ xᵢ²  (unnormalised)
    """
    rms:           float
    kurtosis:      float
    skewness:      float
    crest_factor:  float
    peak_to_peak:  float
    variance:      float
    std:           float
    shape_factor:  float
    impulse_factor: float
    margin_factor: float
    mean_abs:      float
    energy:        float


def compute_time_features(signal: np.ndarray) -> TimeDomainFeatures:
    """
    Compute all time-domain features from a 1-D signal array.

    Args:
        signal: 1-D numpy array of sensor values (e.g. acceleration m/s²).

    Returns:
        :class:`TimeDomainFeatures` dataclass.

    Raises:
        ValueError: If signal is empty or not 1-D.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size == 0:
        raise ValueError("Cannot compute features from an empty signal array.")

    n = len(signal)
    mu = np.mean(signal)
    sigma = np.std(signal, ddof=0)
    peak = float(np.max(np.abs(signal)))

    # ── RMS ──────────────────────────────────────────────────────────────────
    rms_val = float(np.sqrt(np.mean(signal ** 2)))

    # ── Kurtosis (Fisher definition: Gaussian → 3) ────────────────────────────
    if sigma > 0:
        kurtosis_val = float(np.mean(((signal - mu) / sigma) ** 4))
        skewness_val = float(np.mean(((signal - mu) / sigma) ** 3))
    else:
        kurtosis_val = 0.0
        skewness_val = 0.0

    # ── Crest factor ─────────────────────────────────────────────────────────
    crest = peak / rms_val if rms_val > 0 else 0.0

    # ── Peak-to-peak ─────────────────────────────────────────────────────────
    p2p = float(np.max(signal) - np.min(signal))

    # ── Shape factor = RMS / mean(|x|) ───────────────────────────────────────
    mean_abs = float(np.mean(np.abs(signal)))
    shape = rms_val / mean_abs if mean_abs > 0 else 0.0

    # ── Impulse factor = peak / mean(|x|) ────────────────────────────────────
    impulse = peak / mean_abs if mean_abs > 0 else 0.0

    # ── Margin factor = peak / (mean(√|x|))² ─────────────────────────────────
    mean_sqrt_abs = float(np.mean(np.sqrt(np.abs(signal))))
    margin = peak / (mean_sqrt_abs ** 2) if mean_sqrt_abs > 0 else 0.0

    return TimeDomainFeatures(
        rms=rms_val,
        kurtosis=kurtosis_val,
        skewness=skewness_val,
        crest_factor=crest,
        peak_to_peak=p2p,
        variance=float(sigma ** 2),
        std=float(sigma),
        shape_factor=shape,
        impulse_factor=impulse,
        margin_factor=margin,
        mean_abs=mean_abs,
        energy=float(np.sum(signal ** 2)),
    )


def compute_time_features_batch(
    signals: np.ndarray,
) -> list[TimeDomainFeatures]:
    """
    Vectorised batch computation over a 2-D array.

    Args:
        signals: Array of shape (n_windows, window_length).

    Returns:
        List of :class:`TimeDomainFeatures`, one per row.
    """
    signals = np.atleast_2d(np.asarray(signals, dtype=np.float64))
    return [compute_time_features(row) for row in signals]


def features_to_dict(f: TimeDomainFeatures, prefix: str = "") -> dict[str, float]:
    """
    Convert a :class:`TimeDomainFeatures` to a flat dictionary.

    Args:
        f:      Feature dataclass.
        prefix: Optional prefix appended to every key (e.g. ``"vib_"``).

    Returns:
        ``{prefix + field_name: value}`` mapping.
    """
    import dataclasses
    return {
        f"{prefix}{field.name}": getattr(f, field.name)
        for field in dataclasses.fields(f)
    }
