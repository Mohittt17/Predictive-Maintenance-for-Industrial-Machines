"""
Sliding-window feature extraction over sensor time-series.

Applies the time-domain, frequency-domain, and wavelet feature extractors
over consecutive windows of SensorReading objects, producing one FeatureVector
per window.

Window strategy
---------------
    |<-- size -->|
    |            |
  t=0         t=size
       |<-- size -->|
       t=stride    t=stride+size
    ...

The stride controls overlap:
    stride = 1    → maximum overlap (slowest, richest feature density)
    stride = size → no overlap (fastest, fewest windows)
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

from src.ingestion.schema import FeatureVector, SensorReading
from src.signal_processing.time_domain import compute_time_features, features_to_dict as td_to_dict
from src.signal_processing.frequency_domain import (
    compute_frequency_features,
    features_to_dict as fd_to_dict,
)
from src.signal_processing.wavelet import compute_wavelet_features, features_to_dict as wt_to_dict
from src.utils.config import CONFIG
from src.utils.logger import get_logger

logger = get_logger(__name__)

_WIN = CONFIG.window
_SIG = CONFIG.signal


def _extract_column(readings: list[SensorReading], attr: str) -> np.ndarray:
    return np.array([getattr(r, attr) for r in readings], dtype=np.float64)


def _safe_median(values: list[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None and not math.isnan(v)]
    return float(np.median(vals)) if vals else None


def _safe_mode_int(values: list[Optional[int]]) -> Optional[int]:
    counts: dict[int, int] = {}
    for v in values:
        if v is not None:
            counts[v] = counts.get(v, 0) + 1
    return max(counts, key=counts.get) if counts else None  # type: ignore


def extract_window_features(
    readings: list[SensorReading],
    sampling_rate: float = _SIG.sampling_rate_hz,
    wavelet: str = _SIG.wavelet,
    wavelet_level: int = _SIG.wavelet_levels,
    band_ranges: Optional[list[tuple[float, float]]] = None,
    shaft_speed_hz: Optional[float] = None,
) -> FeatureVector:
    """
    Compute the full feature vector from a list of SensorReadings (one window).

    Args:
        readings:       Window of SensorReading objects.
        sampling_rate:  Hz — used for FFT and wavelet frequency axis.
        wavelet:        PyWavelets wavelet name.
        wavelet_level:  DWT decomposition depth.
        band_ranges:    Bearing frequency bands for energy extraction.
        shaft_speed_hz: Shaft frequency for harmonic energy computation.

    Returns:
        :class:`FeatureVector` populated with time, frequency, and wavelet features.

    Raises:
        ValueError: If readings list is empty.
    """
    if not readings:
        raise ValueError("readings list must be non-empty.")

    band_ranges = band_ranges or [tuple(b) for b in _SIG.bearing_freq_bands]
    window_start = readings[0].timestamp
    window_end   = readings[-1].timestamp
    machine_id   = readings[0].machine_id

    # ── Extract raw arrays ────────────────────────────────────────────────────
    vib_x = _extract_column(readings, "vibration_x")
    vib_y = _extract_column(readings, "vibration_y")
    vib_z = _extract_column(readings, "vibration_z")
    temp  = _extract_column(readings, "temperature")
    curr  = _extract_column(readings, "current")
    pres  = _extract_column(readings, "pressure")
    acou  = _extract_column(readings, "acoustic")
    rpm   = _extract_column(readings, "rpm")
    load  = _extract_column(readings, "load")

    # Composite vibration: magnitude  = √(x²+y²+z²)  per sample
    vib_mag = np.sqrt(vib_x**2 + vib_y**2 + vib_z**2)

    # Shaft speed in Hz (for harmonic energy; use mean RPM if not provided)
    if shaft_speed_hz is None and rpm.mean() > 0:
        shaft_speed_hz = float(rpm.mean()) / 60.0

    # ── Time-domain features (vibration magnitude) ────────────────────────────
    td = compute_time_features(vib_mag)
    td_acou = compute_time_features(acou)
    td_curr = compute_time_features(curr)

    # ── Frequency-domain features ─────────────────────────────────────────────
    try:
        fd = compute_frequency_features(
            vib_mag,
            sampling_rate=sampling_rate,
            band_ranges=band_ranges,
            shaft_speed_hz=shaft_speed_hz,
        )
        fd_dict = fd_to_dict(fd, prefix="vib_")
    except Exception as e:
        logger.warning(f"FFT failed for window {window_end}: {e}")
        fd_dict = {}

    # ── Wavelet features ──────────────────────────────────────────────────────
    try:
        import pywt  # noqa: F401
        wt = compute_wavelet_features(vib_mag, wavelet=wavelet, level=wavelet_level)
        wt_dict = wt_to_dict(wt, prefix="vib_")
        # Extract named band energies for the FeatureVector dataclass fields
        d1_e = wt.get_band("D1"); d1_energy = d1_e.energy if d1_e else float("nan")
        d2_e = wt.get_band("D2"); d2_energy = d2_e.energy if d2_e else float("nan")
        d3_e = wt.get_band("D3"); d3_energy = d3_e.energy if d3_e else float("nan")
        d4_e = wt.get_band("D4"); d4_energy = d4_e.energy if d4_e else float("nan")
        d5_e = wt.get_band("D5"); d5_energy = d5_e.energy if d5_e else float("nan")
        approx = wt.get_band(f"A{wavelet_level}")
        approx_energy = approx.energy if approx else float("nan")
    except ImportError:
        logger.warning("PyWavelets not installed; skipping wavelet features.")
        d1_energy = d2_energy = d3_energy = d4_energy = d5_energy = approx_energy = float("nan")

    # ── Band energies from FFT (for FeatureVector named fields) ───────────────
    if fd_dict:
        bearing_band1 = fd.band_energies.get(tuple(band_ranges[0]), float("nan")) if fd_dict and len(band_ranges) > 0 else float("nan")
        bearing_band2 = fd.band_energies.get(tuple(band_ranges[1]), float("nan")) if fd_dict and len(band_ranges) > 1 else float("nan")
        imbalance_1x  = fd.harmonic_energies.get(1, float("nan")) if fd_dict else float("nan")
        imbalance_2x  = fd.harmonic_energies.get(2, float("nan")) if fd_dict else float("nan")
        dominant_freq  = fd.dominant_freq
        spec_entropy   = fd.spectral_entropy
        spec_centroid  = fd.spectral_centroid
    else:
        bearing_band1 = bearing_band2 = imbalance_1x = imbalance_2x = float("nan")
        dominant_freq = spec_entropy = spec_centroid = float("nan")

    # ── Propagate RUL / labels (median of window) ─────────────────────────────
    rul_list   = [r.rul_hours for r in readings]
    label_list = [r.health_label for r in readings]
    rul_val    = _safe_median(rul_list)
    label_val  = _safe_mode_int(label_list)

    return FeatureVector(
        machine_id=machine_id,
        window_start=window_start,
        window_end=window_end,
        n_samples=len(readings),
        # Time-domain
        vib_rms=td.rms,
        vib_kurtosis=td.kurtosis,
        vib_crest_factor=td.crest_factor,
        vib_skewness=td.skewness,
        vib_peak_to_peak=td.peak_to_peak,
        vib_variance=td.variance,
        vib_std=td.std,
        temp_mean=float(np.mean(temp)),
        temp_std=float(np.std(temp)),
        temp_max=float(np.max(temp)),
        curr_mean=float(np.mean(curr)),
        curr_std=float(np.std(curr)),
        curr_rms=td_curr.rms,
        pres_mean=float(np.mean(pres)),
        pres_std=float(np.std(pres)),
        acou_rms=td_acou.rms,
        acou_kurtosis=td_acou.kurtosis,
        acou_peak=float(np.max(np.abs(acou))),
        rpm_mean=float(np.mean(rpm)),
        load_mean=float(np.mean(load)),
        # Frequency-domain
        vib_dominant_freq=dominant_freq,
        vib_spectral_entropy=spec_entropy,
        vib_spectral_centroid=spec_centroid,
        bearing_band1_energy=bearing_band1,
        bearing_band2_energy=bearing_band2,
        imbalance_1x_energy=imbalance_1x,
        imbalance_2x_energy=imbalance_2x,
        # Wavelet
        vib_wavelet_d1_energy=d1_energy,
        vib_wavelet_d2_energy=d2_energy,
        vib_wavelet_d3_energy=d3_energy,
        vib_wavelet_d4_energy=d4_energy,
        vib_wavelet_d5_energy=d5_energy,
        vib_wavelet_approx_energy=approx_energy,
        # Derived (filled in later by Health Engine)
        health_index=float("nan"),
        # Labels
        rul_hours=rul_val,
        health_label=label_val,
        dataset=readings[0].dataset,
    )


def sliding_window(
    readings: list[SensorReading],
    window_size: int = _WIN.size,
    stride: int = _WIN.stride,
    min_samples: int = _WIN.min_samples,
    **kwargs,
) -> list[FeatureVector]:
    """
    Apply :func:`extract_window_features` over a sliding window.

    Args:
        readings:     Flat list of SensorReadings for ONE machine,
                      sorted by timestamp.
        window_size:  Number of readings per window.
        stride:       Step size between windows.
        min_samples:  Minimum readings to form a valid window.
        **kwargs:     Forwarded to :func:`extract_window_features`.

    Returns:
        List of :class:`FeatureVector`, one per valid window.
    """
    if len(readings) < min_samples:
        logger.warning(
            f"Machine {readings[0].machine_id if readings else '?'}: "
            f"only {len(readings)} readings, less than min_samples={min_samples}."
        )
        return []

    vectors: list[FeatureVector] = []
    n = len(readings)

    for start in range(0, n - window_size + 1, stride):
        window = readings[start : start + window_size]
        if len(window) < min_samples:
            continue
        try:
            fv = extract_window_features(window, **kwargs)
            vectors.append(fv)
        except Exception as e:
            logger.warning(f"Window starting at {start} failed: {e}")

    logger.info(
        f"sliding_window: {len(readings)} readings → {len(vectors)} windows "
        f"(size={window_size}, stride={stride})"
    )
    return vectors
