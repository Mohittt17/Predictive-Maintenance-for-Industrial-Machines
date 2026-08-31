"""
Frequency-domain signal feature extraction using FFT.

The Fast Fourier Transform converts a time-domain signal x(t) into the
frequency domain X(f), revealing which frequencies carry energy.  For
rotating machinery this is essential: bearing faults generate energy at
specific characteristic frequencies (BPFO, BPFI, BSF, FTF) that are
invisible in the raw time-domain signal.

Key features extracted
----------------------
- Power spectral density (PSD)
- Dominant frequency (frequency of peak power)
- Spectral entropy (Shannon entropy of normalised PSD — high entropy → flat,
  noisy spectrum; low entropy → concentrated spectral peaks)
- Spectral centroid (weighted mean frequency)
- Band-pass energy in configurable frequency bands (for BPFO/BPFI detection)
- Harmonic energy at N× shaft speed (for imbalance / misalignment)

References
----------
Brandt, A. (2011). Noise and Vibration Analysis. Wiley.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FrequencyFeatures:
    """
    Frequency-domain features extracted from one signal window.

    Attributes
    ----------
    dominant_freq       : Frequency (Hz) of the highest-power bin
    dominant_freq_power : Power at the dominant frequency
    spectral_entropy    : Shannon entropy of normalised PSD (nats)
    spectral_centroid   : PSD-weighted mean frequency (Hz)
    spectral_variance   : PSD-weighted frequency variance
    rms_freq            : RMS of the magnitude spectrum
    band_energies       : Dict mapping (f_low, f_high) → integrated PSD energy
    harmonic_energies   : Dict mapping harmonic order → energy at N×f_shaft
    total_power         : Total power (integral of PSD)
    """
    dominant_freq:       float
    dominant_freq_power: float
    spectral_entropy:    float
    spectral_centroid:   float
    spectral_variance:   float
    rms_freq:            float
    band_energies:       dict   # {(f_low, f_high): energy}
    harmonic_energies:   dict   # {harmonic_order: energy}
    total_power:         float


def compute_fft(
    signal: np.ndarray,
    sampling_rate: float,
    n: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one-sided Power Spectral Density via FFT.

    Uses scipy.signal.welch is NOT used here intentionally — we compute a
    single-window FFT for direct correspondence with wavelet features
    (same window, same time span).

    Args:
        signal:       1-D signal array.
        sampling_rate: Hz.
        n:            FFT points (defaults to len(signal)).

    Returns:
        Tuple of (frequencies, psd) both 1-D arrays of the same length.
        Only positive frequencies are returned.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    n_fft = n or len(signal)

    # Apply Hann window to reduce spectral leakage
    window = np.hanning(len(signal))
    windowed = signal * window
    # Zero-pad to n_fft if needed
    if n_fft > len(windowed):
        windowed = np.pad(windowed, (0, n_fft - len(windowed)))

    spectrum = np.fft.rfft(windowed, n=n_fft)
    freqs    = np.fft.rfftfreq(n_fft, d=1.0 / sampling_rate)

    # One-sided PSD: double the two-sided power (except DC and Nyquist)
    psd = (np.abs(spectrum) ** 2) / n_fft
    if n_fft % 2 == 0:
        psd[1:-1] *= 2.0
    else:
        psd[1:] *= 2.0

    return freqs, psd


def compute_frequency_features(
    signal: np.ndarray,
    sampling_rate: float,
    n_fft: int | None = None,
    band_ranges: Sequence[tuple[float, float]] | None = None,
    shaft_speed_hz: float | None = None,
    n_harmonics: int = 3,
) -> FrequencyFeatures:
    """
    Extract frequency-domain features from a 1-D signal.

    Args:
        signal:         1-D signal array.
        sampling_rate:  Sampling frequency in Hz.
        n_fft:          FFT length (default: next power of 2 ≥ len(signal)).
        band_ranges:    List of (f_low, f_high) tuples for band energy.
                        E.g. [(100, 200), (200, 500)] for bearing bands.
        shaft_speed_hz: Fundamental shaft rotation frequency (Hz).
                        Required to compute harmonic energies.
        n_harmonics:    Number of harmonics to compute (1× … n×).

    Returns:
        :class:`FrequencyFeatures` dataclass.
    """
    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size == 0:
        raise ValueError("Cannot compute frequency features from an empty signal.")

    # Default: next power of 2
    if n_fft is None:
        n_fft = int(2 ** np.ceil(np.log2(len(signal))))

    freqs, psd = compute_fft(signal, sampling_rate=sampling_rate, n=n_fft)
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0   # frequency resolution

    # ── Total power ───────────────────────────────────────────────────────────
    total_power = float(np.sum(psd) * df)

    # ── Dominant frequency ────────────────────────────────────────────────────
    peak_idx = int(np.argmax(psd))
    dominant_freq       = float(freqs[peak_idx])
    dominant_freq_power = float(psd[peak_idx])

    # ── Spectral entropy (Shannon, base e) ────────────────────────────────────
    psd_norm = psd / (np.sum(psd) + 1e-12)
    spectral_entropy = float(-np.sum(psd_norm * np.log(psd_norm + 1e-12)))

    # ── Spectral centroid ─────────────────────────────────────────────────────
    spectral_centroid = float(np.sum(freqs * psd_norm))

    # ── Spectral variance ─────────────────────────────────────────────────────
    spectral_variance = float(np.sum(((freqs - spectral_centroid) ** 2) * psd_norm))

    # ── RMS of magnitude spectrum ─────────────────────────────────────────────
    rms_freq = float(np.sqrt(np.mean(np.abs(np.fft.rfft(signal, n=n_fft)) ** 2)))

    # ── Band energies ─────────────────────────────────────────────────────────
    band_energies: dict = {}
    if band_ranges:
        for (f_low, f_high) in band_ranges:
            mask = (freqs >= f_low) & (freqs < f_high)
            energy = float(np.sum(psd[mask]) * df) if mask.any() else 0.0
            band_energies[(f_low, f_high)] = energy

    # ── Harmonic energies ─────────────────────────────────────────────────────
    harmonic_energies: dict = {}
    if shaft_speed_hz is not None and shaft_speed_hz > 0:
        half_bin = df / 2.0
        for h in range(1, n_harmonics + 1):
            target_freq = h * shaft_speed_hz
            mask = np.abs(freqs - target_freq) <= half_bin * 2
            energy = float(np.sum(psd[mask]) * df) if mask.any() else 0.0
            harmonic_energies[h] = energy

    return FrequencyFeatures(
        dominant_freq=dominant_freq,
        dominant_freq_power=dominant_freq_power,
        spectral_entropy=spectral_entropy,
        spectral_centroid=spectral_centroid,
        spectral_variance=spectral_variance,
        rms_freq=rms_freq,
        band_energies=band_energies,
        harmonic_energies=harmonic_energies,
        total_power=total_power,
    )


def features_to_dict(
    f: FrequencyFeatures,
    prefix: str = "",
    include_bands: bool = True,
    include_harmonics: bool = True,
) -> dict[str, float]:
    """
    Flatten a :class:`FrequencyFeatures` into a ``{key: value}`` dict.

    Band energies are keyed as ``{prefix}band_{f_low}_{f_high}_energy``.
    Harmonic energies are keyed as ``{prefix}harmonic_{n}x_energy``.
    """
    d: dict[str, float] = {
        f"{prefix}dominant_freq":       f.dominant_freq,
        f"{prefix}dominant_freq_power": f.dominant_freq_power,
        f"{prefix}spectral_entropy":    f.spectral_entropy,
        f"{prefix}spectral_centroid":   f.spectral_centroid,
        f"{prefix}spectral_variance":   f.spectral_variance,
        f"{prefix}rms_freq":            f.rms_freq,
        f"{prefix}total_power":         f.total_power,
    }
    if include_bands:
        for (fl, fh), energy in f.band_energies.items():
            key = f"{prefix}band_{int(fl)}_{int(fh)}_energy"
            d[key] = energy
    if include_harmonics:
        for h, energy in f.harmonic_energies.items():
            d[f"{prefix}harmonic_{h}x_energy"] = energy
    return d
