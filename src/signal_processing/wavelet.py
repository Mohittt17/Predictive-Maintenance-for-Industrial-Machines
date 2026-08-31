"""
Wavelet-domain signal feature extraction using Discrete Wavelet Transform (DWT).

Unlike FFT, wavelets decompose a signal into both frequency AND time information.
This is critical for non-stationary industrial signals where a fault may manifest
as a short-duration, high-frequency transient — something FFT averages away.

Pipeline
--------
Vibration signal
    │
    ▼  DWT (e.g. db4, 5 levels)
    ├── Approximation (A5) — low-frequency content
    ├── Detail D5 — low-mid frequency
    ├── Detail D4
    ├── Detail D3
    ├── Detail D2
    └── Detail D1 — highest frequency content

For each sub-band we compute:
    - Energy (Σ cᵢ²)
    - Energy ratio (band energy / total energy)
    - Shannon entropy of the coefficient distribution
    - RMS of coefficients
    - Kurtosis of coefficients (sensitive to bearing fault impulses)

References
----------
Mallat, S. (1999). A Wavelet Tour of Signal Processing. Academic Press.
Peng, Z., Chu, F. (2004). Application of the wavelet transform in machine
condition monitoring and fault diagnostics. Mechanical Systems and Signal
Processing, 18(2), 199–221.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False
    pywt = None  # type: ignore


@dataclass(frozen=True)
class WaveletBandFeatures:
    """Features for a single DWT sub-band (approximation or detail level)."""
    name:          str     # e.g. "D1", "D2", "A5"
    energy:        float
    energy_ratio:  float   # fraction of total signal energy
    entropy:       float   # Shannon entropy of coefficient distribution
    rms:           float
    kurtosis:      float
    n_coefficients: int


@dataclass
class WaveletFeatures:
    """
    All wavelet features for one signal window.

    Attributes
    ----------
    bands          : One :class:`WaveletBandFeatures` per DWT level + approximation.
    total_energy   : Sum of all band energies.
    wavelet        : Wavelet family used (e.g. "db4").
    level          : Decomposition depth.
    """
    bands:        list[WaveletBandFeatures] = field(default_factory=list)
    total_energy: float = 0.0
    wavelet:      str   = "db4"
    level:        int   = 5

    def get_band(self, name: str) -> Optional[WaveletBandFeatures]:
        """Return the band with the given name, or None."""
        for b in self.bands:
            if b.name == name:
                return b
        return None


def _band_entropy(coeffs: np.ndarray) -> float:
    """Shannon entropy of the normalised squared coefficient distribution."""
    sq = coeffs ** 2
    total = sq.sum()
    if total == 0:
        return 0.0
    p = sq / total
    return float(-np.sum(p * np.log(p + 1e-12)))


def _band_kurtosis(coeffs: np.ndarray) -> float:
    sigma = np.std(coeffs)
    if sigma == 0:
        return 0.0
    return float(np.mean(((coeffs - np.mean(coeffs)) / sigma) ** 4))


def compute_wavelet_features(
    signal: np.ndarray,
    wavelet: str = "db4",
    level: int = 5,
) -> WaveletFeatures:
    """
    Compute DWT-based features from a 1-D signal.

    Args:
        signal:  1-D signal array (e.g. vibration m/s²).
        wavelet: PyWavelets wavelet name (default: "db4").
        level:   Decomposition depth (default: 5).

    Returns:
        :class:`WaveletFeatures` with one band per detail level + approximation.

    Raises:
        ImportError:  If PyWavelets is not installed.
        ValueError:   If signal is too short for the requested decomposition level.
    """
    if not _PYWT_AVAILABLE:
        raise ImportError(
            "PyWavelets is required for wavelet analysis.\n"
            "Install with: pip install PyWavelets"
        )

    signal = np.asarray(signal, dtype=np.float64).ravel()
    if signal.size == 0:
        raise ValueError("Cannot compute wavelet features from an empty signal.")

    # Clamp level to maximum feasible for this signal length
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(wavelet))
    level = min(level, max_level)

    # Decompose: returns [cA_n, cD_n, cD_{n-1}, …, cD_1]
    coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
    # coeffs[0]  = approximation at level `level`
    # coeffs[1]  = detail at level `level`   (D_level)
    # coeffs[-1] = detail at level 1         (D1, highest freq)

    # Compute total energy across all bands
    total_energy = sum(float(np.sum(c ** 2)) for c in coeffs)
    if total_energy == 0:
        total_energy = 1e-12

    bands: list[WaveletBandFeatures] = []

    for idx, band_coeffs in enumerate(coeffs):
        if idx == 0:
            name = f"A{level}"
        else:
            name = f"D{level - idx + 1}"

        energy = float(np.sum(band_coeffs ** 2))
        bands.append(WaveletBandFeatures(
            name=name,
            energy=energy,
            energy_ratio=energy / total_energy,
            entropy=_band_entropy(band_coeffs),
            rms=float(np.sqrt(np.mean(band_coeffs ** 2))),
            kurtosis=_band_kurtosis(band_coeffs),
            n_coefficients=len(band_coeffs),
        ))

    return WaveletFeatures(
        bands=bands,
        total_energy=total_energy,
        wavelet=wavelet,
        level=level,
    )


def features_to_dict(
    f: WaveletFeatures,
    prefix: str = "",
    include_entropy: bool = True,
    include_kurtosis: bool = True,
) -> dict[str, float]:
    """
    Flatten a :class:`WaveletFeatures` into a ``{key: value}`` dict.

    Keys follow the pattern ``{prefix}wavelet_{band}_{metric}``.
    E.g. ``"vib_wavelet_D1_energy"``, ``"vib_wavelet_A5_energy_ratio"``.
    """
    d: dict[str, float] = {f"{prefix}wavelet_total_energy": f.total_energy}
    for band in f.bands:
        b = band.name
        d[f"{prefix}wavelet_{b}_energy"]       = band.energy
        d[f"{prefix}wavelet_{b}_energy_ratio"] = band.energy_ratio
        d[f"{prefix}wavelet_{b}_rms"]          = band.rms
        if include_entropy:
            d[f"{prefix}wavelet_{b}_entropy"]  = band.entropy
        if include_kurtosis:
            d[f"{prefix}wavelet_{b}_kurtosis"] = band.kurtosis
    return d
