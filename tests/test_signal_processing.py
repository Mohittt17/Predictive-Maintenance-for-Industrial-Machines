"""
Unit tests for signal processing and feature engineering (Milestone 2).

Covers:
  - Time-domain features: correctness of RMS, kurtosis, crest factor, etc.
  - Frequency-domain features: FFT sanity, band energies, spectral entropy
  - Wavelet features: energy conservation, band naming, kurtosis
  - Sliding-window extraction: shape, stride, min_samples
  - Multimodal: DataFrame conversion, machine-aware split, imputation, normalisation
  - Feature store: save/load roundtrip, list_datasets, delete
"""
from __future__ import annotations

import math
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from src.signal_processing.time_domain import (
    compute_time_features,
    compute_time_features_batch,
    features_to_dict as td_to_dict,
    TimeDomainFeatures,
)
from src.signal_processing.frequency_domain import (
    compute_fft,
    compute_frequency_features,
    features_to_dict as fd_to_dict,
)
from src.feature_engineering.multimodal import (
    feature_vectors_to_dataframe,
    get_feature_matrix,
    machine_aware_split,
    impute_features,
    normalise_features,
    get_feature_cols,
)
from src.ingestion.schema import SensorReading, FeatureVector


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_readings(n: int, machine_id: str = "M-001") -> list[SensorReading]:
    rng = np.random.default_rng(0)
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return [
        SensorReading(
            machine_id=machine_id,
            timestamp=base + timedelta(seconds=i),
            vibration_x=float(rng.normal(0, 0.5)),
            vibration_y=float(rng.normal(0, 0.3)),
            vibration_z=float(rng.normal(0, 0.2)),
            temperature=float(80 + rng.normal(0, 2)),
            current=float(5 + rng.normal(0, 0.1)),
            pressure=float(2.5 + rng.normal(0, 0.05)),
            acoustic=float(40 + rng.normal(0, 1)),
            rpm=1500.0,
            load=80.0,
            rul_hours=float(100 - i),
            health_label=0,
        )
        for i in range(n)
    ]


def _make_feature_vector(machine_id: str = "M-001", rul: float = 80.0) -> FeatureVector:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return FeatureVector(
        machine_id=machine_id,
        window_start=base,
        window_end=base + timedelta(seconds=30),
        n_samples=30,
        vib_rms=0.5,
        vib_kurtosis=3.1,
        vib_crest_factor=2.5,
        vib_skewness=0.1,
        vib_peak_to_peak=2.0,
        vib_variance=0.25,
        vib_std=0.5,
        temp_mean=80.0,
        temp_std=2.0,
        temp_max=85.0,
        curr_mean=5.0,
        curr_std=0.1,
        curr_rms=5.0,
        pres_mean=2.5,
        pres_std=0.05,
        acou_rms=40.0,
        acou_kurtosis=3.0,
        acou_peak=45.0,
        rpm_mean=1500.0,
        load_mean=80.0,
        rul_hours=rul,
        health_label=0,
        dataset="test",
    )


# ─── Time-Domain Tests ────────────────────────────────────────────────────────

class TestTimeDomain:

    def test_rms_sine(self):
        t = np.linspace(0, 2 * np.pi, 1000)
        s = np.sin(t)
        f = compute_time_features(s)
        assert f.rms == pytest.approx(1 / np.sqrt(2), rel=1e-2)

    def test_kurtosis_gaussian(self):
        rng = np.random.default_rng(42)
        s = rng.standard_normal(10_000)
        f = compute_time_features(s)
        # Fisher kurtosis of Gaussian = 3
        assert abs(f.kurtosis - 3.0) < 0.2

    def test_kurtosis_impulse_high(self):
        # Impulsive signal has kurtosis >> 3
        s = np.zeros(1000)
        s[500] = 10.0
        f = compute_time_features(s)
        assert f.kurtosis > 100

    def test_crest_factor_sine(self):
        t = np.linspace(0, 2 * np.pi, 10_000)
        s = np.sin(t)
        f = compute_time_features(s)
        # CF of sine = sqrt(2) ≈ 1.414
        assert f.crest_factor == pytest.approx(np.sqrt(2), rel=2e-2)

    def test_peak_to_peak(self):
        s = np.array([-2.0, 0.0, 3.0])
        f = compute_time_features(s)
        assert f.peak_to_peak == pytest.approx(5.0)

    def test_zero_std_no_nan(self):
        s = np.ones(100)
        f = compute_time_features(s)
        assert not math.isnan(f.kurtosis)
        assert not math.isnan(f.crest_factor)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            compute_time_features(np.array([]))

    def test_batch_shape(self):
        windows = np.random.randn(5, 200)
        results = compute_time_features_batch(windows)
        assert len(results) == 5
        assert all(isinstance(r, TimeDomainFeatures) for r in results)

    def test_to_dict_keys(self):
        s = np.random.randn(100)
        f = compute_time_features(s)
        d = td_to_dict(f, prefix="vib_")
        assert "vib_rms" in d
        assert "vib_kurtosis" in d
        assert "vib_crest_factor" in d

    def test_energy_positive(self):
        s = np.random.randn(500)
        f = compute_time_features(s)
        assert f.energy > 0


# ─── Frequency-Domain Tests ───────────────────────────────────────────────────

class TestFrequencyDomain:

    def test_fft_returns_positive_freqs(self):
        sr = 1000.0
        t = np.linspace(0, 1, int(sr), endpoint=False)
        s = np.sin(2 * np.pi * 100 * t)
        freqs, psd = compute_fft(s, sampling_rate=sr)
        assert np.all(freqs >= 0)
        assert len(freqs) == len(psd)

    def test_dominant_freq_sine(self):
        sr = 1000.0
        t = np.linspace(0, 1, int(sr), endpoint=False)
        s = np.sin(2 * np.pi * 50 * t)
        f = compute_frequency_features(s, sampling_rate=sr)
        assert abs(f.dominant_freq - 50.0) < 2.0   # within 2 Hz

    def test_spectral_entropy_range(self):
        sr = 1000.0
        s = np.random.randn(1000)
        f = compute_frequency_features(s, sampling_rate=sr)
        assert f.spectral_entropy > 0

    def test_total_power_positive(self):
        sr = 1000.0
        s = np.random.randn(1000)
        f = compute_frequency_features(s, sampling_rate=sr)
        assert f.total_power > 0

    def test_band_energy_computed(self):
        sr = 10_000.0
        s = np.random.randn(10_000)
        bands = [(100.0, 500.0), (500.0, 2000.0)]
        f = compute_frequency_features(s, sampling_rate=sr, band_ranges=bands)
        assert len(f.band_energies) == 2
        for energy in f.band_energies.values():
            assert energy >= 0

    def test_harmonic_energy_computed(self):
        sr = 1000.0
        t = np.linspace(0, 1, int(sr), endpoint=False)
        shaft_hz = 25.0
        s = np.sin(2 * np.pi * shaft_hz * t)  # fundamental only
        f = compute_frequency_features(s, sampling_rate=sr, shaft_speed_hz=shaft_hz, n_harmonics=3)
        assert 1 in f.harmonic_energies
        assert 2 in f.harmonic_energies
        # 1× harmonic energy should be dominant
        assert f.harmonic_energies[1] > f.harmonic_energies.get(2, 0)

    def test_to_dict_keys(self):
        sr = 1000.0
        s = np.random.randn(500)
        f = compute_frequency_features(s, sampling_rate=sr)
        d = fd_to_dict(f, prefix="vib_")
        assert "vib_dominant_freq" in d
        assert "vib_spectral_entropy" in d

    def test_empty_raises(self):
        from src.signal_processing.frequency_domain import compute_frequency_features
        with pytest.raises(ValueError):
            compute_frequency_features(np.array([]), sampling_rate=1000.0)


# ─── Wavelet Tests ────────────────────────────────────────────────────────────

class TestWavelet:

    def test_wavelet_import_or_skip(self):
        try:
            import pywt
        except ImportError:
            pytest.skip("PyWavelets not installed")

    def test_energy_conservation(self):
        try:
            import pywt
        except ImportError:
            pytest.skip("PyWavelets not installed")
        from src.signal_processing.wavelet import compute_wavelet_features
        rng = np.random.default_rng(0)
        s = rng.standard_normal(1024)
        wf = compute_wavelet_features(s, wavelet="db4", level=5)
        # Sum of band energies should equal total_energy
        band_sum = sum(b.energy for b in wf.bands)
        assert band_sum == pytest.approx(wf.total_energy, rel=1e-6)

    def test_band_names(self):
        try:
            import pywt
        except ImportError:
            pytest.skip("PyWavelets not installed")
        from src.signal_processing.wavelet import compute_wavelet_features
        s = np.random.randn(512)
        wf = compute_wavelet_features(s, wavelet="db4", level=3)
        names = {b.name for b in wf.bands}
        assert "A3" in names
        assert "D1" in names

    def test_impulse_kurtosis_high(self):
        try:
            import pywt
        except ImportError:
            pytest.skip("PyWavelets not installed")
        from src.signal_processing.wavelet import compute_wavelet_features
        s = np.zeros(1024)
        s[512] = 50.0
        wf = compute_wavelet_features(s)
        # At least one detail band should have high kurtosis
        detail_kurtosis = [b.kurtosis for b in wf.bands if b.name.startswith("D")]
        assert max(detail_kurtosis) > 10

    def test_energy_ratio_sums_to_one(self):
        try:
            import pywt
        except ImportError:
            pytest.skip("PyWavelets not installed")
        from src.signal_processing.wavelet import compute_wavelet_features
        s = np.random.randn(512)
        wf = compute_wavelet_features(s)
        ratio_sum = sum(b.energy_ratio for b in wf.bands)
        assert ratio_sum == pytest.approx(1.0, rel=1e-6)


# ─── Sliding Window Tests ─────────────────────────────────────────────────────

class TestSlidingWindow:

    def test_n_windows_no_overlap(self):
        from src.feature_engineering.window import sliding_window
        readings = _make_readings(100)
        vectors = sliding_window(readings, window_size=10, stride=10, min_samples=5)
        assert len(vectors) == 10

    def test_n_windows_stride_1(self):
        from src.feature_engineering.window import sliding_window
        readings = _make_readings(50)
        vectors = sliding_window(readings, window_size=10, stride=1, min_samples=5)
        # Should produce 50 - 10 + 1 = 41 windows
        assert len(vectors) == 41

    def test_window_machine_id_preserved(self):
        from src.feature_engineering.window import sliding_window
        readings = _make_readings(30, machine_id="TEST-999")
        vectors = sliding_window(readings, window_size=10, stride=10, min_samples=5)
        assert all(v.machine_id == "TEST-999" for v in vectors)

    def test_too_few_readings_returns_empty(self):
        from src.feature_engineering.window import sliding_window
        readings = _make_readings(3)
        vectors = sliding_window(readings, window_size=10, stride=1, min_samples=5)
        assert vectors == []

    def test_feature_vector_fields_populated(self):
        from src.feature_engineering.window import sliding_window
        readings = _make_readings(50)
        vectors = sliding_window(readings, window_size=20, stride=20, min_samples=10)
        assert len(vectors) > 0
        fv = vectors[0]
        assert fv.vib_rms > 0
        assert not math.isnan(fv.vib_kurtosis)
        assert fv.n_samples == 20


# ─── Multimodal Tests ─────────────────────────────────────────────────────────

class TestMultimodal:

    def _make_fvs(self, n_machines: int = 5, n_per_machine: int = 20) -> list[FeatureVector]:
        fvs = []
        for i in range(n_machines):
            for j in range(n_per_machine):
                fvs.append(_make_feature_vector(machine_id=f"M-{i:03d}", rul=float(100 - j)))
        return fvs

    def test_dataframe_shape(self):
        fvs = self._make_fvs(n_machines=3, n_per_machine=10)
        df = feature_vectors_to_dataframe(fvs)
        assert len(df) == 30
        assert "machine_id" in df.columns
        assert "vib_rms" in df.columns

    def test_machine_aware_split_no_leakage(self):
        fvs = self._make_fvs(n_machines=10, n_per_machine=5)
        df = feature_vectors_to_dataframe(fvs)
        train, test = machine_aware_split(df, test_machine_fraction=0.2, random_state=0)
        train_machines = set(train["machine_id"].unique())
        test_machines  = set(test["machine_id"].unique())
        assert train_machines.isdisjoint(test_machines)

    def test_impute_no_nans(self):
        fvs = self._make_fvs()
        df = feature_vectors_to_dataframe(fvs)
        # Inject NaN into training data
        df.loc[0, "vib_rms"] = float("nan")
        train, test = machine_aware_split(df)
        train_imp, test_imp, fill = impute_features(train, test)
        assert train_imp["vib_rms"].isna().sum() == 0

    def test_normalise_zero_mean(self):
        fvs = self._make_fvs(n_machines=10, n_per_machine=20)
        df = feature_vectors_to_dataframe(fvs)
        train, test = machine_aware_split(df)
        train_n, _, _ = normalise_features(train, test)
        # Training RMS column should have mean ≈ 0 after normalisation
        assert abs(train_n["vib_rms"].mean()) < 0.1

    def test_get_feature_cols_non_empty(self):
        cols = get_feature_cols()
        assert len(cols) > 0
        assert "vib_rms" in cols
        assert "machine_id" not in cols

    def test_feature_matrix_shape(self):
        fvs = self._make_fvs(n_machines=2, n_per_machine=5)
        df = feature_vectors_to_dataframe(fvs)
        X = get_feature_matrix(df)
        assert X.shape[0] == 10
        assert X.shape[1] > 0


# ─── Feature Store Tests ──────────────────────────────────────────────────────

class TestFeatureStore:

    def test_save_load_roundtrip(self, tmp_path):
        from src.feature_engineering.feature_store import FeatureStore
        fvs = [_make_feature_vector("M-001"), _make_feature_vector("M-002")]
        store = FeatureStore(base_dir=tmp_path)
        store.save(fvs, dataset_name="test_ds")
        df = store.load_as_dataframe("test_ds")
        assert len(df) == 2
        assert set(df["machine_id"].unique()) == {"M-001", "M-002"}

    def test_list_datasets(self, tmp_path):
        from src.feature_engineering.feature_store import FeatureStore
        fvs = [_make_feature_vector()]
        store = FeatureStore(base_dir=tmp_path)
        store.save(fvs, dataset_name="ds_alpha")
        store.save(fvs, dataset_name="ds_beta")
        datasets = store.list_datasets()
        assert "ds_alpha" in datasets
        assert "ds_beta" in datasets

    def test_load_missing_raises(self, tmp_path):
        from src.feature_engineering.feature_store import FeatureStore
        store = FeatureStore(base_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            store.load_as_dataframe("does_not_exist")

    def test_delete(self, tmp_path):
        from src.feature_engineering.feature_store import FeatureStore
        fvs = [_make_feature_vector()]
        store = FeatureStore(base_dir=tmp_path)
        store.save(fvs, dataset_name="to_delete")
        assert "to_delete" in store.list_datasets()
        store.delete("to_delete")
        assert "to_delete" not in store.list_datasets()

    def test_empty_save_warning(self, tmp_path, caplog):
        import logging
        from src.feature_engineering.feature_store import FeatureStore
        store = FeatureStore(base_dir=tmp_path)
        with caplog.at_level(logging.WARNING):
            store.save([], dataset_name="empty_ds")
        assert "empty" in caplog.text.lower()
