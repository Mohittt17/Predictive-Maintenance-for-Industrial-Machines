"""
Unit tests for dataset loaders (Milestone 1).

These tests verify:
  1. Schema correctness — every field is present and typed correctly.
  2. RUL sanity — values are finite, non-negative, and within expected range.
  3. Health label correctness — labels map to 0–4 only.
  4. Shape / ordering — machine_ids are populated; readings are non-empty.
  5. FileNotFoundError behaviour — graceful error when data is absent.

Tests run with synthetic / fixture data so they pass WITHOUT the actual
NASA or XJTU downloads.  Once real data is downloaded, remove the
@pytest.mark.skipif decorators on the integration tests at the bottom.
"""
from __future__ import annotations

import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ingestion.schema import (
    RawDataset,
    SensorReading,
    FeatureVector,
    MaintenanceRecommendation,
)
from src.ingestion.cmapss_loader import (
    _compute_train_rul,
    _rul_to_health_label,
    _read_txt,
    to_dataframe,
)
from src.ingestion.ims_loader import _rms, _kurtosis_val, _parse_timestamp
from src.ingestion.xjtu_loader import _crest_factor, _rul_to_health_label as xjtu_health_label


# ─── Schema Tests ─────────────────────────────────────────────────────────────

class TestSensorReadingSchema:
    """SensorReading dataclass contract."""

    def test_default_optional_fields(self):
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        r = SensorReading(
            machine_id="TEST-001", timestamp=ts,
            vibration_x=0.1, vibration_y=0.2, vibration_z=0.3,
            temperature=75.0, current=5.0, pressure=2.5, acoustic=40.0,
            rpm=1500.0, load=80.0,
        )
        assert r.rul_hours is None
        assert r.health_label is None
        assert r.dataset == "unknown"

    def test_all_fields_set(self):
        ts = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        r = SensorReading(
            machine_id="M-07", timestamp=ts,
            vibration_x=1.5, vibration_y=0.8, vibration_z=0.3,
            temperature=92.0, current=7.2, pressure=3.1, acoustic=55.0,
            rpm=2100.0, load=95.0,
            rul_hours=56.0, health_label=3, dataset="IMS-Set1-B3",
        )
        assert r.machine_id == "M-07"
        assert r.rul_hours == pytest.approx(56.0)
        assert r.health_label == 3
        assert r.dataset == "IMS-Set1-B3"

    def test_health_label_range(self):
        for label in range(5):
            r = SensorReading(
                machine_id="X", timestamp=datetime.now(timezone.utc),
                vibration_x=0, vibration_y=0, vibration_z=0,
                temperature=0, current=0, pressure=0, acoustic=0,
                rpm=0, load=0, health_label=label,
            )
            assert 0 <= r.health_label <= 4


class TestFeatureVectorSchema:
    """FeatureVector dataclass — nan defaults and field presence."""

    def test_nan_defaults(self):
        fv = FeatureVector(
            machine_id="X",
            window_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            window_end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            n_samples=30,
        )
        assert math.isnan(fv.vib_rms)
        assert math.isnan(fv.health_index)
        assert fv.rul_hours is None

    def test_set_features(self):
        fv = FeatureVector(
            machine_id="M-07",
            window_start=datetime(2024, 1, 1, tzinfo=timezone.utc),
            window_end=datetime(2024, 1, 1, tzinfo=timezone.utc),
            n_samples=30,
            vib_rms=0.045,
            vib_kurtosis=3.21,
            health_index=72.5,
            rul_hours=80.0,
        )
        assert fv.vib_rms == pytest.approx(0.045)
        assert fv.health_index == pytest.approx(72.5)


class TestRawDatasetSchema:
    """RawDataset container properties."""

    def _make_dataset(self, n: int) -> RawDataset:
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        readings = [
            SensorReading(
                machine_id=f"M-{i:03d}", timestamp=ts,
                vibration_x=0.1, vibration_y=0.1, vibration_z=0.1,
                temperature=70.0, current=4.0, pressure=2.0, acoustic=35.0,
                rpm=1500.0, load=80.0,
            )
            for i in range(n)
        ]
        machine_ids = list({r.machine_id for r in readings})
        return RawDataset(name="test", readings=readings, machine_ids=machine_ids)

    def test_n_readings(self):
        ds = self._make_dataset(100)
        assert ds.n_readings == 100

    def test_n_machines(self):
        ds = self._make_dataset(50)
        assert ds.n_machines == 50   # all different IDs

    def test_empty_dataset(self):
        ds = RawDataset(name="empty", readings=[], machine_ids=[])
        assert ds.n_readings == 0
        assert ds.n_machines == 0


# ─── C-MAPSS Helper Tests ────────────────────────────────────────────────────

class TestCmapssHelpers:
    """Unit tests for C-MAPSS internal helper functions."""

    def _make_df(self, n_units: int = 3, cycles_per_unit: int = 10) -> pd.DataFrame:
        """Minimal C-MAPSS-style DataFrame."""
        rows = []
        for uid in range(1, n_units + 1):
            for c in range(1, cycles_per_unit + 1):
                rows.append({"unit": uid, "cycle": c})
        return pd.DataFrame(rows)

    def test_compute_train_rul_last_cycle_is_zero(self):
        df = self._make_df(n_units=2, cycles_per_unit=20)
        rul = _compute_train_rul(df, max_rul=125.0)
        # Last cycle of each unit should have RUL = 0
        for uid in [1, 2]:
            last_idx = df[df["unit"] == uid].index[-1]
            assert rul[last_idx] == pytest.approx(0.0)

    def test_compute_train_rul_first_cycle_clipped(self):
        df = self._make_df(n_units=1, cycles_per_unit=200)
        rul = _compute_train_rul(df, max_rul=125.0)
        assert rul.max() == pytest.approx(125.0)

    def test_rul_to_health_label_boundaries(self):
        assert _rul_to_health_label(100.0) == 0   # healthy
        assert _rul_to_health_label(60.0)  == 1   # minor
        assert _rul_to_health_label(30.0)  == 2   # moderate
        assert _rul_to_health_label(10.0)  == 3   # severe
        assert _rul_to_health_label(2.0)   == 4   # critical
        assert _rul_to_health_label(0.0)   == 4   # failure

    def test_rul_to_health_label_none(self):
        assert _rul_to_health_label(None) is None

    def test_to_dataframe_columns(self):
        """to_dataframe should produce a DataFrame with the canonical columns."""
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
        readings = [
            SensorReading(
                machine_id="X-001", timestamp=ts,
                vibration_x=0.1, vibration_y=0.2, vibration_z=0.3,
                temperature=80.0, current=5.0, pressure=2.5, acoustic=45.0,
                rpm=1500.0, load=80.0, rul_hours=50.0, health_label=1,
            )
        ]
        ds = RawDataset(name="test", readings=readings, machine_ids=["X-001"])
        df = to_dataframe(ds)
        expected_cols = {
            "machine_id", "timestamp", "vibration_x", "vibration_y", "vibration_z",
            "temperature", "current", "pressure", "acoustic", "rpm", "load",
            "rul_hours", "health_label", "dataset",
        }
        assert expected_cols.issubset(set(df.columns))
        assert len(df) == 1


# ─── IMS Helper Tests ─────────────────────────────────────────────────────────

class TestIMSHelpers:
    """Unit tests for IMS internal helpers."""

    def test_rms_zero(self):
        arr = np.zeros(100)
        assert _rms(arr) == pytest.approx(0.0)

    def test_rms_constant(self):
        arr = np.ones(100) * 3.0
        assert _rms(arr) == pytest.approx(3.0)

    def test_rms_sine(self):
        t = np.linspace(0, 2 * np.pi, 1000)
        sine = np.sin(t)
        # RMS of sin is 1/sqrt(2) ≈ 0.707
        assert _rms(sine) == pytest.approx(1.0 / np.sqrt(2), rel=1e-2)

    def test_kurtosis_gaussian(self):
        rng = np.random.default_rng(42)
        arr = rng.standard_normal(10_000)
        # Kurtosis of a Gaussian should be ~3
        k = _kurtosis_val(arr)
        assert abs(k - 3.0) < 0.2

    def test_kurtosis_zero_std(self):
        arr = np.ones(50)
        assert _kurtosis_val(arr) == pytest.approx(0.0)

    def test_parse_timestamp_valid(self):
        ts = _parse_timestamp("2003.10.22.12.06.24")
        assert ts is not None
        assert ts.year == 2003
        assert ts.month == 10
        assert ts.day == 22

    def test_parse_timestamp_invalid(self):
        ts = _parse_timestamp("not_a_timestamp.csv")
        assert ts is None


# ─── XJTU Helper Tests ────────────────────────────────────────────────────────

class TestXJTUHelpers:
    """Unit tests for XJTU-SY helpers."""

    def test_crest_factor_sine(self):
        t = np.linspace(0, 2 * np.pi, 1000)
        sine = np.sin(t)
        # Crest factor of sin = peak / RMS = 1 / (1/sqrt(2)) = sqrt(2)
        cf = _crest_factor(sine)
        assert cf == pytest.approx(np.sqrt(2), rel=1e-2)

    def test_crest_factor_zero(self):
        assert _crest_factor(np.zeros(100)) == pytest.approx(0.0)

    def test_xjtu_health_label_healthy(self):
        assert xjtu_health_label(remaining=95, total=100) == 0

    def test_xjtu_health_label_failure(self):
        assert xjtu_health_label(remaining=0, total=100) == 4

    def test_xjtu_health_label_moderate(self):
        assert xjtu_health_label(remaining=30, total=100) == 2


# ─── FileNotFoundError Tests ──────────────────────────────────────────────────

class TestLoaderFileNotFound:
    """Loaders must raise FileNotFoundError when data is missing."""

    def test_cmapss_missing_raises(self, tmp_path):
        from src.ingestion.cmapss_loader import load_cmapss
        with pytest.raises(FileNotFoundError, match="data/README.md"):
            load_cmapss(subset="FD001", data_dir=tmp_path)

    def test_ims_missing_raises(self, tmp_path):
        from src.ingestion.ims_loader import load_ims
        with pytest.raises(FileNotFoundError):
            load_ims(run="Set1", bearing_channel=1, data_dir=tmp_path)

    def test_xjtu_missing_raises(self, tmp_path):
        from src.ingestion.xjtu_loader import load_xjtu
        with pytest.raises(FileNotFoundError):
            load_xjtu(condition="Condition_1", bearing="Bearing1_1", data_dir=tmp_path)


# ─── Integration Tests (skipped if data not downloaded) ──────────────────────

@pytest.mark.skip(reason="Requires NASA C-MAPSS download — see data/README.md")
def test_cmapss_fd001_integration():
    from src.ingestion.cmapss_loader import load_cmapss
    train, test = load_cmapss("FD001")
    assert train.n_readings > 0
    assert train.n_machines == 100
    assert all(r.rul_hours is not None for r in train.readings)
    assert all(0 <= r.health_label <= 4 for r in train.readings if r.health_label is not None)


@pytest.mark.skip(reason="Requires NASA IMS download — see data/README.md")
def test_ims_set1_bearing1_integration():
    from src.ingestion.ims_loader import load_ims
    ds = load_ims("Set1", bearing_channel=1)
    assert ds.n_readings > 0
    assert len(ds.machine_ids) == 1


@pytest.mark.skip(reason="Requires XJTU-SY download — see data/README.md")
def test_xjtu_condition1_integration():
    from src.ingestion.xjtu_loader import load_xjtu
    ds = load_xjtu("Condition_1", "Bearing1_1")
    assert ds.n_readings > 0
