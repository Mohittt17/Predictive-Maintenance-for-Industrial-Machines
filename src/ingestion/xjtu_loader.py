"""
XJTU-SY Bearing Dataset loader.

The XJTU-SY dataset contains accelerated bearing run-to-failure data from
Xi'an Jiaotong University. Two accelerometers (horizontal + vertical) are
mounted on each bearing housing, sampled at 25.6 kHz, with 32,768 samples
per 1.28-second file.

Dataset structure
-----------------
Three operating conditions (25 Hz / 35 Hz / 45 Hz shaft speed):
  Condition 1 (2100 RPM): Bearings 1_1 to 1_5
  Condition 2 (2250 RPM): Bearings 2_1 to 2_5
  Condition 3 (2400 RPM): Bearings 3_1 to 3_5

Each bearing directory contains: Bearing_i_j/
  └── <minute>.csv  (two columns: horizontal_acc, vertical_acc)

RUL is computed from the number of remaining minute-files until the last
recorded file (failure).

References
----------
Wang, B. et al. (2018). Hybrid prognostics of aluminium electrolytic
capacitors in DC–DC converters. IEEE Transactions on Industrial Electronics.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.ingestion.schema import RawDataset, SensorReading
from src.utils.config import XJTU_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SAMPLING_RATE_HZ = 25_600.0
_CONDITION_RPM = {
    "Condition_1": 2100.0,
    "Condition_2": 2250.0,
    "Condition_3": 2400.0,
}
_CONDITION_LOAD_NM = {     # radial load in N (for metadata)
    "Condition_1": 12.0,
    "Condition_2": 11.0,
    "Condition_3": 10.0,
}


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


def _kurtosis_val(arr: np.ndarray) -> float:
    sigma = np.std(arr)
    if sigma == 0:
        return 0.0
    return float(np.mean(((arr - np.mean(arr)) / sigma) ** 4))


def _crest_factor(arr: np.ndarray) -> float:
    rms_val = _rms(arr)
    if rms_val == 0:
        return 0.0
    return float(np.max(np.abs(arr)) / rms_val)


def _load_minute_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (horizontal, vertical) acceleration arrays from one minute file."""
    try:
        df = pd.read_csv(path, header=0)
        horiz = df.iloc[:, 0].values.astype(np.float32)
        vert  = df.iloc[:, 1].values.astype(np.float32)
        return horiz, vert
    except Exception as e:
        logger.warning(f"Failed to parse {path.name}: {e}")
        n = 32_768
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)


def _rul_to_health_label(remaining: int, total: int) -> int:
    frac = remaining / max(total, 1)
    if frac > 0.80:
        return 0
    if frac > 0.50:
        return 1
    if frac > 0.25:
        return 2
    if frac > 0.05:
        return 3
    return 4


def load_xjtu(
    condition: str = "Condition_1",
    bearing: str = "Bearing1_1",
    data_dir: Optional[Path] = None,
    start_time: Optional[datetime] = None,
) -> RawDataset:
    """
    Load one XJTU-SY bearing run.

    Args:
        condition:  "Condition_1", "Condition_2", or "Condition_3".
        bearing:    Bearing sub-directory name, e.g. "Bearing1_1".
        data_dir:   Root directory containing the condition sub-directories.
        start_time: Synthetic start timestamp (default: 2024-01-01 UTC).

    Returns:
        :class:`RawDataset` with one SensorReading per minute file.
    """
    data_dir   = data_dir or XJTU_DIR
    bear_dir   = Path(data_dir) / condition / bearing

    if not bear_dir.exists():
        raise FileNotFoundError(
            f"XJTU-SY directory not found: {bear_dir}\n"
            "→ Run the download instructions in data/README.md first."
        )

    minute_files = sorted(
        [f for f in bear_dir.glob("*.csv")],
        key=lambda f: int(f.stem) if f.stem.isdigit() else 0,
    )
    if not minute_files:
        raise FileNotFoundError(f"No CSV files in {bear_dir}")

    logger.info(
        f"Loading XJTU-SY {condition}/{bearing}: {len(minute_files)} minute files"
    )

    rpm   = _CONDITION_RPM.get(condition, 2100.0)
    total = len(minute_files)
    start = start_time or datetime(2024, 1, 1, tzinfo=timezone.utc)
    machine_id = f"XJTU-{condition}-{bearing}"

    readings: list[SensorReading] = []
    for idx, mf in enumerate(minute_files):
        ts      = start + timedelta(minutes=idx)
        h, v    = _load_minute_file(mf)
        remaining = total - 1 - idx
        rul_hours = remaining / 60.0   # each file = 1 minute

        reading = SensorReading(
            machine_id=machine_id,
            timestamp=ts,
            vibration_x=_rms(h),
            vibration_y=_rms(v),
            vibration_z=_kurtosis_val(h),
            temperature=0.0,         # XJTU has no temp sensor
            current=0.0,
            pressure=0.0,
            acoustic=_crest_factor(h),    # crest factor as acoustic proxy
            rpm=rpm,
            load=100.0,
            rul_hours=rul_hours,
            health_label=_rul_to_health_label(remaining, total),
            dataset=f"XJTU-{condition}-{bearing}",
        )
        readings.append(reading)

    logger.info(f"XJTU {machine_id}: {len(readings)} readings loaded")

    return RawDataset(
        name=machine_id,
        readings=readings,
        machine_ids=[machine_id],
        metadata={
            "condition": condition,
            "bearing": bearing,
            "rpm": rpm,
            "sampling_rate_hz": _SAMPLING_RATE_HZ,
            "total_minutes": total,
            "source": "XJTU-SY Bearing Dataset",
        },
    )


def load_all_xjtu(data_dir: Optional[Path] = None) -> dict[str, RawDataset]:
    """
    Attempt to load all XJTU-SY bearings across all conditions.
    Missing directories are silently skipped with a warning.
    """
    results: dict[str, RawDataset] = {}
    for cond_num in range(1, 4):
        cond = f"Condition_{cond_num}"
        for bear_num in range(1, 6):
            bearing = f"Bearing{cond_num}_{bear_num}"
            key = f"{cond}-{bearing}"
            try:
                results[key] = load_xjtu(condition=cond, bearing=bearing, data_dir=data_dir)
            except FileNotFoundError as e:
                warnings.warn(str(e), stacklevel=2)
    return results
