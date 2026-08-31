"""
NASA IMS Bearing Dataset loader.

The IMS dataset contains continuous vibration data from four bearings on a
rotating shaft running at 2000 RPM, recorded until failure.

Dataset structure
-----------------
Three test-to-failure runs:
  Set 1: Bearings 1–4  (ch1=B1_x, ch2=B2_x, ch3=B3_x, ch4=B4_x)
  Set 2: Bearings 1–4  (ch1–4)
  Set 3: Bearings 1–4  (ch1–4)

Each file is a 1-second snapshot at 20,480 Hz → 20,480 samples.
Files are named as timestamps: YYYY.MM.DD.HH.mm.ss

Failure info
------------
  Set 1: Bearing 3 (outer race) and Bearing 4 (rolling element)
  Set 2: Bearing 1 (outer race)
  Set 3: Bearing 3 (outer race)

We extract per-file statistical features (not raw waveforms) for the
canonical schema to keep memory tractable.


References
----------
Qiu, H. et al. (2006). Wavelet filter-based weak signature detection method
and its application on rolling element bearing prognostics. JSV.
"""
from __future__ import annotations

import re
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.ingestion.schema import RawDataset, SensorReading
from src.utils.config import IMS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SAMPLING_RATE_HZ = 20_480.0
_MACHINE_RPM = 2000.0
_N_CHANNELS = 4   # one per bearing

# Failure timestamps used to compute RUL per bearing/run
# (approximate — derived from original paper and dataset README)
_FAILURE_TIMESTAMPS: dict[str, dict[int, str]] = {
    "Set1": {
        3: "2003.11.25.23.39.56",
        4: "2003.11.25.23.39.56",
    },
    "Set2": {
        1: "2004.02.19.06.22.39",
    },
    "Set3": {
        3: "2004.04.08.09.27.46",
    },
}

_TS_FMT = "%Y.%m.%d.%H.%M.%S"


def _parse_timestamp(filename: str) -> Optional[datetime]:
    """Parse an IMS snapshot filename into a UTC datetime.

    IMS snapshot files are named as bare timestamps (e.g. ``2003.10.22.12.06.24``)
    with no file extension.  Using ``Path.stem`` would incorrectly strip the last
    numeric segment as an "extension", so we strip only a trailing ``.csv`` (if
    present) and parse the remainder verbatim.
    """
    name = Path(filename).name
    # Strip .csv suffix if present (e.g. from synthetic/test fixtures)
    if name.lower().endswith(".csv"):
        name = name[:-4]
    try:
        return datetime.strptime(name, _TS_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)))


def _kurtosis_val(signal: np.ndarray) -> float:
    mu = np.mean(signal)
    sigma = np.std(signal)
    if sigma == 0:
        return 0.0
    return float(np.mean(((signal - mu) / sigma) ** 4))


def _load_snapshot(path: Path) -> np.ndarray:
    """Load one IMS snapshot file → shape (samples, channels)."""
    try:
        data = pd.read_csv(path, sep=r"\s+", header=None).values.astype(np.float32)
        return data
    except Exception as e:
        logger.warning(f"Could not parse {path.name}: {e}")
        return np.zeros((20_480, _N_CHANNELS), dtype=np.float32)


def _compute_rul_map(
    timestamps: list[datetime],
    failure_ts: Optional[datetime],
) -> list[Optional[float]]:
    """
    Compute RUL in hours for each snapshot given a failure timestamp.
    Returns None for each snapshot if failure timestamp is unknown.
    """
    if failure_ts is None:
        return [None] * len(timestamps)

    total_seconds = (failure_ts - timestamps[0]).total_seconds()
    ruls = []
    for ts in timestamps:
        elapsed = (ts - timestamps[0]).total_seconds()
        rul_secs = max(0.0, total_seconds - elapsed)
        ruls.append(rul_secs / 3600.0)  # → hours
    return ruls


def _rul_to_health_label(rul: Optional[float], total_hours: float) -> Optional[int]:
    """Map RUL fraction to health label 0–4."""
    if rul is None or total_hours <= 0:
        return None
    frac = rul / total_hours
    if frac > 0.80:
        return 0   # Healthy
    if frac > 0.50:
        return 1   # Minor degradation
    if frac > 0.25:
        return 2   # Moderate
    if frac > 0.05:
        return 3   # Severe
    return 4       # Critical / failure


def load_ims(
    run: str = "Set1",
    bearing_channel: int = 1,
    data_dir: Optional[Path] = None,
) -> RawDataset:
    """
    Load one run/bearing combination from the IMS dataset.

    Args:
        run:              "Set1", "Set2", or "Set3".
        bearing_channel:  1–4 (channel index).
        data_dir:         Root directory containing Set1/, Set2/, Set3/.

    Returns:
        :class:`RawDataset` with one SensorReading per snapshot file,
        where each reading contains statistical summaries of the 1-second
        waveform.

    Raises:
        FileNotFoundError: If the run directory does not exist.
    """
    data_dir = data_dir or IMS_DIR
    run_dir  = Path(data_dir) / run

    if not run_dir.exists():
        raise FileNotFoundError(
            f"IMS directory not found: {run_dir}\n"
            "→ Run the download instructions in data/README.md first."
        )

    snapshot_files = sorted(
        [f for f in run_dir.iterdir() if re.match(r"\d{4}\.\d{2}\.\d{2}", f.name)],
        key=lambda f: f.name,
    )
    if not snapshot_files:
        raise FileNotFoundError(f"No snapshot files found in {run_dir}")

    logger.info(f"Loading IMS {run} bearing {bearing_channel}: {len(snapshot_files)} snapshots")

    timestamps = [_parse_timestamp(f.name) for f in snapshot_files]
    valid_ts   = [ts for ts in timestamps if ts is not None]

    # ── RUL computation ───────────────────────────────────────────────────────
    failure_ts_str = _FAILURE_TIMESTAMPS.get(run, {}).get(bearing_channel)
    failure_ts: Optional[datetime] = None
    if failure_ts_str:
        failure_ts = datetime.strptime(failure_ts_str, _TS_FMT).replace(tzinfo=timezone.utc)

    rul_list = _compute_rul_map(valid_ts, failure_ts)
    total_hours = (valid_ts[-1] - valid_ts[0]).total_seconds() / 3600.0 if valid_ts else 1.0

    machine_id = f"IMS-{run}-B{bearing_channel}"
    readings: list[SensorReading] = []
    chan_idx = bearing_channel - 1  # 0-indexed

    for i, (snap_file, ts) in enumerate(zip(snapshot_files, timestamps)):
        if ts is None:
            continue
        waveform = _load_snapshot(snap_file)
        if waveform.shape[1] <= chan_idx:
            continue
        channel = waveform[:, chan_idx]

        rul_val = rul_list[i]

        reading = SensorReading(
            machine_id=machine_id,
            timestamp=ts,
            # Map vibration channel RMS to vibration_x; raw waveform not stored
            vibration_x=_rms(channel),
            vibration_y=_kurtosis_val(channel),         # kurtosis as y proxy
            vibration_z=float(np.max(np.abs(channel))), # peak as z proxy
            temperature=0.0,     # IMS has no temperature channel
            current=0.0,
            pressure=0.0,
            acoustic=_rms(channel),   # reuse vibration RMS as acoustic proxy
            rpm=_MACHINE_RPM,
            load=100.0,              # constant load in IMS
            rul_hours=rul_val,
            health_label=_rul_to_health_label(rul_val, total_hours),
            dataset=f"IMS-{run}-B{bearing_channel}",
        )
        readings.append(reading)

    logger.info(
        f"IMS {run} B{bearing_channel}: {len(readings)} readings, "
        f"duration ≈ {total_hours:.1f}h"
    )

    return RawDataset(
        name=machine_id,
        readings=readings,
        machine_ids=[machine_id],
        metadata={
            "run": run,
            "bearing_channel": bearing_channel,
            "sampling_rate_hz": _SAMPLING_RATE_HZ,
            "rpm": _MACHINE_RPM,
            "failure_bearing": bearing_channel in _FAILURE_TIMESTAMPS.get(run, {}),
            "failure_timestamp": failure_ts_str,
            "source": "NASA IMS Bearing Dataset",
        },
    )


def load_all_ims(data_dir: Optional[Path] = None) -> dict[str, RawDataset]:
    """
    Load all available IMS runs and bearings.

    Returns
    -------
    Dict mapping "<run>-B<channel>" → RawDataset.
    Missing runs are skipped with a warning.
    """
    results: dict[str, RawDataset] = {}
    for run, bearings in _FAILURE_TIMESTAMPS.items():
        for b in range(1, 5):
            key = f"{run}-B{b}"
            try:
                results[key] = load_ims(run=run, bearing_channel=b, data_dir=data_dir)
            except FileNotFoundError as e:
                warnings.warn(str(e), stacklevel=2)
    return results
