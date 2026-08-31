"""
NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) loader.

The C-MAPSS dataset contains run-to-failure simulation data for turbofan
engines across four sub-datasets (FD001–FD004).

File format (space-delimited, no header)
-----------------------------------------
Col  1     : unit number (machine_id)
Col  2     : time in cycles
Cols 3–5   : operational settings (op1, op2, op3) → mapped to rpm, load, pressure
Cols 6–26  : 21 sensor measurements

We use the following sensor mappings to the canonical schema:
    sensor_2  → temperature (LPC outlet temperature)
    sensor_4  → temperature (HPC outlet temperature) — averaged
    sensor_7  → pressure    (HPC outlet pressure)
    sensor_11 → vibration_x (bypass ratio proxy)
    sensor_12 → vibration_y (bleed enthalpy proxy)
    sensor_14 → vibration_z (HPT coolant bleed proxy)
    sensor_21 → acoustic    (ratio of burner fuel to air)
    sensor_9  → current     (physical fan speed proxy)

Ground-truth RUL
-----------------
The test set RUL is provided in RUL_FDxxx.txt files (one value per engine).
For the training set we compute RUL as: max_cycle - current_cycle.

References
----------
Saxena, A. et al. (2008). Damage propagation modeling for aircraft engine
run-to-failure simulation. ICIIS.
"""
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.ingestion.schema import RawDataset, SensorReading
from src.utils.config import CMAPSS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Column names from the C-MAPSS README
_HEADER = [
    "unit", "cycle",
    "op1", "op2", "op3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

# Sensor-to-canonical mapping  (sensor_index → canonical_field)
_SENSOR_MAP = {
    "sensor_2":  "temperature",
    "sensor_7":  "pressure",
    "sensor_9":  "current",
    "sensor_11": "vibration_x",
    "sensor_12": "vibration_y",
    "sensor_14": "vibration_z",
    "sensor_21": "acoustic",
}

# Operational-setting mapping
_OP_MAP = {
    "op1": "rpm",
    "op2": "load",
}

_SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def _read_txt(path: Path) -> pd.DataFrame:
    """Read a space-delimited C-MAPSS file into a DataFrame."""
    df = pd.read_csv(path, sep=r"\s+", header=None, names=_HEADER, engine="python")
    df.dropna(axis=1, how="all", inplace=True)   # trailing empty cols
    return df


def _compute_train_rul(df: pd.DataFrame, max_rul: float = 125.0) -> pd.Series:
    """
    Compute piece-wise linear RUL for training set rows.
    RUL is clipped at max_rul (standard C-MAPSS convention).
    """
    max_cycles = df.groupby("unit")["cycle"].max()
    rul = df.apply(lambda r: max_cycles[r["unit"]] - r["cycle"], axis=1)
    return rul.clip(upper=max_rul).astype(float)


def _df_to_readings(
    df: pd.DataFrame,
    subset: str,
    split: str,
) -> list[SensorReading]:
    """Convert a C-MAPSS DataFrame into canonical SensorReading objects."""
    readings: list[SensorReading] = []

    for _, row in df.iterrows():
        # Synthesise a pseudo-timestamp: treat each cycle as 1 minute
        ts = datetime(2024, 1, 1, tzinfo=timezone.utc).replace(
            second=0,
            microsecond=int(row["cycle"]) * 60 * 1_000_000 % 1_000_000,
        )

        reading = SensorReading(
            machine_id=f"CMAPSS-{subset}-{int(row['unit']):04d}",
            timestamp=ts,
            vibration_x=float(row.get("sensor_11", 0.0)),
            vibration_y=float(row.get("sensor_12", 0.0)),
            vibration_z=float(row.get("sensor_14", 0.0)),
            temperature=float(row.get("sensor_2", 0.0)),
            current=float(row.get("sensor_9", 0.0)),
            pressure=float(row.get("sensor_7", 0.0)),
            acoustic=float(row.get("sensor_21", 0.0)),
            rpm=float(row.get("op1", 0.0)),
            load=float(row.get("op2", 0.0)),
            rul_hours=float(row["rul"]) if "rul" in row and not np.isnan(row["rul"]) else None,
            health_label=_rul_to_health_label(row.get("rul")),
            dataset=f"CMAPSS-{subset}-{split}",
        )
        readings.append(reading)

    return readings


def _rul_to_health_label(rul: Optional[float]) -> Optional[int]:
    """
    Map RUL (cycles) to a coarse health label:
      0 = Healthy (RUL > 80)
      1 = Minor degradation (40 < RUL ≤ 80)
      2 = Moderate (20 < RUL ≤ 40)
      3 = Severe (5 < RUL ≤ 20)
      4 = Critical / failure (RUL ≤ 5)
    """
    if rul is None or np.isnan(rul):
        return None
    if rul > 80:
        return 0
    if rul > 40:
        return 1
    if rul > 20:
        return 2
    if rul > 5:
        return 3
    return 4


def load_cmapss(
    subset: str = "FD001",
    data_dir: Optional[Path] = None,
    max_rul: float = 125.0,
) -> tuple[RawDataset, RawDataset]:
    """
    Load one C-MAPSS sub-dataset (train + test splits).

    Args:
        subset:   One of "FD001", "FD002", "FD003", "FD004".
        data_dir: Path to the directory containing the raw .txt files.
                  Defaults to ``data/raw/cmapss/``.
        max_rul:  Maximum RUL cap applied to the training set (standard = 125).

    Returns:
        Tuple of (train_dataset, test_dataset) as :class:`RawDataset`.

    Raises:
        FileNotFoundError: If the expected files are not found.
    """
    if subset not in _SUBSETS:
        raise ValueError(f"subset must be one of {_SUBSETS}; got '{subset}'")

    data_dir = data_dir or CMAPSS_DIR
    data_dir = Path(data_dir)

    train_path = data_dir / f"train_{subset}.txt"
    test_path  = data_dir / f"test_{subset}.txt"
    rul_path   = data_dir / f"RUL_{subset}.txt"

    for p in (train_path, test_path, rul_path):
        if not p.exists():
            raise FileNotFoundError(
                f"C-MAPSS file not found: {p}\n"
                "→ Run the download instructions in data/README.md first."
            )

    logger.info(f"Loading C-MAPSS {subset} from {data_dir}")

    # ── Training set ──────────────────────────────────────────────────────────
    train_df = _read_txt(train_path)
    train_df["rul"] = _compute_train_rul(train_df, max_rul=max_rul)
    train_readings = _df_to_readings(train_df, subset, "train")

    # ── Test set ──────────────────────────────────────────────────────────────
    test_df = _read_txt(test_path)
    rul_values = pd.read_csv(rul_path, header=None, names=["rul"]).clip(upper=max_rul)

    # Attach RUL to the last row of each unit in the test set
    test_df["rul"] = float("nan")
    unit_ids = test_df["unit"].unique()
    for uid, rul_val in zip(unit_ids, rul_values["rul"].values):
        last_idx = test_df[test_df["unit"] == uid].index[-1]
        test_df.loc[last_idx, "rul"] = float(rul_val)

    test_readings = _df_to_readings(test_df, subset, "test")

    train_machine_ids = sorted({r.machine_id for r in train_readings})
    test_machine_ids  = sorted({r.machine_id for r in test_readings})

    logger.info(
        f"C-MAPSS {subset} loaded: "
        f"train={len(train_readings):,} rows/{len(train_machine_ids)} engines, "
        f"test={len(test_readings):,} rows/{len(test_machine_ids)} engines"
    )

    train_ds = RawDataset(
        name=f"CMAPSS-{subset}-train",
        readings=train_readings,
        machine_ids=train_machine_ids,
        metadata={
            "subset": subset,
            "split": "train",
            "max_rul": max_rul,
            "n_sensors": 21,
            "source": "NASA Prognostics Center of Excellence",
        },
    )
    test_ds = RawDataset(
        name=f"CMAPSS-{subset}-test",
        readings=test_readings,
        machine_ids=test_machine_ids,
        metadata={
            "subset": subset,
            "split": "test",
            "max_rul": max_rul,
            "n_sensors": 21,
            "source": "NASA Prognostics Center of Excellence",
        },
    )
    return train_ds, test_ds


def load_all_cmapss(
    data_dir: Optional[Path] = None,
    max_rul: float = 125.0,
) -> dict[str, tuple[RawDataset, RawDataset]]:
    """
    Load all four C-MAPSS sub-datasets.

    Returns
    -------
    Dict mapping subset name → (train_ds, test_ds).
    Subsets that are not yet downloaded are silently skipped with a warning.
    """
    results: dict[str, tuple[RawDataset, RawDataset]] = {}
    for subset in _SUBSETS:
        try:
            results[subset] = load_cmapss(subset=subset, data_dir=data_dir, max_rul=max_rul)
        except FileNotFoundError as e:
            warnings.warn(str(e), stacklevel=2)
    return results


def to_dataframe(dataset: RawDataset) -> pd.DataFrame:
    """
    Convert a :class:`RawDataset` of SensorReadings into a tidy DataFrame.
    Useful for EDA and feature engineering.
    """
    rows = []
    for r in dataset.readings:
        rows.append({
            "machine_id":  r.machine_id,
            "timestamp":   r.timestamp,
            "vibration_x": r.vibration_x,
            "vibration_y": r.vibration_y,
            "vibration_z": r.vibration_z,
            "temperature": r.temperature,
            "current":     r.current,
            "pressure":    r.pressure,
            "acoustic":    r.acoustic,
            "rpm":         r.rpm,
            "load":        r.load,
            "rul_hours":   r.rul_hours,
            "health_label": r.health_label,
            "dataset":     r.dataset,
        })
    return pd.DataFrame(rows)
