"""
Multimodal sensor fusion — assembles FeatureVectors into ML-ready arrays.

Handles:
  - NaN imputation (median strategy per feature, fitted on training data)
  - Feature normalisation (StandardScaler, fitted on training data)
  - Conversion to numpy arrays / pandas DataFrames for downstream models
  - Train / test split that respects machine boundaries (no data leakage
    between machines)
"""
from __future__ import annotations

import dataclasses
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.ingestion.schema import FeatureVector
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Columns excluded from the feature matrix (identifiers / labels)
_META_COLS = {
    "machine_id", "window_start", "window_end", "n_samples",
    "health_index",   # computed later by Health Engine
    "rul_hours", "health_label", "dataset",
}

# Numeric feature columns (all FeatureVector fields except meta)
_FEATURE_FIELDS: list[str] = [
    f.name for f in dataclasses.fields(FeatureVector)
    if f.name not in _META_COLS
]


def feature_vectors_to_dataframe(vectors: Sequence[FeatureVector]) -> pd.DataFrame:
    """
    Convert a list of FeatureVectors to a pandas DataFrame.

    The resulting DataFrame has one row per window and all canonical
    feature columns.

    Args:
        vectors: Sequence of :class:`FeatureVector`.

    Returns:
        DataFrame with columns matching :class:`FeatureVector` fields.
    """
    rows = []
    for fv in vectors:
        row: dict = {}
        for f in dataclasses.fields(fv):
            val = getattr(fv, f.name)
            row[f.name] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    # Ensure datetime columns are proper dtype
    for col in ("window_start", "window_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True)
    return df


def get_feature_matrix(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    drop_nan_cols: bool = False,
) -> np.ndarray:
    """
    Extract the numeric feature matrix X from a DataFrame.

    Args:
        df:             DataFrame produced by :func:`feature_vectors_to_dataframe`.
        feature_cols:   Subset of columns to include (default: all feature fields).
        drop_nan_cols:  If True, drop columns that are all-NaN.

    Returns:
        2-D float array of shape (n_windows, n_features).
    """
    cols = feature_cols or [c for c in _FEATURE_FIELDS if c in df.columns]
    X = df[cols].values.astype(np.float64)
    if drop_nan_cols:
        nan_mask = np.all(np.isnan(X), axis=0)
        if nan_mask.any():
            dropped = [c for c, m in zip(cols, nan_mask) if m]
            logger.warning(f"Dropping all-NaN feature columns: {dropped}")
            X = X[:, ~nan_mask]
    return X


def machine_aware_split(
    df: pd.DataFrame,
    test_machine_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a multi-machine DataFrame into train / test sets such that
    all windows from a given machine end up in one split only.

    This prevents data leakage: the model never sees future windows of a
    machine it was trained on.

    Args:
        df:                     Full feature DataFrame (must have 'machine_id').
        test_machine_fraction:  Fraction of machines assigned to test.
        random_state:           Random seed.

    Returns:
        (train_df, test_df) tuple.
    """
    rng = np.random.default_rng(random_state)
    machines = list(df["machine_id"].unique())   # plain list → shuffle is well-defined
    rng.shuffle(machines)
    n_test = max(1, int(len(machines) * test_machine_fraction))
    test_machines  = set(machines[:n_test])
    train_machines = set(machines[n_test:])

    train_df = df[df["machine_id"].isin(train_machines)].copy()
    test_df  = df[df["machine_id"].isin(test_machines)].copy()

    logger.info(
        f"machine_aware_split: {len(train_machines)} train machines "
        f"({len(train_df)} windows), "
        f"{len(test_machines)} test machines ({len(test_df)} windows)"
    )
    return train_df, test_df


def impute_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
    strategy: str = "median",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """
    Impute NaN values using statistics computed on training data only.

    Args:
        train_df:     Training feature DataFrame.
        test_df:      Test feature DataFrame.
        feature_cols: Columns to impute (default: all feature fields).
        strategy:     "median" or "mean".

    Returns:
        (train_imputed, test_imputed, fill_values) tuple.
        ``fill_values`` is a dict mapping column → imputation value (for
        serialisation / reproducibility).
    """
    cols = feature_cols or [c for c in _FEATURE_FIELDS if c in train_df.columns]
    fill_values: dict[str, float] = {}

    for col in cols:
        if strategy == "median":
            val = float(train_df[col].median())
        else:
            val = float(train_df[col].mean())
        fill_values[col] = val if not np.isnan(val) else 0.0

    train_imputed = train_df.copy()
    test_imputed  = test_df.copy()
    for col, val in fill_values.items():
        train_imputed[col] = train_imputed[col].fillna(val)
        test_imputed[col]  = test_imputed[col].fillna(val)

    return train_imputed, test_imputed, fill_values


def normalise_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Standardise features to zero mean and unit variance.
    Statistics are computed on training data only.

    Args:
        train_df:     Training feature DataFrame (NaN-free).
        test_df:      Test feature DataFrame.
        feature_cols: Columns to normalise.

    Returns:
        (train_norm, test_norm, scaler_params) where ``scaler_params`` is
        a dict ``{col: {"mean": …, "std": …}}`` for serialisation.
    """
    cols = feature_cols or [c for c in _FEATURE_FIELDS if c in train_df.columns]
    scaler_params: dict = {}

    train_norm = train_df.copy()
    test_norm  = test_df.copy()

    for col in cols:
        mu  = float(train_df[col].mean())
        sig = float(train_df[col].std())
        scaler_params[col] = {"mean": mu, "std": sig}
        if sig > 0:
            train_norm[col] = (train_df[col] - mu) / sig
            test_norm[col]  = (test_df[col]  - mu) / sig
        else:
            # Zero-variance feature → zero-fill (prevents NaN from 0/0)
            train_norm[col] = 0.0
            test_norm[col]  = 0.0

    return train_norm, test_norm, scaler_params


def get_feature_cols() -> list[str]:
    """Return the canonical ordered list of numeric feature column names."""
    return list(_FEATURE_FIELDS)
