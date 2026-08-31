"""
Parquet-backed feature store for the Predictive Maintenance pipeline.

The feature store provides a simple on-disk cache so that expensive sliding-
window feature extraction does not need to be repeated across runs.

Storage backend
---------------
The store auto-selects the best available serialisation engine:
  1. Parquet via pyarrow  (preferred — columnar, fast, small files)
  2. Parquet via fastparquet (alternative Parquet engine)
  3. Pickle               (fallback — no extra deps, works on Python 3.14+)

Layout
------
  data/processed/
    features/
      <dataset_name>/
        <machine_id>.parquet   ← pyarrow / fastparquet
        <machine_id>.pkl       ← pickle fallback

Usage
-----
  store = FeatureStore()
  store.save(vectors, dataset_name="CMAPSS-FD001-train")
  vectors = store.load(dataset_name="CMAPSS-FD001-train")
  df      = store.load_as_dataframe(dataset_name="CMAPSS-FD001-train")
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.ingestion.schema import FeatureVector
from src.feature_engineering.multimodal import feature_vectors_to_dataframe
from src.utils.config import PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

_FEATURES_DIR = PROCESSED_DATA_DIR / "features"


# ─── Backend detection ────────────────────────────────────────────────────────

def _detect_parquet_engine() -> Optional[str]:
    """Return the best available Parquet engine, or None."""
    for engine in ("pyarrow", "fastparquet"):
        try:
            import importlib
            importlib.import_module(engine.replace("fastparquet", "fastparquet"))
            return engine
        except ImportError:
            continue
    return None


_PARQUET_ENGINE: Optional[str] = _detect_parquet_engine()

if _PARQUET_ENGINE:
    logger.debug(f"FeatureStore: using Parquet engine '{_PARQUET_ENGINE}'")
else:
    logger.warning(
        "Neither pyarrow nor fastparquet is available.  "
        "FeatureStore will use pickle (.pkl) as fallback. "
        "Install pyarrow for production use: pip install pyarrow"
    )


class FeatureStore:
    """
    Parquet-backed persistent feature store.

    Each dataset is stored as a directory of per-machine Parquet files,
    enabling efficient machine-level loading without reading the full dataset.

    Args:
        base_dir: Root directory for feature storage.
                  Defaults to ``data/processed/features/``.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = Path(base_dir or _FEATURES_DIR)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _dataset_dir(self, dataset_name: str) -> Path:
        return self.base_dir / dataset_name.replace("/", "_").replace(" ", "_")

    # ── Internal I/O helpers ──────────────────────────────────────────────────

    @staticmethod
    def _save_df(df: pd.DataFrame, path: Path) -> None:
        """Persist a DataFrame using the best available engine."""
        if _PARQUET_ENGINE:
            df.to_parquet(path.with_suffix(".parquet"), index=False, engine=_PARQUET_ENGINE)
        else:
            df.to_pickle(path.with_suffix(".pkl"))

    @staticmethod
    def _load_df(path: Path) -> pd.DataFrame:
        """Load a previously saved DataFrame, trying both formats."""
        parquet_path = path.with_suffix(".parquet")
        pickle_path  = path.with_suffix(".pkl")
        if parquet_path.exists() and _PARQUET_ENGINE:
            return pd.read_parquet(parquet_path, engine=_PARQUET_ENGINE)
        if pickle_path.exists():
            return pd.read_pickle(pickle_path)
        raise FileNotFoundError(f"No feature file found at {path} (tried .parquet and .pkl)")

    @staticmethod
    def _glob_files(directory: Path):
        """Return all feature files in a directory (both .parquet and .pkl)."""
        return sorted(
            list(directory.glob("*.parquet")) + list(directory.glob("*.pkl")),
            key=lambda p: p.stem,
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(
        self,
        vectors: list[FeatureVector],
        dataset_name: str,
        overwrite: bool = True,
    ) -> Path:
        """
        Persist a list of FeatureVectors to Parquet.

        One Parquet file is written per machine_id.  Existing files for the
        same machine are overwritten if ``overwrite=True``, otherwise they are
        skipped.

        Args:
            vectors:      List of :class:`FeatureVector` (may span multiple machines).
            dataset_name: Logical name (used as sub-directory name).
            overwrite:    Whether to overwrite existing files.

        Returns:
            Path to the dataset directory.
        """
        if not vectors:
            logger.warning(f"save called with empty vectors list for '{dataset_name}'")
            return self._dataset_dir(dataset_name)

        dataset_dir = self._dataset_dir(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)

        df = feature_vectors_to_dataframe(vectors)

        n_written = 0
        for machine_id, group in df.groupby("machine_id"):
            safe_id = str(machine_id).replace("/", "_").replace(" ", "_")
            base_path = dataset_dir / safe_id
            # Determine actual path based on engine
            actual_path = base_path.with_suffix(".parquet" if _PARQUET_ENGINE else ".pkl")
            if actual_path.exists() and not overwrite:
                logger.debug(f"Skipping existing: {actual_path.name}")
                continue
            self._save_df(group, base_path)
            n_written += 1

        logger.info(
            f"FeatureStore.save: wrote {n_written} files for '{dataset_name}' "
            f"({len(vectors)} vectors, {df['machine_id'].nunique()} machines)"
        )
        return dataset_dir

    # ── Read ──────────────────────────────────────────────────────────────────

    def load_as_dataframe(
        self,
        dataset_name: str,
        machine_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Load stored features as a pandas DataFrame.

        Args:
            dataset_name: Name used when saving.
            machine_ids:  Optional filter — only load these machines.

        Returns:
            Concatenated DataFrame, sorted by (machine_id, window_start).

        Raises:
            FileNotFoundError: If the dataset directory does not exist.
        """
        dataset_dir = self._dataset_dir(dataset_name)
        if not dataset_dir.exists():
            raise FileNotFoundError(
                f"Feature store has no dataset '{dataset_name}' at {dataset_dir}.\n"
                "Run feature extraction first."
            )

        parquet_files = self._glob_files(dataset_dir)
        if not parquet_files:
            raise FileNotFoundError(f"No feature files found in {dataset_dir}")

        frames = []
        for pf in parquet_files:
            machine_id = pf.stem.replace("_", "-")
            if machine_ids and machine_id not in machine_ids:
                continue
            try:
                frames.append(self._load_df(pf))
            except Exception as e:
                logger.warning(f"Could not load {pf.name}: {e}")

        if not frames:
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        if "window_start" in df.columns:
            df.sort_values(["machine_id", "window_start"], inplace=True)
            df.reset_index(drop=True, inplace=True)

        logger.info(
            f"FeatureStore.load: '{dataset_name}' → {len(df)} rows, "
            f"{df['machine_id'].nunique()} machines"
        )
        return df

    def load(
        self,
        dataset_name: str,
        machine_ids: Optional[list[str]] = None,
    ) -> list[FeatureVector]:
        """
        Load stored features as a list of FeatureVectors.

        Note: This reconstructs FeatureVector objects from the DataFrame.
        For large datasets, working directly with the DataFrame is more
        memory-efficient.
        """
        df = self.load_as_dataframe(dataset_name=dataset_name, machine_ids=machine_ids)
        return _dataframe_to_feature_vectors(df)

    # ── Metadata ──────────────────────────────────────────────────────────────

    def list_datasets(self) -> list[str]:
        """Return names of all stored datasets."""
        return [d.name for d in self.base_dir.iterdir() if d.is_dir()]

    def dataset_info(self, dataset_name: str) -> dict:
        """Return basic metadata about a stored dataset."""
        dataset_dir = self._dataset_dir(dataset_name)
        if not dataset_dir.exists():
            return {}
        files = self._glob_files(dataset_dir)
        total_rows = 0
        for f in files:
            try:
                total_rows += self._load_df(f).shape[0]
            except Exception:
                pass
        return {
            "dataset_name": dataset_name,
            "n_machines": len(files),
            "total_windows": total_rows,
            "path": str(dataset_dir),
        }

    def delete(self, dataset_name: str) -> None:
        """Delete a stored dataset (all Parquet files in the directory)."""
        import shutil
        dataset_dir = self._dataset_dir(dataset_name)
        if dataset_dir.exists():
            shutil.rmtree(dataset_dir)
            logger.info(f"Deleted feature store dataset: {dataset_name}")


def _dataframe_to_feature_vectors(df: pd.DataFrame) -> list[FeatureVector]:
    """Reconstruct FeatureVector objects from a DataFrame row by row."""
    fields = {f.name: f for f in dataclasses.fields(FeatureVector)}
    vectors = []
    for _, row in df.iterrows():
        kwargs: dict = {}
        for fname, fobj in fields.items():
            if fname in row:
                val = row[fname]
                # Restore None for nullable fields
                if isinstance(val, float) and np.isnan(val):
                    if fobj.default is None or (
                        hasattr(fobj, "default_factory") and fobj.default is dataclasses.MISSING
                    ):
                        val = None
                kwargs[fname] = val
            else:
                kwargs[fname] = fobj.default
        vectors.append(FeatureVector(**kwargs))
    return vectors
