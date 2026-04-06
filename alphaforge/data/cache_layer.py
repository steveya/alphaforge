"""Unified cache layer for PIT and non-PIT data.

Stores data in DuckDB tables with source isolation and staleness tracking.
PIT data goes to ``cache_pit_observations``, non-PIT to
``cache_market_observations``, and freshness metadata to ``cache_manifest``.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import duckdb
import pandas as pd

from .types import CacheManifest, StaleDataWarning

_PIT_TABLE = "cache_pit_observations"
_MARKET_TABLE = "cache_market_observations"
_MANIFEST_TABLE = "cache_manifest"


class CacheLayer:
    """Unified cache backed by a DuckDB connection.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        An open DuckDB connection (in-memory or file-backed).
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn
        self._ensure_tables()

    # ------------------------------------------------------------------
    # Table setup
    # ------------------------------------------------------------------

    def _ensure_tables(self) -> None:
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_PIT_TABLE} (
                series_key  TEXT NOT NULL,
                obs_date    TIMESTAMP NOT NULL,
                asof_utc    TIMESTAMP NOT NULL,
                value       DOUBLE,
                source      TEXT NOT NULL,
                dataset     TEXT NOT NULL
            )
        """)
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_MARKET_TABLE} (
                series_key  TEXT NOT NULL,
                obs_date    TIMESTAMP NOT NULL,
                value       DOUBLE,
                source      TEXT NOT NULL,
                dataset     TEXT NOT NULL
            )
        """)
        self._conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {_MANIFEST_TABLE} (
                dataset      TEXT NOT NULL,
                source       TEXT NOT NULL,
                populated_at TIMESTAMP NOT NULL,
                row_count    INTEGER NOT NULL,
                entity_keys  TEXT,
                min_obs_date TIMESTAMP,
                max_obs_date TIMESTAMP,
                UNIQUE(dataset, source)
            )
        """)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    def store(
        self,
        df: pd.DataFrame,
        *,
        dataset: str,
        source: str,
        is_pit: bool,
    ) -> None:
        """Persist data to the appropriate cache table and update manifest."""
        if df.empty:
            return

        table = _PIT_TABLE if is_pit else _MARKET_TABLE
        now = datetime.now(timezone.utc)

        # Build insert frame with required columns
        insert_df = df.copy()
        insert_df["source"] = source
        insert_df["dataset"] = dataset

        if is_pit:
            cols = ["series_key", "obs_date", "asof_utc", "value", "source", "dataset"]
        else:
            cols = ["series_key", "obs_date", "value", "source", "dataset"]

        insert_df = insert_df[cols]

        # Upsert: for simplicity, append (no dedup at cache level — adapters
        # are responsible for providing clean data).
        self._conn.register("_cache_insert", insert_df)
        col_list = ", ".join(cols)
        self._conn.execute(
            f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM _cache_insert"
        )
        self._conn.unregister("_cache_insert")

        # Update manifest — query aggregate stats from the full table
        # Get total row count for this dataset+source in the table
        total_rows_row = self._conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE dataset = ? AND source = ?",
            [dataset, source],
        ).fetchone()
        total_rows = int(total_rows_row[0]) if total_rows_row is not None else 0

        # Get all entity keys for this dataset+source
        all_keys = self._conn.execute(
            f"SELECT DISTINCT series_key FROM {table} WHERE dataset = ? AND source = ? ORDER BY series_key",
            [dataset, source],
        ).fetchdf()["series_key"].tolist()

        all_min_row = self._conn.execute(
            f"SELECT MIN(obs_date) FROM {table} WHERE dataset = ? AND source = ?",
            [dataset, source],
        ).fetchone()
        all_min = all_min_row[0] if all_min_row is not None else None
        all_max_row = self._conn.execute(
            f"SELECT MAX(obs_date) FROM {table} WHERE dataset = ? AND source = ?",
            [dataset, source],
        ).fetchone()
        all_max = all_max_row[0] if all_max_row is not None else None

        entity_keys_str = ",".join(all_keys)

        # Upsert manifest
        self._conn.execute(
            f"DELETE FROM {_MANIFEST_TABLE} WHERE dataset = ? AND source = ?",
            [dataset, source],
        )
        self._conn.execute(
            f"""INSERT INTO {_MANIFEST_TABLE}
                (dataset, source, populated_at, row_count, entity_keys, min_obs_date, max_obs_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
            [dataset, source, now, total_rows, entity_keys_str, all_min, all_max],
        )

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def lookup(
        self,
        *,
        series_key: str,
        dataset: str,
        source: str,
        is_pit: bool,
        max_staleness: Optional[timedelta] = None,
    ) -> Optional[tuple[pd.DataFrame, datetime]]:
        """Look up cached data for a series.

        Returns
        -------
        (DataFrame, cached_at) if found, else None.

        If *max_staleness* is ``timedelta(0)``, always returns None (force miss).
        If cache is older than *max_staleness*, emits :class:`StaleDataWarning`
        but still returns data.
        """
        # Check manifest for this dataset+source
        manifest_row = self._conn.execute(
            f"SELECT populated_at FROM {_MANIFEST_TABLE} WHERE dataset = ? AND source = ?",
            [dataset, source],
        ).fetchone()

        if manifest_row is None:
            return None

        cached_at = manifest_row[0]
        if isinstance(cached_at, str):
            cached_at = pd.Timestamp(cached_at).to_pydatetime()
        if not isinstance(cached_at, datetime):
            cached_at = pd.Timestamp(cached_at).to_pydatetime()
        if cached_at.tzinfo is None:
            cached_at = cached_at.replace(tzinfo=timezone.utc)

        # Staleness check
        if max_staleness is not None:
            age = datetime.now(timezone.utc) - cached_at
            if max_staleness == timedelta(0):
                return None
            if age > max_staleness:
                warnings.warn(
                    f"Stale cache for {dataset}/{source}: "
                    f"age={age}, threshold={max_staleness}",
                    StaleDataWarning,
                    stacklevel=2,
                )

        # Query the data
        table = _PIT_TABLE if is_pit else _MARKET_TABLE
        df = self._conn.execute(
            f"SELECT * FROM {table} WHERE series_key = ? AND dataset = ? AND source = ?",
            [series_key, dataset, source],
        ).fetchdf()

        if df.empty:
            return None

        # Drop internal columns before returning
        drop_cols = [c for c in ("source", "dataset") if c in df.columns]
        df = df.drop(columns=drop_cols)

        return df, cached_at

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def get_manifest(
        self, *, dataset: str, source: str
    ) -> Optional[CacheManifest]:
        """Return the manifest for a dataset+source, or None."""
        row = self._conn.execute(
            f"SELECT populated_at, row_count, entity_keys, min_obs_date, max_obs_date "
            f"FROM {_MANIFEST_TABLE} WHERE dataset = ? AND source = ?",
            [dataset, source],
        ).fetchone()

        if row is None:
            return None

        populated_at = row[0]
        if isinstance(populated_at, str):
            populated_at = pd.Timestamp(populated_at).to_pydatetime()
        if not isinstance(populated_at, datetime):
            populated_at = pd.Timestamp(populated_at).to_pydatetime()
        if populated_at.tzinfo is None:
            populated_at = populated_at.replace(tzinfo=timezone.utc)

        entity_keys_str = row[2] or ""
        entity_keys = [k for k in entity_keys_str.split(",") if k]

        min_obs = row[3]
        max_obs = row[4]
        # Convert to date
        if min_obs is not None:
            if hasattr(min_obs, "date"):
                min_obs = min_obs.date() if callable(min_obs.date) else min_obs
            else:
                min_obs = pd.Timestamp(min_obs).date()
        else:
            min_obs = date.min
        if max_obs is not None:
            if hasattr(max_obs, "date"):
                max_obs = max_obs.date() if callable(max_obs.date) else max_obs
            else:
                max_obs = pd.Timestamp(max_obs).date()
        else:
            max_obs = date.min

        return CacheManifest(
            dataset=dataset,
            source=source,
            entity_keys=entity_keys,
            asof_range=(min_obs, max_obs),
            populated_at=populated_at,
            row_count=row[1],
        )

    # ------------------------------------------------------------------
    # Test helpers
    # ------------------------------------------------------------------

    def _force_manifest_age(
        self, *, dataset: str, source: str, age: timedelta
    ) -> None:
        """Test helper: backdate the manifest's populated_at."""
        old_time = datetime.now(timezone.utc) - age
        self._conn.execute(
            f"UPDATE {_MANIFEST_TABLE} SET populated_at = ? "
            f"WHERE dataset = ? AND source = ?",
            [old_time, dataset, source],
        )
