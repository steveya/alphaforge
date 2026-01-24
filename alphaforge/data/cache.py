from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha1
from pathlib import Path
from typing import Any, Optional, Protocol

import pandas as pd


class CacheBackend(Protocol):
    def get(self, key: str) -> Optional[pd.DataFrame]: ...

    def set(self, key: str, df: pd.DataFrame) -> None: ...


@dataclass
class FileCacheBackend:
    """Simple file-based cache using pandas pickle serialization."""

    base_dir: Path
    suffix: str = ".pkl"

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        digest = sha1(key.encode("utf-8")).hexdigest()
        return self.base_dir / f"{digest}{self.suffix}"

    def get(self, key: str) -> Optional[pd.DataFrame]:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            return pd.read_pickle(path)
        except Exception:
            return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        path = self._key_to_path(key)
        df.to_pickle(path)


@dataclass
class DuckDBCacheBackend:
    """DuckDB-backed cache storing canonical OHLCV rows by cache key."""

    database_path: Path
    table_name: str = "tiingo_cache"
    _duckdb: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            import duckdb  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ImportError("duckdb is required for DuckDBCacheBackend") from exc
        self._duckdb = duckdb
        self._ensure_table()

    def _connect(self):
        return self._duckdb.connect(str(self.database_path))

    def _ensure_table(self) -> None:
        with self._connect() as con:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    cache_key VARCHAR,
                    date TIMESTAMP,
                    entity_id VARCHAR,
                    asof_utc TIMESTAMP,
                    open DOUBLE,
                    high DOUBLE,
                    low DOUBLE,
                    close DOUBLE,
                    volume DOUBLE
                )
                """
            )
            con.execute(
                f"ALTER TABLE {self.table_name} ADD COLUMN IF NOT EXISTS asof_utc TIMESTAMP"
            )

    def get(self, key: str) -> Optional[pd.DataFrame]:
        with self._connect() as con:
            df = con.execute(
                f"SELECT date, entity_id, asof_utc, open, high, low, close, volume "
                f"FROM {self.table_name} WHERE cache_key = ? "
                f"ORDER BY date, entity_id",
                [key],
            ).fetch_df()
        return None if df.empty else df

    def set(self, key: str, df: pd.DataFrame) -> None:
        df_to_store = df.copy()
        df_to_store["cache_key"] = key
        with self._connect() as con:
            con.execute(
                f"DELETE FROM {self.table_name} WHERE cache_key = ?",
                [key],
            )
            con.register("df_to_store", df_to_store)
            con.execute(
                f"INSERT INTO {self.table_name} SELECT "
                f"cache_key, date, entity_id, asof_utc, open, high, low, close, volume "
                f"FROM df_to_store"
            )
