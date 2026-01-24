from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Optional, Protocol

import pandas as pd


class RawDataStore(Protocol):
    def get(self, key: str) -> Optional[pd.DataFrame]: ...

    def set(self, key: str, df: pd.DataFrame) -> None: ...


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class FileRawDataStore:
    """File-based raw data cache store (one parquet per key)."""

    root: str
    suffix: str = ".parquet"

    def __post_init__(self) -> None:
        self.root_path = Path(self.root).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)

    def _key_to_path(self, key: str) -> Path:
        digest = sha1(key.encode("utf-8")).hexdigest()
        return self.root_path / f"{digest}{self.suffix}"

    def get(self, key: str) -> Optional[pd.DataFrame]:
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        path = self._key_to_path(key)
        df.to_parquet(path, index=False)


@dataclass
class DuckDBRawDataStore:
    """DuckDB-indexed raw data cache store (parquet payloads)."""

    root: str
    duckdb_path: Optional[str] = None
    table_name: str = "raw_cache"

    def __post_init__(self) -> None:
        self.root_path = Path(self.root).resolve()
        self.root_path.mkdir(parents=True, exist_ok=True)
        if self.duckdb_path is None:
            self.duckdb_path = str(self.root_path / "raw_cache.duckdb")
        self._init_db()

    def _conn(self):
        import duckdb

        return duckdb.connect(self.duckdb_path)

    def _init_db(self) -> None:
        with self._conn() as con:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    cache_key VARCHAR PRIMARY KEY,
                    payload_path VARCHAR NOT NULL,
                    created_utc VARCHAR NOT NULL
                );
                """
            )

    def _payload_path(self, key: str) -> Path:
        digest = sha1(key.encode("utf-8")).hexdigest()
        return self.root_path / f"{digest}.parquet"

    def get(self, key: str) -> Optional[pd.DataFrame]:
        with self._conn() as con:
            row = con.execute(
                f"SELECT payload_path FROM {self.table_name} WHERE cache_key = ?",
                [key],
            ).fetchone()
        if row is None:
            return None
        path = Path(row[0])
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        path = self._payload_path(key)
        df.to_parquet(path, index=False)
        with self._conn() as con:
            con.execute(
                f"""
                INSERT INTO {self.table_name}(cache_key, payload_path, created_utc)
                VALUES (?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    payload_path=excluded.payload_path,
                    created_utc=excluded.created_utc
                """,
                [key, str(path), _utc_now_iso()],
            )
