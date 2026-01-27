from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import pandas as pd

_PIT_TABLE = "pit_observations"


def _ts_utc(value: pd.Timestamp | str | None) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def ensure_pit_table(conn) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_PIT_TABLE} (
            series_key TEXT NOT NULL,
            obs_date TIMESTAMP NOT NULL,
            asof_utc TIMESTAMP NOT NULL,
            value DOUBLE,
            release_time_utc TIMESTAMP,
            revision_id TEXT,
            source TEXT,
            meta_json TEXT,
            ingested_utc TIMESTAMP NOT NULL DEFAULT now()
        );
        """
    )
    conn.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS pit_unique_key
        ON {_PIT_TABLE}(series_key, obs_date, asof_utc);
        """
    )
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS pit_series_obs
        ON {_PIT_TABLE}(series_key, obs_date);
        """
    )
    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS pit_series_asof
        ON {_PIT_TABLE}(series_key, asof_utc);
        """
    )


def _normalize_datetime_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], utc=True)
    return out


@dataclass
class PITAccessor:
    conn: object

    def __post_init__(self) -> None:
        ensure_pit_table(self.conn)

    def upsert_pit_observations(self, df: pd.DataFrame) -> None:
        required = {"series_key", "obs_date", "asof_utc", "value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        normalized = _normalize_datetime_columns(
            df, ["obs_date", "asof_utc", "release_time_utc", "ingested_utc"]
        )
        normalized = normalized.copy()
        if "ingested_utc" not in normalized.columns:
            normalized["ingested_utc"] = pd.Timestamp.now("UTC")

        columns = [
            "series_key",
            "obs_date",
            "asof_utc",
            "value",
            "release_time_utc",
            "revision_id",
            "source",
            "meta_json",
            "ingested_utc",
        ]
        for col in columns:
            if col not in normalized.columns:
                normalized[col] = pd.NA
        normalized = normalized[columns]

        self.conn.register("pit_incoming", normalized)
        self.conn.execute(
            f"""
            INSERT INTO {_PIT_TABLE} ({", ".join(columns)})
            SELECT {", ".join(columns)} FROM pit_incoming
            ON CONFLICT(series_key, obs_date, asof_utc) DO UPDATE SET
                value=excluded.value,
                release_time_utc=excluded.release_time_utc,
                revision_id=excluded.revision_id,
                source=excluded.source,
                meta_json=excluded.meta_json,
                ingested_utc=excluded.ingested_utc;
            """
        )
        self.conn.unregister("pit_incoming")

    def get_snapshot(
        self,
        series_key: str,
        asof: pd.Timestamp,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        method: Literal["latest_leq"] = "latest_leq",
    ) -> pd.Series:
        if method != "latest_leq":
            raise ValueError(f"Unsupported snapshot method: {method}")
        asof_ts = _ts_utc(asof)
        start_ts = _ts_utc(start)
        end_ts = _ts_utc(end)

        filters = ["series_key = ?", "asof_utc <= ?"]
        params: list[object] = [series_key, asof_ts]
        if start_ts is not None:
            filters.append("obs_date >= ?")
            params.append(start_ts)
        if end_ts is not None:
            filters.append("obs_date <= ?")
            params.append(end_ts)

        where_clause = " AND ".join(filters)
        query = f"""
            SELECT obs_date, value
            FROM (
                SELECT
                    obs_date,
                    value,
                    ROW_NUMBER() OVER (
                        PARTITION BY obs_date
                        ORDER BY asof_utc DESC
                    ) AS rn
                FROM {_PIT_TABLE}
                WHERE {where_clause}
            ) ranked
            WHERE rn = 1
            ORDER BY obs_date
        """
        df = self.conn.execute(query, params).fetch_df()
        if df.empty:
            return pd.Series(dtype="float64", name=series_key)
        series = pd.Series(
            df["value"].to_numpy(),
            index=pd.to_datetime(df["obs_date"], utc=True),
            name=series_key,
        )
        series.index.name = "obs_date"
        return series

    def get_revision_timeline(
        self,
        series_key: str,
        obs_date: pd.Timestamp,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
    ) -> pd.Series:
        obs_ts = _ts_utc(obs_date)
        start_ts = _ts_utc(start_asof)
        end_ts = _ts_utc(end_asof)

        filters = ["series_key = ?", "obs_date = ?"]
        params: list[object] = [series_key, obs_ts]
        if start_ts is not None:
            filters.append("asof_utc >= ?")
            params.append(start_ts)
        if end_ts is not None:
            filters.append("asof_utc <= ?")
            params.append(end_ts)

        where_clause = " AND ".join(filters)
        query = f"""
            SELECT asof_utc, value
            FROM {_PIT_TABLE}
            WHERE {where_clause}
            ORDER BY asof_utc ASC
        """
        df = self.conn.execute(query, params).fetch_df()
        if df.empty:
            return pd.Series(dtype="float64", name=series_key)
        series = pd.Series(
            df["value"].to_numpy(),
            index=pd.to_datetime(df["asof_utc"], utc=True),
            name=series_key,
        )
        series.index.name = "asof_utc"
        return series
