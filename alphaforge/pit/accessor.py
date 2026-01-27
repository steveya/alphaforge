from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import duckdb
import pandas as pd

from alphaforge.time.ref_period import RefPeriod, RefFreq

_PIT_TABLE = "pit_observations"


def to_utc_naive(value):
    """Convert datetimes to UTC-naive for DuckDB storage and predicates."""
    if value is None:
        return None
    if isinstance(value, pd.Series):
        ts = pd.to_datetime(value, utc=True)
        return ts.dt.tz_convert("UTC").dt.tz_localize(None)
    if isinstance(value, pd.Index):
        ts = pd.to_datetime(value, utc=True)
        return ts.tz_convert("UTC").tz_localize(None)
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.tz_localize(None)


def to_utc_aware(value):
    """Convert datetimes to UTC-aware for PIT accessor outputs."""
    if value is None:
        return None
    if isinstance(value, pd.Series):
        ts = pd.to_datetime(value)
        return ts.dt.tz_localize("UTC") if ts.dt.tz is None else ts.dt.tz_convert("UTC")
    if isinstance(value, pd.Index):
        ts = pd.to_datetime(value)
        return ts.tz_localize("UTC") if ts.tz is None else ts.tz_convert("UTC")
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def ensure_pit_table(conn: duckdb.DuckDBPyConnection) -> None:
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
            ingested_utc TIMESTAMP NOT NULL DEFAULT now(),
            UNIQUE(series_key, obs_date, asof_utc)
        );
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


def _normalize_datetime_columns(
    df: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            # Normalize to naive UTC to avoid local-time shifts in DuckDB TIMESTAMP.
            out[col] = to_utc_naive(out[col])
    return out


@dataclass
class PITAccessor:
    conn: duckdb.DuckDBPyConnection

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
        if "ingested_utc" not in normalized.columns:
            normalized["ingested_utc"] = to_utc_naive(pd.Timestamp.now("UTC"))

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
        try:
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
        finally:
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
        # DuckDB stores TIMESTAMP without tz; pass UTC-naive parameters.
        asof_ts = to_utc_naive(asof)
        start_ts = to_utc_naive(start)
        end_ts = to_utc_naive(end)

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
        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return pd.Series(dtype="float64", name=series_key)
        series = pd.Series(
            df["value"].to_numpy(),
            index=to_utc_aware(df["obs_date"]),
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
        # DuckDB stores TIMESTAMP without tz; pass UTC-naive parameters.
        obs_ts = to_utc_naive(obs_date)
        start_ts = to_utc_naive(start_asof)
        end_ts = to_utc_naive(end_asof)

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
        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return pd.Series(dtype="float64", name=series_key)
        series = pd.Series(
            df["value"].to_numpy(),
            index=to_utc_aware(df["asof_utc"]),
            name=series_key,
        )
        series.index.name = "asof_utc"
        return series

    def get_revision_timeline_ref(
        self,
        series_key: str,
        ref: str | RefPeriod,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        freq: RefFreq | None = None,
    ) -> pd.Series:
        """Resolve reference period to obs_date and return revision timeline."""
        ref_period = RefPeriod.parse(ref) if isinstance(ref, str) else ref
        if freq is not None and freq != ref_period.freq:
            raise ValueError(
                "Reference period frequency does not match requested freq."
            )
        obs_date = ref_period.end_obs_date()
        return self.get_revision_timeline(
            series_key, obs_date, start_asof=start_asof, end_asof=end_asof
        )

    def get_snapshot_ref(
        self,
        series_key: str,
        asof: pd.Timestamp,
        start_ref: str | RefPeriod | None = None,
        end_ref: str | RefPeriod | None = None,
        *,
        freq: RefFreq | None = None,
    ) -> pd.Series:
        """
        Snapshot query using reference period keys for start/end obs_date bounds.

        start_ref/end_ref map to the end timestamp of the reference period.
        """

        def _resolve(ref_value: str | RefPeriod | None) -> pd.Timestamp | None:
            if ref_value is None:
                return None
            if isinstance(ref_value, RefPeriod):
                ref_period = ref_value
            else:
                ref_period = RefPeriod.parse(ref_value)
            if freq is not None and ref_period.freq != freq:
                raise ValueError(
                    "Reference period frequency does not match requested freq."
                )
            return ref_period.end_obs_date()

        start_ts = _resolve(start_ref)
        end_ts = _resolve(end_ref)
        return self.get_snapshot(series_key, asof, start=start_ts, end=end_ts)
