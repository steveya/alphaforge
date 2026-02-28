from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Literal, Sequence

import duckdb
import pandas as pd

from alphaforge.time.ref_period import RefFreq, RefPeriod

from .guards import ReleaseLagPolicy, effective_asof
from .transforms import (
    PITTransformResult,
    PITTransformSpec,
    apply_obs_path_transform,
    apply_revision_path_transform,
    resolve_engine,
    serialize_params_for_lineage,
    validate_transform_spec,
)

_PIT_TABLE = "pit_observations"
_PIT_TRANSFORMS_TABLE = "pit_transforms"
_PIT_TRANSFORM_RUNS_TABLE = "pit_transform_runs"


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

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_PIT_TRANSFORMS_TABLE} (
            transform_id TEXT PRIMARY KEY,
            output_series_key TEXT NOT NULL,
            input_series_keys_json TEXT NOT NULL,
            axis TEXT NOT NULL,
            op TEXT NOT NULL,
            params_json TEXT NOT NULL,
            engine TEXT NOT NULL,
            spec_hash TEXT NOT NULL,
            created_utc TIMESTAMP NOT NULL DEFAULT now()
        );
        """
    )

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_PIT_TRANSFORM_RUNS_TABLE} (
            run_id TEXT PRIMARY KEY,
            transform_id TEXT NOT NULL,
            start_obs TIMESTAMP,
            end_obs TIMESTAMP,
            start_asof TIMESTAMP,
            end_asof TIMESTAMP,
            rows_written BIGINT NOT NULL,
            status TEXT NOT NULL,
            started_utc TIMESTAMP NOT NULL,
            finished_utc TIMESTAMP NOT NULL
        );
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS pit_transform_runs_transform_id
        ON {_PIT_TRANSFORM_RUNS_TABLE}(transform_id, started_utc);
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

    def _list_candidate_asofs(
        self,
        series_key: str,
        *,
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
    ) -> pd.DatetimeIndex:
        filters = ["series_key = ?"]
        params: list[object] = [series_key]

        if start_obs is not None:
            filters.append("obs_date >= ?")
            params.append(to_utc_naive(start_obs))
        if end_obs is not None:
            filters.append("obs_date <= ?")
            params.append(to_utc_naive(end_obs))
        if start_asof is not None:
            filters.append("asof_utc >= ?")
            params.append(to_utc_naive(start_asof))
        if end_asof is not None:
            filters.append("asof_utc <= ?")
            params.append(to_utc_naive(end_asof))

        where_clause = " AND ".join(filters)
        query = f"""
            SELECT DISTINCT asof_utc
            FROM {_PIT_TABLE}
            WHERE {where_clause}
            ORDER BY asof_utc ASC
        """
        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return pd.DatetimeIndex([], tz="UTC")
        return pd.DatetimeIndex(to_utc_aware(df["asof_utc"]))

    def _upsert_transform_metadata(
        self,
        spec: PITTransformSpec,
        engine_used: str,
    ) -> None:
        transform_id = spec.transform_id()
        self.conn.execute(
            f"""
            INSERT INTO {_PIT_TRANSFORMS_TABLE} (
                transform_id,
                output_series_key,
                input_series_keys_json,
                axis,
                op,
                params_json,
                engine,
                spec_hash,
                created_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(transform_id) DO UPDATE SET
                output_series_key=excluded.output_series_key,
                input_series_keys_json=excluded.input_series_keys_json,
                axis=excluded.axis,
                op=excluded.op,
                params_json=excluded.params_json,
                engine=excluded.engine,
                spec_hash=excluded.spec_hash,
                created_utc=excluded.created_utc
            """,
            [
                transform_id,
                spec.output_series_key,
                json.dumps([spec.input_series_key]),
                spec.axis,
                spec.op,
                serialize_params_for_lineage(spec.params),
                engine_used,
                spec.spec_hash(),
                to_utc_naive(pd.Timestamp.now("UTC")),
            ],
        )

    def _insert_transform_run(
        self,
        *,
        transform_id: str,
        start_obs: pd.Timestamp | None,
        end_obs: pd.Timestamp | None,
        start_asof: pd.Timestamp | None,
        end_asof: pd.Timestamp | None,
        rows_written: int,
        status: str,
        started_utc: pd.Timestamp,
        finished_utc: pd.Timestamp,
    ) -> None:
        self.conn.execute(
            f"""
            INSERT INTO {_PIT_TRANSFORM_RUNS_TABLE} (
                run_id,
                transform_id,
                start_obs,
                end_obs,
                start_asof,
                end_asof,
                rows_written,
                status,
                started_utc,
                finished_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                str(uuid.uuid4()),
                transform_id,
                to_utc_naive(start_obs),
                to_utc_naive(end_obs),
                to_utc_naive(start_asof),
                to_utc_naive(end_asof),
                int(rows_written),
                status,
                to_utc_naive(started_utc),
                to_utc_naive(finished_utc),
            ],
        )

    def _delete_transformed_rows(
        self,
        *,
        output_series_key: str,
        start_obs: pd.Timestamp | None,
        end_obs: pd.Timestamp | None,
        start_asof: pd.Timestamp | None,
        end_asof: pd.Timestamp | None,
    ) -> None:
        filters = ["series_key = ?"]
        params: list[object] = [output_series_key]

        if start_obs is not None:
            filters.append("obs_date >= ?")
            params.append(to_utc_naive(start_obs))
        if end_obs is not None:
            filters.append("obs_date <= ?")
            params.append(to_utc_naive(end_obs))
        if start_asof is not None:
            filters.append("asof_utc >= ?")
            params.append(to_utc_naive(start_asof))
        if end_asof is not None:
            filters.append("asof_utc <= ?")
            params.append(to_utc_naive(end_asof))

        where_clause = " AND ".join(filters)
        self.conn.execute(f"DELETE FROM {_PIT_TABLE} WHERE {where_clause}", params)

    @staticmethod
    def _lineage_meta(
        *,
        transform_id: str,
        spec: PITTransformSpec,
        engine_used: str,
        source_asof: pd.Timestamp | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "transform_id": transform_id,
            "input_series_key": spec.input_series_key,
            "op": spec.op,
            "axis": spec.axis,
            "engine": engine_used,
            "params": spec.sanitized_params(),
            "spec_hash": spec.spec_hash(),
            "experimental": bool(spec.axis == "revision_path"),
        }
        if source_asof is not None:
            payload["source_asof_utc"] = source_asof.isoformat()
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _build_obs_path_rows(
        self,
        *,
        spec: PITTransformSpec,
        start_obs: pd.Timestamp | None,
        end_obs: pd.Timestamp | None,
        start_asof: pd.Timestamp | None,
        end_asof: pd.Timestamp | None,
        lag_policy: ReleaseLagPolicy | None,
        transform_id: str,
        engine_used: str,
    ) -> pd.DataFrame:
        asof_values = self._list_candidate_asofs(
            spec.input_series_key,
            start_obs=start_obs,
            end_obs=end_obs,
            start_asof=start_asof,
            end_asof=end_asof,
        )
        chunks: list[pd.DataFrame] = []

        start_obs_utc = to_utc_aware(start_obs) if start_obs is not None else None
        end_obs_utc = to_utc_aware(end_obs) if end_obs is not None else None

        for asof in asof_values:
            source_asof = asof
            if lag_policy is not None:
                source_asof = effective_asof(source_asof, spec.input_series_key, lag_policy)

            if source_asof > asof:
                raise ValueError(
                    "Causality violation: effective source_asof is later than output asof. "
                    f"source_asof={source_asof}, output_asof={asof}"
                )

            snapshot = self.get_snapshot(
                spec.input_series_key,
                source_asof,
                start=start_obs,
                end=end_obs,
            )
            if snapshot.empty:
                continue

            transformed = apply_obs_path_transform(snapshot, spec).dropna()
            if transformed.empty:
                continue

            if start_obs_utc is not None:
                transformed = transformed[transformed.index >= start_obs_utc]
            if end_obs_utc is not None:
                transformed = transformed[transformed.index <= end_obs_utc]
            if transformed.empty:
                continue

            meta_json = self._lineage_meta(
                transform_id=transform_id,
                spec=spec,
                engine_used=engine_used,
                source_asof=source_asof,
            )
            chunks.append(
                pd.DataFrame(
                    {
                        "series_key": spec.output_series_key,
                        "obs_date": transformed.index,
                        "asof_utc": [asof] * len(transformed),
                        "value": transformed.to_numpy(),
                        "source": [f"pit_transform:{transform_id}"] * len(transformed),
                        "meta_json": [meta_json] * len(transformed),
                    }
                )
            )

        if not chunks:
            return pd.DataFrame(
                columns=["series_key", "obs_date", "asof_utc", "value", "source", "meta_json"]
            )
        return pd.concat(chunks, ignore_index=True)

    def _fetch_revision_rows(
        self,
        *,
        series_key: str,
        start_obs: pd.Timestamp | None,
        end_obs: pd.Timestamp | None,
        start_asof: pd.Timestamp | None,
        end_asof: pd.Timestamp | None,
    ) -> pd.DataFrame:
        filters = ["series_key = ?"]
        params: list[object] = [series_key]

        if start_obs is not None:
            filters.append("obs_date >= ?")
            params.append(to_utc_naive(start_obs))
        if end_obs is not None:
            filters.append("obs_date <= ?")
            params.append(to_utc_naive(end_obs))
        if start_asof is not None:
            filters.append("asof_utc >= ?")
            params.append(to_utc_naive(start_asof))
        if end_asof is not None:
            filters.append("asof_utc <= ?")
            params.append(to_utc_naive(end_asof))

        where_clause = " AND ".join(filters)
        query = f"""
            SELECT obs_date, asof_utc, value
            FROM {_PIT_TABLE}
            WHERE {where_clause}
            ORDER BY obs_date, asof_utc
        """
        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return df

        df["obs_date"] = to_utc_aware(df["obs_date"])
        df["asof_utc"] = to_utc_aware(df["asof_utc"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df

    def _build_revision_path_rows(
        self,
        *,
        spec: PITTransformSpec,
        start_obs: pd.Timestamp | None,
        end_obs: pd.Timestamp | None,
        start_asof: pd.Timestamp | None,
        end_asof: pd.Timestamp | None,
        transform_id: str,
        engine_used: str,
    ) -> pd.DataFrame:
        raw = self._fetch_revision_rows(
            series_key=spec.input_series_key,
            start_obs=start_obs,
            end_obs=end_obs,
            start_asof=start_asof,
            end_asof=end_asof,
        )
        if raw.empty:
            return pd.DataFrame(
                columns=["series_key", "obs_date", "asof_utc", "value", "source", "meta_json"]
            )

        start_asof_utc = to_utc_aware(start_asof) if start_asof is not None else None
        end_asof_utc = to_utc_aware(end_asof) if end_asof is not None else None

        chunks: list[pd.DataFrame] = []
        for obs_date, group in raw.groupby("obs_date", dropna=False):
            timeline = pd.Series(
                group["value"].to_numpy(),
                index=pd.DatetimeIndex(group["asof_utc"]),
                name=spec.input_series_key,
            ).sort_index()
            transformed = apply_revision_path_transform(timeline, spec).dropna()
            if transformed.empty:
                continue

            if start_asof_utc is not None:
                transformed = transformed[transformed.index >= start_asof_utc]
            if end_asof_utc is not None:
                transformed = transformed[transformed.index <= end_asof_utc]
            if transformed.empty:
                continue

            meta_json = self._lineage_meta(
                transform_id=transform_id,
                spec=spec,
                engine_used=engine_used,
            )
            chunks.append(
                pd.DataFrame(
                    {
                        "series_key": spec.output_series_key,
                        "obs_date": [obs_date] * len(transformed),
                        "asof_utc": transformed.index,
                        "value": transformed.to_numpy(),
                        "source": [f"pit_transform:{transform_id}"] * len(transformed),
                        "meta_json": [meta_json] * len(transformed),
                    }
                )
            )

        if not chunks:
            return pd.DataFrame(
                columns=["series_key", "obs_date", "asof_utc", "value", "source", "meta_json"]
            )
        return pd.concat(chunks, ignore_index=True)

    def apply_transform(
        self,
        spec: PITTransformSpec,
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        persist: bool = True,
        overwrite: bool = False,
        lag_policy: ReleaseLagPolicy | None = None,
    ) -> PITTransformResult:
        validate_transform_spec(spec)

        started_utc = pd.Timestamp.now(tz="UTC")
        engine_used = resolve_engine(spec)
        transform_id = spec.transform_id()

        if spec.axis == "obs_path":
            result_df = self._build_obs_path_rows(
                spec=spec,
                start_obs=start_obs,
                end_obs=end_obs,
                start_asof=start_asof,
                end_asof=end_asof,
                lag_policy=lag_policy,
                transform_id=transform_id,
                engine_used=engine_used,
            )
        elif spec.axis == "revision_path":
            result_df = self._build_revision_path_rows(
                spec=spec,
                start_obs=start_obs,
                end_obs=end_obs,
                start_asof=start_asof,
                end_asof=end_asof,
                transform_id=transform_id,
                engine_used=engine_used,
            )
        else:
            raise ValueError(f"Unsupported transform axis: {spec.axis}")

        if persist:
            if overwrite:
                self._delete_transformed_rows(
                    output_series_key=spec.output_series_key,
                    start_obs=start_obs,
                    end_obs=end_obs,
                    start_asof=start_asof,
                    end_asof=end_asof,
                )
            if not result_df.empty:
                self.upsert_pit_observations(result_df)

            self._upsert_transform_metadata(spec, engine_used)

        finished_utc = pd.Timestamp.now(tz="UTC")
        status = "success" if persist else "dry_run"

        if persist:
            self._insert_transform_run(
                transform_id=transform_id,
                start_obs=start_obs,
                end_obs=end_obs,
                start_asof=start_asof,
                end_asof=end_asof,
                rows_written=int(len(result_df)),
                status=status,
                started_utc=started_utc,
                finished_utc=finished_utc,
            )

        return PITTransformResult(
            transform_id=transform_id,
            output_series_key=spec.output_series_key,
            rows_written=int(len(result_df)),
            engine_used=engine_used,
            run_started_utc=started_utc,
            run_finished_utc=finished_utc,
        )

    def list_transforms(self, output_series_key: str | None = None) -> pd.DataFrame:
        query = f"""
            SELECT
                transform_id,
                output_series_key,
                input_series_keys_json,
                axis,
                op,
                params_json,
                engine,
                spec_hash,
                created_utc
            FROM {_PIT_TRANSFORMS_TABLE}
        """
        params: list[object] = []
        if output_series_key is not None:
            query += " WHERE output_series_key = ?"
            params.append(output_series_key)
        query += " ORDER BY created_utc DESC, transform_id"

        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return df
        df["created_utc"] = to_utc_aware(df["created_utc"])
        return df
