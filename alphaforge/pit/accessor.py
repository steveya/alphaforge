from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import duckdb
import pandas as pd

from alphaforge.time.ref_period import RefFreq, RefPeriod

from .exceptions import (
    PITCausalityError,
    PITContractError,
    PITExperimentalFeatureError,
    PITUnsupportedOperationError,
    PITValidationError,
)
from .guards import ReleaseLagPolicy, effective_asof
from .transforms import (
    EngineMismatchPolicy,
    PITEngineResolution,
    PITTransformResult,
    PITTransformSpec,
    apply_binary_obs_path_transform,
    apply_obs_path_transform,
    apply_revision_path_transform,
    coerce_transform_spec,
    resolve_engine,
    serialize_params_for_lineage,
    validate_transform_spec,
)
from .validation import validate_pit_observations

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

    def upsert_pit_observations(self, df: pd.DataFrame, *, strict: bool = True) -> None:
        report = validate_pit_observations(df)
        if report.missing_required_columns:
            raise PITContractError(
                f"Missing required columns: {sorted(report.missing_required_columns)}"
            )

        if strict and report.has_errors:
            raise PITValidationError(report.to_error_message())

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
            raise PITUnsupportedOperationError(f"Unsupported snapshot method: {method}")

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
        ref_period = RefPeriod.parse(ref) if isinstance(ref, str) else ref
        if freq is not None and freq != ref_period.freq:
            raise PITContractError("Reference period frequency does not match requested freq.")
        obs_date = ref_period.end_obs_date()
        return self.get_revision_timeline(
            series_key,
            obs_date,
            start_asof=start_asof,
            end_asof=end_asof,
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
        def _resolve(ref_value: str | RefPeriod | None) -> pd.Timestamp | None:
            if ref_value is None:
                return None
            if isinstance(ref_value, RefPeriod):
                ref_period = ref_value
            else:
                ref_period = RefPeriod.parse(ref_value)
            if freq is not None and ref_period.freq != freq:
                raise PITContractError("Reference period frequency does not match requested freq.")
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
                json.dumps(self._input_series_keys_for_spec(spec)),
                spec.axis,
                spec.op,
                serialize_params_for_lineage(spec.sanitized_params()),
                engine_used,
                spec.spec_hash(),
                to_utc_naive(pd.Timestamp.now("UTC")),
            ],
        )

    @staticmethod
    def _input_series_keys_for_spec(spec: PITTransformSpec) -> list[str]:
        keys = [spec.input_series_key]
        if spec.op == "binary":
            right_key = str(spec.sanitized_params().get("right_series_key", "")).strip()
            if right_key:
                keys.append(right_key)
        return keys

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
        engine_resolution: PITEngineResolution,
        source_asof: pd.Timestamp | None = None,
        source_asof_by_series: Mapping[str, pd.Timestamp] | None = None,
    ) -> str:
        payload: dict[str, object] = {
            "transform_id": transform_id,
            "input_series_key": spec.input_series_key,
            "input_series_keys": PITAccessor._input_series_keys_for_spec(spec),
            "op": spec.op,
            "axis": spec.axis,
            "engine": engine_resolution.engine_used,
            "engine_requested": engine_resolution.engine_requested,
            "params": spec.sanitized_params(),
            "spec_hash": spec.spec_hash(),
            "experimental": bool(spec.axis == "revision_path"),
        }
        if engine_resolution.fallback_reason is not None:
            payload["fallback_reason"] = engine_resolution.fallback_reason
        if source_asof is not None:
            payload["source_asof_utc"] = source_asof.isoformat()
        if source_asof_by_series:
            payload["source_asof_by_series_utc"] = {
                key: ts.isoformat() for key, ts in source_asof_by_series.items()
            }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _list_candidate_asofs_multi(
        self,
        series_keys: Sequence[str],
        *,
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
    ) -> pd.DatetimeIndex:
        unique_keys = sorted({str(k) for k in series_keys if str(k).strip()})
        if not unique_keys:
            return pd.DatetimeIndex([], tz="UTC")

        placeholders = ", ".join(["?"] * len(unique_keys))
        filters = [f"series_key IN ({placeholders})"]
        params: list[object] = [*unique_keys]

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
        engine_resolution: PITEngineResolution,
    ) -> pd.DataFrame:
        input_keys = self._input_series_keys_for_spec(spec)
        if len(input_keys) == 1:
            asof_values = self._list_candidate_asofs(
                input_keys[0],
                start_obs=start_obs,
                end_obs=end_obs,
                start_asof=start_asof,
                end_asof=end_asof,
            )
        else:
            asof_values = self._list_candidate_asofs_multi(
                input_keys,
                start_obs=start_obs,
                end_obs=end_obs,
                start_asof=start_asof,
                end_asof=end_asof,
            )
        chunks: list[pd.DataFrame] = []

        start_obs_utc = to_utc_aware(start_obs) if start_obs is not None else None
        end_obs_utc = to_utc_aware(end_obs) if end_obs is not None else None

        for asof in asof_values:
            source_asof_by_series: dict[str, pd.Timestamp] = {}
            for series_key in input_keys:
                effective = asof
                if lag_policy is not None:
                    effective = effective_asof(effective, series_key, lag_policy)
                if effective > asof:
                    raise PITCausalityError(
                        "Causality violation: effective source_asof is later than output asof. "
                        f"series_key={series_key}, source_asof={effective}, output_asof={asof}"
                    )
                source_asof_by_series[series_key] = effective

            if spec.op == "binary":
                right_series_key = str(spec.sanitized_params().get("right_series_key", "")).strip()
                if not right_series_key:
                    raise PITValidationError(
                        "binary transform requires params['right_series_key']."
                    )
                left_snapshot = self.get_snapshot(
                    spec.input_series_key,
                    source_asof_by_series[spec.input_series_key],
                    start=start_obs,
                    end=end_obs,
                )
                right_snapshot = self.get_snapshot(
                    right_series_key,
                    source_asof_by_series[right_series_key],
                    start=start_obs,
                    end=end_obs,
                )
                transformed = apply_binary_obs_path_transform(
                    left_snapshot,
                    right_snapshot,
                    spec,
                ).dropna()
            else:
                snapshot = self.get_snapshot(
                    spec.input_series_key,
                    source_asof_by_series[spec.input_series_key],
                    start=start_obs,
                    end=end_obs,
                )
                if snapshot.empty:
                    continue
                transformed = apply_obs_path_transform(
                    snapshot,
                    spec,
                    engine=engine_resolution.engine_used,
                ).dropna()

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
                engine_resolution=engine_resolution,
                source_asof=source_asof_by_series.get(spec.input_series_key),
                source_asof_by_series=source_asof_by_series,
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
        engine_resolution: PITEngineResolution,
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
            transformed = apply_revision_path_transform(
                timeline,
                spec,
                engine=engine_resolution.engine_used,
            ).dropna()
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
                engine_resolution=engine_resolution,
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

    def _materialize_transform_rows(
        self,
        spec: PITTransformSpec | Mapping[str, Any],
        start_obs: pd.Timestamp | None,
        end_obs: pd.Timestamp | None,
        start_asof: pd.Timestamp | None,
        end_asof: pd.Timestamp | None,
        *,
        lag_policy: ReleaseLagPolicy | None,
        allow_experimental: bool,
        on_engine_mismatch: EngineMismatchPolicy,
    ) -> tuple[PITTransformSpec, pd.DataFrame, PITEngineResolution]:
        spec_obj = coerce_transform_spec(spec)
        validate_transform_spec(spec_obj)

        if spec_obj.axis == "revision_path" and not allow_experimental:
            raise PITExperimentalFeatureError(
                "axis='revision_path' is experimental. Set allow_experimental=True to enable it."
            )

        engine_resolution = resolve_engine(
            spec_obj,
            on_engine_mismatch=on_engine_mismatch,
        )
        transform_id = spec_obj.transform_id()

        if spec_obj.axis == "obs_path":
            result_df = self._build_obs_path_rows(
                spec=spec_obj,
                start_obs=start_obs,
                end_obs=end_obs,
                start_asof=start_asof,
                end_asof=end_asof,
                lag_policy=lag_policy,
                transform_id=transform_id,
                engine_resolution=engine_resolution,
            )
        elif spec_obj.axis == "revision_path":
            result_df = self._build_revision_path_rows(
                spec=spec_obj,
                start_obs=start_obs,
                end_obs=end_obs,
                start_asof=start_asof,
                end_asof=end_asof,
                transform_id=transform_id,
                engine_resolution=engine_resolution,
            )
        else:
            raise PITUnsupportedOperationError(f"Unsupported transform axis: {spec_obj.axis}")

        return spec_obj, result_df, engine_resolution

    def preview_transform(
        self,
        spec: PITTransformSpec | Mapping[str, Any],
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        lag_policy: ReleaseLagPolicy | None = None,
        allow_experimental: bool = False,
        on_engine_mismatch: EngineMismatchPolicy = "error",
    ) -> pd.DataFrame:
        _, result_df, _ = self._materialize_transform_rows(
            spec,
            start_obs,
            end_obs,
            start_asof,
            end_asof,
            lag_policy=lag_policy,
            allow_experimental=allow_experimental,
            on_engine_mismatch=on_engine_mismatch,
        )
        return result_df.copy()

    def apply_transform(
        self,
        spec: PITTransformSpec | Mapping[str, Any],
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        persist: bool = True,
        overwrite: bool = False,
        lag_policy: ReleaseLagPolicy | None = None,
        allow_experimental: bool = False,
        on_engine_mismatch: EngineMismatchPolicy = "error",
    ) -> PITTransformResult:
        started_utc = pd.Timestamp.now(tz="UTC")

        spec_obj, result_df, engine_resolution = self._materialize_transform_rows(
            spec,
            start_obs,
            end_obs,
            start_asof,
            end_asof,
            lag_policy=lag_policy,
            allow_experimental=allow_experimental,
            on_engine_mismatch=on_engine_mismatch,
        )

        if persist:
            if overwrite:
                self._delete_transformed_rows(
                    output_series_key=spec_obj.output_series_key,
                    start_obs=start_obs,
                    end_obs=end_obs,
                    start_asof=start_asof,
                    end_asof=end_asof,
                )
            if not result_df.empty:
                # Transform outputs can intentionally use period-end labels that are
                # later than asof_utc for certain operators (for example resample).
                self.upsert_pit_observations(result_df, strict=False)

            self._upsert_transform_metadata(spec_obj, engine_resolution.engine_used)

        finished_utc = pd.Timestamp.now(tz="UTC")
        status = "success" if persist else "dry_run"

        if persist:
            self._insert_transform_run(
                transform_id=spec_obj.transform_id(),
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
            transform_id=spec_obj.transform_id(),
            output_series_key=spec_obj.output_series_key,
            rows_written=int(len(result_df)),
            engine_used=engine_resolution.engine_used,
            run_started_utc=started_utc,
            run_finished_utc=finished_utc,
            engine_requested=engine_resolution.engine_requested,
            fallback_reason=engine_resolution.fallback_reason,
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
