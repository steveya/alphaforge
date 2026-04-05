from __future__ import annotations

import ast
import hashlib
import json
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import duckdb
import pandas as pd
from pandas.tseries.offsets import MonthEnd

from alphaforge.time.ref_period import (
    ObsDateAnchor,
    RefFreq,
    RefPeriod,
    coerce_ref_period,
    normalize_obs_date_anchor,
    normalize_ref_freq,
)

from .exceptions import (
    PITCausalityError,
    PITContractError,
    PITExperimentalFeatureError,
    PITUnsupportedOperationError,
    PITValidationError,
    PITValidationWarning,
)
from .guards import ReleaseLagPolicy, effective_asof
from .models import (
    PITExpressionGraphResult,
    PITExpressionGraphSpec,
    PITExpressionNode,
    ReleaseRecord,
    ReleaseSelectionPolicy,
    SnapshotSeriesSpec,
    coerce_expression_graph_spec,
    coerce_snapshot_series_spec,
    normalize_release_selection_policy,
)
from .pipelines import (
    PITPipelineResult,
    PITPipelineSpec,
    coerce_pipeline_spec,
)
from .queries import (
    RefRevisionQuery,
    RefSnapshotQuery,
    coerce_ref_revision_query,
    coerce_ref_snapshot_query,
)
from .ref_entity import make_ref_entity_id
from .transforms import (
    EngineMismatchPolicy,
    PITEngineResolution,
    PITTransformResult,
    PITTransformSpec,
    apply_binary_obs_path_transform,
    apply_coalesce_obs_path_transform,
    apply_obs_path_transform,
    apply_revision_path_transform,
    apply_splice_obs_path_transform,
    coerce_transform_spec,
    resolve_engine,
    serialize_params_for_lineage,
    transform_input_series_keys,
    validate_transform_spec,
)
from .validation import validate_pit_observations

_PIT_TABLE = "pit_observations"
_PIT_TRANSFORMS_TABLE = "pit_transforms"
_PIT_TRANSFORM_RUNS_TABLE = "pit_transform_runs"
_PIT_PIPELINES_TABLE = "pit_pipelines"
_PIT_PIPELINE_RUNS_TABLE = "pit_pipeline_runs"
_PIT_EXPR_GRAPHS_TABLE = "pit_expression_graphs"
_PIT_EXPR_GRAPH_RUNS_TABLE = "pit_expression_graph_runs"


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

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_PIT_PIPELINES_TABLE} (
            pipeline_id TEXT PRIMARY KEY,
            spec_hash TEXT NOT NULL,
            spec_json TEXT NOT NULL,
            description TEXT,
            created_utc TIMESTAMP NOT NULL DEFAULT now()
        );
        """
    )

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_PIT_PIPELINE_RUNS_TABLE} (
            run_id TEXT PRIMARY KEY,
            pipeline_id TEXT NOT NULL,
            start_obs TIMESTAMP,
            end_obs TIMESTAMP,
            start_asof TIMESTAMP,
            end_asof TIMESTAMP,
            incremental BOOLEAN NOT NULL,
            requested_since_asof TIMESTAMP,
            effective_start_asof TIMESTAMP,
            requested_since_run_id TEXT,
            max_output_asof TIMESTAMP,
            rows_written BIGINT NOT NULL,
            step_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            started_utc TIMESTAMP NOT NULL,
            finished_utc TIMESTAMP NOT NULL
        );
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS pit_pipeline_runs_pipeline_id
        ON {_PIT_PIPELINE_RUNS_TABLE}(pipeline_id, started_utc);
        """
    )

    conn.execute(
        f"""
        ALTER TABLE {_PIT_PIPELINE_RUNS_TABLE}
        ADD COLUMN IF NOT EXISTS max_output_asof TIMESTAMP
        """
    )

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_PIT_EXPR_GRAPHS_TABLE} (
            graph_id TEXT PRIMARY KEY,
            spec_hash TEXT NOT NULL,
            spec_json TEXT NOT NULL,
            description TEXT,
            created_utc TIMESTAMP NOT NULL DEFAULT now()
        );
        """
    )

    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {_PIT_EXPR_GRAPH_RUNS_TABLE} (
            run_id TEXT PRIMARY KEY,
            graph_id TEXT NOT NULL,
            start_obs TIMESTAMP,
            end_obs TIMESTAMP,
            start_asof TIMESTAMP,
            end_asof TIMESTAMP,
            incremental BOOLEAN NOT NULL,
            requested_since_asof TIMESTAMP,
            effective_start_asof TIMESTAMP,
            requested_since_run_id TEXT,
            max_output_asof TIMESTAMP,
            rows_written BIGINT NOT NULL,
            node_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            started_utc TIMESTAMP NOT NULL,
            finished_utc TIMESTAMP NOT NULL
        );
        """
    )

    conn.execute(
        f"""
        CREATE INDEX IF NOT EXISTS pit_expression_graph_runs_graph_id
        ON {_PIT_EXPR_GRAPH_RUNS_TABLE}(graph_id, started_utc);
        """
    )

    conn.execute(
        f"""
        ALTER TABLE {_PIT_EXPR_GRAPH_RUNS_TABLE}
        ADD COLUMN IF NOT EXISTS max_output_asof TIMESTAMP
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


def _resolve_ingestion_policy(strict: bool | str) -> Literal["error", "warn", "coerce"]:
    if isinstance(strict, bool):
        return "error" if strict else "warn"
    mode = str(strict).strip().lower()
    if mode not in {"error", "warn", "coerce"}:
        raise PITContractError(
            "strict must be bool or one of {'error', 'warn', 'coerce'}."
        )
    return mode  # type: ignore[return-value]


def _coerce_pit_observations(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["__input_order"] = range(len(out))

    required = ["series_key", "obs_date", "asof_utc", "value"]
    for col in required:
        if col not in out.columns:
            raise PITContractError(f"Missing required columns: {sorted({col})}")

    out["obs_date"] = pd.to_datetime(out["obs_date"], errors="coerce", utc=True)
    out["asof_utc"] = pd.to_datetime(out["asof_utc"], errors="coerce", utc=True)
    if "release_time_utc" in out.columns:
        out["release_time_utc"] = pd.to_datetime(
            out["release_time_utc"], errors="coerce", utc=True
        )
    if "ingested_utc" in out.columns:
        out["ingested_utc"] = pd.to_datetime(out["ingested_utc"], errors="coerce", utc=True)

    # Null required rows after coercion are irrecoverable and removed deterministically.
    out = out.dropna(subset=required)

    # Drop PIT-future rows (obs date known after its as-of timestamp).
    out = out[out["obs_date"] <= out["asof_utc"]]

    # Keep the last input row for duplicate PIT keys.
    out = out.sort_values("__input_order")
    out = out.drop_duplicates(subset=["series_key", "obs_date", "asof_utc"], keep="last")
    out = out.sort_values("__input_order").drop(columns=["__input_order"]).reset_index(drop=True)
    return out


def _expression_hash(expression: str) -> str:
    return hashlib.sha256(expression.strip().encode("utf-8")).hexdigest()


def _ensure_utc_index(idx: pd.Index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(idx))
    return out.tz_localize("UTC") if out.tz is None else out.tz_convert("UTC")


def _coerce_series_numeric(s: pd.Series) -> pd.Series:
    out = pd.to_numeric(pd.Series(s).copy(), errors="coerce")
    out.index = _ensure_utc_index(out.index)
    return out.sort_index()


def _validate_expression_ast(
    node: ast.AST,
    aliases: set[str],
) -> None:
    if isinstance(node, ast.Expression):
        _validate_expression_ast(node.body, aliases)
        return

    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            raise PITValidationError("Expression supports only +, -, *, /.")  # pragma: no cover
        _validate_expression_ast(node.left, aliases)
        _validate_expression_ast(node.right, aliases)
        return

    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise PITValidationError("Expression unary operators support only + and -.")
        _validate_expression_ast(node.operand, aliases)
        return

    if isinstance(node, ast.Name):
        if node.id not in aliases:
            raise PITValidationError(f"Unknown expression alias: '{node.id}'.")
        return

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise PITValidationError("Expression call must use a simple function name.")
        if node.func.id not in {"lag", "diff"}:
            raise PITValidationError("Expression supports only lag(alias, n) and diff(alias, n).")
        if len(node.args) != 2:
            raise PITValidationError(
                f"Expression function '{node.func.id}' expects exactly 2 arguments."
            )
        arg0 = node.args[0]
        arg1 = node.args[1]
        if not isinstance(arg0, ast.Name) or arg0.id not in aliases:
            raise PITValidationError(
                f"Expression function '{node.func.id}' first argument must be an input alias."
            )
        if not isinstance(arg1, ast.Constant) or not isinstance(arg1.value, int):
            raise PITValidationError(
                f"Expression function '{node.func.id}' second argument must be an integer."
            )
        if int(arg1.value) <= 0:
            raise PITValidationError(
                f"Expression function '{node.func.id}' periods must be > 0."
            )
        return

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return
        raise PITValidationError("Expression constants must be numeric.")

    raise PITValidationError("Expression contains unsupported syntax.")


def _eval_expression_ast(
    node: ast.AST,
    env: Mapping[str, pd.Series],
    *,
    join: Literal["inner", "left", "right", "outer"],
    fill_value: float | None,
) -> pd.Series | float:
    if isinstance(node, ast.Expression):
        return _eval_expression_ast(node.body, env, join=join, fill_value=fill_value)

    if isinstance(node, ast.Name):
        return _coerce_series_numeric(env[node.id])

    if isinstance(node, ast.Constant):
        return float(node.value)  # type: ignore[arg-type]

    if isinstance(node, ast.UnaryOp):
        value = _eval_expression_ast(node.operand, env, join=join, fill_value=fill_value)
        if isinstance(value, pd.Series):
            return value if isinstance(node.op, ast.UAdd) else -value
        return value if isinstance(node.op, ast.UAdd) else -float(value)

    if isinstance(node, ast.Call):
        func_name = str(node.func.id)  # type: ignore[attr-defined]
        alias = str(node.args[0].id)  # type: ignore[attr-defined]
        periods = int(node.args[1].value)  # type: ignore[attr-defined]
        base = _coerce_series_numeric(env[alias])
        if func_name == "lag":
            return base.shift(periods=periods)
        if func_name == "diff":
            return base.diff(periods=periods)
        raise PITValidationError(f"Unsupported expression function '{func_name}'.")

    if isinstance(node, ast.BinOp):
        left = _eval_expression_ast(node.left, env, join=join, fill_value=fill_value)
        right = _eval_expression_ast(node.right, env, join=join, fill_value=fill_value)

        if isinstance(left, pd.Series) and isinstance(right, pd.Series):
            left_aligned, right_aligned = left.align(right, join=join)
            if fill_value is not None:
                left_aligned = left_aligned.fillna(float(fill_value))
                right_aligned = right_aligned.fillna(float(fill_value))
            if isinstance(node.op, ast.Add):
                out = left_aligned + right_aligned
            elif isinstance(node.op, ast.Sub):
                out = left_aligned - right_aligned
            elif isinstance(node.op, ast.Mult):
                out = left_aligned * right_aligned
            elif isinstance(node.op, ast.Div):
                out = left_aligned / right_aligned
                out = out.replace([float("inf"), float("-inf")], pd.NA)
            else:  # pragma: no cover
                raise PITValidationError("Unsupported expression binary operator.")
            return pd.Series(out)

        if isinstance(left, pd.Series):
            right_num = float(right)
            if fill_value is not None:
                left = left.fillna(float(fill_value))
            if isinstance(node.op, ast.Add):
                return left + right_num
            if isinstance(node.op, ast.Sub):
                return left - right_num
            if isinstance(node.op, ast.Mult):
                return left * right_num
            if isinstance(node.op, ast.Div):
                out = left / right_num
                return out.replace([float("inf"), float("-inf")], pd.NA)
            raise PITValidationError("Unsupported expression binary operator.")  # pragma: no cover

        if isinstance(right, pd.Series):
            left_num = float(left)
            if fill_value is not None:
                right = right.fillna(float(fill_value))
            if isinstance(node.op, ast.Add):
                return left_num + right
            if isinstance(node.op, ast.Sub):
                return left_num - right
            if isinstance(node.op, ast.Mult):
                return left_num * right
            if isinstance(node.op, ast.Div):
                out = left_num / right
                return out.replace([float("inf"), float("-inf")], pd.NA)
            raise PITValidationError("Unsupported expression binary operator.")  # pragma: no cover

        left_num = float(left)
        right_num = float(right)
        if isinstance(node.op, ast.Add):
            return left_num + right_num
        if isinstance(node.op, ast.Sub):
            return left_num - right_num
        if isinstance(node.op, ast.Mult):
            return left_num * right_num
        if isinstance(node.op, ast.Div):
            return left_num / right_num
        raise PITValidationError("Unsupported expression binary operator.")  # pragma: no cover

    raise PITValidationError("Expression contains unsupported syntax.")  # pragma: no cover


def _evaluate_expression_series(
    expression: str,
    env: Mapping[str, pd.Series],
    *,
    join: Literal["inner", "left", "right", "outer"],
    fill_value: float | None,
) -> pd.Series:
    try:
        parsed = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise PITValidationError(f"Invalid expression syntax: {exc.msg}") from exc
    _validate_expression_ast(parsed, set(env.keys()))
    out = _eval_expression_ast(parsed, env, join=join, fill_value=fill_value)
    if not isinstance(out, pd.Series):
        raise PITValidationError("Expression must resolve to a series output.")
    out = _coerce_series_numeric(out).sort_index()
    return out


@dataclass
class PITAccessor:
    conn: duckdb.DuckDBPyConnection

    @classmethod
    def open(cls, root: str | Path) -> "PITAccessor":
        """Open a PIT accessor from a DuckDBParquetStore root."""
        from alphaforge.store.duckdb_parquet import DuckDBParquetStore

        store = DuckDBParquetStore(root=str(root))
        return cls(store.conn())

    def __post_init__(self) -> None:
        ensure_pit_table(self.conn)

    def upsert_pit_observations(
        self,
        df: pd.DataFrame,
        *,
        strict: bool | Literal["error", "warn", "coerce"] = True,
    ) -> None:
        policy = _resolve_ingestion_policy(strict)
        report = validate_pit_observations(df)
        if report.missing_required_columns:
            raise PITContractError(
                f"Missing required columns: {sorted(report.missing_required_columns)}"
            )

        if policy == "error" and report.has_errors:
            raise PITValidationError(report.to_error_message())

        if policy == "warn" and report.has_errors:
            warnings.warn(
                f"PIT validation warning: {report.to_error_message()}",
                PITValidationWarning,
                stacklevel=2,
            )

        incoming = df
        if policy == "coerce":
            incoming = _coerce_pit_observations(df)
            repaired_report = validate_pit_observations(incoming)
            if repaired_report.missing_required_columns:
                raise PITContractError(
                    f"Missing required columns: {sorted(repaired_report.missing_required_columns)}"
                )
            if repaired_report.has_errors:
                raise PITValidationError(
                    "strict='coerce' failed to repair all PIT validation issues: "
                    f"{repaired_report.to_error_message()}"
                )

            dropped_rows = int(len(df) - len(incoming))
            if report.has_errors or dropped_rows > 0:
                detail = report.to_error_message() if report.has_errors else "none"
                warnings.warn(
                    "PIT validation warning: strict='coerce' repaired input rows "
                    f"(dropped_rows={dropped_rows}, original_issues={detail}).",
                    PITValidationWarning,
                    stacklevel=2,
                )

        normalized = _normalize_datetime_columns(
            incoming, ["obs_date", "asof_utc", "release_time_utc", "ingested_utc"]
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

        df = self._get_snapshot_rows_with_source_asof(
            series_key,
            asof,
            start=start,
            end=end,
        )
        if df.empty:
            return pd.Series(dtype="float64", name=series_key)

        series = pd.Series(
            df["value"].to_numpy(),
            index=to_utc_aware(df["obs_date"]),
            name=series_key,
        )
        series.index.name = "obs_date"
        return series

    def get_snapshot_multi(
        self,
        series_keys: Sequence[str],
        asof: pd.Timestamp,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        method: Literal["latest_leq"] = "latest_leq",
    ) -> pd.DataFrame:
        if method != "latest_leq":
            raise PITUnsupportedOperationError(f"Unsupported snapshot method: {method}")

        unique_keys = sorted({str(k) for k in series_keys if str(k).strip()})
        if not unique_keys:
            return pd.DataFrame(
                {
                    "series_key": pd.Series(dtype="object"),
                    "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                    "source_asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "value": pd.Series(dtype="float64"),
                }
            )

        asof_ts = to_utc_naive(asof)
        start_ts = to_utc_naive(start)
        end_ts = to_utc_naive(end)
        placeholders = ", ".join(["?"] * len(unique_keys))
        filters = [f"series_key IN ({placeholders})", "asof_utc <= ?"]
        params: list[object] = [*unique_keys, asof_ts]
        if start_ts is not None:
            filters.append("obs_date >= ?")
            params.append(start_ts)
        if end_ts is not None:
            filters.append("obs_date <= ?")
            params.append(end_ts)

        where_clause = " AND ".join(filters)
        query = f"""
            SELECT series_key, obs_date, source_asof_utc, value
            FROM (
                SELECT
                    series_key,
                    obs_date,
                    asof_utc AS source_asof_utc,
                    value,
                    ROW_NUMBER() OVER (
                        PARTITION BY series_key, obs_date
                        ORDER BY asof_utc DESC
                    ) AS rn
                FROM {_PIT_TABLE}
                WHERE {where_clause}
            ) ranked
            WHERE rn = 1
            ORDER BY series_key, obs_date
        """
        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return pd.DataFrame(
                {
                    "series_key": pd.Series(dtype="object"),
                    "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                    "source_asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "value": pd.Series(dtype="float64"),
                }
            )

        out = df.copy()
        out["obs_date"] = to_utc_aware(out["obs_date"])
        out["source_asof_utc"] = to_utc_aware(out["source_asof_utc"])
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out.reset_index(drop=True)

    def _get_snapshot_rows_with_source_asof(
        self,
        series_key: str,
        asof: pd.Timestamp,
        *,
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
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
            SELECT obs_date, source_asof_utc, value
            FROM (
                SELECT
                    obs_date,
                    asof_utc AS source_asof_utc,
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
            return df

        df["obs_date"] = to_utc_aware(df["obs_date"])
        df["source_asof_utc"] = to_utc_aware(df["source_asof_utc"])
        return df

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

    def get_revision_path(
        self,
        series_key: str,
        obs_date: pd.Timestamp,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
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
            SELECT series_key, obs_date, asof_utc, value, revision_id
            FROM {_PIT_TABLE}
            WHERE {where_clause}
            ORDER BY asof_utc ASC
        """
        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return pd.DataFrame(
                {
                    "series_key": pd.Series(dtype="object"),
                    "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                    "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "value": pd.Series(dtype="float64"),
                    "revision_id": pd.Series(dtype="object"),
                }
            )

        out = df.copy()
        out["obs_date"] = to_utc_aware(out["obs_date"])
        out["asof_utc"] = to_utc_aware(out["asof_utc"])
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out.reset_index(drop=True)

    def get_revision_path_multi(self, requests: pd.DataFrame) -> pd.DataFrame:
        required = {"request_id", "series_key", "obs_date"}
        missing = required - set(requests.columns)
        if missing:
            raise PITContractError(
                f"Revision path requests missing required columns: {sorted(missing)}"
            )

        if requests.empty:
            return pd.DataFrame(
                {
                    "request_id": pd.Series(dtype="object"),
                    "series_key": pd.Series(dtype="object"),
                    "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                    "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "value": pd.Series(dtype="float64"),
                    "revision_id": pd.Series(dtype="object"),
                }
            )

        req = requests.copy()
        req["obs_date"] = pd.to_datetime(req["obs_date"], utc=True, errors="coerce")
        if "start_asof" in req.columns:
            req["start_asof"] = pd.to_datetime(req["start_asof"], utc=True, errors="coerce")
        else:
            req["start_asof"] = pd.NaT
        if "end_asof" in req.columns:
            req["end_asof"] = pd.to_datetime(req["end_asof"], utc=True, errors="coerce")
        else:
            req["end_asof"] = pd.NaT
        req = req.dropna(subset=["request_id", "series_key", "obs_date"]).copy()
        if req.empty:
            return pd.DataFrame(
                {
                    "request_id": pd.Series(dtype="object"),
                    "series_key": pd.Series(dtype="object"),
                    "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                    "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "value": pd.Series(dtype="float64"),
                    "revision_id": pd.Series(dtype="object"),
                }
            )

        normalized = req.loc[
            :, ["request_id", "series_key", "obs_date", "start_asof", "end_asof"]
        ].copy()
        normalized["obs_date"] = to_utc_naive(normalized["obs_date"])
        normalized["start_asof"] = to_utc_naive(normalized["start_asof"])
        normalized["end_asof"] = to_utc_naive(normalized["end_asof"])
        self.conn.register("pit_revision_path_requests", normalized)
        try:
            df = self.conn.execute(
                f"""
                SELECT
                    r.request_id,
                    p.series_key,
                    p.obs_date,
                    p.asof_utc,
                    p.value,
                    p.revision_id
                FROM pit_revision_path_requests r
                INNER JOIN {_PIT_TABLE} p
                    ON p.series_key = r.series_key
                   AND p.obs_date = r.obs_date
                WHERE (r.start_asof IS NULL OR p.asof_utc >= r.start_asof)
                  AND (r.end_asof IS NULL OR p.asof_utc <= r.end_asof)
                ORDER BY r.request_id, p.asof_utc ASC
                """
            ).fetchdf()
        finally:
            self.conn.unregister("pit_revision_path_requests")

        if df.empty:
            return pd.DataFrame(
                {
                    "request_id": pd.Series(dtype="object"),
                    "series_key": pd.Series(dtype="object"),
                    "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                    "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "value": pd.Series(dtype="float64"),
                    "revision_id": pd.Series(dtype="object"),
                }
            )

        out = df.copy()
        out["obs_date"] = to_utc_aware(out["obs_date"])
        out["asof_utc"] = to_utc_aware(out["asof_utc"])
        out["value"] = pd.to_numeric(out["value"], errors="coerce")
        return out.reset_index(drop=True)

    def get_revision_timeline_ref(
        self,
        series_key: str,
        ref: object,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        freq: RefFreq | None = None,
        obs_date_anchor: ObsDateAnchor | str = "end",
    ) -> pd.Series:
        ref_period = self._resolve_ref_period(ref, freq=freq, obs_date_anchor=obs_date_anchor)
        obs_date = ref_period.obs_date(anchor=obs_date_anchor)
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
        start_ref: object | None = None,
        end_ref: object | None = None,
        *,
        freq: RefFreq | None = None,
        obs_date_anchor: ObsDateAnchor | str = "end",
    ) -> pd.Series:
        def _resolve(ref_value: object | None) -> pd.Timestamp | None:
            if ref_value is None:
                return None
            return self._resolve_ref_period(
                ref_value,
                freq=freq,
                obs_date_anchor=obs_date_anchor,
            ).obs_date(anchor=obs_date_anchor)

        start_ts = _resolve(start_ref)
        end_ts = _resolve(end_ref)
        return self.get_snapshot(series_key, asof, start=start_ts, end=end_ts)

    @staticmethod
    def _resolve_ref_period(
        ref: object,
        freq: RefFreq | None = None,
        obs_date_anchor: ObsDateAnchor | str = "end",
    ) -> RefPeriod:
        try:
            return coerce_ref_period(ref, freq=freq, obs_date_anchor=obs_date_anchor)
        except ValueError as exc:
            raise PITContractError(str(exc)) from exc

    def snapshot_ref(
        self,
        query: RefSnapshotQuery | Mapping[str, Any],
    ) -> pd.Series:
        query_obj = coerce_ref_snapshot_query(query)
        obs_date_anchor = normalize_obs_date_anchor(query_obj.obs_date_anchor)
        freq = normalize_ref_freq(query_obj.freq)
        snap = self.get_snapshot_ref(
            query_obj.series_key,
            query_obj.asof,
            start_ref=query_obj.start_ref,
            end_ref=query_obj.end_ref,
            freq=freq,
            obs_date_anchor=obs_date_anchor,
        )
        if snap.empty:
            return pd.Series(
                index=pd.Index([], dtype="object", name="ref_period"),
                dtype="float64",
                name=query_obj.series_key,
            )

        ref_index = [
            self._resolve_ref_period(
                obs_date,
                freq=freq,
                obs_date_anchor=obs_date_anchor,
            )
            for obs_date in snap.index
        ]
        if len(ref_index) != len(set(ref_index)):
            raise PITContractError(
                "Ref snapshot query produced duplicate reference periods. "
                "Check the requested frequency and obs_date_anchor."
            )

        out = pd.Series(
            snap.to_numpy(),
            index=pd.Index(ref_index, name="ref_period"),
            name=query_obj.series_key,
        )
        out.attrs["freq"] = freq
        out.attrs["obs_date_anchor"] = obs_date_anchor
        return out

    def revisions_ref(
        self,
        query: RefRevisionQuery | Mapping[str, Any],
    ) -> pd.Series:
        query_obj = coerce_ref_revision_query(query)
        obs_date_anchor = normalize_obs_date_anchor(query_obj.obs_date_anchor)
        freq = normalize_ref_freq(query_obj.freq)
        ref_period = self._resolve_ref_period(
            query_obj.ref,
            freq=freq,
            obs_date_anchor=obs_date_anchor,
        )
        series = self.get_revision_timeline_ref(
            query_obj.series_key,
            ref_period,
            start_asof=query_obj.start_asof,
            end_asof=query_obj.end_asof,
            freq=freq,
            obs_date_anchor=obs_date_anchor,
        )
        series.name = make_ref_entity_id(query_obj.series_key, ref_period)
        series.attrs["ref_period"] = ref_period
        series.attrs["obs_date_anchor"] = obs_date_anchor
        return series

    def list_release_stream(
        self,
        series_key: str,
        ref: object,
        asof: pd.Timestamp | None = None,
        *,
        freq: RefFreq | None = None,
    ) -> pd.DataFrame:
        ref_period = self._resolve_ref_period(ref, freq=freq)
        obs_date = ref_period.end_obs_date()

        filters = ["series_key = ?", "obs_date = ?"]
        params: list[object] = [series_key, to_utc_naive(obs_date)]
        if asof is not None:
            filters.append("asof_utc <= ?")
            params.append(to_utc_naive(asof))

        where_clause = " AND ".join(filters)
        query = f"""
            SELECT series_key, obs_date, asof_utc, value, revision_id
            FROM {_PIT_TABLE}
            WHERE {where_clause}
            ORDER BY asof_utc ASC
        """
        out = self.conn.execute(query, params).fetchdf()
        if out.empty:
            return pd.DataFrame(
                columns=[
                    "series_key",
                    "ref_key",
                    "obs_date",
                    "asof_utc",
                    "release_rank",
                    "value",
                    "revision_id",
                    "is_first",
                    "is_latest",
                ]
            )

        out["obs_date"] = to_utc_aware(out["obs_date"])
        out["asof_utc"] = to_utc_aware(out["asof_utc"])
        out["release_rank"] = range(1, len(out) + 1)
        out["ref_key"] = ref_period.to_key()
        out["is_first"] = out["release_rank"] == 1
        out["is_latest"] = out["release_rank"] == int(out["release_rank"].max())

        cols = [
            "series_key",
            "ref_key",
            "obs_date",
            "asof_utc",
            "release_rank",
            "value",
            "revision_id",
            "is_first",
            "is_latest",
        ]
        return out[cols].reset_index(drop=True)

    def resolve_release(
        self,
        series_key: str,
        ref: object,
        *,
        policy: ReleaseSelectionPolicy | Mapping[str, Any] | str = "latest",
        asof: pd.Timestamp | None = None,
        freq: RefFreq | None = None,
    ) -> ReleaseRecord | None:
        stream = self.list_release_stream(series_key, ref, asof=asof, freq=freq)
        if stream.empty:
            return None

        mode, value = normalize_release_selection_policy(policy)
        selected = stream

        if mode == "first":
            row = selected.iloc[0]
        elif mode == "latest":
            row = selected.iloc[-1]
        elif mode == "rank":
            rank = int(value)  # type: ignore[arg-type]
            subset = selected[selected["release_rank"] == rank]
            if subset.empty:
                return None
            row = subset.iloc[0]
        elif mode == "horizon":
            horizon = pd.Timedelta(value)
            cutoff = pd.Timestamp(selected.iloc[0]["obs_date"]) + horizon
            if asof is not None:
                asof_ts = to_utc_aware(asof)
                if asof_ts < cutoff:
                    cutoff = asof_ts
            subset = selected[pd.to_datetime(selected["asof_utc"], utc=True) <= cutoff]
            if subset.empty:
                return None
            row = subset.iloc[-1]
        else:
            raise PITContractError(f"Unsupported release policy mode: {mode}")

        return ReleaseRecord(
            series_key=str(row["series_key"]),
            ref_key=str(row["ref_key"]),
            obs_date=pd.Timestamp(row["obs_date"]),
            asof_utc=pd.Timestamp(row["asof_utc"]),
            release_rank=int(row["release_rank"]),
            value=(
                float(row["value"])
                if ("value" in row and pd.notna(row["value"]))
                else None
            ),
            revision_id=(
                str(row["revision_id"])
                if ("revision_id" in row and pd.notna(row["revision_id"]))
                else None
            ),
        )

    def _snapshot_with_release_policy(
        self,
        series_key: str,
        asof: pd.Timestamp,
        *,
        policy: ReleaseSelectionPolicy | Mapping[str, Any] | str = "latest",
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.Series:
        rows = self._snapshot_rows_with_release_policy(
            series_key,
            asof,
            policy=policy,
            start=start,
            end=end,
        )
        if rows.empty:
            return pd.Series(dtype="float64", name=series_key)

        out = pd.Series(
            rows["value"].to_numpy(),
            index=pd.DatetimeIndex(rows["obs_date"]),
            name=series_key,
        ).sort_index()
        out.index.name = "obs_date"
        return out

    def _snapshot_rows_with_release_policy(
        self,
        series_key: str,
        asof: pd.Timestamp,
        *,
        policy: ReleaseSelectionPolicy | Mapping[str, Any] | str = "latest",
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        mode, value = normalize_release_selection_policy(policy)
        if mode == "latest":
            return self._get_snapshot_rows_with_source_asof(
                series_key,
                asof,
                start=start,
                end=end,
            )

        filters = ["series_key = ?", "asof_utc <= ?"]
        params: list[object] = [series_key, to_utc_naive(asof)]
        if start is not None:
            filters.append("obs_date >= ?")
            params.append(to_utc_naive(start))
        if end is not None:
            filters.append("obs_date <= ?")
            params.append(to_utc_naive(end))

        where_clause = " AND ".join(filters)
        query = f"""
            SELECT obs_date, asof_utc, value
            FROM {_PIT_TABLE}
            WHERE {where_clause}
            ORDER BY obs_date ASC, asof_utc ASC
        """
        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return pd.DataFrame(
                {
                    "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                    "source_asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "value": pd.Series(dtype="float64"),
                }
            )

        df["obs_date"] = to_utc_aware(df["obs_date"])
        df["asof_utc"] = to_utc_aware(df["asof_utc"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        selected_chunks: list[pd.DataFrame] = []
        for obs_date, group in df.groupby("obs_date", sort=True):
            group = group.sort_values("asof_utc")
            if mode == "first":
                selected = group.iloc[[0]]
            elif mode == "rank":
                rank = int(value)  # type: ignore[arg-type]
                if rank > len(group):
                    continue
                selected = group.iloc[[rank - 1]]
            elif mode == "horizon":
                horizon = pd.Timedelta(value)
                cutoff = pd.Timestamp(obs_date) + horizon
                cutoff = min(cutoff, to_utc_aware(asof))
                window = group[group["asof_utc"] <= cutoff]
                if window.empty:
                    continue
                selected = window.iloc[[-1]]
            else:
                raise PITContractError(f"Unsupported release policy mode: {mode}")
            selected_chunks.append(selected)

        if not selected_chunks:
            return pd.DataFrame(
                {
                    "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                    "source_asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                    "value": pd.Series(dtype="float64"),
                }
            )

        selected_df = pd.concat(selected_chunks, ignore_index=True).rename(
            columns={"asof_utc": "source_asof_utc"}
        )
        return selected_df[["obs_date", "source_asof_utc", "value"]].sort_values(
            "obs_date"
        ).reset_index(drop=True)

    @staticmethod
    def _align_snapshot_index(
        index: pd.DatetimeIndex,
        align: Literal["month_end", "quarter_end"],
    ) -> pd.DatetimeIndex:
        idx = pd.DatetimeIndex(index)
        idx = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
        naive = idx.tz_localize(None)
        if align == "month_end":
            aligned = naive + MonthEnd(0)
        elif align == "quarter_end":
            aligned = naive.to_period("Q").to_timestamp(how="end").normalize()
        else:
            raise PITContractError("align must be one of {'month_end', 'quarter_end'}.")
        return pd.DatetimeIndex(aligned).tz_localize("UTC")

    @staticmethod
    def _empty_snapshot_panel_long() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "series_key": pd.Series(dtype="object"),
                "series_alias": pd.Series(dtype="object"),
                "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                "source_obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                "source_asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "value": pd.Series(dtype="float64"),
            }
        )

    def _snapshot_bounds_from_spec(
        self,
        spec: SnapshotSeriesSpec,
    ) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        anchor = normalize_obs_date_anchor(spec.obs_date_anchor)
        freq = normalize_ref_freq(spec.freq)

        def _resolve(ref_value: object | None) -> pd.Timestamp | None:
            if ref_value is None:
                return None
            return self._resolve_ref_period(
                ref_value,
                freq=freq,
                obs_date_anchor=anchor,
            ).obs_date(anchor=anchor)

        return _resolve(spec.start_ref), _resolve(spec.end_ref)

    def _snapshot_panel_rows_from_frame(
        self,
        spec: SnapshotSeriesSpec,
        frame: pd.DataFrame,
        *,
        align: Literal["month_end", "quarter_end"],
    ) -> pd.DataFrame:
        if frame.empty:
            return self._empty_snapshot_panel_long()

        rows = frame.copy()
        rows["source_obs_date"] = to_utc_aware(rows["source_obs_date"])
        rows["source_asof_utc"] = to_utc_aware(rows["source_asof_utc"])
        rows["value"] = pd.to_numeric(rows["value"], errors="coerce")
        rows["obs_date"] = self._align_snapshot_index(
            pd.DatetimeIndex(rows["source_obs_date"]),
            align=align,
        )
        rows["series_key"] = spec.series_key
        rows["series_alias"] = spec.alias or spec.series_key
        rows = rows.loc[
            :, ["series_key", "series_alias", "obs_date", "source_obs_date", "source_asof_utc", "value"]
        ]
        rows = rows.sort_values(["series_key", "obs_date", "source_obs_date", "source_asof_utc"])
        rows = rows.drop_duplicates(subset=["series_key", "obs_date"], keep="last")
        return rows.reset_index(drop=True)

    def build_snapshot_panel_long(
        self,
        series_specs: Sequence[SnapshotSeriesSpec | Mapping[str, Any]],
        asof: pd.Timestamp,
        *,
        align: Literal["month_end", "quarter_end"] = "month_end",
    ) -> pd.DataFrame:
        normalized_specs = [coerce_snapshot_series_spec(raw_spec) for raw_spec in series_specs]
        if not normalized_specs:
            return self._empty_snapshot_panel_long()

        aliases = [spec.alias or spec.series_key for spec in normalized_specs]
        if len(set(aliases)) != len(aliases):
            raise PITContractError("Snapshot panel aliases must be unique.")

        latest_batches: dict[
            tuple[pd.Timestamp | None, pd.Timestamp | None],
            list[SnapshotSeriesSpec],
        ] = {}
        pieces: list[pd.DataFrame] = []

        for spec in normalized_specs:
            start_obs, end_obs = self._snapshot_bounds_from_spec(spec)
            mode, _ = normalize_release_selection_policy(spec.release_policy)
            if mode == "latest":
                latest_batches.setdefault((start_obs, end_obs), []).append(spec)
                continue

            rows = self._snapshot_rows_with_release_policy(
                spec.series_key,
                asof=asof,
                policy=spec.release_policy,
                start=start_obs,
                end=end_obs,
            ).rename(columns={"obs_date": "source_obs_date"})
            pieces.append(
                self._snapshot_panel_rows_from_frame(
                    spec,
                    rows,
                    align=align,
                )
            )

        for (start_obs, end_obs), specs in latest_batches.items():
            batch = self.get_snapshot_multi(
                [spec.series_key for spec in specs],
                asof=asof,
                start=start_obs,
                end=end_obs,
            ).rename(columns={"obs_date": "source_obs_date"})
            for spec in specs:
                spec_rows = batch.loc[batch["series_key"] == spec.series_key].copy()
                pieces.append(
                    self._snapshot_panel_rows_from_frame(
                        spec,
                        spec_rows,
                        align=align,
                    )
                )

        if not pieces:
            return self._empty_snapshot_panel_long()

        out = pd.concat(pieces, ignore_index=True)
        if out.empty:
            return self._empty_snapshot_panel_long()
        return out.sort_values(["obs_date", "series_alias"]).reset_index(drop=True)

    def build_snapshot_panel(
        self,
        series_specs: Sequence[SnapshotSeriesSpec | Mapping[str, Any]],
        asof: pd.Timestamp,
        *,
        align: Literal["month_end", "quarter_end"] = "month_end",
        join: Literal["inner", "left", "right", "outer"] = "outer",
    ) -> pd.DataFrame:
        if join not in {"inner", "left", "right", "outer"}:
            raise PITContractError("join must be one of {'inner', 'left', 'right', 'outer'}.")
        normalized_specs = [coerce_snapshot_series_spec(raw_spec) for raw_spec in series_specs]
        if not normalized_specs:
            return pd.DataFrame()
        aliases = [spec.alias or spec.series_key for spec in normalized_specs]

        long_panel = self.build_snapshot_panel_long(
            normalized_specs,
            asof=asof,
            align=align,
        )
        if long_panel.empty:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="obs_date"), columns=aliases)

        panel = (
            long_panel.pivot(index="obs_date", columns="series_alias", values="value")
            .sort_index()
            .reindex(columns=aliases)
        )
        if join == "inner":
            panel = panel.dropna(how="any")
        elif join == "left":
            panel = panel.loc[panel[aliases[0]].notna()]
        elif join == "right":
            panel = panel.loc[panel[aliases[-1]].notna()]

        panel.index = (
            panel.index.tz_convert("UTC")
            if panel.index.tz is not None
            else panel.index.tz_localize("UTC")
        )
        panel.index.name = "obs_date"
        return panel

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
        return transform_input_series_keys(spec)

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

    def _upsert_pipeline_metadata(self, spec: PITPipelineSpec) -> str:
        pipeline_id = spec.resolved_pipeline_id()
        self.conn.execute(
            f"""
            INSERT INTO {_PIT_PIPELINES_TABLE} (
                pipeline_id,
                spec_hash,
                spec_json,
                description,
                created_utc
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(pipeline_id) DO UPDATE SET
                spec_hash=excluded.spec_hash,
                spec_json=excluded.spec_json,
                description=excluded.description,
                created_utc=excluded.created_utc
            """,
            [
                pipeline_id,
                spec.spec_hash(),
                json.dumps(spec.spec_payload(), sort_keys=True, separators=(",", ":")),
                spec.description,
                to_utc_naive(pd.Timestamp.now("UTC")),
            ],
        )
        return pipeline_id

    def _insert_pipeline_run(
        self,
        *,
        run_id: str,
        pipeline_id: str,
        start_obs: pd.Timestamp | None,
        end_obs: pd.Timestamp | None,
        start_asof: pd.Timestamp | None,
        end_asof: pd.Timestamp | None,
        incremental: bool,
        requested_since_asof: pd.Timestamp | None,
        effective_start_asof: pd.Timestamp | None,
        requested_since_run_id: str | None,
        max_output_asof: pd.Timestamp | None,
        rows_written: int,
        step_count: int,
        status: str,
        started_utc: pd.Timestamp,
        finished_utc: pd.Timestamp,
    ) -> None:
        self.conn.execute(
            f"""
            INSERT INTO {_PIT_PIPELINE_RUNS_TABLE} (
                run_id,
                pipeline_id,
                start_obs,
                end_obs,
                start_asof,
                end_asof,
                incremental,
                requested_since_asof,
                effective_start_asof,
                requested_since_run_id,
                max_output_asof,
                rows_written,
                step_count,
                status,
                started_utc,
                finished_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                pipeline_id,
                to_utc_naive(start_obs),
                to_utc_naive(end_obs),
                to_utc_naive(start_asof),
                to_utc_naive(end_asof),
                bool(incremental),
                to_utc_naive(requested_since_asof),
                to_utc_naive(effective_start_asof),
                requested_since_run_id,
                to_utc_naive(max_output_asof),
                int(rows_written),
                int(step_count),
                status,
                to_utc_naive(started_utc),
                to_utc_naive(finished_utc),
            ],
        )

    def _resolve_pipeline_effective_start_asof(
        self,
        *,
        pipeline_id: str,
        incremental: bool,
        start_asof: pd.Timestamp | None,
        since_asof: pd.Timestamp | None,
        since_run_id: str | None,
    ) -> pd.Timestamp | None:
        if not incremental and (since_asof is not None or since_run_id is not None):
            raise PITContractError(
                "since_asof/since_run_id require incremental=True for pipeline execution."
            )

        anchors: list[pd.Timestamp] = []
        if start_asof is not None:
            anchors.append(to_utc_aware(start_asof))
        if since_asof is not None:
            anchors.append(to_utc_aware(since_asof))

        if since_run_id is not None:
            row = self.conn.execute(
                f"""
                SELECT pipeline_id, max_output_asof, finished_utc
                FROM {_PIT_PIPELINE_RUNS_TABLE}
                WHERE run_id = ?
                """,
                [since_run_id],
            ).fetchone()
            if row is None:
                raise PITContractError(f"Unknown PIT pipeline run_id: {since_run_id}")
            run_pipeline_id = str(row[0])
            if run_pipeline_id != pipeline_id:
                raise PITContractError(
                    "since_run_id belongs to a different pipeline. "
                    f"Expected pipeline_id='{pipeline_id}', got '{run_pipeline_id}'."
                )
            explicit_max_asof = row[1]
            if explicit_max_asof is not None:
                anchors.append(to_utc_aware(explicit_max_asof))
            else:
                inferred = self._pipeline_max_output_asof(pipeline_id)
                if inferred is not None:
                    anchors.append(inferred)
                else:
                    anchors.append(to_utc_aware(row[2]))
        elif incremental and since_asof is None:
            latest = self.conn.execute(
                f"""
                SELECT max_output_asof, finished_utc
                FROM {_PIT_PIPELINE_RUNS_TABLE}
                WHERE pipeline_id = ? AND status = 'success'
                ORDER BY started_utc DESC
                LIMIT 1
                """,
                [pipeline_id],
            ).fetchone()
            if latest is not None:
                explicit_max_asof = latest[0]
                if explicit_max_asof is not None:
                    anchors.append(to_utc_aware(explicit_max_asof))
                else:
                    inferred = self._pipeline_max_output_asof(pipeline_id)
                    if inferred is not None:
                        anchors.append(inferred)
                    else:
                        anchors.append(to_utc_aware(latest[1]))

        if not anchors:
            return None
        return max(anchors)

    def _pipeline_max_output_asof(self, pipeline_id: str) -> pd.Timestamp | None:
        row = self.conn.execute(
            f"SELECT spec_json FROM {_PIT_PIPELINES_TABLE} WHERE pipeline_id = ?",
            [pipeline_id],
        ).fetchone()
        if row is None or row[0] is None:
            return None

        try:
            payload = json.loads(str(row[0]))
        except json.JSONDecodeError:
            return None
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list):
            return None

        keys: list[str] = []
        for raw in raw_steps:
            if not isinstance(raw, Mapping):
                continue
            raw_spec = raw.get("spec")
            if not isinstance(raw_spec, Mapping):
                continue
            key = str(raw_spec.get("output_series_key", "")).strip()
            if key:
                keys.append(key)
        keys = sorted(set(keys))
        if not keys:
            return None

        placeholders = ", ".join(["?"] * len(keys))
        max_row = self.conn.execute(
            f"""
            SELECT MAX(asof_utc)
            FROM {_PIT_TABLE}
            WHERE series_key IN ({placeholders})
            """,
            keys,
        ).fetchone()
        if max_row is None or max_row[0] is None:
            return None
        return to_utc_aware(max_row[0])

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
        selected_input_series_key: str | None = None,
        selected_input_asof: pd.Timestamp | None = None,
        lineage_extra: Mapping[str, object] | None = None,
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
        if selected_input_series_key is not None:
            payload["selected_input_series_key"] = selected_input_series_key
        if selected_input_asof is not None:
            payload["selected_input_asof_utc"] = selected_input_asof.isoformat()
        if lineage_extra:
            for key, value in lineage_extra.items():
                if value is None:
                    payload[key] = None
                elif isinstance(value, pd.Timestamp):
                    payload[key] = value.isoformat()
                elif isinstance(value, pd.Timedelta):
                    payload[key] = str(value)
                else:
                    payload[key] = value
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

    def list_union_vintages(
        self,
        series_keys: Sequence[str],
        start: pd.Timestamp | None = None,
        end: pd.Timestamp | None = None,
        *,
        mode: Literal["event", "calendar"] = "event",
        calendar_freq: str = "D",
    ) -> pd.DatetimeIndex:
        unique_keys = sorted({str(k) for k in series_keys if str(k).strip()})

        if mode == "event":
            if not unique_keys:
                return pd.DatetimeIndex([], tz="UTC")

            placeholders = ", ".join(["?"] * len(unique_keys))
            filters = [f"series_key IN ({placeholders})"]
            params: list[object] = [*unique_keys]
            if start is not None:
                filters.append("asof_utc >= ?")
                params.append(to_utc_naive(start))
            if end is not None:
                filters.append("asof_utc <= ?")
                params.append(to_utc_naive(end))

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
            return pd.DatetimeIndex(to_utc_aware(df["asof_utc"])).sort_values().unique()

        if mode == "calendar":
            if start is None or end is None:
                raise PITContractError("mode='calendar' requires both start and end.")
            start_ts = to_utc_aware(start)
            end_ts = to_utc_aware(end)
            if start_ts > end_ts:
                return pd.DatetimeIndex([], tz="UTC")
            return pd.date_range(start=start_ts, end=end_ts, freq=calendar_freq, tz="UTC")

        raise PITContractError("mode must be one of {'event', 'calendar'}.")

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
                continue
            if spec.op == "coalesce":
                snapshots: dict[str, pd.Series] = {}
                source_asof_by_series_obs: dict[str, dict[pd.Timestamp, pd.Timestamp]] = {}
                for series_key in input_keys:
                    snapshot_rows = self._get_snapshot_rows_with_source_asof(
                        series_key,
                        source_asof_by_series[series_key],
                        start=start_obs,
                        end=end_obs,
                    )
                    if snapshot_rows.empty:
                        snapshots[series_key] = pd.Series(dtype="float64", name=series_key)
                        source_asof_by_series_obs[series_key] = {}
                        continue
                    snapshot_series = pd.Series(
                        snapshot_rows["value"].to_numpy(),
                        index=pd.DatetimeIndex(snapshot_rows["obs_date"]),
                        name=series_key,
                    )
                    snapshot_series.index.name = "obs_date"
                    snapshots[series_key] = snapshot_series
                    source_asof_by_series_obs[series_key] = {
                        pd.Timestamp(obs_date): pd.Timestamp(source_asof_utc)
                        for obs_date, source_asof_utc in zip(
                            snapshot_rows["obs_date"],
                            snapshot_rows["source_asof_utc"],
                            strict=False,
                        )
                    }
                transformed_df = apply_coalesce_obs_path_transform(snapshots, spec)
                if transformed_df.empty:
                    continue
                if start_obs_utc is not None:
                    transformed_df = transformed_df[transformed_df.index >= start_obs_utc]
                if end_obs_utc is not None:
                    transformed_df = transformed_df[transformed_df.index <= end_obs_utc]
                if transformed_df.empty:
                    continue

                row_meta = [
                    self._lineage_meta(
                        transform_id=transform_id,
                        spec=spec,
                        engine_resolution=engine_resolution,
                        source_asof=source_asof_by_series.get(spec.input_series_key),
                        source_asof_by_series=source_asof_by_series,
                        selected_input_series_key=(
                            str(selected_key)
                            if pd.notna(selected_key)
                            else None
                        ),
                        selected_input_asof=(
                            source_asof_by_series_obs.get(str(selected_key), {}).get(pd.Timestamp(obs_date))
                            if pd.notna(selected_key)
                            else None
                        ),
                    )
                    for obs_date, selected_key in zip(
                        transformed_df.index,
                        transformed_df["selected_input_series_key"].tolist(),
                        strict=False,
                    )
                ]
                chunks.append(
                    pd.DataFrame(
                        {
                            "series_key": spec.output_series_key,
                            "obs_date": transformed_df.index,
                            "asof_utc": [asof] * len(transformed_df),
                            "value": transformed_df["value"].to_numpy(),
                            "source": [f"pit_transform:{transform_id}"] * len(transformed_df),
                            "meta_json": row_meta,
                        }
                    )
                )
                continue
            if spec.op == "splice":
                right_series_key = str(spec.sanitized_params().get("right_series_key", "")).strip()
                if not right_series_key:
                    raise PITValidationError(
                        "splice transform requires params['right_series_key']."
                    )

                left_rows = self._get_snapshot_rows_with_source_asof(
                    spec.input_series_key,
                    source_asof_by_series[spec.input_series_key],
                    start=start_obs,
                    end=end_obs,
                )
                right_rows = self._get_snapshot_rows_with_source_asof(
                    right_series_key,
                    source_asof_by_series[right_series_key],
                    start=start_obs,
                    end=end_obs,
                )

                left_snapshot = (
                    pd.Series(
                        left_rows["value"].to_numpy(),
                        index=pd.DatetimeIndex(left_rows["obs_date"]),
                        name=spec.input_series_key,
                    )
                    if not left_rows.empty
                    else pd.Series(dtype="float64", name=spec.input_series_key)
                )
                right_snapshot = (
                    pd.Series(
                        right_rows["value"].to_numpy(),
                        index=pd.DatetimeIndex(right_rows["obs_date"]),
                        name=right_series_key,
                    )
                    if not right_rows.empty
                    else pd.Series(dtype="float64", name=right_series_key)
                )

                left_source_asof_by_obs = {
                    pd.Timestamp(obs_date): pd.Timestamp(source_asof_utc)
                    for obs_date, source_asof_utc in zip(
                        left_rows["obs_date"],
                        left_rows["source_asof_utc"],
                        strict=False,
                    )
                }
                right_source_asof_by_obs = {
                    pd.Timestamp(obs_date): pd.Timestamp(source_asof_utc)
                    for obs_date, source_asof_utc in zip(
                        right_rows["obs_date"],
                        right_rows["source_asof_utc"],
                        strict=False,
                    )
                }

                transformed_df = apply_splice_obs_path_transform(
                    left_snapshot,
                    right_snapshot,
                    spec,
                )
                if transformed_df.empty:
                    continue
                if start_obs_utc is not None:
                    transformed_df = transformed_df[transformed_df.index >= start_obs_utc]
                if end_obs_utc is not None:
                    transformed_df = transformed_df[transformed_df.index <= end_obs_utc]
                if transformed_df.empty:
                    continue

                transition_periods = int(spec.sanitized_params().get("transition_periods", 0))
                row_meta = []
                for obs_date, row in transformed_df.iterrows():
                    selected_key = (
                        str(row["selected_input_series_key"])
                        if pd.notna(row["selected_input_series_key"])
                        else None
                    )
                    selected_input_asof: pd.Timestamp | None = None
                    if selected_key == spec.input_series_key:
                        selected_input_asof = left_source_asof_by_obs.get(pd.Timestamp(obs_date))
                    elif selected_key == right_series_key:
                        selected_input_asof = right_source_asof_by_obs.get(pd.Timestamp(obs_date))

                    row_meta.append(
                        self._lineage_meta(
                            transform_id=transform_id,
                            spec=spec,
                            engine_resolution=engine_resolution,
                            source_asof=source_asof_by_series.get(spec.input_series_key),
                            source_asof_by_series=source_asof_by_series,
                            selected_input_series_key=selected_key,
                            selected_input_asof=selected_input_asof,
                            lineage_extra={
                                "splice_state": (
                                    str(row["splice_state"])
                                    if pd.notna(row["splice_state"])
                                    else None
                                ),
                                "splice_left_input_asof_utc": left_source_asof_by_obs.get(
                                    pd.Timestamp(obs_date)
                                ),
                                "splice_right_input_asof_utc": right_source_asof_by_obs.get(
                                    pd.Timestamp(obs_date)
                                ),
                                "splice_anchor_obs_date_utc": (
                                    pd.Timestamp(row["splice_anchor_obs_date"])
                                    if pd.notna(row["splice_anchor_obs_date"])
                                    else None
                                ),
                                "splice_anchor_left_value": (
                                    float(row["splice_anchor_left_value"])
                                    if pd.notna(row["splice_anchor_left_value"])
                                    else None
                                ),
                                "splice_anchor_right_value": (
                                    float(row["splice_anchor_right_value"])
                                    if pd.notna(row["splice_anchor_right_value"])
                                    else None
                                ),
                                "splice_scale": (
                                    float(row["splice_scale"])
                                    if pd.notna(row["splice_scale"])
                                    else None
                                ),
                                "splice_offset": (
                                    float(row["splice_offset"])
                                    if pd.notna(row["splice_offset"])
                                    else None
                                ),
                                "splice_transition_periods": transition_periods,
                                "splice_left_weight": float(row["splice_left_weight"]),
                                "splice_right_weight": float(row["splice_right_weight"]),
                            },
                        )
                    )

                chunks.append(
                    pd.DataFrame(
                        {
                            "series_key": spec.output_series_key,
                            "obs_date": transformed_df.index,
                            "asof_utc": [asof] * len(transformed_df),
                            "value": transformed_df["value"].to_numpy(),
                            "source": [f"pit_transform:{transform_id}"] * len(transformed_df),
                            "meta_json": row_meta,
                        }
                    )
                )
                continue
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

    def explain_transform(
        self,
        spec: PITTransformSpec | Mapping[str, Any],
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        allow_experimental: bool = False,
        on_engine_mismatch: EngineMismatchPolicy = "error",
    ) -> dict[str, Any]:
        spec_obj = coerce_transform_spec(spec)
        validate_transform_spec(spec_obj)

        if spec_obj.axis == "revision_path" and not allow_experimental:
            raise PITExperimentalFeatureError(
                "axis='revision_path' is experimental. Set allow_experimental=True to enable it."
            )

        input_keys = self._input_series_keys_for_spec(spec_obj)
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

        engine_resolution = resolve_engine(
            spec_obj,
            on_engine_mismatch=on_engine_mismatch,
        )

        return {
            "transform_id": spec_obj.transform_id(),
            "spec_hash": spec_obj.spec_hash(),
            "input_series_keys": input_keys,
            "output_series_key": spec_obj.output_series_key,
            "axis": spec_obj.axis,
            "op": spec_obj.op,
            "params": spec_obj.sanitized_params(),
            "engine_requested": engine_resolution.engine_requested,
            "engine_used": engine_resolution.engine_used,
            "fallback_reason": engine_resolution.fallback_reason,
            "candidate_asof_count": int(len(asof_values)),
            "candidate_asof_start_utc": asof_values.min() if len(asof_values) else None,
            "candidate_asof_end_utc": asof_values.max() if len(asof_values) else None,
            "start_obs": to_utc_aware(start_obs) if start_obs is not None else None,
            "end_obs": to_utc_aware(end_obs) if end_obs is not None else None,
            "start_asof": to_utc_aware(start_asof) if start_asof is not None else None,
            "end_asof": to_utc_aware(end_asof) if end_asof is not None else None,
            "experimental": bool(spec_obj.axis == "revision_path"),
        }

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

    def explain_pipeline(
        self,
        spec: PITPipelineSpec | Mapping[str, Any],
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        incremental: bool = False,
        since_asof: pd.Timestamp | None = None,
        since_run_id: str | None = None,
        allow_experimental: bool = False,
        on_engine_mismatch: EngineMismatchPolicy = "error",
    ) -> dict[str, Any]:
        pipeline_spec = coerce_pipeline_spec(spec)
        pipeline_id = pipeline_spec.resolved_pipeline_id()
        ordered_steps = pipeline_spec.ordered_steps()
        effective_start_asof = self._resolve_pipeline_effective_start_asof(
            pipeline_id=pipeline_id,
            incremental=incremental,
            start_asof=start_asof,
            since_asof=since_asof,
            since_run_id=since_run_id,
        )

        step_explanations = [
            {
                "step_name": step.name,
                "depends_on": list(step.depends_on),
                "transform": self.explain_transform(
                    step.normalized_spec(),
                    start_obs=start_obs,
                    end_obs=end_obs,
                    start_asof=effective_start_asof,
                    end_asof=end_asof,
                    allow_experimental=allow_experimental,
                    on_engine_mismatch=on_engine_mismatch,
                ),
            }
            for step in ordered_steps
        ]

        return {
            "pipeline_id": pipeline_id,
            "spec_hash": pipeline_spec.spec_hash(),
            "description": pipeline_spec.description,
            "incremental": incremental,
            "requested_since_asof": to_utc_aware(since_asof) if since_asof is not None else None,
            "requested_since_run_id": since_run_id,
            "effective_start_asof": effective_start_asof,
            "end_asof": to_utc_aware(end_asof) if end_asof is not None else None,
            "start_obs": to_utc_aware(start_obs) if start_obs is not None else None,
            "end_obs": to_utc_aware(end_obs) if end_obs is not None else None,
            "step_count": len(step_explanations),
            "steps": step_explanations,
        }

    def preview_pipeline(
        self,
        spec: PITPipelineSpec | Mapping[str, Any],
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        overwrite: bool = False,
        lag_policy: ReleaseLagPolicy | None = None,
        allow_experimental: bool = False,
        on_engine_mismatch: EngineMismatchPolicy = "error",
        incremental: bool = False,
        since_asof: pd.Timestamp | None = None,
        since_run_id: str | None = None,
        include_intermediate: bool = False,
    ) -> pd.DataFrame:
        pipeline_spec = coerce_pipeline_spec(spec)
        pipeline_id = pipeline_spec.resolved_pipeline_id()
        effective_start_asof = self._resolve_pipeline_effective_start_asof(
            pipeline_id=pipeline_id,
            incremental=incremental,
            start_asof=start_asof,
            since_asof=since_asof,
            since_run_id=since_run_id,
        )
        ordered_steps = pipeline_spec.ordered_steps()

        step_frames: list[pd.DataFrame] = []
        self.conn.execute("BEGIN TRANSACTION")
        try:
            for step in ordered_steps:
                step_spec, result_df, _ = self._materialize_transform_rows(
                    step.normalized_spec(),
                    start_obs,
                    end_obs,
                    effective_start_asof,
                    end_asof,
                    lag_policy=lag_policy,
                    allow_experimental=allow_experimental,
                    on_engine_mismatch=on_engine_mismatch,
                )
                if overwrite:
                    self._delete_transformed_rows(
                        output_series_key=step_spec.output_series_key,
                        start_obs=start_obs,
                        end_obs=end_obs,
                        start_asof=effective_start_asof,
                        end_asof=end_asof,
                    )
                if not result_df.empty:
                    self.upsert_pit_observations(result_df, strict=False)
                    tagged = result_df.copy()
                    tagged["step_name"] = step.name
                    step_frames.append(tagged)
            self.conn.execute("ROLLBACK")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

        if not step_frames:
            base_columns = [
                "series_key",
                "obs_date",
                "asof_utc",
                "value",
                "source",
                "meta_json",
                "step_name",
            ]
            return pd.DataFrame(columns=base_columns)

        if include_intermediate:
            return pd.concat(step_frames, ignore_index=True)
        return step_frames[-1].reset_index(drop=True)

    def apply_pipeline(
        self,
        spec: PITPipelineSpec | Mapping[str, Any],
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        overwrite: bool = False,
        lag_policy: ReleaseLagPolicy | None = None,
        allow_experimental: bool = False,
        on_engine_mismatch: EngineMismatchPolicy = "error",
        incremental: bool = False,
        since_asof: pd.Timestamp | None = None,
        since_run_id: str | None = None,
    ) -> PITPipelineResult:
        started_utc = pd.Timestamp.now(tz="UTC")
        run_id = str(uuid.uuid4())
        pipeline_spec = coerce_pipeline_spec(spec)
        pipeline_id = self._upsert_pipeline_metadata(pipeline_spec)
        effective_start_asof = self._resolve_pipeline_effective_start_asof(
            pipeline_id=pipeline_id,
            incremental=incremental,
            start_asof=start_asof,
            since_asof=since_asof,
            since_run_id=since_run_id,
        )
        ordered_steps = pipeline_spec.ordered_steps()
        step_results: list[PITTransformResult] = []
        total_rows = 0
        max_output_asof: pd.Timestamp | None = None

        try:
            for step in ordered_steps:
                result = self.apply_transform(
                    step.normalized_spec(),
                    start_obs=start_obs,
                    end_obs=end_obs,
                    start_asof=effective_start_asof,
                    end_asof=end_asof,
                    persist=True,
                    overwrite=overwrite,
                    lag_policy=lag_policy,
                    allow_experimental=allow_experimental,
                    on_engine_mismatch=on_engine_mismatch,
                )
                step_results.append(result)
                total_rows += int(result.rows_written)
                if result.rows_written > 0:
                    row = self.conn.execute(
                        f"""
                        SELECT MAX(asof_utc)
                        FROM {_PIT_TABLE}
                        WHERE series_key = ? AND source = ?
                        """,
                        [
                            result.output_series_key,
                            f"pit_transform:{result.transform_id}",
                        ],
                    ).fetchone()
                    if row is not None and row[0] is not None:
                        candidate = to_utc_aware(row[0])
                        if max_output_asof is None or candidate > max_output_asof:
                            max_output_asof = candidate
            status = "success"
        except Exception:
            finished_utc = pd.Timestamp.now(tz="UTC")
            self._insert_pipeline_run(
                run_id=run_id,
                pipeline_id=pipeline_id,
                start_obs=start_obs,
                end_obs=end_obs,
                start_asof=start_asof,
                end_asof=end_asof,
                incremental=incremental,
                requested_since_asof=since_asof,
                effective_start_asof=effective_start_asof,
                requested_since_run_id=since_run_id,
                max_output_asof=max_output_asof,
                rows_written=total_rows,
                step_count=len(step_results),
                status="failed",
                started_utc=started_utc,
                finished_utc=finished_utc,
            )
            raise

        finished_utc = pd.Timestamp.now(tz="UTC")
        self._insert_pipeline_run(
            run_id=run_id,
            pipeline_id=pipeline_id,
            start_obs=start_obs,
            end_obs=end_obs,
            start_asof=start_asof,
            end_asof=end_asof,
            incremental=incremental,
            requested_since_asof=since_asof,
            effective_start_asof=effective_start_asof,
            requested_since_run_id=since_run_id,
            max_output_asof=max_output_asof,
            rows_written=total_rows,
            step_count=len(step_results),
            status=status,
            started_utc=started_utc,
            finished_utc=finished_utc,
        )
        return PITPipelineResult(
            pipeline_id=pipeline_id,
            run_id=run_id,
            status=status,
            step_results=tuple(step_results),
            rows_written=total_rows,
            run_started_utc=started_utc,
            run_finished_utc=finished_utc,
            incremental=incremental,
            effective_start_asof=effective_start_asof,
        )

    def _upsert_expression_graph_metadata(self, spec: PITExpressionGraphSpec) -> str:
        graph_id = spec.resolved_graph_id()
        self.conn.execute(
            f"""
            INSERT INTO {_PIT_EXPR_GRAPHS_TABLE} (
                graph_id,
                spec_hash,
                spec_json,
                description,
                created_utc
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(graph_id) DO UPDATE SET
                spec_hash=excluded.spec_hash,
                spec_json=excluded.spec_json,
                description=excluded.description,
                created_utc=excluded.created_utc
            """,
            [
                graph_id,
                spec.spec_hash(),
                json.dumps(spec.spec_payload(), sort_keys=True, separators=(",", ":")),
                spec.description,
                to_utc_naive(pd.Timestamp.now("UTC")),
            ],
        )
        return graph_id

    def _insert_expression_graph_run(
        self,
        *,
        run_id: str,
        graph_id: str,
        start_obs: pd.Timestamp | None,
        end_obs: pd.Timestamp | None,
        start_asof: pd.Timestamp | None,
        end_asof: pd.Timestamp | None,
        incremental: bool,
        requested_since_asof: pd.Timestamp | None,
        effective_start_asof: pd.Timestamp | None,
        requested_since_run_id: str | None,
        max_output_asof: pd.Timestamp | None,
        rows_written: int,
        node_count: int,
        status: str,
        started_utc: pd.Timestamp,
        finished_utc: pd.Timestamp,
    ) -> None:
        self.conn.execute(
            f"""
            INSERT INTO {_PIT_EXPR_GRAPH_RUNS_TABLE} (
                run_id,
                graph_id,
                start_obs,
                end_obs,
                start_asof,
                end_asof,
                incremental,
                requested_since_asof,
                effective_start_asof,
                requested_since_run_id,
                max_output_asof,
                rows_written,
                node_count,
                status,
                started_utc,
                finished_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                graph_id,
                to_utc_naive(start_obs),
                to_utc_naive(end_obs),
                to_utc_naive(start_asof),
                to_utc_naive(end_asof),
                bool(incremental),
                to_utc_naive(requested_since_asof),
                to_utc_naive(effective_start_asof),
                requested_since_run_id,
                to_utc_naive(max_output_asof),
                int(rows_written),
                int(node_count),
                status,
                to_utc_naive(started_utc),
                to_utc_naive(finished_utc),
            ],
        )

    def _expression_graph_max_output_asof(self, graph_id: str) -> pd.Timestamp | None:
        row = self.conn.execute(
            f"SELECT spec_json FROM {_PIT_EXPR_GRAPHS_TABLE} WHERE graph_id = ?",
            [graph_id],
        ).fetchone()
        if row is None or row[0] is None:
            return None

        try:
            payload = json.loads(str(row[0]))
        except json.JSONDecodeError:
            return None

        raw_nodes = payload.get("nodes")
        if not isinstance(raw_nodes, list):
            return None

        keys: list[str] = []
        for raw in raw_nodes:
            if not isinstance(raw, Mapping):
                continue
            key = str(raw.get("output_series_key", "")).strip()
            if key:
                keys.append(key)
        keys = sorted(set(keys))
        if not keys:
            return None

        placeholders = ", ".join(["?"] * len(keys))
        max_row = self.conn.execute(
            f"""
            SELECT MAX(asof_utc)
            FROM {_PIT_TABLE}
            WHERE series_key IN ({placeholders})
            """,
            keys,
        ).fetchone()
        if max_row is None or max_row[0] is None:
            return None
        return to_utc_aware(max_row[0])

    def _resolve_expression_graph_effective_start_asof(
        self,
        *,
        graph_id: str,
        incremental: bool,
        start_asof: pd.Timestamp | None,
        since_asof: pd.Timestamp | None,
        since_run_id: str | None,
    ) -> pd.Timestamp | None:
        if not incremental and (since_asof is not None or since_run_id is not None):
            raise PITContractError(
                "since_asof/since_run_id require incremental=True for expression graph execution."
            )

        anchors: list[pd.Timestamp] = []
        if start_asof is not None:
            anchors.append(to_utc_aware(start_asof))
        if since_asof is not None:
            anchors.append(to_utc_aware(since_asof))

        if since_run_id is not None:
            row = self.conn.execute(
                f"""
                SELECT graph_id, max_output_asof, finished_utc
                FROM {_PIT_EXPR_GRAPH_RUNS_TABLE}
                WHERE run_id = ?
                """,
                [since_run_id],
            ).fetchone()
            if row is None:
                raise PITContractError(f"Unknown PIT expression graph run_id: {since_run_id}")
            run_graph_id = str(row[0])
            if run_graph_id != graph_id:
                raise PITContractError(
                    "since_run_id belongs to a different expression graph. "
                    f"Expected graph_id='{graph_id}', got '{run_graph_id}'."
                )
            explicit_max_asof = row[1]
            if explicit_max_asof is not None:
                anchors.append(to_utc_aware(explicit_max_asof))
            else:
                inferred = self._expression_graph_max_output_asof(graph_id)
                if inferred is not None:
                    anchors.append(inferred)
                else:
                    anchors.append(to_utc_aware(row[2]))
        elif incremental and since_asof is None:
            latest = self.conn.execute(
                f"""
                SELECT max_output_asof, finished_utc
                FROM {_PIT_EXPR_GRAPH_RUNS_TABLE}
                WHERE graph_id = ? AND status = 'success'
                ORDER BY started_utc DESC
                LIMIT 1
                """,
                [graph_id],
            ).fetchone()
            if latest is not None:
                explicit_max_asof = latest[0]
                if explicit_max_asof is not None:
                    anchors.append(to_utc_aware(explicit_max_asof))
                else:
                    inferred = self._expression_graph_max_output_asof(graph_id)
                    if inferred is not None:
                        anchors.append(inferred)
                    else:
                        anchors.append(to_utc_aware(latest[1]))

        if not anchors:
            return None
        return max(anchors)

    @staticmethod
    def _expression_lineage_meta(
        *,
        graph_id: str,
        node: PITExpressionNode,
        source_asof_by_series: Mapping[str, pd.Timestamp],
    ) -> str:
        payload: dict[str, object] = {
            "graph_id": graph_id,
            "node_name": node.name,
            "output_series_key": node.output_series_key,
            "expression": node.expression,
            "expression_hash": _expression_hash(node.expression),
            "inputs": dict(node.inputs),
            "depends_on": list(node.depends_on),
            "join": node.join,
            "fill_value": node.fill_value,
            "source_asof_by_series_utc": {
                k: to_utc_aware(v).isoformat() for k, v in source_asof_by_series.items()
            },
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def _materialize_expression_node_rows(
        self,
        *,
        graph_id: str,
        node: PITExpressionNode,
        start_obs: pd.Timestamp | None,
        end_obs: pd.Timestamp | None,
        start_asof: pd.Timestamp | None,
        end_asof: pd.Timestamp | None,
        lag_policy: ReleaseLagPolicy | None,
    ) -> pd.DataFrame:
        input_series_keys = sorted({str(v) for v in node.inputs.values() if str(v).strip()})
        asof_values = self.list_union_vintages(
            input_series_keys,
            start=start_asof,
            end=end_asof,
            mode="event",
        )
        if len(asof_values) == 0:
            return pd.DataFrame(
                columns=["series_key", "obs_date", "asof_utc", "value", "source", "meta_json"]
            )

        start_obs_utc = to_utc_aware(start_obs) if start_obs is not None else None
        end_obs_utc = to_utc_aware(end_obs) if end_obs is not None else None
        chunks: list[pd.DataFrame] = []

        for asof in asof_values:
            source_asof_by_series: dict[str, pd.Timestamp] = {}
            env: dict[str, pd.Series] = {}
            for alias, series_key in node.inputs.items():
                effective = asof
                if lag_policy is not None:
                    effective = effective_asof(effective, series_key, lag_policy)
                if effective > asof:
                    raise PITCausalityError(
                        "Causality violation: effective source_asof is later than output asof. "
                        f"series_key={series_key}, source_asof={effective}, output_asof={asof}"
                    )
                source_asof_by_series[series_key] = effective
                env[alias] = self.get_snapshot(
                    series_key,
                    effective,
                    start=start_obs,
                    end=end_obs,
                )

            transformed = _evaluate_expression_series(
                node.expression,
                env,
                join=node.join,
                fill_value=node.fill_value,
            ).dropna()
            if transformed.empty:
                continue

            if start_obs_utc is not None:
                transformed = transformed[transformed.index >= start_obs_utc]
            if end_obs_utc is not None:
                transformed = transformed[transformed.index <= end_obs_utc]
            if transformed.empty:
                continue

            lineage = self._expression_lineage_meta(
                graph_id=graph_id,
                node=node,
                source_asof_by_series=source_asof_by_series,
            )
            chunks.append(
                pd.DataFrame(
                    {
                        "series_key": node.output_series_key,
                        "obs_date": transformed.index,
                        "asof_utc": [asof] * len(transformed),
                        "value": transformed.to_numpy(),
                        "source": [f"pit_expr_graph:{graph_id}:{node.name}"] * len(transformed),
                        "meta_json": [lineage] * len(transformed),
                    }
                )
            )

        if not chunks:
            return pd.DataFrame(
                columns=["series_key", "obs_date", "asof_utc", "value", "source", "meta_json"]
            )
        return pd.concat(chunks, ignore_index=True)

    def explain_expression_graph(
        self,
        spec: PITExpressionGraphSpec | Mapping[str, Any],
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        incremental: bool = False,
        since_asof: pd.Timestamp | None = None,
        since_run_id: str | None = None,
    ) -> dict[str, Any]:
        graph_spec = coerce_expression_graph_spec(spec)
        graph_id = graph_spec.resolved_graph_id()
        ordered_nodes = graph_spec.ordered_nodes()
        effective_start_asof = self._resolve_expression_graph_effective_start_asof(
            graph_id=graph_id,
            incremental=incremental,
            start_asof=start_asof,
            since_asof=since_asof,
            since_run_id=since_run_id,
        )

        node_items: list[dict[str, object]] = []
        for node in ordered_nodes:
            input_series_keys = sorted({str(v) for v in node.inputs.values() if str(v).strip()})
            asof_values = self.list_union_vintages(
                input_series_keys,
                start=effective_start_asof,
                end=end_asof,
                mode="event",
            )
            node_items.append(
                {
                    "node_name": node.name,
                    "depends_on": list(node.depends_on),
                    "output_series_key": node.output_series_key,
                    "expression": node.expression,
                    "expression_hash": _expression_hash(node.expression),
                    "inputs": dict(node.inputs),
                    "join": node.join,
                    "fill_value": node.fill_value,
                    "candidate_asof_count": int(len(asof_values)),
                    "candidate_asof_start_utc": asof_values.min() if len(asof_values) else None,
                    "candidate_asof_end_utc": asof_values.max() if len(asof_values) else None,
                    "engine_requested": "python",
                    "engine_used": "python",
                }
            )

        return {
            "graph_id": graph_id,
            "spec_hash": graph_spec.spec_hash(),
            "description": graph_spec.description,
            "incremental": incremental,
            "requested_since_asof": to_utc_aware(since_asof) if since_asof is not None else None,
            "requested_since_run_id": since_run_id,
            "effective_start_asof": effective_start_asof,
            "end_asof": to_utc_aware(end_asof) if end_asof is not None else None,
            "start_obs": to_utc_aware(start_obs) if start_obs is not None else None,
            "end_obs": to_utc_aware(end_obs) if end_obs is not None else None,
            "node_count": len(node_items),
            "nodes": node_items,
        }

    def preview_expression_graph(
        self,
        spec: PITExpressionGraphSpec | Mapping[str, Any],
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        overwrite: bool = False,
        lag_policy: ReleaseLagPolicy | None = None,
        incremental: bool = False,
        since_asof: pd.Timestamp | None = None,
        since_run_id: str | None = None,
        include_intermediate: bool = False,
    ) -> pd.DataFrame:
        graph_spec = coerce_expression_graph_spec(spec)
        graph_id = graph_spec.resolved_graph_id()
        ordered_nodes = graph_spec.ordered_nodes()
        effective_start_asof = self._resolve_expression_graph_effective_start_asof(
            graph_id=graph_id,
            incremental=incremental,
            start_asof=start_asof,
            since_asof=since_asof,
            since_run_id=since_run_id,
        )

        node_frames: list[pd.DataFrame] = []
        self.conn.execute("BEGIN TRANSACTION")
        try:
            for node in ordered_nodes:
                result_df = self._materialize_expression_node_rows(
                    graph_id=graph_id,
                    node=node,
                    start_obs=start_obs,
                    end_obs=end_obs,
                    start_asof=effective_start_asof,
                    end_asof=end_asof,
                    lag_policy=lag_policy,
                )
                if overwrite:
                    self._delete_transformed_rows(
                        output_series_key=node.output_series_key,
                        start_obs=start_obs,
                        end_obs=end_obs,
                        start_asof=effective_start_asof,
                        end_asof=end_asof,
                    )
                if not result_df.empty:
                    self.upsert_pit_observations(result_df, strict=False)
                    tagged = result_df.copy()
                    tagged["node_name"] = node.name
                    node_frames.append(tagged)
            self.conn.execute("ROLLBACK")
        except Exception:
            self.conn.execute("ROLLBACK")
            raise

        if not node_frames:
            base_columns = [
                "series_key",
                "obs_date",
                "asof_utc",
                "value",
                "source",
                "meta_json",
                "node_name",
            ]
            return pd.DataFrame(columns=base_columns)

        if include_intermediate:
            return pd.concat(node_frames, ignore_index=True)
        return node_frames[-1].reset_index(drop=True)

    def apply_expression_graph(
        self,
        spec: PITExpressionGraphSpec | Mapping[str, Any],
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        overwrite: bool = False,
        lag_policy: ReleaseLagPolicy | None = None,
        incremental: bool = False,
        since_asof: pd.Timestamp | None = None,
        since_run_id: str | None = None,
    ) -> PITExpressionGraphResult:
        started_utc = pd.Timestamp.now(tz="UTC")
        run_id = str(uuid.uuid4())
        graph_spec = coerce_expression_graph_spec(spec)
        graph_id = self._upsert_expression_graph_metadata(graph_spec)
        ordered_nodes = graph_spec.ordered_nodes()
        effective_start_asof = self._resolve_expression_graph_effective_start_asof(
            graph_id=graph_id,
            incremental=incremental,
            start_asof=start_asof,
            since_asof=since_asof,
            since_run_id=since_run_id,
        )

        total_rows = 0
        node_rows_written: dict[str, int] = {}
        max_output_asof: pd.Timestamp | None = None

        try:
            for node in ordered_nodes:
                result_df = self._materialize_expression_node_rows(
                    graph_id=graph_id,
                    node=node,
                    start_obs=start_obs,
                    end_obs=end_obs,
                    start_asof=effective_start_asof,
                    end_asof=end_asof,
                    lag_policy=lag_policy,
                )
                if overwrite:
                    self._delete_transformed_rows(
                        output_series_key=node.output_series_key,
                        start_obs=start_obs,
                        end_obs=end_obs,
                        start_asof=effective_start_asof,
                        end_asof=end_asof,
                    )
                rows = int(len(result_df))
                node_rows_written[node.name] = rows
                if rows > 0:
                    self.upsert_pit_observations(result_df, strict=False)
                    row = self.conn.execute(
                        f"""
                        SELECT MAX(asof_utc)
                        FROM {_PIT_TABLE}
                        WHERE series_key = ? AND source = ?
                        """,
                        [
                            node.output_series_key,
                            f"pit_expr_graph:{graph_id}:{node.name}",
                        ],
                    ).fetchone()
                    if row is not None and row[0] is not None:
                        candidate = to_utc_aware(row[0])
                        if max_output_asof is None or candidate > max_output_asof:
                            max_output_asof = candidate
                total_rows += rows

            status = "success"
        except Exception:
            finished_utc = pd.Timestamp.now(tz="UTC")
            self._insert_expression_graph_run(
                run_id=run_id,
                graph_id=graph_id,
                start_obs=start_obs,
                end_obs=end_obs,
                start_asof=start_asof,
                end_asof=end_asof,
                incremental=incremental,
                requested_since_asof=since_asof,
                effective_start_asof=effective_start_asof,
                requested_since_run_id=since_run_id,
                max_output_asof=max_output_asof,
                rows_written=total_rows,
                node_count=len(node_rows_written),
                status="failed",
                started_utc=started_utc,
                finished_utc=finished_utc,
            )
            raise

        finished_utc = pd.Timestamp.now(tz="UTC")
        self._insert_expression_graph_run(
            run_id=run_id,
            graph_id=graph_id,
            start_obs=start_obs,
            end_obs=end_obs,
            start_asof=start_asof,
            end_asof=end_asof,
            incremental=incremental,
            requested_since_asof=since_asof,
            effective_start_asof=effective_start_asof,
            requested_since_run_id=since_run_id,
            max_output_asof=max_output_asof,
            rows_written=total_rows,
            node_count=len(node_rows_written),
            status=status,
            started_utc=started_utc,
            finished_utc=finished_utc,
        )

        return PITExpressionGraphResult(
            graph_id=graph_id,
            run_id=run_id,
            status=status,
            rows_written=total_rows,
            node_rows_written=node_rows_written,
            run_started_utc=started_utc,
            run_finished_utc=finished_utc,
            incremental=incremental,
            effective_start_asof=effective_start_asof,
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

    @staticmethod
    def _empty_series_lineage() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "series_key": pd.Series(dtype="object"),
                "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "value": pd.Series(dtype="float64"),
                "source": pd.Series(dtype="object"),
                "meta_json": pd.Series(dtype="object"),
                "lineage_kind": pd.Series(dtype="object"),
                "transform_id": pd.Series(dtype="object"),
                "graph_id": pd.Series(dtype="object"),
                "node_name": pd.Series(dtype="object"),
                "op": pd.Series(dtype="object"),
                "axis": pd.Series(dtype="object"),
                "engine": pd.Series(dtype="object"),
                "engine_requested": pd.Series(dtype="object"),
                "experimental": pd.Series(dtype="bool"),
                "input_series_keys": pd.Series(dtype="object"),
                "source_asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "selected_input_series_key": pd.Series(dtype="object"),
                "selected_input_asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "source_asof_by_series_utc": pd.Series(dtype="object"),
                "max_source_asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "causality_status": pd.Series(dtype="object"),
            }
        )

    @staticmethod
    def _parse_lineage_payload(meta_json: object) -> dict[str, Any]:
        if meta_json is None or pd.isna(meta_json):
            return {}
        try:
            payload = json.loads(str(meta_json))
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _parse_lineage_timestamp(value: object) -> pd.Timestamp | None:
        if value is None:
            return None
        try:
            return to_utc_aware(value)
        except (TypeError, ValueError):
            return None

    def _parse_lineage_timestamp_map(self, value: object) -> dict[str, pd.Timestamp]:
        if not isinstance(value, Mapping):
            return {}
        out: dict[str, pd.Timestamp] = {}
        for key, raw in value.items():
            parsed = self._parse_lineage_timestamp(raw)
            if parsed is not None:
                out[str(key)] = parsed
        return out

    @staticmethod
    def _lineage_kind(payload: Mapping[str, Any]) -> str:
        if "transform_id" in payload:
            return "transform"
        if "graph_id" in payload:
            return "expression_graph"
        if payload:
            return "derived"
        return "raw"

    def get_series_lineage(
        self,
        series_key: str,
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        limit: int | None = 500,
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
            SELECT series_key, obs_date, asof_utc, value, source, meta_json
            FROM {_PIT_TABLE}
            WHERE {where_clause}
            ORDER BY obs_date ASC, asof_utc ASC
        """
        if limit is not None:
            if limit <= 0:
                raise PITContractError("get_series_lineage limit must be > 0 when provided.")
            query += " LIMIT ?"
            params.append(int(limit))

        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return self._empty_series_lineage()

        df["obs_date"] = to_utc_aware(df["obs_date"])
        df["asof_utc"] = to_utc_aware(df["asof_utc"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

        records: list[dict[str, Any]] = []
        for row in df.itertuples(index=False):
            payload = self._parse_lineage_payload(row.meta_json)
            lineage_kind = self._lineage_kind(payload)
            input_series_keys = payload.get("input_series_keys")
            if not isinstance(input_series_keys, list):
                input_key = payload.get("input_series_key")
                input_series_keys = [input_key] if input_key is not None else []
            input_series_keys = tuple(str(key) for key in input_series_keys if str(key).strip())

            source_asof = self._parse_lineage_timestamp(payload.get("source_asof_utc"))
            selected_input_asof = self._parse_lineage_timestamp(
                payload.get("selected_input_asof_utc")
            )
            source_asof_by_series = self._parse_lineage_timestamp_map(
                payload.get("source_asof_by_series_utc")
            )
            max_candidates = [
                ts
                for ts in [source_asof, selected_input_asof, *source_asof_by_series.values()]
                if ts is not None
            ]
            max_source_asof = max(max_candidates) if max_candidates else None

            if lineage_kind == "raw":
                causality_status = "raw"
            elif max_source_asof is None:
                causality_status = "unknown"
            elif max_source_asof > pd.Timestamp(row.asof_utc):
                causality_status = "violation"
            elif bool(payload.get("experimental")):
                causality_status = "experimental"
            else:
                causality_status = "ok"

            records.append(
                {
                    "series_key": row.series_key,
                    "obs_date": pd.Timestamp(row.obs_date),
                    "asof_utc": pd.Timestamp(row.asof_utc),
                    "value": row.value,
                    "source": row.source,
                    "meta_json": row.meta_json,
                    "lineage_kind": lineage_kind,
                    "transform_id": payload.get("transform_id"),
                    "graph_id": payload.get("graph_id"),
                    "node_name": payload.get("node_name"),
                    "op": payload.get("op"),
                    "axis": payload.get("axis"),
                    "engine": payload.get("engine"),
                    "engine_requested": payload.get("engine_requested"),
                    "experimental": bool(payload.get("experimental", False)),
                    "input_series_keys": input_series_keys,
                    "source_asof_utc": source_asof,
                    "selected_input_series_key": payload.get("selected_input_series_key"),
                    "selected_input_asof_utc": selected_input_asof,
                    "source_asof_by_series_utc": source_asof_by_series,
                    "max_source_asof_utc": max_source_asof,
                    "causality_status": causality_status,
                }
            )

        return pd.DataFrame(records)

    def explain_series(
        self,
        series_key: str,
        start_obs: pd.Timestamp | None = None,
        end_obs: pd.Timestamp | None = None,
        start_asof: pd.Timestamp | None = None,
        end_asof: pd.Timestamp | None = None,
        *,
        limit: int | None = 500,
    ) -> dict[str, Any]:
        lineage = self.get_series_lineage(
            series_key,
            start_obs=start_obs,
            end_obs=end_obs,
            start_asof=start_asof,
            end_asof=end_asof,
            limit=limit,
        )
        if lineage.empty:
            return {
                "series_key": series_key,
                "row_count": 0,
                "derived_row_count": 0,
                "lineage_kinds": [],
                "input_series_keys": [],
                "transform_ids": [],
                "graph_ids": [],
                "causality_status_counts": {},
                "causality_safe": True,
            }

        derived = lineage[lineage["lineage_kind"] != "raw"]
        input_series_keys = sorted(
            {
                key
                for keys in lineage["input_series_keys"]
                for key in (keys if isinstance(keys, tuple) else tuple())
            }
        )
        transform_ids = sorted(
            {str(value) for value in lineage["transform_id"].dropna().tolist() if str(value).strip()}
        )
        graph_ids = sorted(
            {str(value) for value in lineage["graph_id"].dropna().tolist() if str(value).strip()}
        )
        status_counts = {
            str(key): int(value)
            for key, value in lineage["causality_status"].value_counts(dropna=False).items()
        }
        causality_safe = (
            status_counts.get("violation", 0) == 0
            and status_counts.get("unknown", 0) == 0
            and status_counts.get("experimental", 0) == 0
        )

        return {
            "series_key": series_key,
            "row_count": int(len(lineage)),
            "derived_row_count": int(len(derived)),
            "lineage_kinds": sorted({str(value) for value in lineage["lineage_kind"].tolist()}),
            "input_series_keys": input_series_keys,
            "transform_ids": transform_ids,
            "graph_ids": graph_ids,
            "causality_status_counts": status_counts,
            "causality_safe": causality_safe,
            "latest_output_asof_utc": lineage["asof_utc"].max(),
            "latest_source_asof_utc": lineage["max_source_asof_utc"].dropna().max()
            if lineage["max_source_asof_utc"].notna().any()
            else None,
        }

    def list_pipelines(self, pipeline_id: str | None = None) -> pd.DataFrame:
        query = f"""
            SELECT
                pipeline_id,
                spec_hash,
                spec_json,
                description,
                created_utc
            FROM {_PIT_PIPELINES_TABLE}
        """
        params: list[object] = []
        if pipeline_id is not None:
            query += " WHERE pipeline_id = ?"
            params.append(pipeline_id)
        query += " ORDER BY created_utc DESC, pipeline_id"

        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return df
        df["created_utc"] = to_utc_aware(df["created_utc"])
        return df

    def list_pipeline_runs(
        self,
        pipeline_id: str | None = None,
        *,
        limit: int | None = 100,
    ) -> pd.DataFrame:
        query = f"""
            SELECT
                run_id,
                pipeline_id,
                start_obs,
                end_obs,
                start_asof,
                end_asof,
                incremental,
                requested_since_asof,
                effective_start_asof,
                requested_since_run_id,
                max_output_asof,
                rows_written,
                step_count,
                status,
                started_utc,
                finished_utc
            FROM {_PIT_PIPELINE_RUNS_TABLE}
        """
        params: list[object] = []
        clauses: list[str] = []
        if pipeline_id is not None:
            clauses.append("pipeline_id = ?")
            params.append(pipeline_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY started_utc DESC, run_id"
        if limit is not None:
            if limit <= 0:
                raise PITContractError("list_pipeline_runs limit must be > 0 when provided.")
            query += " LIMIT ?"
            params.append(int(limit))

        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return df
        dt_cols = [
            "start_obs",
            "end_obs",
            "start_asof",
            "end_asof",
            "requested_since_asof",
            "effective_start_asof",
            "max_output_asof",
            "started_utc",
            "finished_utc",
        ]
        for col in dt_cols:
            if col in df.columns:
                df[col] = to_utc_aware(df[col])
        return df

    def list_expression_graphs(self, graph_id: str | None = None) -> pd.DataFrame:
        query = f"""
            SELECT
                graph_id,
                spec_hash,
                spec_json,
                description,
                created_utc
            FROM {_PIT_EXPR_GRAPHS_TABLE}
        """
        params: list[object] = []
        if graph_id is not None:
            query += " WHERE graph_id = ?"
            params.append(graph_id)
        query += " ORDER BY created_utc DESC, graph_id"

        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return df
        df["created_utc"] = to_utc_aware(df["created_utc"])
        return df

    def list_expression_graph_runs(
        self,
        graph_id: str | None = None,
        *,
        limit: int | None = 100,
    ) -> pd.DataFrame:
        query = f"""
            SELECT
                run_id,
                graph_id,
                start_obs,
                end_obs,
                start_asof,
                end_asof,
                incremental,
                requested_since_asof,
                effective_start_asof,
                requested_since_run_id,
                max_output_asof,
                rows_written,
                node_count,
                status,
                started_utc,
                finished_utc
            FROM {_PIT_EXPR_GRAPH_RUNS_TABLE}
        """
        params: list[object] = []
        clauses: list[str] = []
        if graph_id is not None:
            clauses.append("graph_id = ?")
            params.append(graph_id)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY started_utc DESC, run_id"
        if limit is not None:
            if limit <= 0:
                raise PITContractError(
                    "list_expression_graph_runs limit must be > 0 when provided."
                )
            query += " LIMIT ?"
            params.append(int(limit))

        df = self.conn.execute(query, params).fetchdf()
        if df.empty:
            return df
        dt_cols = [
            "start_obs",
            "end_obs",
            "start_asof",
            "end_asof",
            "requested_since_asof",
            "effective_start_asof",
            "max_output_asof",
            "started_utc",
            "finished_utc",
        ]
        for col in dt_cols:
            if col in df.columns:
                df[col] = to_utc_aware(df[col])
        return df
