from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, TypedDict, cast

import duckdb
import pandas as pd

from .exceptions import (
    PITContractError,
    PITEngineError,
    PITUnsupportedOperationError,
    PITValidationError,
)

TransformAxis = Literal["obs_path", "revision_path"]
TransformOp = Literal[
    "resample",
    "aggregate",
    "rolling",
    "expanding",
    "lag",
    "diff",
    "path_apply",
]
TransformEngine = Literal["auto", "duckdb", "python"]
EngineMismatchPolicy = Literal["error", "fallback"]
RuntimeEngine = Literal["duckdb", "python"]
AggName = Literal["first", "last", "min", "max", "mean", "sum"]


class ResampleParams(TypedDict, total=False):
    rule: str
    agg: AggName


class AggregateParams(TypedDict, total=False):
    rule: str
    agg: AggName


class RollingParams(TypedDict, total=False):
    window: int
    min_periods: int
    agg: AggName


class ExpandingParams(TypedDict, total=False):
    min_periods: int
    agg: AggName


class LagParams(TypedDict, total=False):
    periods: int


class DiffParams(TypedDict, total=False):
    periods: int


class PathApplyParams(TypedDict, total=False):
    udf_name: str
    func: Callable[[pd.Series], Any]


TransformParams = (
    ResampleParams
    | AggregateParams
    | RollingParams
    | ExpandingParams
    | LagParams
    | DiffParams
    | PathApplyParams
    | dict[str, Any]
)

_ALLOWED_AXIS_OPS: dict[TransformAxis, tuple[TransformOp, ...]] = {
    "obs_path": (
        "resample",
        "aggregate",
        "rolling",
        "expanding",
        "lag",
        "diff",
        "path_apply",
    ),
    "revision_path": (
        "rolling",
        "expanding",
        "lag",
        "diff",
    ),
}

_ALLOWED_AGGS: set[str] = {"first", "last", "min", "max", "mean", "sum"}
_ALLOWED_PARAM_KEYS: dict[TransformOp, set[str]] = {
    "resample": {"rule", "agg"},
    "aggregate": {"rule", "agg"},
    "rolling": {"window", "min_periods", "agg"},
    "expanding": {"min_periods", "agg"},
    "lag": {"periods"},
    "diff": {"periods"},
    "path_apply": {"udf_name", "func"},
}


@dataclass(frozen=True)
class PITEngineResolution:
    engine_requested: TransformEngine
    engine_used: RuntimeEngine
    fallback_reason: str | None = None


@dataclass(frozen=True)
class PITTransformSpec:
    input_series_key: str
    output_series_key: str
    axis: TransformAxis = "obs_path"
    op: TransformOp = "resample"
    params: dict[str, Any] = field(default_factory=dict)
    engine: TransformEngine = "auto"

    def normalized_params(self, *, include_callable: bool = False) -> dict[str, Any]:
        return normalize_transform_params(
            self.op,
            self.params,
            include_callable=include_callable,
        )

    def sanitized_params(self) -> dict[str, Any]:
        """Params suitable for hashing and lineage serialization."""
        return self.normalized_params(include_callable=False)

    def spec_payload(self) -> dict[str, Any]:
        return {
            "input_series_key": self.input_series_key,
            "output_series_key": self.output_series_key,
            "axis": self.axis,
            "op": self.op,
            "params": self.sanitized_params(),
            "engine": self.engine,
        }

    def spec_hash(self) -> str:
        payload = json.dumps(self.spec_payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def transform_id(self) -> str:
        digest = self.spec_hash()[:16]
        return f"{self.output_series_key}:{digest}"


@dataclass(frozen=True)
class PITTransformResult:
    transform_id: str
    output_series_key: str
    rows_written: int
    engine_used: str
    run_started_utc: pd.Timestamp
    run_finished_utc: pd.Timestamp
    engine_requested: str
    fallback_reason: str | None = None


def _supported_combinations_text() -> str:
    items: list[str] = []
    for axis, ops_tuple in _ALLOWED_AXIS_OPS.items():
        ops = ", ".join(ops_tuple)
        items.append(f"{axis}: {ops}")
    return "; ".join(items)


def coerce_transform_spec(spec: PITTransformSpec | Mapping[str, Any]) -> PITTransformSpec:
    if isinstance(spec, PITTransformSpec):
        return spec

    if not isinstance(spec, Mapping):
        raise PITContractError(
            "Transform spec must be PITTransformSpec or a mapping with transform fields."
        )

    allowed_keys = {"input_series_key", "output_series_key", "axis", "op", "params", "engine"}
    unknown_keys = sorted(set(spec.keys()) - allowed_keys)
    if unknown_keys:
        raise PITContractError(f"Unknown transform spec keys: {unknown_keys}")

    payload = dict(spec)
    if "params" not in payload:
        payload["params"] = {}

    try:
        return PITTransformSpec(**payload)
    except TypeError as exc:
        raise PITContractError(f"Invalid transform spec mapping: {exc}") from exc


def _normalize_resample_rule(rule: str) -> str:
    aliases = {
        "M": "ME",
        "Q": "QE",
        "A": "YE",
        "Y": "YE",
    }
    return aliases.get(rule.upper(), rule.upper())


def _validate_agg(agg: str) -> str:
    out = agg.lower()
    if out not in _ALLOWED_AGGS:
        allowed_agg = ", ".join(sorted(_ALLOWED_AGGS))
        raise PITValidationError(f"Unsupported aggregation '{agg}'. Allowed: {allowed_agg}")
    return out


def normalize_transform_params(
    op: TransformOp,
    params: Mapping[str, Any],
    *,
    include_callable: bool = False,
) -> dict[str, Any]:
    if not isinstance(params, Mapping):
        raise PITValidationError("Transform params must be a mapping.")

    unknown_keys = sorted(set(params.keys()) - _ALLOWED_PARAM_KEYS[op])
    if unknown_keys:
        raise PITValidationError(
            f"Unknown params for op='{op}': {unknown_keys}. "
            f"Allowed keys: {sorted(_ALLOWED_PARAM_KEYS[op])}"
        )

    out: dict[str, Any] = {}

    if op == "resample":
        rule = str(params.get("rule", "")).strip()
        if not rule:
            raise PITValidationError("resample requires params['rule'].")
        out["rule"] = _normalize_resample_rule(rule)
        out["agg"] = _validate_agg(str(params.get("agg", "last")))

    elif op == "aggregate":
        if "rule" in params and params["rule"] is not None:
            rule = str(params["rule"]).strip()
            if not rule:
                raise PITValidationError("aggregate params['rule'] must be non-empty when set.")
            out["rule"] = _normalize_resample_rule(rule)
        out["agg"] = _validate_agg(str(params.get("agg", "last")))

    elif op == "rolling":
        window = int(params.get("window", 1))
        min_periods = int(params.get("min_periods", window))
        if window <= 0:
            raise PITValidationError("rolling requires params['window'] > 0.")
        if min_periods <= 0:
            raise PITValidationError("rolling requires params['min_periods'] > 0.")
        out["window"] = window
        out["min_periods"] = min_periods
        out["agg"] = _validate_agg(str(params.get("agg", "mean")))

    elif op == "expanding":
        min_periods = int(params.get("min_periods", 1))
        if min_periods <= 0:
            raise PITValidationError("expanding requires params['min_periods'] > 0.")
        out["min_periods"] = min_periods
        out["agg"] = _validate_agg(str(params.get("agg", "mean")))

    elif op in {"lag", "diff"}:
        periods = int(params.get("periods", 1))
        if periods <= 0:
            raise PITValidationError(f"{op} requires params['periods'] > 0.")
        out["periods"] = periods

    elif op == "path_apply":
        udf_name = str(params.get("udf_name", "")).strip()
        if not udf_name:
            raise PITValidationError("path_apply requires params['udf_name'] for lineage.")
        out["udf_name"] = udf_name

        func = params.get("func")
        if not callable(func):
            raise PITValidationError("path_apply requires params['func'] callable.")
        if include_callable:
            out["func"] = func

    else:
        raise PITUnsupportedOperationError(f"Unsupported PIT transform op: {op}")

    return out


def validate_transform_spec(spec: PITTransformSpec) -> None:
    if not spec.input_series_key:
        raise PITContractError("input_series_key is required.")
    if not spec.output_series_key:
        raise PITContractError("output_series_key is required.")

    if spec.axis not in _ALLOWED_AXIS_OPS:
        allowed_axes = ", ".join(sorted(_ALLOWED_AXIS_OPS))
        raise PITUnsupportedOperationError(
            f"Unsupported transform axis: '{spec.axis}'. Allowed: {allowed_axes}"
        )

    allowed = _ALLOWED_AXIS_OPS[spec.axis]
    if spec.op not in allowed:
        combos = _supported_combinations_text()
        raise PITUnsupportedOperationError(
            "Unsupported op/axis combination: "
            f"axis='{spec.axis}', op='{spec.op}'. "
            f"Supported combinations -> {combos}"
        )

    normalize_transform_params(spec.op, spec.params, include_callable=True)


def _is_duckdb_rule_supported(rule: str) -> bool:
    return rule in {"ME", "QE", "YE"}


def duckdb_supports_spec(spec: PITTransformSpec) -> bool:
    if spec.op == "path_apply":
        return False

    params = spec.normalized_params(include_callable=False)

    if spec.op == "resample":
        rule = str(params.get("rule", ""))
        return _is_duckdb_rule_supported(rule)

    if spec.op == "aggregate":
        rule_value = params.get("rule")
        if rule_value is None:
            return True
        return _is_duckdb_rule_supported(str(rule_value))

    return spec.op in {"rolling", "expanding", "lag", "diff"}


def resolve_engine(
    spec: PITTransformSpec,
    *,
    on_engine_mismatch: EngineMismatchPolicy = "error",
) -> PITEngineResolution:
    requested = spec.engine
    if requested == "python":
        return PITEngineResolution(
            engine_requested=requested,
            engine_used="python",
            fallback_reason=None,
        )

    if requested == "auto":
        return PITEngineResolution(
            engine_requested=requested,
            engine_used="duckdb" if duckdb_supports_spec(spec) else "python",
            fallback_reason=None,
        )

    if requested == "duckdb":
        if duckdb_supports_spec(spec):
            return PITEngineResolution(
                engine_requested=requested,
                engine_used="duckdb",
                fallback_reason=None,
            )
        if on_engine_mismatch == "error":
            raise PITEngineError(
                "Requested engine='duckdb' is not supported for this PIT transform spec. "
                "Use engine='python' or set on_engine_mismatch='fallback'."
            )
        return PITEngineResolution(
            engine_requested=requested,
            engine_used="python",
            fallback_reason="duckdb_unsupported_for_spec",
        )

    raise PITEngineError(f"Unsupported transform engine: {requested}")


def _coerce_utc_index(idx: pd.Index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(idx))
    if out.tz is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _apply_named_agg(obj: Any, agg: str) -> pd.Series:
    if not hasattr(obj, agg):
        raise PITUnsupportedOperationError(f"Unsupported aggregation: {agg}")
    out = getattr(obj, agg)()
    if not isinstance(out, pd.Series):
        raise PITContractError("Aggregation did not return a pandas Series.")
    return out


def _aggregate_scalar(series: pd.Series, agg: str) -> float:
    if agg == "first":
        non_null = series.dropna()
        value = non_null.iloc[0] if not non_null.empty else float("nan")
        return float(value) if pd.notna(value) else float("nan")
    if agg == "last":
        non_null = series.dropna()
        value = non_null.iloc[-1] if not non_null.empty else float("nan")
        return float(value) if pd.notna(value) else float("nan")

    if not hasattr(series, agg):
        raise PITUnsupportedOperationError(f"Unsupported aggregation: {agg}")
    value = getattr(series, agg)()
    return float(value) if pd.notna(value) else float("nan")


def _window_agg_expr(agg: str) -> str:
    if agg == "sum":
        return "SUM(value)"
    if agg == "mean":
        return "AVG(value)"
    if agg == "min":
        return "MIN(value)"
    if agg == "max":
        return "MAX(value)"
    if agg == "first":
        return "FIRST_VALUE(value)"
    if agg == "last":
        return "LAST_VALUE(value)"
    raise PITUnsupportedOperationError(f"Unsupported aggregation: {agg}")


def _grouped_agg_expr(agg: str) -> str:
    if agg == "sum":
        return "SUM(value)"
    if agg == "mean":
        return "AVG(value)"
    if agg == "min":
        return "MIN(value)"
    if agg == "max":
        return "MAX(value)"
    if agg == "first":
        return "list_extract(list(value ORDER BY ts) FILTER (WHERE value IS NOT NULL), 1)"
    if agg == "last":
        return "list_extract(list(value ORDER BY ts DESC) FILTER (WHERE value IS NOT NULL), 1)"
    raise PITUnsupportedOperationError(f"Unsupported aggregation: {agg}")


def _bucket_end_expr(rule: str) -> str:
    if rule == "ME":
        return "(DATE_TRUNC('month', ts)::DATE + INTERVAL '1 month' - INTERVAL '1 day')::TIMESTAMP"
    if rule == "QE":
        return "(DATE_TRUNC('quarter', ts)::DATE + INTERVAL '3 month' - INTERVAL '1 day')::TIMESTAMP"
    if rule == "YE":
        return "(DATE_TRUNC('year', ts)::DATE + INTERVAL '1 year' - INTERVAL '1 day')::TIMESTAMP"
    raise PITUnsupportedOperationError(
        f"DuckDB engine does not support resample/aggregate rule '{rule}'. "
        "Supported rules: ME, QE, YE."
    )


def _to_duckdb_frame(s: pd.Series) -> pd.DataFrame:
    idx = _coerce_utc_index(s.index).tz_localize(None)
    return pd.DataFrame({"ts": idx, "value": _as_numeric(s).to_numpy()})


def _run_duckdb_series_query(df: pd.DataFrame, query: str) -> pd.Series:
    conn = duckdb.connect()
    conn.register("series_input", df)
    try:
        out = conn.execute(query).fetchdf()
    finally:
        conn.unregister("series_input")
        conn.close()

    if out.empty:
        return pd.Series(dtype="float64")

    out = out.dropna(subset=["ts"])
    if out.empty:
        return pd.Series(dtype="float64")

    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    series = pd.Series(
        pd.to_numeric(out["value"], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(out["ts"]),
    ).sort_index()
    return series


def _apply_series_op_python(s: pd.Series, spec: PITTransformSpec) -> pd.Series:
    op = spec.op
    params = spec.normalized_params(include_callable=True)

    if op == "resample":
        transformed = _apply_named_agg(s.resample(params["rule"]), params["agg"])

    elif op == "aggregate":
        aggregate_rule = params.get("rule")
        agg = str(params["agg"])
        if aggregate_rule is not None:
            transformed = _apply_named_agg(s.resample(str(aggregate_rule)), agg)
        else:
            if s.empty:
                transformed = pd.Series(dtype="float64")
            else:
                transformed = pd.Series(
                    [_aggregate_scalar(s, agg)],
                    index=pd.DatetimeIndex([s.index.max()]),
                )

    elif op == "rolling":
        transformed = _apply_named_agg(
            s.rolling(
                window=int(params["window"]),
                min_periods=int(params["min_periods"]),
            ),
            str(params["agg"]),
        )

    elif op == "expanding":
        transformed = _apply_named_agg(
            s.expanding(min_periods=int(params["min_periods"])),
            str(params["agg"]),
        )

    elif op == "lag":
        transformed = s.shift(periods=int(params["periods"]))

    elif op == "diff":
        transformed = s.diff(periods=int(params["periods"]))

    elif op == "path_apply":
        func = cast(Callable[[pd.Series], Any], params["func"])
        applied = func(s.copy())
        if isinstance(applied, pd.Series):
            transformed = applied
        else:
            transformed = pd.Series(applied, index=s.index)

    else:
        raise PITUnsupportedOperationError(f"Unsupported PIT transform op: {op}")

    out = pd.Series(transformed).sort_index()
    out.index = _coerce_utc_index(out.index)
    out = _as_numeric(out)
    out.name = spec.output_series_key
    return out


def _apply_series_op_duckdb(s: pd.Series, spec: PITTransformSpec) -> pd.Series:
    op = spec.op
    params = spec.normalized_params(include_callable=True)
    if s.empty:
        return pd.Series(dtype="float64", name=spec.output_series_key)

    df = _to_duckdb_frame(s)

    if op == "lag":
        periods = int(params["periods"])
        query = f"""
            SELECT ts, LAG(value, {periods}) OVER (ORDER BY ts) AS value
            FROM series_input
            ORDER BY ts
        """
    elif op == "diff":
        periods = int(params["periods"])
        query = f"""
            SELECT ts, value - LAG(value, {periods}) OVER (ORDER BY ts) AS value
            FROM series_input
            ORDER BY ts
        """
    elif op == "rolling":
        window = int(params["window"])
        min_periods = int(params["min_periods"])
        agg_expr = _window_agg_expr(str(params["agg"]))
        query = f"""
            SELECT
                ts,
                CASE
                    WHEN COUNT(value) OVER w >= {min_periods}
                    THEN {agg_expr} OVER w
                    ELSE NULL
                END AS value
            FROM series_input
            WINDOW w AS (
                ORDER BY ts
                ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW
            )
            ORDER BY ts
        """
    elif op == "expanding":
        min_periods = int(params["min_periods"])
        agg_expr = _window_agg_expr(str(params["agg"]))
        query = f"""
            SELECT
                ts,
                CASE
                    WHEN COUNT(value) OVER w >= {min_periods}
                    THEN {agg_expr} OVER w
                    ELSE NULL
                END AS value
            FROM series_input
            WINDOW w AS (
                ORDER BY ts
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            )
            ORDER BY ts
        """
    elif op == "resample":
        rule = str(params["rule"])
        bucket_expr = _bucket_end_expr(rule)
        agg_expr = _grouped_agg_expr(str(params["agg"]))
        query = f"""
            SELECT {bucket_expr} AS ts, {agg_expr} AS value
            FROM series_input
            GROUP BY 1
            ORDER BY 1
        """
    elif op == "aggregate":
        agg_expr = _grouped_agg_expr(str(params["agg"]))
        rule_value = params.get("rule")
        if rule_value is None:
            query = f"""
                SELECT MAX(ts) AS ts, {agg_expr} AS value
                FROM series_input
            """
        else:
            bucket_expr = _bucket_end_expr(str(rule_value))
            query = f"""
                SELECT {bucket_expr} AS ts, {agg_expr} AS value
                FROM series_input
                GROUP BY 1
                ORDER BY 1
            """
    else:
        raise PITUnsupportedOperationError(
            f"DuckDB engine does not support PIT op='{op}' for axis='{spec.axis}'."
        )

    out = _run_duckdb_series_query(df, query)
    out.index = _coerce_utc_index(out.index)
    out.name = spec.output_series_key
    return out


def apply_obs_path_transform(
    snapshot: pd.Series,
    spec: PITTransformSpec,
    *,
    engine: RuntimeEngine = "python",
) -> pd.Series:
    """Apply a PIT transform over an obs_date-indexed snapshot path."""
    if spec.axis != "obs_path":
        raise PITContractError("apply_obs_path_transform requires axis='obs_path'.")
    validate_transform_spec(spec)

    s = snapshot.copy().sort_index()
    s = _as_numeric(s)
    s.index = _coerce_utc_index(s.index)
    if engine == "duckdb":
        return _apply_series_op_duckdb(s, spec)
    return _apply_series_op_python(s, spec)


def apply_revision_path_transform(
    timeline: pd.Series,
    spec: PITTransformSpec,
    *,
    engine: RuntimeEngine = "python",
) -> pd.Series:
    """Apply a PIT transform over an asof_utc-indexed revision timeline."""
    if spec.axis != "revision_path":
        raise PITContractError("apply_revision_path_transform requires axis='revision_path'.")
    validate_transform_spec(spec)

    s = timeline.copy().sort_index()
    s = _as_numeric(s)
    s.index = _coerce_utc_index(s.index)
    if engine == "duckdb":
        return _apply_series_op_duckdb(s, spec)
    return _apply_series_op_python(s, spec)


def serialize_params_for_lineage(params: Mapping[str, Any]) -> str:
    serializable: dict[str, Any] = {}
    for key, value in params.items():
        if callable(value):
            continue
        if isinstance(value, pd.Timestamp):
            serializable[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            serializable[key] = str(value)
        else:
            serializable[key] = value
    return json.dumps(serializable, sort_keys=True, separators=(",", ":"))


def ensure_callable(path_func: Callable[[pd.Series], Any]) -> Callable[[pd.Series], Any]:
    if not callable(path_func):
        raise PITValidationError("Provided path function is not callable.")
    return path_func
