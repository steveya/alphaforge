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
    "pct_change",
    "ffill",
    "binary",
    "coalesce",
    "splice",
    "path_apply",
]
TransformEngine = Literal["auto", "duckdb", "python"]
EngineMismatchPolicy = Literal["error", "fallback"]
RuntimeEngine = Literal["duckdb", "python"]
AggName = Literal["first", "last", "min", "max", "mean", "sum", "count", "std", "var"]
BinaryOperator = Literal["add", "sub", "mul", "div"]
JoinMode = Literal["inner", "left", "right", "outer"]
SpliceAdjustment = Literal["ratio", "add"]
TransformInputKind = Literal["single", "multi"]


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


class PctChangeParams(TypedDict, total=False):
    periods: int


class FFillParams(TypedDict, total=False):
    limit: int


class BinaryParams(TypedDict, total=False):
    right_series_key: str
    operator: BinaryOperator
    join: JoinMode
    fill_value: float


class CoalesceParams(TypedDict, total=False):
    other_series_keys: list[str] | tuple[str, ...] | str


class SpliceParams(TypedDict, total=False):
    right_series_key: str
    adjustment: SpliceAdjustment
    transition_periods: int
    join: JoinMode


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
    | PctChangeParams
    | FFillParams
    | BinaryParams
    | CoalesceParams
    | SpliceParams
    | PathApplyParams
    | dict[str, Any]
)


@dataclass
class TransformOperatorDef:
    allowed_axes: tuple[TransformAxis, ...]
    param_keys: set[str]
    normalize: Callable[[Mapping[str, Any], bool], dict[str, Any]]
    input_series_keys: Callable[[PITTransformSpec], list[str]]
    input_kind: TransformInputKind = "single"
    duckdb_supports: Callable[[dict[str, Any]], bool] | None = None
    python_runner: Callable[[pd.Series, PITTransformSpec, dict[str, Any]], pd.Series] | None = None
    duckdb_runner: Callable[[pd.Series, PITTransformSpec, dict[str, Any]], pd.Series] | None = None


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


def _get_operator_def(op: TransformOp) -> TransformOperatorDef:
    try:
        return _OPERATORS[op]
    except KeyError as exc:
        raise PITUnsupportedOperationError(f"Unsupported PIT transform op: {op}") from exc


def _default_input_series_keys(spec: PITTransformSpec) -> list[str]:
    return [spec.input_series_key]


def _binary_input_series_keys(spec: PITTransformSpec) -> list[str]:
    keys = [spec.input_series_key]
    right_key = str(spec.sanitized_params().get("right_series_key", "")).strip()
    if right_key:
        keys.append(right_key)
    return keys


def _coalesce_input_series_keys(spec: PITTransformSpec) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for key in [spec.input_series_key, *spec.sanitized_params().get("other_series_keys", [])]:
        text = str(key).strip()
        if not text or text in seen:
            continue
        keys.append(text)
        seen.add(text)
    return keys


def _splice_input_series_keys(spec: PITTransformSpec) -> list[str]:
    keys = [spec.input_series_key]
    right_key = str(spec.sanitized_params().get("right_series_key", "")).strip()
    if right_key:
        keys.append(right_key)
    return keys


def transform_input_series_keys(spec: PITTransformSpec) -> list[str]:
    return _get_operator_def(spec.op).input_series_keys(spec)


def transform_input_kind(spec: PITTransformSpec) -> TransformInputKind:
    return _get_operator_def(spec.op).input_kind


def _supported_combinations_text() -> str:
    ops_by_axis: dict[TransformAxis, list[str]] = {"obs_path": [], "revision_path": []}
    for op_name, op_def in _OPERATORS.items():
        for axis in op_def.allowed_axes:
            ops_by_axis[axis].append(op_name)
    items = [f"{axis}: {', '.join(ops)}" for axis, ops in ops_by_axis.items()]
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


def _validate_positive_int(*, value: Any, name: str) -> int:
    out = int(value)
    if out <= 0:
        raise PITValidationError(f"{name} must be > 0.")
    return out


def _validate_non_negative_int(*, value: Any, name: str) -> int:
    out = int(value)
    if out < 0:
        raise PITValidationError(f"{name} must be >= 0.")
    return out


def _normalize_resample_params(params: Mapping[str, Any], include_callable: bool) -> dict[str, Any]:
    del include_callable
    rule = str(params.get("rule", "")).strip()
    if not rule:
        raise PITValidationError("resample requires params['rule'].")
    return {
        "rule": _normalize_resample_rule(rule),
        "agg": _validate_agg(str(params.get("agg", "last"))),
    }


def _normalize_aggregate_params(params: Mapping[str, Any], include_callable: bool) -> dict[str, Any]:
    del include_callable
    out: dict[str, Any] = {"agg": _validate_agg(str(params.get("agg", "last")))}
    if "rule" in params and params["rule"] is not None:
        rule = str(params["rule"]).strip()
        if not rule:
            raise PITValidationError("aggregate params['rule'] must be non-empty when set.")
        out["rule"] = _normalize_resample_rule(rule)
    return out


def _normalize_rolling_params(params: Mapping[str, Any], include_callable: bool) -> dict[str, Any]:
    del include_callable
    window = _validate_positive_int(value=params.get("window", 1), name="rolling params['window']")
    min_periods = _validate_positive_int(
        value=params.get("min_periods", window),
        name="rolling params['min_periods']",
    )
    return {
        "window": window,
        "min_periods": min_periods,
        "agg": _validate_agg(str(params.get("agg", "mean"))),
    }


def _normalize_expanding_params(
    params: Mapping[str, Any],
    include_callable: bool,
) -> dict[str, Any]:
    del include_callable
    min_periods = _validate_positive_int(
        value=params.get("min_periods", 1),
        name="expanding params['min_periods']",
    )
    return {
        "min_periods": min_periods,
        "agg": _validate_agg(str(params.get("agg", "mean"))),
    }


def _normalize_periods_params(
    op: TransformOp,
    params: Mapping[str, Any],
    include_callable: bool,
) -> dict[str, Any]:
    del include_callable
    periods = _validate_positive_int(value=params.get("periods", 1), name=f"{op} params['periods']")
    return {"periods": periods}


def _normalize_ffill_params(params: Mapping[str, Any], include_callable: bool) -> dict[str, Any]:
    del include_callable
    out: dict[str, Any] = {}
    if "limit" in params and params["limit"] is not None:
        out["limit"] = _validate_positive_int(value=params["limit"], name="ffill params['limit']")
    return out


def _normalize_binary_params(params: Mapping[str, Any], include_callable: bool) -> dict[str, Any]:
    del include_callable
    right_series_key = str(params.get("right_series_key", "")).strip()
    if not right_series_key:
        raise PITValidationError("binary requires params['right_series_key'].")

    operator = str(params.get("operator", "sub")).strip().lower()
    if operator not in {"add", "sub", "mul", "div"}:
        raise PITValidationError(
            "binary requires params['operator'] in ['add', 'sub', 'mul', 'div']."
        )

    join = str(params.get("join", "inner")).strip().lower()
    if join not in {"inner", "left", "right", "outer"}:
        raise PITValidationError(
            "binary requires params['join'] in ['inner', 'left', 'right', 'outer']."
        )

    out: dict[str, Any] = {
        "right_series_key": right_series_key,
        "operator": operator,
        "join": join,
    }
    if "fill_value" in params and params["fill_value"] is not None:
        out["fill_value"] = float(params["fill_value"])
    return out


def _normalize_coalesce_params(params: Mapping[str, Any], include_callable: bool) -> dict[str, Any]:
    del include_callable
    raw_values = params.get("other_series_keys")
    if isinstance(raw_values, str):
        items = [raw_values]
    elif isinstance(raw_values, (list, tuple)):
        items = list(raw_values)
    else:
        raise PITValidationError("coalesce requires params['other_series_keys'] list/tuple.")

    cleaned: list[str] = []
    seen: set[str] = set()
    for value in items:
        text = str(value).strip()
        if not text or text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    if not cleaned:
        raise PITValidationError("coalesce requires at least one non-empty other_series_key.")
    return {"other_series_keys": cleaned}


def _normalize_splice_params(params: Mapping[str, Any], include_callable: bool) -> dict[str, Any]:
    del include_callable
    right_series_key = str(params.get("right_series_key", "")).strip()
    if not right_series_key:
        raise PITValidationError("splice requires params['right_series_key'].")

    adjustment = str(params.get("adjustment", "")).strip().lower()
    if adjustment not in {"ratio", "add"}:
        raise PITValidationError("splice requires params['adjustment'] in ['ratio', 'add'].")

    join = str(params.get("join", "outer")).strip().lower()
    if join not in {"inner", "left", "right", "outer"}:
        raise PITValidationError(
            "splice requires params['join'] in ['inner', 'left', 'right', 'outer']."
        )

    transition_periods = _validate_non_negative_int(
        value=params.get("transition_periods", 0),
        name="splice params['transition_periods']",
    )

    return {
        "right_series_key": right_series_key,
        "adjustment": adjustment,
        "transition_periods": transition_periods,
        "join": join,
    }


def _normalize_path_apply_params(
    params: Mapping[str, Any],
    include_callable: bool,
) -> dict[str, Any]:
    udf_name = str(params.get("udf_name", "")).strip()
    if not udf_name:
        raise PITValidationError("path_apply requires params['udf_name'] for lineage.")

    func = params.get("func")
    if not callable(func):
        raise PITValidationError("path_apply requires params['func'] callable.")

    out: dict[str, Any] = {"udf_name": udf_name}
    if include_callable:
        out["func"] = func
    return out


def normalize_transform_params(
    op: TransformOp,
    params: Mapping[str, Any],
    *,
    include_callable: bool = False,
) -> dict[str, Any]:
    if not isinstance(params, Mapping):
        raise PITValidationError("Transform params must be a mapping.")

    operator = _get_operator_def(op)
    unknown_keys = sorted(set(params.keys()) - operator.param_keys)
    if unknown_keys:
        raise PITValidationError(
            f"Unknown params for op='{op}': {unknown_keys}. "
            f"Allowed keys: {sorted(operator.param_keys)}"
        )
    return operator.normalize(params, include_callable)


def validate_transform_spec(spec: PITTransformSpec) -> None:
    if not spec.input_series_key:
        raise PITContractError("input_series_key is required.")
    if not spec.output_series_key:
        raise PITContractError("output_series_key is required.")

    operator = _get_operator_def(spec.op)
    if spec.axis not in operator.allowed_axes:
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
    operator = _get_operator_def(spec.op)
    if operator.duckdb_supports is None:
        return False
    return operator.duckdb_supports(spec.normalized_params(include_callable=False))


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
    if agg == "count":
        return "COUNT(value)"
    if agg == "std":
        return "STDDEV_SAMP(value)"
    if agg == "var":
        return "VAR_SAMP(value)"
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
    if agg == "count":
        return "COUNT(value)"
    if agg == "std":
        return "STDDEV_SAMP(value)"
    if agg == "var":
        return "VAR_SAMP(value)"
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


def _run_resample_python(s: pd.Series, spec: PITTransformSpec, params: dict[str, Any]) -> pd.Series:
    del spec
    return _apply_named_agg(s.resample(params["rule"]), params["agg"])


def _run_aggregate_python(s: pd.Series, spec: PITTransformSpec, params: dict[str, Any]) -> pd.Series:
    del spec
    aggregate_rule = params.get("rule")
    agg = str(params["agg"])
    if aggregate_rule is not None:
        return _apply_named_agg(s.resample(str(aggregate_rule)), agg)
    if s.empty:
        return pd.Series(dtype="float64")
    return pd.Series([_aggregate_scalar(s, agg)], index=pd.DatetimeIndex([s.index.max()]))


def _run_rolling_python(s: pd.Series, spec: PITTransformSpec, params: dict[str, Any]) -> pd.Series:
    del spec
    return _apply_named_agg(
        s.rolling(
            window=int(params["window"]),
            min_periods=int(params["min_periods"]),
        ),
        str(params["agg"]),
    )


def _run_expanding_python(s: pd.Series, spec: PITTransformSpec, params: dict[str, Any]) -> pd.Series:
    del spec
    return _apply_named_agg(
        s.expanding(min_periods=int(params["min_periods"])),
        str(params["agg"]),
    )


def _run_lag_python(s: pd.Series, spec: PITTransformSpec, params: dict[str, Any]) -> pd.Series:
    del spec
    return s.shift(periods=int(params["periods"]))


def _run_diff_python(s: pd.Series, spec: PITTransformSpec, params: dict[str, Any]) -> pd.Series:
    del spec
    return s.diff(periods=int(params["periods"]))


def _run_pct_change_python(
    s: pd.Series,
    spec: PITTransformSpec,
    params: dict[str, Any],
) -> pd.Series:
    del spec
    return s.pct_change(periods=int(params["periods"]), fill_method=None)


def _run_ffill_python(s: pd.Series, spec: PITTransformSpec, params: dict[str, Any]) -> pd.Series:
    del spec
    limit = params.get("limit")
    return s.ffill(limit=int(limit) if limit is not None else None)


def _run_path_apply_python(
    s: pd.Series,
    spec: PITTransformSpec,
    params: dict[str, Any],
) -> pd.Series:
    del spec
    func = cast(Callable[[pd.Series], Any], params["func"])
    applied = func(s.copy())
    if isinstance(applied, pd.Series):
        return applied
    return pd.Series(applied, index=s.index)


def _run_lag_duckdb(s: pd.Series, spec: PITTransformSpec, params: dict[str, Any]) -> pd.Series:
    del spec
    df = _to_duckdb_frame(s)
    periods = int(params["periods"])
    query = f"""
        SELECT ts, LAG(value, {periods}) OVER (ORDER BY ts) AS value
        FROM series_input
        ORDER BY ts
    """
    return _run_duckdb_series_query(df, query)


def _run_diff_duckdb(s: pd.Series, spec: PITTransformSpec, params: dict[str, Any]) -> pd.Series:
    del spec
    df = _to_duckdb_frame(s)
    periods = int(params["periods"])
    query = f"""
        SELECT ts, value - LAG(value, {periods}) OVER (ORDER BY ts) AS value
        FROM series_input
        ORDER BY ts
    """
    return _run_duckdb_series_query(df, query)


def _run_pct_change_duckdb(
    s: pd.Series,
    spec: PITTransformSpec,
    params: dict[str, Any],
) -> pd.Series:
    del spec
    df = _to_duckdb_frame(s)
    periods = int(params["periods"])
    query = f"""
        WITH lagged AS (
            SELECT
                ts,
                value,
                LAG(value, {periods}) OVER (ORDER BY ts) AS prev_value
            FROM series_input
        )
        SELECT
            ts,
            CASE
                WHEN prev_value IS NULL THEN NULL
                ELSE (value / prev_value) - 1
            END AS value
        FROM lagged
        ORDER BY ts
    """
    return _run_duckdb_series_query(df, query)


def _run_rolling_duckdb(
    s: pd.Series,
    spec: PITTransformSpec,
    params: dict[str, Any],
) -> pd.Series:
    del spec
    df = _to_duckdb_frame(s)
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
    return _run_duckdb_series_query(df, query)


def _run_expanding_duckdb(
    s: pd.Series,
    spec: PITTransformSpec,
    params: dict[str, Any],
) -> pd.Series:
    del spec
    df = _to_duckdb_frame(s)
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
    return _run_duckdb_series_query(df, query)


def _run_resample_duckdb(
    s: pd.Series,
    spec: PITTransformSpec,
    params: dict[str, Any],
) -> pd.Series:
    del spec
    df = _to_duckdb_frame(s)
    rule = str(params["rule"])
    bucket_expr = _bucket_end_expr(rule)
    agg_expr = _grouped_agg_expr(str(params["agg"]))
    query = f"""
        SELECT {bucket_expr} AS ts, {agg_expr} AS value
        FROM series_input
        GROUP BY 1
        ORDER BY 1
    """
    return _run_duckdb_series_query(df, query)


def _run_aggregate_duckdb(
    s: pd.Series,
    spec: PITTransformSpec,
    params: dict[str, Any],
) -> pd.Series:
    del spec
    df = _to_duckdb_frame(s)
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
    return _run_duckdb_series_query(df, query)


def _apply_series_runner(
    s: pd.Series,
    spec: PITTransformSpec,
    *,
    engine: RuntimeEngine,
) -> pd.Series:
    operator = _get_operator_def(spec.op)
    if operator.input_kind != "single":
        raise PITUnsupportedOperationError(
            f"{spec.op} transform requires multiple input snapshots and must be applied "
            "through PITAccessor.apply_transform."
        )

    params = spec.normalized_params(include_callable=(engine == "python"))
    if engine == "duckdb":
        runner = operator.duckdb_runner
        if runner is None:
            raise PITUnsupportedOperationError(
                f"DuckDB engine does not support PIT op='{spec.op}' for axis='{spec.axis}'."
            )
    else:
        runner = operator.python_runner
        if runner is None:
            raise PITUnsupportedOperationError(f"Unsupported PIT transform op: {spec.op}")

    transformed = runner(s, spec, params)
    out = pd.Series(transformed).sort_index()
    out.index = _coerce_utc_index(out.index)
    out = _as_numeric(out)
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
    return _apply_series_runner(s, spec, engine=engine)


def apply_binary_obs_path_transform(
    left_snapshot: pd.Series,
    right_snapshot: pd.Series,
    spec: PITTransformSpec,
) -> pd.Series:
    """Apply a binary transform over two obs_date-indexed PIT snapshots."""
    if spec.axis != "obs_path":
        raise PITContractError("apply_binary_obs_path_transform requires axis='obs_path'.")
    if spec.op != "binary":
        raise PITContractError("apply_binary_obs_path_transform requires op='binary'.")
    validate_transform_spec(spec)

    params = spec.normalized_params(include_callable=False)
    operator = str(params["operator"])
    join_mode = str(params["join"])
    fill_value = params.get("fill_value")

    left = _as_numeric(left_snapshot.copy().sort_index())
    right = _as_numeric(right_snapshot.copy().sort_index())
    left.index = _coerce_utc_index(left.index)
    right.index = _coerce_utc_index(right.index)

    left_aligned, right_aligned = left.align(right, join=join_mode)
    if fill_value is not None:
        left_aligned = left_aligned.fillna(float(fill_value))
        right_aligned = right_aligned.fillna(float(fill_value))

    if operator == "add":
        out = left_aligned + right_aligned
    elif operator == "sub":
        out = left_aligned - right_aligned
    elif operator == "mul":
        out = left_aligned * right_aligned
    elif operator == "div":
        out = left_aligned / right_aligned
        out = out.replace([float("inf"), float("-inf")], pd.NA)
    else:
        raise PITValidationError(
            f"Unsupported binary operator '{operator}'. Expected one of: add, sub, mul, div."
        )

    out = pd.Series(out).sort_index()
    out.index = _coerce_utc_index(out.index)
    out = _as_numeric(out)
    out.name = spec.output_series_key
    return out


def _aligned_multi_input_series(
    left_snapshot: pd.Series,
    right_snapshot: pd.Series,
    *,
    join_mode: JoinMode,
) -> tuple[pd.Series, pd.Series]:
    left = _as_numeric(left_snapshot.copy().sort_index())
    right = _as_numeric(right_snapshot.copy().sort_index())
    left.index = _coerce_utc_index(left.index)
    right.index = _coerce_utc_index(right.index)
    return left.align(right, join=join_mode)


def apply_splice_obs_path_transform(
    left_snapshot: pd.Series,
    right_snapshot: pd.Series,
    spec: PITTransformSpec,
) -> pd.DataFrame:
    """Apply an adjusted PIT splice over two obs_date-indexed snapshots."""
    if spec.axis != "obs_path":
        raise PITContractError("apply_splice_obs_path_transform requires axis='obs_path'.")
    if spec.op != "splice":
        raise PITContractError("apply_splice_obs_path_transform requires op='splice'.")
    validate_transform_spec(spec)

    params = spec.normalized_params(include_callable=False)
    join_mode = cast(JoinMode, params["join"])
    adjustment = cast(SpliceAdjustment, params["adjustment"])
    transition_periods = int(params["transition_periods"])
    right_series_key = str(params["right_series_key"])

    left_aligned, right_aligned = _aligned_multi_input_series(
        left_snapshot,
        right_snapshot,
        join_mode=join_mode,
    )

    left_non_null = left_aligned.dropna()
    right_non_null = right_aligned.dropna()
    if right_non_null.empty:
        out = pd.DataFrame(
            {
                "value": left_aligned,
                "selected_input_series_key": pd.Series(
                    [
                        spec.input_series_key if pd.notna(value) else pd.NA
                        for value in left_aligned.to_numpy()
                    ],
                    index=left_aligned.index,
                    dtype="object",
                ),
                "splice_state": pd.Series(
                    ["left" if pd.notna(value) else pd.NA for value in left_aligned.to_numpy()],
                    index=left_aligned.index,
                    dtype="object",
                ),
                "splice_anchor_obs_date": pd.NaT,
                "splice_anchor_left_value": pd.NA,
                "splice_anchor_right_value": pd.NA,
                "splice_scale": pd.NA,
                "splice_offset": pd.NA,
                "splice_left_weight": 1.0,
                "splice_right_weight": 0.0,
            }
        )
        out = out.dropna(subset=["value"])
        out.index = _coerce_utc_index(out.index)
        return out.sort_index()

    handoff_obs_date = pd.Timestamp(right_non_null.index.min())
    overlap_index = left_non_null.index.intersection(right_non_null.index).sort_values()

    anchor_obs_date: pd.Timestamp | None = None
    anchor_left_value: float | None = None
    anchor_right_value: float | None = None
    splice_scale: float | None = None
    splice_offset: float | None = None
    calibrated = False

    if len(overlap_index) > 0:
        anchor_obs_date = pd.Timestamp(overlap_index.max())
        anchor_left = left_aligned.loc[anchor_obs_date]
        anchor_right = right_aligned.loc[anchor_obs_date]
        if pd.notna(anchor_left) and pd.notna(anchor_right):
            anchor_left_value = float(anchor_left)
            anchor_right_value = float(anchor_right)
            if adjustment == "ratio":
                if anchor_right_value != 0.0:
                    splice_scale = anchor_left_value / anchor_right_value
                    calibrated = True
            else:
                splice_offset = anchor_left_value - anchor_right_value
                calibrated = True

    blended_obs_dates = [
        pd.Timestamp(obs_date)
        for obs_date in left_aligned.index
        if obs_date >= handoff_obs_date
    ][:transition_periods]
    transition_positions = {
        obs_date: pos for pos, obs_date in enumerate(blended_obs_dates)
    }

    values: list[float | None] = []
    selected_keys: list[str | None] = []
    states: list[str | None] = []
    left_weights: list[float] = []
    right_weights: list[float] = []

    for obs_date, left_value, right_value in zip(
        left_aligned.index,
        left_aligned.to_numpy(),
        right_aligned.to_numpy(),
        strict=False,
    ):
        obs_ts = pd.Timestamp(obs_date)

        if obs_ts < handoff_obs_date:
            values.append(float(left_value) if pd.notna(left_value) else None)
            selected_keys.append(spec.input_series_key if pd.notna(left_value) else None)
            states.append("left" if pd.notna(left_value) else None)
            left_weights.append(1.0)
            right_weights.append(0.0)
            continue

        if not calibrated:
            values.append(None)
            selected_keys.append(None)
            states.append("uncalibrated")
            left_weights.append(0.0)
            right_weights.append(0.0)
            continue

        adjusted_right_value: float | None
        if pd.isna(right_value):
            adjusted_right_value = None
        elif adjustment == "ratio":
            adjusted_right_value = float(right_value) * cast(float, splice_scale)
        else:
            adjusted_right_value = float(right_value) + cast(float, splice_offset)

        if transition_periods > 0 and obs_ts in transition_positions:
            pos = transition_positions[obs_ts]
            right_weight = float((pos + 1) / (transition_periods + 1))
            left_weight = float(1.0 - right_weight)
            if pd.notna(left_value) and adjusted_right_value is not None:
                value = (left_weight * float(left_value)) + (right_weight * adjusted_right_value)
                values.append(value)
            else:
                values.append(None)
            selected_keys.append(None)
            states.append("transition")
            left_weights.append(left_weight)
            right_weights.append(right_weight)
            continue

        values.append(adjusted_right_value)
        selected_keys.append(right_series_key if adjusted_right_value is not None else None)
        states.append("right" if adjusted_right_value is not None else None)
        left_weights.append(0.0)
        right_weights.append(1.0 if adjusted_right_value is not None else 0.0)

    out = pd.DataFrame(
        {
            "value": pd.Series(values, index=left_aligned.index, dtype="float64"),
            "selected_input_series_key": pd.Series(
                selected_keys,
                index=left_aligned.index,
                dtype="object",
            ),
            "splice_state": pd.Series(states, index=left_aligned.index, dtype="object"),
            "splice_anchor_obs_date": anchor_obs_date,
            "splice_anchor_left_value": anchor_left_value,
            "splice_anchor_right_value": anchor_right_value,
            "splice_scale": splice_scale,
            "splice_offset": splice_offset,
            "splice_left_weight": left_weights,
            "splice_right_weight": right_weights,
        }
    )
    out = out.dropna(subset=["value"])
    out.index = _coerce_utc_index(out.index)
    return out.sort_index()


def apply_coalesce_obs_path_transform(
    snapshots: Mapping[str, pd.Series],
    spec: PITTransformSpec,
) -> pd.DataFrame:
    """Apply a coalesce/splice transform over ordered PIT snapshots."""
    if spec.axis != "obs_path":
        raise PITContractError("apply_coalesce_obs_path_transform requires axis='obs_path'.")
    if spec.op != "coalesce":
        raise PITContractError("apply_coalesce_obs_path_transform requires op='coalesce'.")
    validate_transform_spec(spec)

    ordered_keys = transform_input_series_keys(spec)
    union_index = pd.DatetimeIndex([], tz="UTC")
    aligned_inputs: dict[str, pd.Series] = {}

    for key in ordered_keys:
        series = snapshots.get(key)
        if series is None or series.empty:
            prepared = pd.Series(dtype="float64")
        else:
            prepared = _as_numeric(series.copy().sort_index())
            prepared.index = _coerce_utc_index(prepared.index)
        aligned_inputs[key] = prepared
        if not prepared.empty:
            union_index = union_index.union(prepared.index)

    if len(union_index) == 0:
        return pd.DataFrame(
            {
                "value": pd.Series(dtype="float64"),
                "selected_input_series_key": pd.Series(dtype="object"),
            }
        )

    matrix = pd.DataFrame(
        {key: aligned_inputs[key].reindex(union_index) for key in ordered_keys},
        index=union_index,
    )
    any_value = matrix.notna().any(axis=1)
    selected = pd.Series(pd.NA, index=matrix.index, dtype="object")
    if any_value.any():
        selected.loc[any_value] = matrix.loc[any_value].notna().idxmax(axis=1)

    value = matrix.bfill(axis=1).iloc[:, 0]
    out = pd.DataFrame(
        {
            "value": _as_numeric(value),
            "selected_input_series_key": selected,
        },
        index=matrix.index,
    )
    out = out.dropna(subset=["value"])
    out.index = _coerce_utc_index(out.index)
    return out.sort_index()


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
    return _apply_series_runner(s, spec, engine=engine)


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


_ALLOWED_AGGS: set[str] = {
    "count",
    "first",
    "last",
    "max",
    "mean",
    "min",
    "std",
    "sum",
    "var",
}

_OPERATORS: dict[TransformOp, TransformOperatorDef] = {
    "resample": TransformOperatorDef(
        allowed_axes=("obs_path",),
        param_keys={"rule", "agg"},
        normalize=_normalize_resample_params,
        input_series_keys=_default_input_series_keys,
        duckdb_supports=lambda params: _is_duckdb_rule_supported(str(params["rule"])),
        python_runner=_run_resample_python,
        duckdb_runner=_run_resample_duckdb,
    ),
    "aggregate": TransformOperatorDef(
        allowed_axes=("obs_path",),
        param_keys={"rule", "agg"},
        normalize=_normalize_aggregate_params,
        input_series_keys=_default_input_series_keys,
        duckdb_supports=lambda params: (
            True if params.get("rule") is None else _is_duckdb_rule_supported(str(params["rule"]))
        ),
        python_runner=_run_aggregate_python,
        duckdb_runner=_run_aggregate_duckdb,
    ),
    "rolling": TransformOperatorDef(
        allowed_axes=("obs_path", "revision_path"),
        param_keys={"window", "min_periods", "agg"},
        normalize=_normalize_rolling_params,
        input_series_keys=_default_input_series_keys,
        duckdb_supports=lambda params: True,
        python_runner=_run_rolling_python,
        duckdb_runner=_run_rolling_duckdb,
    ),
    "expanding": TransformOperatorDef(
        allowed_axes=("obs_path", "revision_path"),
        param_keys={"min_periods", "agg"},
        normalize=_normalize_expanding_params,
        input_series_keys=_default_input_series_keys,
        duckdb_supports=lambda params: True,
        python_runner=_run_expanding_python,
        duckdb_runner=_run_expanding_duckdb,
    ),
    "lag": TransformOperatorDef(
        allowed_axes=("obs_path", "revision_path"),
        param_keys={"periods"},
        normalize=lambda params, include_callable: _normalize_periods_params(
            "lag", params, include_callable
        ),
        input_series_keys=_default_input_series_keys,
        duckdb_supports=lambda params: True,
        python_runner=_run_lag_python,
        duckdb_runner=_run_lag_duckdb,
    ),
    "diff": TransformOperatorDef(
        allowed_axes=("obs_path", "revision_path"),
        param_keys={"periods"},
        normalize=lambda params, include_callable: _normalize_periods_params(
            "diff", params, include_callable
        ),
        input_series_keys=_default_input_series_keys,
        duckdb_supports=lambda params: True,
        python_runner=_run_diff_python,
        duckdb_runner=_run_diff_duckdb,
    ),
    "pct_change": TransformOperatorDef(
        allowed_axes=("obs_path",),
        param_keys={"periods"},
        normalize=lambda params, include_callable: _normalize_periods_params(
            "pct_change", params, include_callable
        ),
        input_series_keys=_default_input_series_keys,
        duckdb_supports=lambda params: True,
        python_runner=_run_pct_change_python,
        duckdb_runner=_run_pct_change_duckdb,
    ),
    "ffill": TransformOperatorDef(
        allowed_axes=("obs_path",),
        param_keys={"limit"},
        normalize=_normalize_ffill_params,
        input_series_keys=_default_input_series_keys,
        duckdb_supports=lambda params: False,
        python_runner=_run_ffill_python,
    ),
    "binary": TransformOperatorDef(
        allowed_axes=("obs_path",),
        param_keys={"right_series_key", "operator", "join", "fill_value"},
        normalize=_normalize_binary_params,
        input_series_keys=_binary_input_series_keys,
        input_kind="multi",
        duckdb_supports=lambda params: False,
    ),
    "coalesce": TransformOperatorDef(
        allowed_axes=("obs_path",),
        param_keys={"other_series_keys"},
        normalize=_normalize_coalesce_params,
        input_series_keys=_coalesce_input_series_keys,
        input_kind="multi",
        duckdb_supports=lambda params: False,
    ),
    "splice": TransformOperatorDef(
        allowed_axes=("obs_path",),
        param_keys={"right_series_key", "adjustment", "transition_periods", "join"},
        normalize=_normalize_splice_params,
        input_series_keys=_splice_input_series_keys,
        input_kind="multi",
        duckdb_supports=lambda params: False,
    ),
    "path_apply": TransformOperatorDef(
        allowed_axes=("obs_path",),
        param_keys={"udf_name", "func"},
        normalize=_normalize_path_apply_params,
        input_series_keys=_default_input_series_keys,
        duckdb_supports=lambda params: False,
        python_runner=_run_path_apply_python,
    ),
}
