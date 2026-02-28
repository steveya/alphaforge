from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, cast

import pandas as pd

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
_ALLOWED_AGGS = {"first", "last", "min", "max", "mean", "sum"}


@dataclass(frozen=True)
class PITTransformSpec:
    input_series_key: str
    output_series_key: str
    axis: TransformAxis = "obs_path"
    op: TransformOp = "resample"
    params: dict[str, Any] = field(default_factory=dict)
    engine: TransformEngine = "auto"

    def sanitized_params(self) -> dict[str, Any]:
        """Params suitable for hashing/lineage serialization."""
        out: dict[str, Any] = {}
        for key, value in self.params.items():
            if callable(value):
                continue
            if isinstance(value, pd.Timestamp):
                out[key] = value.isoformat()
            elif isinstance(value, pd.Timedelta):
                out[key] = str(value)
            else:
                out[key] = value
        return out

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


def _supported_combinations_text() -> str:
    items = []
    for axis in ("obs_path", "revision_path"):
        ops = ", ".join(_ALLOWED_AXIS_OPS[axis])
        items.append(f"{axis}: {ops}")
    return "; ".join(items)


def validate_transform_spec(spec: PITTransformSpec) -> None:
    if not spec.input_series_key:
        raise ValueError("input_series_key is required.")
    if not spec.output_series_key:
        raise ValueError("output_series_key is required.")

    allowed = _ALLOWED_AXIS_OPS[spec.axis]
    if spec.op not in allowed:
        combos = _supported_combinations_text()
        raise ValueError(
            "Unsupported op/axis combination: "
            f"axis='{spec.axis}', op='{spec.op}'. "
            f"Supported combinations -> {combos}"
        )

    if spec.op in {"resample", "aggregate"}:
        agg = str(spec.params.get("agg", "last")).lower()
        if agg not in _ALLOWED_AGGS:
            allowed_agg = ", ".join(sorted(_ALLOWED_AGGS))
            raise ValueError(f"Unsupported aggregation '{agg}'. Allowed: {allowed_agg}")

    if spec.op == "resample":
        rule = str(spec.params.get("rule", "")).strip()
        if not rule:
            raise ValueError("resample requires params['rule'].")

    if spec.op == "rolling":
        window = int(spec.params.get("window", 1))
        min_periods = int(spec.params.get("min_periods", window))
        if window <= 0:
            raise ValueError("rolling requires params['window'] > 0.")
        if min_periods <= 0:
            raise ValueError("rolling requires params['min_periods'] > 0.")

    if spec.op == "expanding":
        min_periods = int(spec.params.get("min_periods", 1))
        if min_periods <= 0:
            raise ValueError("expanding requires params['min_periods'] > 0.")

    if spec.op in {"lag", "diff"}:
        periods = int(spec.params.get("periods", 1))
        if periods <= 0:
            raise ValueError(f"{spec.op} requires params['periods'] > 0.")

    if spec.op == "path_apply":
        udf_name = str(spec.params.get("udf_name", "")).strip()
        func = spec.params.get("func")
        if not udf_name:
            raise ValueError("path_apply requires params['udf_name'] for lineage.")
        if not callable(func):
            raise ValueError("path_apply requires params['func'] callable.")


def resolve_engine(spec: PITTransformSpec) -> Literal["duckdb", "python"]:
    if spec.engine == "duckdb":
        return "duckdb"
    if spec.engine == "python":
        return "python"
    # auto
    return "python" if spec.op == "path_apply" else "duckdb"


def _coerce_utc_index(idx: pd.Index) -> pd.DatetimeIndex:
    out = pd.DatetimeIndex(pd.to_datetime(idx))
    if out.tz is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out


def _as_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_resample_rule(rule: str) -> str:
    aliases = {
        "M": "ME",
        "Q": "QE",
        "A": "YE",
        "Y": "YE",
    }
    return aliases.get(rule.upper(), rule)


def _apply_named_agg(obj: Any, agg: str) -> pd.Series:
    if not hasattr(obj, agg):
        raise ValueError(f"Unsupported aggregation: {agg}")
    out = getattr(obj, agg)()
    if not isinstance(out, pd.Series):
        raise TypeError("Aggregation did not return a pandas Series.")
    return out


def _aggregate_scalar(series: pd.Series, agg: str) -> float:
    if not hasattr(series, agg):
        raise ValueError(f"Unsupported aggregation: {agg}")
    value = getattr(series, agg)()
    return float(value) if pd.notna(value) else float("nan")


def _apply_series_op(s: pd.Series, spec: PITTransformSpec) -> pd.Series:
    op = spec.op
    params = spec.params

    if op == "resample":
        rule = _normalize_resample_rule(str(params.get("rule", "")))
        agg = str(params.get("agg", "last")).lower()
        transformed = _apply_named_agg(s.resample(rule), agg)

    elif op == "aggregate":
        agg = str(params.get("agg", "last")).lower()
        aggregate_rule = params.get("rule")
        if aggregate_rule is not None:
            transformed = _apply_named_agg(
                s.resample(_normalize_resample_rule(str(aggregate_rule))),
                agg,
            )
        else:
            if s.empty:
                transformed = pd.Series(dtype="float64")
            else:
                transformed = pd.Series(
                    [_aggregate_scalar(s, agg)],
                    index=pd.DatetimeIndex([s.index.max()]),
                )

    elif op == "rolling":
        window = int(params.get("window", 1))
        min_periods = int(params.get("min_periods", window))
        agg = str(params.get("agg", "mean")).lower()
        transformed = _apply_named_agg(
            s.rolling(window=window, min_periods=min_periods),
            agg,
        )

    elif op == "expanding":
        min_periods = int(params.get("min_periods", 1))
        agg = str(params.get("agg", "mean")).lower()
        transformed = _apply_named_agg(
            s.expanding(min_periods=min_periods),
            agg,
        )

    elif op == "lag":
        periods = int(params.get("periods", 1))
        transformed = s.shift(periods=periods)

    elif op == "diff":
        periods = int(params.get("periods", 1))
        transformed = s.diff(periods=periods)

    elif op == "path_apply":
        func_value = params.get("func")
        if not callable(func_value):
            raise ValueError("path_apply requires params['func'] callable.")
        func = cast(Callable[[pd.Series], Any], func_value)
        applied = func(s.copy())
        if isinstance(applied, pd.Series):
            transformed = applied
        else:
            transformed = pd.Series(applied, index=s.index)

    else:
        raise ValueError(f"Unsupported PIT transform op: {op}")

    out = pd.Series(transformed).sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = _coerce_utc_index(out.index)
    else:
        out.index = _coerce_utc_index(out.index)

    out = _as_numeric(out)
    out.name = spec.output_series_key
    return out


def apply_obs_path_transform(
    snapshot: pd.Series,
    spec: PITTransformSpec,
) -> pd.Series:
    """Apply a PIT transform over an obs_date-indexed snapshot path."""
    if spec.axis != "obs_path":
        raise ValueError("apply_obs_path_transform requires axis='obs_path'.")
    validate_transform_spec(spec)

    s = snapshot.copy().sort_index()
    s = _as_numeric(s)
    s.index = _coerce_utc_index(s.index)
    return _apply_series_op(s, spec)


def apply_revision_path_transform(
    timeline: pd.Series,
    spec: PITTransformSpec,
) -> pd.Series:
    """Apply a PIT transform over an asof_utc-indexed revision timeline."""
    if spec.axis != "revision_path":
        raise ValueError("apply_revision_path_transform requires axis='revision_path'.")
    validate_transform_spec(spec)

    s = timeline.copy().sort_index()
    s = _as_numeric(s)
    s.index = _coerce_utc_index(s.index)
    return _apply_series_op(s, spec)


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
        raise ValueError("Provided path function is not callable.")
    return path_func
