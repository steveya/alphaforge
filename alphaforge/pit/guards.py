from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from .exceptions import PITContractError


def _to_utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize("UTC")
    else:
        out = out.tz_convert("UTC")
    return out


@dataclass(frozen=True)
class ReleaseLagPolicy:
    """Policy for lag-adjusting effective asof timestamps per series."""

    default_lag: pd.Timedelta = pd.Timedelta(0)
    per_series_lag: Mapping[str, pd.Timedelta] = field(default_factory=dict)
    cutoff_hour_utc: int | None = None
    embargo_until_utc: Mapping[str, pd.Timestamp] = field(default_factory=dict)


def effective_asof(
    asof: pd.Timestamp,
    series_key: str,
    policy: ReleaseLagPolicy,
) -> pd.Timestamp:
    """Compute effective asof after lag/cutoff/embargo policy adjustments."""
    base = _to_utc(asof)
    lag = policy.per_series_lag.get(series_key, policy.default_lag)
    eff = base - lag

    if policy.cutoff_hour_utc is not None:
        if not 0 <= policy.cutoff_hour_utc <= 23:
            raise PITContractError("cutoff_hour_utc must be between 0 and 23.")
        if eff.hour < policy.cutoff_hour_utc:
            eff = eff.floor("D") - pd.Timedelta(seconds=1)

    embargo = policy.embargo_until_utc.get(series_key)
    if embargo is not None:
        embargo_ts = _to_utc(embargo)
        if eff > embargo_ts:
            eff = embargo_ts

    return eff


def pit_leakage_report(
    df: pd.DataFrame,
    ts_col: str = "obs_date",
    asof_col: str = "asof_utc",
) -> pd.DataFrame:
    """Structured leakage and PIT hygiene diagnostics for PIT-shaped data."""
    if ts_col not in df.columns or asof_col not in df.columns:
        raise PITContractError(f"Expected columns '{ts_col}' and '{asof_col}'.")

    out = df.copy()
    ts = pd.to_datetime(out[ts_col])
    asof = pd.to_datetime(out[asof_col])

    ts_tz_issues = int(ts.dt.tz is None)
    asof_tz_issues = int(asof.dt.tz is None)

    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")

    if asof.dt.tz is None:
        asof = asof.dt.tz_localize("UTC")
    else:
        asof = asof.dt.tz_convert("UTC")

    future_mask = ts > asof
    future_rows = int(future_mask.sum())

    duplicate_key_rows = 0
    if {"series_key", "obs_date", "asof_utc"}.issubset(out.columns):
        duplicate_key_rows = int(
            out.duplicated(subset=["series_key", "obs_date", "asof_utc"]).sum()
        )

    non_monotone_pairs = 0
    if {"series_key", "obs_date", "asof_utc"}.issubset(out.columns):
        grouped = out.copy()
        grouped["asof_utc"] = asof
        for _, group in grouped.groupby(["series_key", "obs_date"], dropna=False, sort=False):
            # Keep input order to catch non-monotone ingestion/timeline sequences.
            if not group["asof_utc"].is_monotonic_increasing:
                non_monotone_pairs += 1

    return pd.DataFrame(
        [
            {
                "rows": int(len(out)),
                "future_rows": future_rows,
                "future_pct": float(future_rows / len(out)) if len(out) else 0.0,
                "duplicate_key_rows": duplicate_key_rows,
                "non_monotone_timelines": non_monotone_pairs,
                "ts_tz_issues": ts_tz_issues,
                "asof_tz_issues": asof_tz_issues,
            }
        ]
    )
