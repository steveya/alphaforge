"""Consolidated timestamp and date conversion utilities.

This module provides canonical implementations of UTC timestamp coercion
and date normalization for PIT data operations.
"""

from __future__ import annotations

from datetime import date

import pandas as pd


def coerce_utc_timestamp(value: date | pd.Timestamp | str) -> pd.Timestamp:
    """Coerce a date-like value to a UTC-aware pandas Timestamp.

    Accepts date, pd.Timestamp (tz-naive or tz-aware), or parseable string.
    """
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def normalize_utc_day(value: date | pd.Timestamp) -> date:
    """Normalize a date-like value to a UTC midnight date object."""
    return coerce_utc_timestamp(value).normalize().date()


def to_utc_naive(
    values: pd.Series | pd.DatetimeIndex | object,
) -> pd.Series | pd.DatetimeIndex | pd.Timestamp:
    """Convert datetime-like values to UTC-naive timestamps.

    Handles pd.Series, pd.DatetimeIndex, and scalar values.
    """
    if isinstance(values, pd.Series) and pd.api.types.is_datetime64_any_dtype(values):
        parsed = values
    elif isinstance(values, pd.DatetimeIndex):
        parsed = values
    else:
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if isinstance(values, pd.Series):
        if parsed.dt.tz is None:
            parsed = parsed.dt.tz_localize("UTC")
        else:
            parsed = parsed.dt.tz_convert("UTC")
        return parsed.dt.tz_localize(None)
    if isinstance(parsed, pd.DatetimeIndex):
        if parsed.tz is None:
            parsed = parsed.tz_localize("UTC")
        else:
            parsed = parsed.tz_convert("UTC")
        return parsed.tz_localize(None)
    if pd.isna(parsed):
        return parsed
    return parsed.tz_convert("UTC").tz_localize(None)


def to_month_end(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a timestamp to the UTC-aware end of its month."""
    ts = coerce_utc_timestamp(ts)
    naive = ts.tz_localize(None)
    month_end = naive.to_period("M").end_time.normalize()
    return month_end.tz_localize("UTC")


def to_quarter_end_ts(ts: pd.Timestamp) -> pd.Timestamp:
    """Convert a timestamp to the UTC-aware end of its quarter."""
    ts = coerce_utc_timestamp(ts)
    naive = ts.tz_localize(None)
    quarter_end = naive.to_period("Q").end_time.normalize()
    return quarter_end.tz_localize("UTC")
