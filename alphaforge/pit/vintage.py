"""Vintage selection and lookahead validation utilities.

Moved from nowcast-data to alphaforge as generic PIT infrastructure.
"""
from __future__ import annotations

from datetime import date, datetime

import pandas as pd


def select_vintage_for_asof(
    vintages: list[date | datetime | pd.Timestamp],
    asof_date: date | datetime | pd.Timestamp,
) -> date | datetime | pd.Timestamp | None:
    """Select the latest vintage not after asof_date.

    Returns None if no vintage is available before asof_date.
    """
    if not vintages:
        return None

    norm_vintages = [_normalize_date(v) for v in vintages]
    norm_asof = _normalize_date(asof_date)

    sorted_vintages = sorted(zip(norm_vintages, vintages))

    selected_original = None
    for norm_v, orig_v in sorted_vintages:
        if norm_v <= norm_asof:
            selected_original = orig_v
        else:
            break

    return selected_original


def validate_no_lookahead(
    vintage_date: date | datetime | pd.Timestamp,
    asof_date: date | datetime | pd.Timestamp,
) -> bool:
    """Return True if vintage_date <= asof_date (no lookahead bias)."""
    return _normalize_date(vintage_date) <= _normalize_date(asof_date)


def _normalize_date(d: date | datetime | pd.Timestamp) -> pd.Timestamp:
    """Normalize date to pd.Timestamp for comparison."""
    if isinstance(d, pd.Timestamp):
        if d.tzinfo is not None:
            return d.tz_localize(None)
        return d
    return pd.Timestamp(d)
