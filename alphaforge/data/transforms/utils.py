"""Shared PIT transform utilities."""

from __future__ import annotations

from typing import Sequence

import pandas as pd

_PIT_OUTPUT_COLUMNS = ["series_key", "obs_date", "asof_utc", "value", "source"]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN where denominator is zero."""
    return numerator / denominator.replace(0, float("nan"))


def melt_to_pit_format(
    df: pd.DataFrame,
    entity_col: str,
    obs_date_col: str,
    asof_col: str,
    value_vars: Sequence[str],
    key_prefix: str,
    source_name: str,
) -> pd.DataFrame:
    """Melt a wide DataFrame into PIT observation long format.

    Parameters
    ----------
    df : Wide DataFrame with one row per (entity, date).
    entity_col : Column containing the entity identifier.
    obs_date_col : Column containing the observation date.
    asof_col : Column containing the as-of / publication timestamp.
    value_vars : Metric columns to melt into long format.
    key_prefix : Prefix for series_key (e.g. "cftc.cot.tff.").
    source_name : Value for the 'source' column.

    Returns
    -------
    DataFrame with columns: series_key, obs_date, asof_utc, value, source
    """
    if df.empty:
        return pd.DataFrame(columns=_PIT_OUTPUT_COLUMNS)

    present_vars = [v for v in value_vars if v in df.columns]
    if not present_vars:
        return pd.DataFrame(columns=_PIT_OUTPUT_COLUMNS)

    id_vars = [entity_col, obs_date_col, asof_col]
    melted = df[id_vars + present_vars].melt(
        id_vars=id_vars,
        value_vars=present_vars,
        var_name="metric",
        value_name="value",
    )

    melted["series_key"] = key_prefix + melted[entity_col] + "." + melted["metric"]
    melted["source"] = source_name

    return melted.rename(
        columns={obs_date_col: "obs_date", asof_col: "asof_utc"}
    )[_PIT_OUTPUT_COLUMNS].reset_index(drop=True)
