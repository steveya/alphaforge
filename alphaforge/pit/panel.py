"""PIT panel building utilities.

Generic functions for assembling aligned panels from PIT snapshots.
Used by positioning (SignalContext) and nowcast-data (benchmark PIT builder).
"""
from __future__ import annotations

from typing import Sequence

import pandas as pd

from alphaforge.pit.accessor import PITAccessor


def build_pit_panel(
    pit: PITAccessor,
    series_keys: dict[str, str],
    asof: pd.Timestamp,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    align_freq: str | None = None,
) -> pd.DataFrame:
    """Build an aligned wide panel from PIT snapshots.

    Parameters
    ----------
    pit : PITAccessor
    series_keys : mapping from desired column name to PIT series_key
    asof : point-in-time cutoff
    start, end : date range
    align_freq : if set, reindex to this frequency (with ffill for slower series)

    Returns
    -------
    Wide DataFrame with index=obs_date, columns=column_name.
    """
    series: dict[str, pd.Series] = {}
    for col_name, skey in series_keys.items():
        snap = pit.get_snapshot(skey, asof, start=start, end=end)
        series[col_name] = snap

    if not series:
        return pd.DataFrame()

    df = pd.DataFrame(series)

    if align_freq is not None:
        new_idx = pd.date_range(
            start=df.index.min(), end=df.index.max(), freq=align_freq
        )
        df = df.reindex(new_idx).ffill()

    df.index.name = "obs_date"
    return df


def build_pit_panel_long(
    pit: PITAccessor,
    series_specs: Sequence[dict],
    asof: pd.Timestamp,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build a long-format panel (obs_date, entity_id, value columns).

    Parameters
    ----------
    series_specs : list of dicts with keys:
        - name: column alias
        - series_key: PIT series key
        - entity_id: (optional) entity identifier
    asof, start, end : PIT query params

    Returns
    -------
    Long DataFrame with columns: obs_date, entity_id, and one column per spec name.
    """
    rows: list[dict] = []

    for spec in series_specs:
        name = spec["name"]
        skey = spec["series_key"]
        entity_id = spec.get("entity_id", name)

        snap = pit.get_snapshot(skey, asof, start=start, end=end)
        for d, v in snap.items():
            rows.append({
                "obs_date": d,
                "entity_id": entity_id,
                name: v,
            })

    if not rows:
        return pd.DataFrame(columns=["obs_date", "entity_id"])

    return pd.DataFrame(rows)


def long_to_wide(
    df: pd.DataFrame,
    index_col: str = "obs_date",
    columns_col: str = "series_key",
    values_col: str = "value",
) -> pd.DataFrame:
    """Pivot long PIT DataFrame to wide format."""
    return df.pivot_table(
        index=index_col,
        columns=columns_col,
        values=values_col,
        aggfunc="first",
    )
