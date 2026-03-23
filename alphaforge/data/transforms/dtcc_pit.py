"""Convert DTCC PPD daily output to PIT observation format.

Moved from positioning to alphaforge — this transform is source-specific
(DTCC PPD CSV format) and needed by any consumer of DTCC data.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .utils import melt_to_pit_format

_DEFAULT_METRICS = (
    "trade_count",
    "notional_sum",
    "price_mean",
    "dv01_proxy_sum",
    "notional_median",
)


def dtcc_daily_to_pit_observations(
    df: pd.DataFrame,
    metrics: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Convert DTCC PPD daily fetch output to PIT observation rows.

    Parameters
    ----------
    df : DataFrame from DTCCPPDSource.fetch(table="dtcc.ppd.daily") with
        columns: date, entity_id, asof_utc, trade_count, notional_sum, etc.
    metrics : which series to emit.  Defaults to all five.

    Returns
    -------
    DataFrame with columns: series_key, obs_date, asof_utc, value, source
    """
    if df.empty:
        return pd.DataFrame(
            columns=["series_key", "obs_date", "asof_utc", "value", "source"]
        )

    chosen = list(metrics) if metrics is not None else list(_DEFAULT_METRICS)

    work = df.copy()
    work["obs_date"] = work["date"]

    return melt_to_pit_format(
        df=work,
        entity_col="entity_id",
        obs_date_col="obs_date",
        asof_col="asof_utc",
        value_vars=chosen,
        key_prefix="dtcc.ppd.daily.",
        source_name="dtcc_ppd",
    )
