"""Convert CFTC CoT output to PIT observation format.

Moved from positioning to alphaforge — this transform is source-specific
(CFTC CSV format) and needed by any consumer of CoT data.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .utils import melt_to_pit_format, safe_divide

_DEFAULT_METRICS = (
    "long_positions",
    "short_positions",
    "net_positions",
    "net_pct_oi",
    "open_interest",
)


def cot_to_pit_observations(
    df: pd.DataFrame,
    metrics: Sequence[str] | None = None,
    *,
    key_prefix: str = "cftc.cot.tff.",
    source_name: str = "cftc_cot",
) -> pd.DataFrame:
    """Convert CFTC CoT fetch output to PIT observation rows.

    Parameters
    ----------
    df : DataFrame from a CFTC CoT source fetch() with columns:
        date (publication date, Friday), entity_id, long_positions,
        short_positions, open_interest.
    metrics : which series to emit.  Defaults to all five.
    key_prefix : Prefix for generated ``series_key`` values.
    source_name : Value for the output ``source`` column.

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

    # publication_date (Friday) is in the 'date' column.
    # Report date (Tuesday) = publication_date - 3 business days.
    work["asof_utc"] = work["date"]
    work["obs_date"] = work["date"].map(
        lambda d: d - pd.tseries.offsets.BDay(3) if pd.notna(d) else pd.NaT
    )

    # Compute derived metrics
    if "net_positions" in chosen or "net_pct_oi" in chosen:
        work["net_positions"] = work["long_positions"] - work["short_positions"]
    if "net_pct_oi" in chosen:
        work["net_pct_oi"] = safe_divide(work["net_positions"], work["open_interest"])

    return melt_to_pit_format(
        df=work,
        entity_col="entity_id",
        obs_date_col="obs_date",
        asof_col="asof_utc",
        value_vars=chosen,
        key_prefix=key_prefix,
        source_name=source_name,
    )
