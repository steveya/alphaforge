from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Sequence
import pandas as pd

# incoming column name for time is often 'date'; index level name will be 'ts_utc'
DATE = "date"
TS = "ts_utc"
ENTITY = "entity_id"


def _coerce_dt_aware(x: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(x))
    # If tz-naive, localize to UTC; otherwise convert to UTC
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx


def _coerce_ts_to_aware(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass
class PanelFrame:
    """Canonical panel: MultiIndex (ts_utc, entity_id). Dates stored tz-aware (UTC)."""

    df: pd.DataFrame

    @staticmethod
    def from_long(
        df: pd.DataFrame, time_col: str = DATE, entity_col: str = ENTITY
    ) -> "PanelFrame":
        if time_col not in df.columns or entity_col not in df.columns:
            raise ValueError(f"Expected columns {time_col} and {entity_col}.")
        out = df.copy()

        # normalize datetime to tz-aware UTC (keep tz-aware dtype)
        out[time_col] = _coerce_dt_aware(out[time_col])
        out[entity_col] = out[entity_col].astype(str)

        out = out.set_index([time_col, entity_col]).sort_index()
        # name time level 'ts_utc' to make intention explicit
        out.index.set_names([TS, ENTITY], inplace=True)
        return PanelFrame(out)

    def slice(
        self, start=None, end=None, entities: Optional[Sequence[str]] = None
    ) -> "PanelFrame":
        out = self.df

        # time level is tz-aware UTC
        d = pd.DatetimeIndex(out.index.get_level_values(TS))

        if start is not None:
            s = _coerce_ts_to_aware(start)
            out = out[d >= s]
            d = pd.DatetimeIndex(out.index.get_level_values(TS))  # update
        if end is not None:
            e = _coerce_ts_to_aware(end)
            out = out[d <= e]

        if entities is not None:
            ents = out.index.get_level_values(ENTITY)
            out = out[ents.isin([str(x) for x in entities])]

        return PanelFrame(out)

    def ensure_sorted(self) -> "PanelFrame":
        return PanelFrame(self.df.sort_index())
