"""Reusable short-rate research dataset builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from .context import DataContext
from .query import Query

DEFAULT_TREASURY_SERIES: dict[float, str] = {
    0.25: "DGS3MO",
    0.50: "DGS6MO",
    1.00: "DGS1",
    2.00: "DGS2",
    5.00: "DGS5",
    10.00: "DGS10",
}

DEFAULT_MACRO_SERIES: dict[str, str] = {
    "cpi": "CPIAUCSL",
    "industrial_production": "INDPRO",
    "policy_rate": "FEDFUNDS",
}


@dataclass
class ShortRateDataset:
    """Container for a constructed short-rate research dataset."""

    yields: pd.DataFrame
    surveys: pd.DataFrame | None = None
    macro: pd.DataFrame | None = None
    short_rate: pd.DataFrame | None = None
    benchmark: pd.DataFrame | None = None
    metadata: dict[str, object] = field(default_factory=dict)


def _normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    return out.sort_index()


def _coerce_utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _fetch_long(
    ctx: DataContext,
    *,
    source: str,
    table: str,
    columns: list[str],
    entities: list[str] | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if source not in ctx.sources:
        raise KeyError(f"Unknown source {source!r}. Available sources: {list(ctx.sources)}")
    return ctx.sources[source].fetch(
        Query(
            table=table,
            columns=columns,
            entities=entities,
            start=start,
            end=end,
        )
    )


def _wide_from_long(
    df: pd.DataFrame,
    *,
    time_col: str,
    entity_col: str,
    value_col: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(index=time_col, columns=entity_col, values=value_col, aggfunc="last")
    return _normalize_index(wide)


def _monthly_last(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_index(frame)
    return frame.resample("ME").last().dropna(how="all")


def _weekly_friday_last(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _normalize_index(frame)
    return frame.resample("W-FRI").last().dropna(how="all")


def _convert_rates_to_decimal(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.astype(float) / 100.0


def _fetch_fred_panel(
    ctx: DataContext,
    *,
    source: str,
    series_map: Mapping[object, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    sort_labels: bool = True,
) -> pd.DataFrame:
    df = _fetch_long(
        ctx,
        source=source,
        table="fred_series",
        columns=["value"],
        entities=list(series_map.values()),
        start=start,
        end=end,
    )
    wide = _wide_from_long(df, time_col="date", entity_col="series_id", value_col="value")
    rename_map = {series_id: column for column, series_id in series_map.items()}
    wide = wide.rename(columns=rename_map)
    if sort_labels:
        ordered = sorted(wide.columns, key=float)
        wide = wide.loc[:, ordered]
    return wide


def _select_spf_anchor_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "horizon_years", "value"])
    keys = (
        df["sheet_name"].astype(str).str.lower().fillna("")
        + " "
        + df["series_name"].astype(str).str.lower().fillna("")
        + " "
        + df["entity_id"].astype(str).str.lower().fillna("")
    )

    short_mask = keys.str.contains(r"tbill|bill")
    long_mask = keys.str.contains(r"tbond|treasury") & keys.str.contains(r"10|long")

    selected = []
    if short_mask.any():
        short = df.loc[short_mask, ["date", "value"]].copy()
        short["horizon_years"] = 1.0
        selected.append(short)
    if long_mask.any():
        long = df.loc[long_mask, ["date", "value"]].copy()
        long["horizon_years"] = 10.0
        selected.append(long)
    if not selected:
        return pd.DataFrame(columns=["date", "horizon_years", "value"])
    return pd.concat(selected, ignore_index=True)


def _build_spf_anchor_panel(
    ctx: DataContext,
    *,
    source: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    df = _fetch_long(
        ctx,
        source=source,
        table="philadelphia.spf.mean_level",
        columns=["value", "sheet_name", "series_name", "survey_period", "release_date"],
        entities=None,
        start=start,
        end=end,
    )
    selected = _select_spf_anchor_rows(df)
    if selected.empty:
        return pd.DataFrame(), {"selected_series": []}
    wide = selected.pivot_table(index="date", columns="horizon_years", values="value", aggfunc="last")
    wide = _monthly_last(wide).ffill().dropna(how="all")
    return _convert_rates_to_decimal(wide), {"selected_series": sorted(map(str, wide.columns))}


def _fetch_frb_benchmark_panel(
    ctx: DataContext,
    *,
    source: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    df = _fetch_long(
        ctx,
        source=source,
        table="frb.term_structure",
        columns=["value", "category", "maturity_years"],
        entities=None,
        start=start,
        end=end,
    )
    if df.empty:
        return pd.DataFrame()
    df = df[df["category"] == "yield_term_premium"].copy()
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(index="date", columns="maturity_years", values="value", aggfunc="last")
    wide = _monthly_last(wide)
    wide.columns = [f"term_premium_{float(col):g}y" for col in wide.columns]
    return _convert_rates_to_decimal(wide)


def _build_macro_panel(
    ctx: DataContext,
    *,
    source: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    macro_series: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    macro_series = DEFAULT_MACRO_SERIES if macro_series is None else dict(macro_series)
    raw = _fetch_fred_panel(
        ctx,
        source=source,
        series_map=macro_series,
        start=start - pd.DateOffset(months=18),
        end=end,
        sort_labels=False,
    )
    raw = _monthly_last(raw).sort_index()
    cpi = raw["cpi"].astype(float)
    ip = raw["industrial_production"].astype(float)
    policy = raw["policy_rate"].astype(float) / 100.0
    inflation = cpi.pct_change(12)
    activity = ip.pct_change(12)
    macro = pd.DataFrame(
        {
            "inflation": inflation,
            "activity": activity,
            "policy_rate": policy,
        }
    )
    start_ts = _coerce_utc_timestamp(start)
    end_ts = _coerce_utc_timestamp(end)
    return macro.loc[
        (macro.index >= start_ts)
        & (macro.index <= end_ts)
    ]


def build_kim_orphanides_dataset(
    ctx: DataContext,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fred_source: str = "fred",
    spf_source: str = "philadelphia_spf",
    frb_source: str = "frb_term_structure",
    yield_series: Mapping[float, str] | None = None,
) -> ShortRateDataset:
    yield_series = DEFAULT_TREASURY_SERIES if yield_series is None else dict(yield_series)
    raw_yields = _fetch_fred_panel(
        ctx,
        source=fred_source,
        series_map=yield_series,
        start=start,
        end=end,
    )
    yields = _convert_rates_to_decimal(_monthly_last(raw_yields))
    surveys, survey_meta = _build_spf_anchor_panel(
        ctx,
        source=spf_source,
        start=start,
        end=end,
    )
    benchmark = _fetch_frb_benchmark_panel(
        ctx,
        source=frb_source,
        start=start,
        end=end,
    )
    short_rate = yields[[min(yield_series)]].rename(columns={min(yield_series): "short_rate"})
    return ShortRateDataset(
        yields=yields,
        surveys=surveys,
        short_rate=short_rate,
        benchmark=benchmark,
        metadata={
            "maturities": list(yields.columns),
            "survey_horizons": list(surveys.columns) if not surveys.empty else [],
            **survey_meta,
        },
    )


def build_policy_rule_dataset(
    ctx: DataContext,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fred_source: str = "fred",
    yield_series: Mapping[float, str] | None = None,
    macro_series: Mapping[str, str] | None = None,
) -> ShortRateDataset:
    yield_series = DEFAULT_TREASURY_SERIES if yield_series is None else dict(yield_series)
    raw_yields = _fetch_fred_panel(
        ctx,
        source=fred_source,
        series_map=yield_series,
        start=start,
        end=end,
    )
    yields = _convert_rates_to_decimal(_monthly_last(raw_yields))
    macro = _build_macro_panel(
        ctx,
        source=fred_source,
        start=start,
        end=end,
        macro_series=macro_series,
    )
    aligned_index = yields.index.intersection(macro.index)
    return ShortRateDataset(
        yields=yields.loc[aligned_index],
        macro=macro.loc[aligned_index],
        short_rate=yields.loc[aligned_index, [min(yield_series)]].rename(
            columns={min(yield_series): "short_rate"}
        ),
        metadata={"maturities": list(yields.columns)},
    )


def build_duan_weekly_dataset(
    ctx: DataContext,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fred_source: str = "fred",
    yield_series: Mapping[float, str] | None = None,
) -> ShortRateDataset:
    yield_series = DEFAULT_TREASURY_SERIES if yield_series is None else dict(yield_series)
    raw_yields = _fetch_fred_panel(
        ctx,
        source=fred_source,
        series_map=yield_series,
        start=start,
        end=end,
    )
    yields = _convert_rates_to_decimal(_weekly_friday_last(raw_yields))
    short_maturity = min(yield_series)
    short_rate = yields[[short_maturity]].rename(columns={short_maturity: "short_rate"})
    return ShortRateDataset(
        yields=yields,
        short_rate=short_rate,
        metadata={"maturities": list(yields.columns), "grid": "W-FRI"},
    )


def build_macro_finance_dataset(
    ctx: DataContext,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fred_source: str = "fred",
    yield_series: Mapping[float, str] | None = None,
    macro_series: Mapping[str, str] | None = None,
) -> ShortRateDataset:
    yield_series = DEFAULT_TREASURY_SERIES if yield_series is None else dict(yield_series)
    raw_yields = _fetch_fred_panel(
        ctx,
        source=fred_source,
        series_map=yield_series,
        start=start,
        end=end,
    )
    yields = _convert_rates_to_decimal(_monthly_last(raw_yields))
    macro = _build_macro_panel(
        ctx,
        source=fred_source,
        start=start,
        end=end,
        macro_series=macro_series,
    )
    aligned_index = yields.index.intersection(macro.index)
    return ShortRateDataset(
        yields=yields.loc[aligned_index],
        macro=macro.loc[aligned_index],
        short_rate=yields.loc[aligned_index, [min(yield_series)]].rename(
            columns={min(yield_series): "short_rate"}
        ),
        metadata={"maturities": list(yields.columns)},
    )
