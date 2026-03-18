from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alphaforge import TradingCalendar
from alphaforge.data.context import DataContext
from alphaforge.data.short_rates import (
    build_duan_weekly_dataset,
    build_kim_orphanides_dataset,
    build_macro_finance_dataset,
    build_policy_rule_dataset,
)
from alphaforge.store.local_parquet import LocalParquetStore


@dataclass
class DummySource:
    tables: dict[str, pd.DataFrame]

    def schemas(self):
        return {}

    def fetch(self, q):
        df = self.tables[q.table].copy()
        if q.entities is not None:
            if "series_id" in df.columns:
                df = df[df["series_id"].isin(q.entities)]
            elif "entity_id" in df.columns:
                df = df[df["entity_id"].isin(q.entities)]
        if q.start is not None:
            df = df[pd.to_datetime(df["date"], utc=True) >= q.start]
        if q.end is not None:
            df = df[pd.to_datetime(df["date"], utc=True) <= q.end]
        return df.reset_index(drop=True)


def _make_ctx(tmp_path) -> DataContext:
    fred_rows = []
    for date, dgs3mo, dgs6mo, dgs1, dgs2, dgs5, dgs10, cpi, indpro, fedfunds in [
        ("2020-01-03", 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 100.0, 100.0, 1.55),
        ("2020-01-31", 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 100.2, 100.2, 1.45),
        ("2020-02-28", 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 100.5, 99.8, 0.95),
        ("2020-03-27", 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 101.0, 99.0, 0.25),
        ("2020-04-24", 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 101.4, 98.8, 0.15),
        ("2020-05-29", 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 101.8, 99.2, 0.10),
        ("2021-01-29", 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 103.0, 101.5, 0.12),
    ]:
        values = {
            "DGS3MO": dgs3mo,
            "DGS6MO": dgs6mo,
            "DGS1": dgs1,
            "DGS2": dgs2,
            "DGS5": dgs5,
            "DGS10": dgs10,
            "CPIAUCSL": cpi,
            "INDPRO": indpro,
            "FEDFUNDS": fedfunds,
        }
        for series_id, value in values.items():
            fred_rows.append(
                {
                    "date": pd.Timestamp(date, tz="UTC"),
                    "series_id": series_id,
                    "value": value,
                    "asof_utc": pd.Timestamp(date, tz="UTC"),
                }
            )

    spf = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-02-14", tz="UTC"), pd.Timestamp("2020-05-15", tz="UTC")] * 2,
            "release_date": [pd.Timestamp("2020-02-14", tz="UTC"), pd.Timestamp("2020-05-15", tz="UTC")] * 2,
            "survey_period": ["2020Q1", "2020Q2", "2020Q1", "2020Q2"],
            "sheet_name": ["Mean Level"] * 4,
            "series_name": ["TBILL", "TBILL", "TBOND10Y", "TBOND10Y"],
            "value": [1.2, 0.9, 2.0, 1.7],
            "entity_id": [
                "spf.mean_level.tbill",
                "spf.mean_level.tbill",
                "spf.mean_level.tbond10y",
                "spf.mean_level.tbond10y",
            ],
            "asof_utc": [pd.Timestamp("2020-02-14", tz="UTC"), pd.Timestamp("2020-05-15", tz="UTC")] * 2,
        }
    )
    frb = pd.DataFrame(
        {
            "date": [pd.Timestamp("2020-01-31", tz="UTC"), pd.Timestamp("2020-02-29", tz="UTC")],
            "value": [0.3, 0.2],
            "category": ["yield_term_premium", "yield_term_premium"],
            "maturity_years": [10.0, 10.0],
            "entity_id": [
                "rates.us.frb.yield_term_premium.10y",
                "rates.us.frb.yield_term_premium.10y",
            ],
            "asof_utc": [pd.Timestamp("2020-01-31", tz="UTC"), pd.Timestamp("2020-02-29", tz="UTC")],
        }
    )

    return DataContext(
        sources={
            "fred": DummySource({"fred_series": pd.DataFrame(fred_rows)}),
            "philadelphia_spf": DummySource({"philadelphia.spf.mean_level": spf}),
            "frb_term_structure": DummySource({"frb.term_structure": frb}),
        },
        calendars={"XNYS": TradingCalendar("XNYS", tz="UTC")},
        store=LocalParquetStore(tmp_path / "store"),
    )


def test_build_kim_dataset(tmp_path) -> None:
    ctx = _make_ctx(tmp_path)
    dataset = build_kim_orphanides_dataset(
        ctx,
        start=pd.Timestamp("2020-01-01", tz="UTC"),
        end=pd.Timestamp("2021-01-31", tz="UTC"),
    )

    assert list(dataset.yields.columns) == [0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    assert set(dataset.surveys.columns) == {1.0, 10.0}
    assert "term_premium_10y" in dataset.benchmark.columns


def test_build_policy_rule_dataset(tmp_path) -> None:
    ctx = _make_ctx(tmp_path)
    dataset = build_policy_rule_dataset(
        ctx,
        start=pd.Timestamp("2020-01-01", tz="UTC"),
        end=pd.Timestamp("2021-01-31", tz="UTC"),
    )

    assert {"inflation", "activity", "policy_rate"} == set(dataset.macro.columns)
    assert dataset.short_rate.columns.tolist() == ["short_rate"]


def test_build_duan_weekly_dataset(tmp_path) -> None:
    ctx = _make_ctx(tmp_path)
    dataset = build_duan_weekly_dataset(
        ctx,
        start=pd.Timestamp("2020-01-01", tz="UTC"),
        end=pd.Timestamp("2020-05-31", tz="UTC"),
    )

    assert not dataset.yields.empty
    assert dataset.short_rate.columns.tolist() == ["short_rate"]


def test_build_macro_finance_dataset(tmp_path) -> None:
    ctx = _make_ctx(tmp_path)
    dataset = build_macro_finance_dataset(
        ctx,
        start=pd.Timestamp("2020-01-01", tz="UTC"),
        end=pd.Timestamp("2021-01-31", tz="UTC"),
    )

    assert {"inflation", "activity", "policy_rate"} == set(dataset.macro.columns)
    assert dataset.yields.index.equals(dataset.macro.index)
