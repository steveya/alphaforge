"""Build the reduced Kim-Orphanides / Kim-Wright dataset with legacy loaders."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from alphaforge import (
    DataContext,
    DuckDBParquetStore,
    FRBTermStructureBenchmarkSource,
    FREDDataSource,
    PhiladelphiaSPFMeanLevelSource,
    TradingCalendar,
    build_kim_orphanides_dataset,
)


def main() -> None:
    fred_api_key = os.environ.get("FRED_API_KEY")
    if not fred_api_key:
        raise RuntimeError("Set FRED_API_KEY before running this example.")

    # This dataset helper still consumes the legacy/raw-loader DataSource shape.
    ctx = DataContext(
        sources={
            "fred": FREDDataSource(api_key=fred_api_key),
            "philadelphia_spf": PhiladelphiaSPFMeanLevelSource(),
            "frb_term_structure": FRBTermStructureBenchmarkSource(),
        },
        calendars={"XNYS": TradingCalendar("XNYS", tz="UTC")},
        store=DuckDBParquetStore(root=Path("./alphaforge_short_rate_store")),
    )

    dataset = build_kim_orphanides_dataset(
        ctx,
        start=pd.Timestamp("1990-01-01", tz="UTC"),
        end=pd.Timestamp("2024-12-31", tz="UTC"),
    )
    print("yields", dataset.yields.tail())
    print("surveys", None if dataset.surveys is None else dataset.surveys.tail())
    print("benchmark", None if dataset.benchmark is None else dataset.benchmark.tail())


if __name__ == "__main__":
    main()
