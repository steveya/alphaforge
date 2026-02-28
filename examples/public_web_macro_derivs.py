from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from alphaforge.data.context import DataContext
from alphaforge.data.public_web import (
    DTCCPPDSource,
    EurexStatsDailySource,
    EzoicAdRevenueDailySource,
)
from alphaforge.data.query import Query
from alphaforge.store.local_parquet import LocalParquetStore


def _dtcc_fixture_provider() -> bytes:
    repo_root = Path(__file__).resolve().parents[1]
    fixture = repo_root / "tests/fixtures/public_web/dtcc_ppd/sample_events.csv"
    csv_data = fixture.read_bytes()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("events.csv", csv_data)
    return buf.getvalue()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    dtcc_payload = _dtcc_fixture_provider()
    dtcc = DTCCPPDSource(
        list_provider=lambda report_type, asset_code: (
            [
                {
                    "fileName": "CFTC_CUMULATIVE_RATES_2026_01_02.zip",
                    "dissemDTM": "2026-01-03T00:00:00Z",
                }
            ]
            if report_type == "cumulative" and asset_code == "IR"
            else []
        ),
        artifact_provider=lambda file_name: dtcc_payload,
        source_mode="cumulative",
    )

    eurex = EurexStatsDailySource(
        stats_url=(
            repo_root / "tests/fixtures/public_web/eurex_market_stats/sample.html"
        ).as_uri()
    )
    ezoic = EzoicAdRevenueDailySource(
        data_url=(repo_root / "tests/fixtures/public_web/ezoic/sample.json").as_uri()
    )

    ctx = DataContext(
        sources={
            dtcc.name: dtcc,
            eurex.name: eurex,
            ezoic.name: ezoic,
        },
        calendars={},
        store=LocalParquetStore(str(repo_root / "alphaforge_demo_store/public_web")),
    )

    q_dtcc = Query(
        table="dtcc.ppd.daily",
        columns=["trade_count", "notional_sum", "price_std"],
        start=pd.Timestamp("2026-01-01", tz="UTC"),
        end=pd.Timestamp("2026-01-05", tz="UTC"),
    )
    q_eurex = Query(
        table="eurex.stats.daily",
        columns=["volume", "open_interest", "product_name"],
    )
    q_ezoic = Query(
        table="ezoic.adrevenue.daily",
        columns=["value", "region"],
        start=pd.Timestamp("2026-01-01", tz="UTC"),
        end=pd.Timestamp("2026-01-10", tz="UTC"),
    )

    dtcc_panel = ctx.fetch_panel(dtcc.name, q_dtcc)
    eurex_panel = ctx.fetch_panel(eurex.name, q_eurex)
    ezoic_panel = ctx.fetch_panel(ezoic.name, q_ezoic)

    print("\nDTCC daily head")
    print(dtcc_panel.df.head())

    print("\nEurex stats head")
    print(eurex_panel.df.head())

    print("\nEzoic ad revenue head")
    print(ezoic_panel.df.head())


if __name__ == "__main__":
    main()
