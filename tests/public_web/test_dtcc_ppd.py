from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from alphaforge.data.public_web.dtcc_ppd import DTCCPPDSource
from alphaforge.data.query import Query


def _zip_fixture_bytes() -> bytes:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures/public_web/dtcc_ppd/sample_events.csv"
    )
    csv_data = fixture.read_bytes()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("events_part1.csv", csv_data)
    return buf.getvalue()


def test_dtcc_events_and_daily_aggregation() -> None:
    payload = _zip_fixture_bytes()

    def list_provider(report_type: str, asset_code: str) -> list[dict]:
        if report_type != "slice" or asset_code != "IR":
            return []
        return [
            {
                "fileName": "CFTC_SLICE_RATES_2026_01_02_1.zip",
                "dissemDTM": "2026-01-02T15:37:19Z",
            }
        ]

    def provider(file_name: str) -> bytes:
        return payload

    source = DTCCPPDSource(
        list_provider=list_provider,
        artifact_provider=provider,
        source_mode="slice",
    )

    events = source.fetch(
        Query(
            table="dtcc.ppd.events",
            columns=["price", "notional", "product", "currency", "tenor"],
            start=pd.Timestamp("2026-01-02", tz="UTC"),
            end=pd.Timestamp("2026-01-02 23:59:59", tz="UTC"),
        )
    )

    assert not events.empty
    assert str(events["ts_utc"].dtype).startswith(("datetime64[ns,", "datetime64[us,"))
    assert events["entity_id"].str.contains("dtccppd").all()

    first_entity = events["entity_id"].iloc[0]
    daily = source.fetch(
        Query(
            table="dtcc.ppd.daily",
            columns=["notional_sum", "trade_count", "price_mean"],
            entities=[first_entity],
            start=pd.Timestamp("2026-01-02", tz="UTC"),
            end=pd.Timestamp("2026-01-03", tz="UTC"),
        )
    )

    assert not daily.empty
    assert daily["entity_id"].nunique() == 1
    assert "trade_count" in daily.columns
    assert int(daily["trade_count"].iloc[0]) >= 1
    assert "notional_sum" in daily.columns
    assert str(daily["asof_utc"].dtype).startswith(("datetime64[ns,", "datetime64[us,"))
