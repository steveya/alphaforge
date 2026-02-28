from __future__ import annotations

from pathlib import Path

from alphaforge.data.public_web.lch_cdsclear_daily import LCHCDSClearDailySource
from alphaforge.data.query import Query


def test_lch_cdsclear_daily_html_parser() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures/public_web/lch_cdsclear/sample.html"
    )
    source = LCHCDSClearDailySource(volumes_url=fixture.as_uri())

    df = source.fetch(
        Query(
            table="lch.cdsclear.daily",
            columns=["metric", "segment"],
        )
    )

    assert not df.empty
    assert {"date", "entity_id", "asof_utc", "value"}.issubset(df.columns)
    assert df["entity_id"].str.contains("lch").all()
