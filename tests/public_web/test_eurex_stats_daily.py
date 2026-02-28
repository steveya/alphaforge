from __future__ import annotations

from pathlib import Path

from alphaforge.data.public_web.eurex_stats_daily import EurexStatsDailySource
from alphaforge.data.query import Query


def test_eurex_stats_daily_html_parser() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures/public_web/eurex_market_stats/sample.html"
    )
    source = EurexStatsDailySource(stats_url=fixture.as_uri())

    df = source.fetch(
        Query(
            table="eurex.stats.daily",
            columns=["product_group", "product_name"],
        )
    )

    assert not df.empty
    assert {"date", "entity_id", "asof_utc", "volume", "open_interest"}.issubset(
        df.columns
    )
    assert df["entity_id"].str.startswith("eurex.").all()
