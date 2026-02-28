from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data.public_web.cftc_swaps_weekly import CFTCWeeklySwapsSource
from alphaforge.data.query import Query


def test_cftc_swaps_weekly_csv_parse_and_filters() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures/public_web/cftc_swaps/sample.csv"
    )
    source = CFTCWeeklySwapsSource(file_urls=[fixture.as_uri()])

    df = source.fetch(
        Query(
            table="cftc.swaps.weekly",
            columns=["metric", "currency"],
            start=pd.Timestamp("2026-01-01", tz="UTC"),
            end=pd.Timestamp("2026-01-05", tz="UTC"),
        )
    )

    assert not df.empty
    assert {"date", "entity_id", "asof_utc", "value"}.issubset(df.columns)
    assert str(df["date"].dtype).startswith("datetime64[ns,")
    assert (df["date"] <= pd.Timestamp("2026-01-05", tz="UTC")).all()
