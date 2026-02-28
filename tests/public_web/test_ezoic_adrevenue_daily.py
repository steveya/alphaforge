from __future__ import annotations

from pathlib import Path

import pandas as pd

from alphaforge.data.public_web.ezoic_adrevenue_daily import EzoicAdRevenueDailySource
from alphaforge.data.query import Query


def test_ezoic_adrevenue_daily_json_parser_and_entity_filter() -> None:
    fixture = (
        Path(__file__).resolve().parents[1] / "fixtures/public_web/ezoic/sample.json"
    )
    source = EzoicAdRevenueDailySource(data_url=fixture.as_uri())

    all_rows = source.fetch(
        Query(
            table="ezoic.adrevenue.daily",
            columns=["region"],
            start=pd.Timestamp("2026-01-01", tz="UTC"),
            end=pd.Timestamp("2026-01-01", tz="UTC"),
        )
    )

    assert not all_rows.empty
    assert {"date", "entity_id", "asof_utc", "value"}.issubset(all_rows.columns)

    entity = all_rows["entity_id"].iloc[0]
    filtered = source.fetch(
        Query(
            table="ezoic.adrevenue.daily",
            columns=["value"],
            entities=[entity],
        )
    )
    assert filtered["entity_id"].nunique() == 1
