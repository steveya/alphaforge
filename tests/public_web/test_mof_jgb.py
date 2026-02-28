from __future__ import annotations

import pandas as pd

from alphaforge.data.public_web.mof_jgb import MOFJGBYieldCurveSource
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def test_mof_jgb_source_fetch() -> None:
    csv = (
        "Date,1Y,2Y,5Y,10Y\n"
        "2020-01-06,-0.10,-0.05,0.01,0.08\n"
        "2020-01-07,-0.09,-0.04,0.02,0.09\n"
    )
    http = FakeHttpClient({"jgbcme.csv": csv.encode()})
    src = MOFJGBYieldCurveSource(http_client=http)

    df = src.fetch(
        Query(
            table="mof.jgb.yields",
            columns=["yield_pct"],
            start=pd.Timestamp("2020-01-01", tz="UTC"),
            end=pd.Timestamp("2020-12-31", tz="UTC"),
        )
    )

    assert not df.empty
    assert {"date", "entity_id", "asof_utc", "yield_pct"}.issubset(df.columns)
