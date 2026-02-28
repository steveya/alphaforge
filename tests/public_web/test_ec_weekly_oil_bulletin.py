from __future__ import annotations

import pandas as pd

from alphaforge.data.public_web.ec_weekly_oil_bulletin import (
    ECWeeklyOilBulletinDataSource,
)
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def test_ec_oil_bulletin_source_fetch(monkeypatch) -> None:
    html = '<a href="https://example.com/with_tax.xlsx">with</a>'
    http = FakeHttpClient({"bulletin.html": html.encode(), "with_tax.xlsx": b"dummy"})
    src = ECWeeklyOilBulletinDataSource(
        http_client=http, bulletin_url="https://example.com/bulletin"
    )

    import alphaforge.data.public_web.ec_weekly_oil_bulletin as mod

    monkeypatch.setattr(
        mod,
        "parse_xlsx_bytes",
        lambda data: pd.DataFrame(
            {
                "date": ["2020-01-06"],
                "product": ["Eurosuper95"],
                "country": ["DE"],
                "price": [1.23],
            }
        ),
    )

    df = src.fetch(Query(table="ec_oil_bulletin_weekly", columns=["value"]))
    assert not df.empty
