from __future__ import annotations

import io
import zipfile

import pandas as pd

from alphaforge.data.public_web.b3_historical_quotes import B3HistoricalQuotesDataSource
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def test_b3_quotes_source_fetch() -> None:
    html = '<a href="https://example.com/COTAHIST_A2020.zip">zip</a>'

    line = [" "] * 220
    line[0:2] = list("01")
    line[2:10] = list("20200102")
    line[12:24] = list("PETR4       ")
    line[56:69] = list(f"{1234500:013d}")
    line[69:82] = list(f"{1250000:013d}")
    line[82:95] = list(f"{1220000:013d}")
    line[108:121] = list(f"{1240000:013d}")
    line[170:188] = list(f"{1000000:018d}")
    txt = "".join(line)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("COTAHIST_A2020.TXT", txt)

    http = FakeHttpClient(
        {"landing.html": html.encode(), "COTAHIST_A2020.zip": buf.getvalue()}
    )
    src = B3HistoricalQuotesDataSource(
        http_client=http, page_url="https://example.com/page"
    )
    df = src.fetch(
        Query(
            table="b3_equity_quotes_daily",
            columns=["close"],
            start=pd.Timestamp("2020-01-01", tz="UTC"),
            end=pd.Timestamp("2020-12-31", tz="UTC"),
        )
    )
    assert not df.empty
    assert {"date", "ticker", "asof_utc", "close"}.issubset(df.columns)
