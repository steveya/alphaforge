from __future__ import annotations

from alphaforge.data.public_web.bcb_sgs import BCBSGSDataSource
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def test_bcb_sgs_source_fetch() -> None:
    csv = "data,valor\n01/01/2020,4.50\n"
    http = FakeHttpClient({"bcdata.sgs.432": csv.encode()})
    src = BCBSGSDataSource(http_client=http)
    df = src.fetch(Query(table="bcb_sgs_series", columns=["value"], entities=["432"]))
    assert len(df) == 1
