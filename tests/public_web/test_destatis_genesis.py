from __future__ import annotations

import json

from alphaforge.data.public_web.destatis_genesis import DestatisGenesisDataSource
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def test_destatis_source_fetch() -> None:
    payload = {"Object": {"Value": [{"time": "202001", "value": "106.8"}]}}
    http = FakeHttpClient({"61111-0001": json.dumps(payload).encode()})
    src = DestatisGenesisDataSource(
        api_key="k",
        http_client=http,
        registry_entries=[
            {
                "entity_id": "DESTATIS:TEST",
                "table_code": "61111-0001",
                "params": {"name": "data", "format": "json"},
            }
        ],
    )
    df = src.fetch(
        Query(table="destatis_series", columns=["value"], entities=["DESTATIS:TEST"])
    )
    assert len(df) == 1
