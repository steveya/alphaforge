from __future__ import annotations

import json

from alphaforge.data.public_web.eia import EIADataSource
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def test_eia_source_fetch() -> None:
    payload = {"response": {"data": [{"period": "2020-01-01", "value": "61.2"}]}}
    http = FakeHttpClient({"petroleum/pri/spt": json.dumps(payload).encode()})
    src = EIADataSource(
        api_key="k",
        http_client=http,
        registry_entries=[
            {"entity_id": "EIA:TEST", "route": "petroleum/pri/spt", "params": {}}
        ],
    )
    df = src.fetch(Query(table="eia_series", columns=["value"], entities=["EIA:TEST"]))
    assert not df.empty
