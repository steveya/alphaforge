from __future__ import annotations

import json

from alphaforge.data.public_web.eurostat import EurostatDataSource
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def test_eurostat_source_fetch() -> None:
    payload = {
        "value": {"0": 105.1},
        "dimension": {"time": {"category": {"label": {"0": "2020-01"}}}},
    }
    http = FakeHttpClient({"prc_hicp_midx": json.dumps(payload).encode()})
    src = EurostatDataSource(
        http_client=http,
        registry_entries=[
            {"entity_id": "EUROSTAT:TEST", "dataset": "prc_hicp_midx", "filters": {}}
        ],
    )
    df = src.fetch(
        Query(table="eurostat_series", columns=["value"], entities=["EUROSTAT:TEST"])
    )
    assert len(df) == 1
