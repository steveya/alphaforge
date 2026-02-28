from __future__ import annotations

import json

from alphaforge.data.public_web.bea import BEADataSource
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def test_bea_source_fetch() -> None:
    payload = {
        "BEAAPI": {
            "Results": {"Data": [{"TimePeriod": "2020-01", "DataValue": "110.2"}]}
        }
    }
    http = FakeHttpClient({"api/data": json.dumps(payload).encode()})
    src = BEADataSource(
        api_key="k",
        http_client=http,
        registry_entries=[
            {
                "entity_id": "BEA:TEST",
                "params": {
                    "datasetname": "NIPA",
                    "TableName": "T",
                    "LineNumber": "1",
                    "Frequency": "M",
                    "Year": "ALL",
                },
            }
        ],
    )
    df = src.fetch(Query(table="bea_series", columns=["value"], entities=["BEA:TEST"]))
    assert len(df) == 1
