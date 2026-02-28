from __future__ import annotations

import json

from alphaforge.data.public_web.ibge_sidra import IBGESidraDataSource
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def test_ibge_sidra_source_fetch() -> None:
    payload = [{"D3C": "202001", "V": "0.21"}]
    http = FakeHttpClient({"sidra": json.dumps(payload).encode()})
    src = IBGESidraDataSource(
        http_client=http,
        registry_entries=[
            {"entity_id": "IBGE:TEST", "url": "https://example.com/sidra"}
        ],
    )
    df = src.fetch(
        Query(table="ibge_sidra_series", columns=["value"], entities=["IBGE:TEST"])
    )
    assert len(df) == 1
