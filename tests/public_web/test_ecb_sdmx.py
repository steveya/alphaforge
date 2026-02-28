from __future__ import annotations

from alphaforge.data.public_web.ecb_sdmx import ECBSDMXDataSource
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def test_ecb_sdmx_source_fetch() -> None:
    csv = "TIME_PERIOD,OBS_VALUE\n2020-01,104.2\n"
    http = FakeHttpClient({"ICP": csv.encode()})
    src = ECBSDMXDataSource(
        http_client=http,
        registry_entries=[
            {
                "entity_id": "ECB:TEST",
                "flowRef": "ICP",
                "key": "M.U2.N.000000.4.INX",
                "params": {"format": "csvdata"},
            }
        ],
    )
    df = src.fetch(
        Query(table="ecb_sdmx_series", columns=["value"], entities=["ECB:TEST"])
    )
    assert len(df) == 1
