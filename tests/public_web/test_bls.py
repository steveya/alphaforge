from __future__ import annotations

from alphaforge.data.public_web.bls import BLSDataSource
from alphaforge.data.query import Query


def test_bls_source_fetch() -> None:
    def provider(series_ids, start_year, end_year):
        return {
            "Results": {
                "series": [
                    {
                        "seriesID": series_ids[0],
                        "data": [{"year": "2020", "period": "M01", "value": "258.678"}],
                    }
                ]
            }
        }

    src = BLSDataSource(response_provider=provider)
    df = src.fetch(
        Query(table="bls_series", columns=["value"], entities=["cuur0000sa0"])
    )
    assert not df.empty
    assert {"date", "entity_id", "asof_utc", "value"}.issubset(df.columns)
