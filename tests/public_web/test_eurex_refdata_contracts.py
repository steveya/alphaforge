from __future__ import annotations

from pathlib import Path

from alphaforge.data.public_web.eurex_refdata_contracts import (
    EurexRefdataContractsSource,
)
from alphaforge.data.query import Query


def test_eurex_refdata_contracts_json_parser() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures/public_web/eurex_refdata/sample.json"
    )
    source = EurexRefdataContractsSource(api_url=fixture.as_uri())

    df = source.fetch(
        Query(
            table="eurex.refdata.contracts",
            columns=["product_name", "currency", "expiry_date"],
        )
    )

    assert not df.empty
    assert {"date", "entity_id", "asof_utc", "symbol", "expiry_date"}.issubset(
        df.columns
    )
    assert df["entity_id"].str.endswith(".eurex").all()
