from __future__ import annotations

from pathlib import Path

from alphaforge.data.public_web.cme_productslate_reference import CMEProductSlateSource
from alphaforge.data.query import Query


def test_cme_productslate_fetch_with_projection_and_entity_filter() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures/public_web/cme_productslate/sample.csv"
    )

    source = CMEProductSlateSource(csv_url=fixture.as_uri())

    all_rows = source.fetch(
        Query(
            table="cme.productslate.reference",
            columns=["product_name", "globex_symbol"],
        )
    )

    assert not all_rows.empty
    assert {"date", "entity_id", "asof_utc", "product_name", "globex_symbol"}.issubset(
        set(all_rows.columns)
    )
    assert all_rows["date"].dt.tz is not None

    entity = all_rows.loc[
        all_rows["product_name"].str.contains("2-Year"), "entity_id"
    ].iloc[0]
    filtered = source.fetch(
        Query(
            table="cme.productslate.reference",
            columns=["product_name"],
            entities=[entity],
        )
    )

    assert filtered["entity_id"].nunique() == 1
    assert filtered["entity_id"].iloc[0] == entity
    assert "product_name" in filtered.columns
