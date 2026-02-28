from __future__ import annotations

from alphaforge.data.public_web.anp_fuel_prices import ANPFuelPricesDataSource
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def test_anp_fuel_source_fetch() -> None:
    html = '<a href="https://example.com/anp.csv">csv</a>'
    csv = "DATA INICIAL,PRODUTO,ESTADO,PRECO MEDIO REVENDA\n2020-01-06,Gasolina Comum,SP,4.39\n"
    http = FakeHttpClient({"landing.html": html.encode(), "anp.csv": csv.encode()})
    src = ANPFuelPricesDataSource(
        http_client=http, landing_url="https://example.com/landing"
    )
    df = src.fetch(Query(table="anp_fuel_prices_weekly", columns=["value"]))
    assert not df.empty
