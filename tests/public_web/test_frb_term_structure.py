from __future__ import annotations

import pandas as pd

from alphaforge.data.public_web.frb_term_structure import FRBTermStructureBenchmarkSource
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def test_frb_term_structure_fetch() -> None:
    csv = (
        "Date,THREEFY0100.B,THREEFYTP0100.B,THREEFY1000.B,THREEFYTP1000.B\n"
        "22-02-2024,4.9738,0.0448,4.3965,0.3464\n"
        "23-02-2024,5.0000,0.0500,4.4200,0.3500\n"
    )
    src = FRBTermStructureBenchmarkSource(
        http_client=FakeHttpClient({"feds200533.csv": csv.encode()})
    )

    df = src.fetch(
        Query(
            table="frb.term_structure",
            columns=["value", "category", "maturity_years"],
            start=pd.Timestamp("2024-02-01", tz="UTC"),
            end=pd.Timestamp("2024-03-01", tz="UTC"),
        )
    )

    assert not df.empty
    assert {"date", "entity_id", "asof_utc", "value", "category", "maturity_years"}.issubset(
        df.columns
    )
    assert set(df["category"]) == {"yield", "yield_term_premium"}
