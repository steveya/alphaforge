from __future__ import annotations

import io

import pandas as pd

from alphaforge.data.public_web.philadelphia_spf import PhiladelphiaSPFMeanLevelSource
from alphaforge.data.query import Query

from ._fake_http import FakeHttpClient


def _sample_workbook_bytes() -> bytes:
    with io.BytesIO() as buffer:
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            pd.DataFrame(
                {
                    "YEAR": [2020, 2020],
                    "QUARTER": [1, 2],
                    "TBILL": [1.25, 1.10],
                    "TBOND10Y": [2.05, 1.90],
                }
            ).to_excel(writer, sheet_name="Mean Level", index=False)
        return buffer.getvalue()


def test_philadelphia_spf_source_fetch() -> None:
    release_text = "2020Q1 release date: 2020-02-14\n2020Q2 release date: 2020-05-15\n"
    src = PhiladelphiaSPFMeanLevelSource(
        http_client=FakeHttpClient(
            {
                "meanlevel.xlsx": _sample_workbook_bytes(),
                "spf-release-dates.txt": release_text.encode(),
            }
        )
    )

    df = src.fetch(
        Query(
            table="philadelphia.spf.mean_level",
            columns=["value", "series_name", "survey_period", "release_date"],
            start=pd.Timestamp("2020-01-01", tz="UTC"),
            end=pd.Timestamp("2020-12-31", tz="UTC"),
        )
    )

    assert not df.empty
    expected = {
        "date",
        "entity_id",
        "asof_utc",
        "series_name",
        "survey_period",
        "release_date",
        "value",
    }
    assert expected.issubset(df.columns)
    assert set(df["survey_period"]) == {"2020Q1", "2020Q2"}
