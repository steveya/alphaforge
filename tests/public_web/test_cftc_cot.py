from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.data.public_web.cftc_cot import (
    CFTCCoTSource,
    CFTCDisaggregatedCoTSource,
    _publication_date,
)
from alphaforge.data.query import Query

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures/public_web/cftc_cot"


def _csv_to_zip_bytes(csv_path: Path) -> bytes:
    """Wrap a CSV fixture into a ZIP archive in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write(csv_path, arcname=csv_path.name)
    return buf.getvalue()


def _csv_text_to_zip_bytes(filename: str, csv_text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, csv_text)
    return buf.getvalue()


class _MockHttpClient:
    """Return pre-built ZIP bytes for any URL request."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def get_bytes(self, *, url: str, source: str, artifact_name: str, **kw) -> bytes:
        return self._payload


class _SequenceHttpClient:
    """Return a per-URL payload so tests can model partial archive failures."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self._payloads = payloads

    def get_bytes(self, *, url: str, source: str, artifact_name: str, **kw) -> bytes:
        return self._payloads[url]


def _make_tff_source() -> CFTCCoTSource:
    csv_path = _FIXTURE_DIR / "sample.csv"
    zip_bytes = _csv_to_zip_bytes(csv_path)
    http = _MockHttpClient(zip_bytes)
    return CFTCCoTSource(
        http_client=http,
        file_urls=["file:///dummy.zip"],
    )


def _make_disagg_source() -> CFTCDisaggregatedCoTSource:
    csv_path = _FIXTURE_DIR / "sample_disagg.csv"
    zip_bytes = _csv_to_zip_bytes(csv_path)
    http = _MockHttpClient(zip_bytes)
    return CFTCDisaggregatedCoTSource(
        http_client=http,
        file_urls=["file:///dummy.zip"],
    )


def _make_tff_legacy_header_source() -> CFTCCoTSource:
    csv_path = _FIXTURE_DIR / "sample.csv"
    csv_text = csv_path.read_text()
    csv_text = csv_text.replace(
        "Report_Date_as_YYYY-MM-DD",
        "Report_Date_as_MM_DD_YYYY",
    )
    zip_bytes = _csv_text_to_zip_bytes("legacy_tff.csv", csv_text)
    http = _MockHttpClient(zip_bytes)
    return CFTCCoTSource(http_client=http, file_urls=["file:///dummy.zip"])


def _make_disagg_legacy_header_source() -> CFTCDisaggregatedCoTSource:
    csv_path = _FIXTURE_DIR / "sample_disagg.csv"
    csv_text = csv_path.read_text()
    csv_text = csv_text.replace(
        "Report_Date_as_YYYY-MM-DD",
        "Report_Date_as_MM_DD_YYYY",
    )
    zip_bytes = _csv_text_to_zip_bytes("legacy_disagg.csv", csv_text)
    http = _MockHttpClient(zip_bytes)
    return CFTCDisaggregatedCoTSource(http_client=http, file_urls=["file:///dummy.zip"])


class TestCFTCCoTSource:
    def test_schemas(self) -> None:
        source = _make_tff_source()
        schemas = source.schemas()
        assert "cftc.cot.tff" in schemas
        schema = schemas["cftc.cot.tff"]
        assert schema.native_freq == "W"
        assert schema.time_semantics == "interval_end"
        assert "long_positions" in schema.required_columns
        assert "short_positions" in schema.required_columns

    def test_fetch_returns_rows(self) -> None:
        source = _make_tff_source()
        df = source.fetch(
            Query(
                table="cftc.cot.tff",
                columns=["long_positions", "short_positions"],
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-02-01", tz="UTC"),
            )
        )
        assert not df.empty
        assert {
            "date",
            "entity_id",
            "asof_utc",
            "long_positions",
            "short_positions",
        }.issubset(df.columns)

    def test_entity_id_format(self) -> None:
        source = _make_tff_source()
        df = source.fetch(
            Query(
                table="cftc.cot.tff",
                columns=["long_positions", "short_positions"],
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-02-01", tz="UTC"),
            )
        )
        entities = set(df["entity_id"].unique())
        # VIX leveraged money should produce this entity
        assert "futures.vix.lev_money.cftc" in entities
        # S&P 500 dealer should produce this entity
        assert "futures.sp500.dealer.cftc" in entities
        # All entities should follow the pattern futures.{code}.{category}.cftc
        for eid in entities:
            parts = eid.split(".")
            assert parts[0] == "futures"
            assert parts[-1] == "cftc"

    def test_entity_filter(self) -> None:
        source = _make_tff_source()
        df = source.fetch(
            Query(
                table="cftc.cot.tff",
                columns=["long_positions", "short_positions"],
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-02-01", tz="UTC"),
                entities=["futures.vix.lev_money.cftc"],
            )
        )
        assert not df.empty
        assert df["entity_id"].eq("futures.vix.lev_money.cftc").all()

    def test_time_filter(self) -> None:
        source = _make_tff_source()
        # Only the first VIX row has report_date 2026-01-06 → pub_date 2026-01-09
        df = source.fetch(
            Query(
                table="cftc.cot.tff",
                columns=["long_positions", "short_positions"],
                start=pd.Timestamp("2026-01-05", tz="UTC"),
                end=pd.Timestamp("2026-01-12", tz="UTC"),
                entities=["futures.vix.lev_money.cftc"],
            )
        )
        assert not df.empty
        # Publication date for 2026-01-06 (Tue) → 2026-01-09 (Fri)
        dates = pd.DatetimeIndex(df["date"]).normalize()
        if dates.tz is not None:
            dates = dates.tz_convert(None)
        assert all(d <= pd.Timestamp("2026-01-12") for d in dates)

    def test_date_column_is_utc(self) -> None:
        source = _make_tff_source()
        df = source.fetch(
            Query(
                table="cftc.cot.tff",
                columns=["long_positions"],
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-02-01", tz="UTC"),
            )
        )
        assert str(df["date"].dtype).startswith("datetime64[") and "UTC" in str(df["date"].dtype)

    def test_publication_date_is_friday(self) -> None:
        """Publication date should always be a Friday (3 bdays after Tuesday report)."""
        source = _make_tff_source()
        df = source.fetch(
            Query(
                table="cftc.cot.tff",
                columns=["long_positions", "short_positions", "publication_date"],
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-02-01", tz="UTC"),
            )
        )
        # The date column IS the publication date
        pub_dates = pd.DatetimeIndex(df["date"])
        day_of_week = pub_dates.dayofweek
        # 4 = Friday
        assert (
            day_of_week == 4
        ).all(), f"Expected all Fridays, got day_of_week={day_of_week.unique()}"

    def test_all_trader_categories(self) -> None:
        source = _make_tff_source()
        df = source.fetch(
            Query(
                table="cftc.cot.tff",
                columns=["long_positions", "short_positions", "trader_category"],
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-02-01", tz="UTC"),
                entities=[
                    "futures.vix.lev_money.cftc",
                    "futures.vix.dealer.cftc",
                    "futures.vix.asset_mgr.cftc",
                    "futures.vix.other_rept.cftc",
                ],
            )
        )
        entities = set(df["entity_id"].unique())
        assert "futures.vix.lev_money.cftc" in entities
        assert "futures.vix.dealer.cftc" in entities
        assert "futures.vix.asset_mgr.cftc" in entities
        assert "futures.vix.other_rept.cftc" in entities

    def test_legacy_report_date_header_is_parsed(self) -> None:
        source = _make_tff_legacy_header_source()
        df = source.fetch(
            Query(
                table="cftc.cot.tff",
                columns=["long_positions"],
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-02-01", tz="UTC"),
                entities=["futures.vix.lev_money.cftc"],
            )
        )
        assert not df.empty
        assert pd.Timestamp(df["date"].iloc[0]) == pd.Timestamp("2026-01-09", tz="UTC")

    def test_year_urls_use_historical_batch_before_2010(self) -> None:
        source = CFTCCoTSource(file_urls=None)
        assert source._year_urls(2008, 2018) == [
            "https://www.cftc.gov/files/dea/history/fin_fut_txt_2006_2016.zip",
            "https://www.cftc.gov/files/dea/history/fut_fin_txt_2017.zip",
            "https://www.cftc.gov/files/dea/history/fut_fin_txt_2018.zip",
        ]

    def test_unknown_table_raises(self) -> None:
        source = _make_tff_source()
        with pytest.raises(ValueError, match="Unknown table"):
            source.fetch(
                Query(
                    table="bad.table",
                    columns=["long_positions"],
                    start=pd.Timestamp("2026-01-01", tz="UTC"),
                    end=pd.Timestamp("2026-02-01", tz="UTC"),
                )
            )

    def test_archive_failures_raise_instead_of_silently_skipping_years(self) -> None:
        csv_path = _FIXTURE_DIR / "sample.csv"
        good_zip = _csv_to_zip_bytes(csv_path)
        source = CFTCCoTSource(
            http_client=_SequenceHttpClient(
                {
                    "file:///broken_2025.zip": b"not-a-zip",
                    "file:///good_2026.zip": good_zip,
                }
            ),
            file_urls=["file:///broken_2025.zip", "file:///good_2026.zip"],
        )

        with pytest.raises(RuntimeError, match="file:///broken_2025.zip"):
            source.fetch(
                Query(
                    table="cftc.cot.tff",
                    columns=["long_positions"],
                    start=pd.Timestamp("2025-01-01", tz="UTC"),
                    end=pd.Timestamp("2026-02-01", tz="UTC"),
                )
            )


class TestCFTCDisaggregatedCoTSource:
    def test_schemas(self) -> None:
        source = _make_disagg_source()
        schemas = source.schemas()
        assert "cftc.cot.disagg" in schemas
        schema = schemas["cftc.cot.disagg"]
        assert schema.native_freq == "W"
        assert schema.time_semantics == "interval_end"
        assert "long_positions" in schema.required_columns
        assert "short_positions" in schema.required_columns

    def test_fetch_returns_rows(self) -> None:
        source = _make_disagg_source()
        df = source.fetch(
            Query(
                table="cftc.cot.disagg",
                columns=["long_positions", "short_positions"],
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-02-01", tz="UTC"),
            )
        )
        assert not df.empty
        assert {
            "date",
            "entity_id",
            "asof_utc",
            "long_positions",
            "short_positions",
        }.issubset(df.columns)

    def test_entity_ids_use_commodity_contract_codes(self) -> None:
        source = _make_disagg_source()
        df = source.fetch(
            Query(
                table="cftc.cot.disagg",
                columns=["long_positions", "short_positions"],
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-02-01", tz="UTC"),
            )
        )
        entities = set(df["entity_id"].unique())
        assert "futures.wheat_srw.prod_merc.cftc" in entities
        assert "futures.gold.swap.cftc" in entities

    def test_entity_filter(self) -> None:
        source = _make_disagg_source()
        df = source.fetch(
            Query(
                table="cftc.cot.disagg",
                columns=["long_positions", "short_positions"],
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-02-01", tz="UTC"),
                entities=["futures.wheat_srw.m_money.cftc"],
            )
        )
        assert not df.empty
        assert df["entity_id"].eq("futures.wheat_srw.m_money.cftc").all()

    def test_missing_spread_columns_are_nan(self) -> None:
        source = _make_disagg_source()
        df = source.fetch(
            Query(
                table="cftc.cot.disagg",
                columns=["spread_positions", "trader_category"],
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-02-01", tz="UTC"),
                entities=["futures.wheat_srw.prod_merc.cftc"],
            )
        )
        assert not df.empty
        assert df["trader_category"].eq("prod_merc").all()
        assert df["spread_positions"].isna().all()

    def test_legacy_report_date_header_is_parsed(self) -> None:
        source = _make_disagg_legacy_header_source()
        df = source.fetch(
            Query(
                table="cftc.cot.disagg",
                columns=["long_positions"],
                start=pd.Timestamp("2026-01-01", tz="UTC"),
                end=pd.Timestamp("2026-02-01", tz="UTC"),
                entities=["futures.wheat_srw.prod_merc.cftc"],
            )
        )
        assert not df.empty
        assert pd.Timestamp(df["date"].iloc[0]) == pd.Timestamp("2026-01-09", tz="UTC")

    def test_year_urls_use_historical_batch_before_2010(self) -> None:
        source = CFTCDisaggregatedCoTSource(file_urls=None)
        assert source._year_urls(2008, 2018) == [
            "https://www.cftc.gov/files/dea/history/fut_disagg_txt_hist_2006_2016.zip",
            "https://www.cftc.gov/files/dea/history/fut_disagg_txt_2017.zip",
            "https://www.cftc.gov/files/dea/history/fut_disagg_txt_2018.zip",
        ]


class TestPublicationDate:
    def test_tuesday_to_friday(self) -> None:
        dates = pd.Series([pd.Timestamp("2026-01-06", tz="UTC")])  # Tuesday
        pub = _publication_date(dates)
        assert pub.iloc[0].dayofweek == 4  # Friday
        assert pub.iloc[0] == pd.Timestamp("2026-01-09", tz="UTC")

    def test_nat_passthrough(self) -> None:
        dates = pd.Series([pd.NaT])
        pub = _publication_date(dates)
        assert pd.isna(pub.iloc[0])
