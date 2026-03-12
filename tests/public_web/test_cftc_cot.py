from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.data.public_web.cftc_cot import CFTCCoTSource, _publication_date
from alphaforge.data.query import Query

_FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures/public_web/cftc_cot"


def _csv_to_zip_bytes(csv_path: Path) -> bytes:
    """Wrap a CSV fixture into a ZIP archive in memory."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.write(csv_path, arcname=csv_path.name)
    return buf.getvalue()


class _MockHttpClient:
    """Return pre-built ZIP bytes for any URL request."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def get_bytes(self, *, url: str, source: str, artifact_name: str, **kw) -> bytes:
        return self._payload


def _make_source() -> CFTCCoTSource:
    csv_path = _FIXTURE_DIR / "sample.csv"
    zip_bytes = _csv_to_zip_bytes(csv_path)
    http = _MockHttpClient(zip_bytes)
    return CFTCCoTSource(
        http_client=http,
        file_urls=["file:///dummy.zip"],
    )


class TestCFTCCoTSource:
    def test_schemas(self) -> None:
        source = _make_source()
        schemas = source.schemas()
        assert "cftc.cot.tff" in schemas
        schema = schemas["cftc.cot.tff"]
        assert schema.native_freq == "W"
        assert schema.time_semantics == "interval_end"
        assert "long_positions" in schema.required_columns
        assert "short_positions" in schema.required_columns

    def test_fetch_returns_rows(self) -> None:
        source = _make_source()
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
        source = _make_source()
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
        source = _make_source()
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
        source = _make_source()
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
        source = _make_source()
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
        source = _make_source()
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
        source = _make_source()
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

    def test_unknown_table_raises(self) -> None:
        source = _make_source()
        with pytest.raises(ValueError, match="Unknown table"):
            source.fetch(
                Query(
                    table="bad.table",
                    columns=["long_positions"],
                    start=pd.Timestamp("2026-01-01", tz="UTC"),
                    end=pd.Timestamp("2026-02-01", tz="UTC"),
                )
            )


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
