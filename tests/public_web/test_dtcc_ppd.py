from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from alphaforge.data.public_web.dtcc_ppd import DTCCPPDSource
from alphaforge.data.query import Query


def _zip_fixture_bytes() -> bytes:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures/public_web/dtcc_ppd/sample_events.csv"
    )
    csv_data = fixture.read_bytes()

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("events_part1.csv", csv_data)
    return buf.getvalue()


_FILE_ENTRY = {
    "fileName": "CFTC_SLICE_RATES_2026_01_02_1.zip",
    "dissemDTM": "2026-01-02T15:37:19Z",
}
_FX_FILE_ENTRY = {
    "fileName": "CFTC_SLICE_FX_2026_01_02_1.zip",
    "dissemDTM": "2026-01-02T10:37:19Z",
}


def _list_provider(report_type: str, asset_code: str) -> list[dict]:
    if report_type == "slice" and asset_code == "IR":
        return [_FILE_ENTRY]
    if report_type == "slice" and asset_code == "FX":
        return [_FX_FILE_ENTRY]
    if report_type == "cumulative" and asset_code == "IR":
        return [_FILE_ENTRY]
    if report_type == "cumulative" and asset_code == "FX":
        return [_FX_FILE_ENTRY]
    return []


def _make_source(**kwargs) -> DTCCPPDSource:
    payload = _zip_fixture_bytes()
    defaults = dict(
        list_provider=_list_provider,
        artifact_provider=lambda fn: payload,
        asset_codes=("IR", "FX"),
    )
    defaults.update(kwargs)
    return DTCCPPDSource(**defaults)


def _fetch_events(source: DTCCPPDSource | None = None, **query_kw) -> pd.DataFrame:
    source = source or _make_source(source_mode="slice")
    defaults = dict(
        table="dtcc.ppd.events",
        columns=["price", "notional", "product", "currency", "tenor", "asset_class"],
        start=pd.Timestamp("2026-01-02", tz="UTC"),
        end=pd.Timestamp("2026-01-03 23:59:59", tz="UTC"),
    )
    defaults.update(query_kw)
    return source.fetch(Query(**defaults))


def _fetch_daily(source: DTCCPPDSource | None = None, **query_kw) -> pd.DataFrame:
    source = source or _make_source(source_mode="cumulative")
    defaults = dict(
        table="dtcc.ppd.daily",
        columns=[
            "trade_count",
            "notional_sum",
            "price_mean",
            "price_std",
            "notional_median",
            "trade_count_large",
            "dv01_proxy_sum",
        ],
        start=pd.Timestamp("2026-01-02", tz="UTC"),
        end=pd.Timestamp("2026-01-03", tz="UTC"),
    )
    defaults.update(query_kw)
    return source.fetch(Query(**defaults))


# ── Original test (updated for expanded fixture) ──────────────────────


def test_dtcc_events_and_daily_aggregation() -> None:
    source = _make_source(source_mode="slice")
    events = _fetch_events(source)

    assert not events.empty
    assert str(events["ts_utc"].dtype).startswith(("datetime64[ns,", "datetime64[us,"))
    assert events["entity_id"].str.contains("dtccppd").all()

    first_entity = events["entity_id"].iloc[0]
    daily = source.fetch(
        Query(
            table="dtcc.ppd.daily",
            columns=["notional_sum", "trade_count", "price_mean"],
            entities=[first_entity],
            start=pd.Timestamp("2026-01-02", tz="UTC"),
            end=pd.Timestamp("2026-01-03", tz="UTC"),
        )
    )

    assert not daily.empty
    assert daily["entity_id"].nunique() == 1
    assert "trade_count" in daily.columns
    assert int(daily["trade_count"].iloc[0]) >= 1
    assert "notional_sum" in daily.columns
    assert str(daily["asof_utc"].dtype).startswith(("datetime64[ns,", "datetime64[us,"))


# ── New tests ─────────────────────────────────────────────────────────


class TestSchemas:
    def test_schemas(self) -> None:
        source = _make_source()
        schemas = source.schemas()
        assert "dtcc.ppd.events" in schemas
        assert "dtcc.ppd.daily" in schemas

        ev = schemas["dtcc.ppd.events"]
        assert ev.time_column == "ts_utc"
        assert ev.entity_column == "entity_id"
        assert ev.native_freq == "D"
        assert ev.time_semantics == "point"

        daily = schemas["dtcc.ppd.daily"]
        assert daily.time_column == "date"
        assert daily.entity_column == "entity_id"
        assert daily.native_freq == "D"
        assert daily.time_semantics == "interval_end"


class TestEntityIdFormat:
    def test_entity_id_format(self) -> None:
        events = _fetch_events()
        pattern = re.compile(r"^dtccppd\.[a-z_]+\.[a-z_]+\.[a-z]+\.\w+$")
        for eid in events["entity_id"].unique():
            assert pattern.match(eid), f"Entity ID does not match pattern: {eid}"
            assert eid == eid.lower(), f"Entity ID has uppercase: {eid}"
            assert " " not in eid, f"Entity ID has spaces: {eid}"


class TestFXEvents:
    def test_fx_events(self) -> None:
        events = _fetch_events()
        fx = events[events["entity_id"].str.startswith("dtccppd.fx.")]
        assert not fx.empty, "No FX events found"
        products = {p.lower().replace(" ", "_") for p in fx["product"].unique()}
        assert "fx_forward" in products or "fx_swap" in products, (
            f"Expected fx_forward or fx_swap in FX products, got {products}"
        )


class TestIRSEvents:
    def test_irs_events(self) -> None:
        events = _fetch_events()
        irs = events[events["entity_id"].str.contains("interest_rate_swap")]
        assert not irs.empty, "No IRS events found"


class TestCCSEvents:
    def test_ccs_events(self) -> None:
        events = _fetch_events()
        ccs = events[events["entity_id"].str.contains("cross_currency_swap")]
        assert not ccs.empty, "No CCS events found"


class TestDailyAggregation:
    def test_daily_aggregation_columns(self) -> None:
        daily = _fetch_daily()
        assert not daily.empty
        required = [
            "trade_count",
            "notional_sum",
            "price_mean",
            "price_std",
            "notional_median",
            "trade_count_large",
            "dv01_proxy_sum",
        ]
        for col in required:
            assert col in daily.columns, f"Missing column: {col}"


class TestDailyDV01Proxy:
    def test_daily_dv01_proxy(self) -> None:
        # Use IR-only source to avoid double-counting from FX payload
        source = _make_source(source_mode="cumulative", asset_codes=("IR",))
        daily = _fetch_daily(source)
        # Find a 10y IRS entity
        irs_10y = daily[
            daily["entity_id"].str.contains("interest_rate_swap")
            & daily["entity_id"].str.contains("10y")
        ]
        assert not irs_10y.empty, "No 10y IRS entity in daily"

        # Compute expected: from fixture, 10y IRS USD rows have
        # notional 8_000_000 + 12_000_000 = 20_000_000
        # duration_scalar for 10y = 8.5
        # dv01_proxy per row = notional * 8.5 * 1e-4
        # total = (8_000_000 + 12_000_000) * 8.5 * 1e-4 = 17_000
        row = irs_10y[irs_10y["entity_id"].str.contains("usd")]
        if not row.empty:
            expected = (8_000_000 + 12_000_000) * 8.5 * 1e-4
            assert abs(row["dv01_proxy_sum"].iloc[0] - expected) < 1.0, (
                f"Expected dv01_proxy_sum ≈ {expected}, got {row['dv01_proxy_sum'].iloc[0]}"
            )


class TestEntityFilter:
    def test_entity_filter(self) -> None:
        events_all = _fetch_events()
        target = events_all["entity_id"].iloc[0]

        source = _make_source(source_mode="slice")
        filtered = source.fetch(
            Query(
                table="dtcc.ppd.events",
                columns=["price", "notional"],
                start=pd.Timestamp("2026-01-02", tz="UTC"),
                end=pd.Timestamp("2026-01-03 23:59:59", tz="UTC"),
                entities=[target],
            )
        )
        assert not filtered.empty
        assert (filtered["entity_id"] == target).all()


class TestTimeFilter:
    def test_time_filter(self) -> None:
        source = _make_source(source_mode="slice")
        # Tight window: only 14:00-15:00 on 2026-01-02
        filtered = source.fetch(
            Query(
                table="dtcc.ppd.events",
                columns=["price", "notional"],
                start=pd.Timestamp("2026-01-02T14:00:00Z"),
                end=pd.Timestamp("2026-01-02T15:00:00Z"),
            )
        )
        if not filtered.empty:
            ts = pd.DatetimeIndex(filtered["ts_utc"])
            assert (ts >= pd.Timestamp("2026-01-02T14:00:00Z")).all()
            assert (ts <= pd.Timestamp("2026-01-02T15:00:00Z")).all()


class TestUnknownTable:
    def test_unknown_table_raises(self) -> None:
        source = _make_source()
        with pytest.raises(ValueError, match="Unknown table"):
            source.fetch(
                Query(
                    table="dtcc.ppd.bad_table",
                    columns=["price"],
                    start=pd.Timestamp("2026-01-02", tz="UTC"),
                    end=pd.Timestamp("2026-01-03", tz="UTC"),
                )
            )
