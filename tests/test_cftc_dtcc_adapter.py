"""Phase 6 – CFTCAdapter & DTCCAdapter: bulk SourceAdapters for positioning data.

TDD: tests written BEFORE implementation.
Uses synthetic data to test transform logic and cache integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import pandas as pd
import pytest

from alphaforge.data.adapter import SourceAdapter
from alphaforge.data.query import Query

# ---------------------------------------------------------------------------
# Synthetic raw data (mimics what CFTCCoTSource/DTCCPPDSource.fetch returns)
# ---------------------------------------------------------------------------


def _raw_cot_df():
    """Mimics CFTCCoTSource.fetch() output: wide format per entity/date."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-10", "2025-01-10", "2025-01-17", "2025-01-17"]),
            "entity_id": ["futures.eur.lev_money.cftc", "futures.gbp.lev_money.cftc",
                          "futures.eur.lev_money.cftc", "futures.gbp.lev_money.cftc"],
            "long_positions": [1000.0, 2000.0, 1100.0, 2100.0],
            "short_positions": [800.0, 1800.0, 900.0, 1900.0],
            "open_interest": [5000.0, 8000.0, 5200.0, 8200.0],
        }
    )


def _raw_disagg_cot_df():
    """Mimics CFTCDisaggregatedCoTSource.fetch() output."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-10", "2025-01-10", "2025-01-17", "2025-01-17"]),
            "entity_id": [
                "futures.wheat_srw.m_money.cftc",
                "futures.gold.swap.cftc",
                "futures.wheat_srw.m_money.cftc",
                "futures.gold.swap.cftc",
            ],
            "long_positions": [3000.0, 5000.0, 3200.0, 5100.0],
            "short_positions": [2800.0, 4200.0, 2900.0, 4300.0],
            "open_interest": [15000.0, 25000.0, 15500.0, 25200.0],
        }
    )


def _raw_dtcc_df():
    """Mimics DTCCPPDSource.fetch() output."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"]),
            "entity_id": ["dtccppd.fx.eur", "dtccppd.fx.gbp",
                          "dtccppd.fx.eur", "dtccppd.fx.gbp"],
            "asof_utc": pd.to_datetime(["2025-01-02", "2025-01-02", "2025-01-03", "2025-01-03"]),
            "trade_count": [100, 200, 110, 210],
            "notional_sum": [1e9, 2e9, 1.1e9, 2.1e9],
            "price_mean": [1.05, 1.26, 1.06, 1.27],
            "dv01_proxy_sum": [50000.0, 60000.0, 52000.0, 62000.0],
            "notional_median": [5e6, 6e6, 5.5e6, 6.5e6],
        }
    )


def _raw_dtcc_family_df():
    """Daily DTCC output with distinct FX and IRS family entity ids."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-02",
                    "2025-01-03",
                    "2025-01-03",
                ]
            ),
            "entity_id": [
                "dtccppd.fx.fx_forward.usd.1m",
                "dtccppd.fx.fx_swap.eur.3m",
                "dtccppd.rates.interest_rate_swap.usd.5y",
                "dtccppd.rates.cross_currency_swap.usd.5y",
                "dtccppd.fx.fx_swap.gbp.6m",
                "dtccppd.rates.interest_rate_swap.eur.10y",
            ],
            "asof_utc": pd.to_datetime(
                [
                    "2025-01-02T15:00:00Z",
                    "2025-01-02T15:00:00Z",
                    "2025-01-02T15:00:00Z",
                    "2025-01-02T15:00:00Z",
                    "2025-01-03T15:00:00Z",
                    "2025-01-03T15:00:00Z",
                ]
            ),
            "trade_count": [100, 90, 60, 40, 120, 55],
            "notional_sum": [10e6, 15e6, 50e6, 35e6, 12e6, 42e6],
            "price_mean": [1.085, 0.0025, 0.032, 0.015, 0.0040, 0.028],
            "price_std": [0.01, 0.002, 0.001, 0.0015, 0.0025, 0.0012],
            "notional_median": [5e6, 7.5e6, 25e6, 17.5e6, 6e6, 21e6],
            "trade_count_large": [4, 5, 3, 2, 6, 3],
            "dv01_proxy_sum": [0.0, 0.0, 120_000.0, 80_000.0, 0.0, 140_000.0],
        }
    )


@dataclass
class _StubDTCCSource:
    table_frames: dict[str, pd.DataFrame]
    queries: list[Query]

    def __init__(self, table_frames: dict[str, pd.DataFrame]) -> None:
        self.table_frames = table_frames
        self.queries = []

    def schemas(self):
        return {}

    def fetch(self, q: Query) -> pd.DataFrame:
        self.queries.append(q)
        return self.table_frames[q.table].copy()


# ---------------------------------------------------------------------------
# Transform tests (functions moved from positioning)
# ---------------------------------------------------------------------------


class TestCotTransform:
    def test_transform_shape(self):
        from alphaforge.data.transforms.cot_pit import cot_to_pit_observations

        df = _raw_cot_df()
        result = cot_to_pit_observations(df)
        # 2 entities × 2 dates × 5 metrics = 20 rows
        assert len(result) == 20

    def test_transform_columns(self):
        from alphaforge.data.transforms.cot_pit import cot_to_pit_observations

        df = _raw_cot_df()
        result = cot_to_pit_observations(df)
        assert list(result.columns) == ["series_key", "obs_date", "asof_utc", "value", "source"]

    def test_obs_date_is_tuesday(self):
        from alphaforge.data.transforms.cot_pit import cot_to_pit_observations

        df = _raw_cot_df()
        result = cot_to_pit_observations(df)
        # obs_date should be Tuesday (3 bdays before Friday pub date)
        for d in result["obs_date"]:
            assert pd.Timestamp(d).dayofweek == 1  # Tuesday

    def test_series_key_format(self):
        from alphaforge.data.transforms.cot_pit import cot_to_pit_observations

        df = _raw_cot_df()
        result = cot_to_pit_observations(df)
        assert any("cftc.cot.tff." in k for k in result["series_key"])

    def test_metric_filtering(self):
        from alphaforge.data.transforms.cot_pit import cot_to_pit_observations

        df = _raw_cot_df()
        result = cot_to_pit_observations(df, metrics=["net_positions"])
        assert len(result) == 4  # 2 entities × 2 dates × 1 metric

    def test_empty_input(self):
        from alphaforge.data.transforms.cot_pit import cot_to_pit_observations

        result = cot_to_pit_observations(pd.DataFrame())
        assert len(result) == 0


class TestDtccTransform:
    def test_transform_shape(self):
        from alphaforge.data.transforms.dtcc_pit import dtcc_daily_to_pit_observations

        df = _raw_dtcc_df()
        result = dtcc_daily_to_pit_observations(df)
        # 2 entities × 2 dates × 5 metrics = 20 rows
        assert len(result) == 20

    def test_transform_columns(self):
        from alphaforge.data.transforms.dtcc_pit import dtcc_daily_to_pit_observations

        df = _raw_dtcc_df()
        result = dtcc_daily_to_pit_observations(df)
        assert list(result.columns) == ["series_key", "obs_date", "asof_utc", "value", "source"]

    def test_series_key_format(self):
        from alphaforge.data.transforms.dtcc_pit import dtcc_daily_to_pit_observations

        df = _raw_dtcc_df()
        result = dtcc_daily_to_pit_observations(df)
        assert any("dtcc.ppd.daily." in k for k in result["series_key"])

    def test_transform_allows_custom_prefix_and_source_name(self):
        from alphaforge.data.transforms.dtcc_pit import dtcc_daily_to_pit_observations

        df = _raw_dtcc_family_df()
        result = dtcc_daily_to_pit_observations(
            df,
            key_prefix="dtcc.fx.",
            source_name="dtcc_ppd_fx",
        )

        assert result["series_key"].str.startswith("dtcc.fx.").all()
        assert result["source"].eq("dtcc_ppd_fx").all()


class TestMeltToPitFormat:
    def test_basic_melt(self):
        from alphaforge.data.transforms.utils import melt_to_pit_format

        df = pd.DataFrame(
            {
                "entity": ["A", "B"],
                "obs": pd.to_datetime(["2025-01-01", "2025-01-01"]),
                "asof": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                "metric1": [10.0, 20.0],
                "metric2": [30.0, 40.0],
            }
        )
        result = melt_to_pit_format(
            df, entity_col="entity", obs_date_col="obs",
            asof_col="asof", value_vars=["metric1", "metric2"],
            key_prefix="test.", source_name="test_src",
        )
        assert len(result) == 4
        assert set(result["source"]) == {"test_src"}
        assert "test.A.metric1" in result["series_key"].values


# ---------------------------------------------------------------------------
# Adapter protocol conformance
# ---------------------------------------------------------------------------


@pytest.fixture
def cftc_adapter(tmp_path):
    from alphaforge.data.sources.cftc import CFTCAdapter

    conn = duckdb.connect(str(tmp_path / "cache.duckdb"))
    return CFTCAdapter(
        raw_fetcher=lambda start, end: _raw_cot_df(),
        cache_conn=conn,
    )


@pytest.fixture
def dtcc_adapter(tmp_path):
    from alphaforge.data.sources.dtcc import DTCCAdapter

    conn = duckdb.connect(str(tmp_path / "cache.duckdb"))
    return DTCCAdapter(
        source=_StubDTCCSource({"dtcc.ppd.daily": _raw_dtcc_df()}),
        cache_conn=conn,
    )


@pytest.fixture
def cftc_multi_adapter(tmp_path):
    from alphaforge.data.sources.cftc import CFTCAdapter

    conn = duckdb.connect(str(tmp_path / "cache.duckdb"))
    return CFTCAdapter(
        raw_fetchers={
            "cot.tff": lambda start, end: _raw_cot_df(),
            "cot.disagg": lambda start, end: _raw_disagg_cot_df(),
        },
        cache_conn=conn,
    )


class TestCFTCAdapterProtocol:
    def test_is_source_adapter(self, cftc_adapter):
        assert isinstance(cftc_adapter, SourceAdapter)

    def test_source_name(self, cftc_adapter):
        assert cftc_adapter.source_name == "cftc"

    def test_datasets(self, cftc_adapter):
        assert "cot.tff" in cftc_adapter.datasets


class TestDTCCAdapterProtocol:
    def test_is_source_adapter(self, dtcc_adapter):
        assert isinstance(dtcc_adapter, SourceAdapter)

    def test_source_name(self, dtcc_adapter):
        assert dtcc_adapter.source_name == "dtcc"

    def test_datasets(self, dtcc_adapter):
        assert "dtcc.ppd" in dtcc_adapter.datasets

    def test_fetch_uses_internal_raw_source_query(self, tmp_path):
        from alphaforge.data.sources.dtcc import DTCCAdapter

        conn = duckdb.connect(str(tmp_path / "cache.duckdb"))
        source = _StubDTCCSource({"dtcc.ppd.daily": _raw_dtcc_df()})
        adapter = DTCCAdapter(source=source, cache_conn=conn)

        q = Query(
            table="dtcc.ppd",
            columns=["value"],
            entities=["dtcc.ppd.daily.dtccppd.fx.eur.trade_count"],
            start="2025-01-01",
            end="2025-01-31",
            asof="2025-01-20",
        )
        result = adapter.fetch(q)

        assert len(source.queries) == 1
        raw_query = source.queries[0]
        assert raw_query.table == "dtcc.ppd.daily"
        assert raw_query.start == pd.Timestamp("2025-01-01", tz="UTC")
        assert raw_query.end == pd.Timestamp("2025-01-31", tz="UTC")
        assert not result.data.empty

    def test_prefetch_uses_internal_raw_source(self, tmp_path):
        from alphaforge.data.sources.dtcc import DTCCAdapter

        conn = duckdb.connect(str(tmp_path / "cache.duckdb"))
        source = _StubDTCCSource({"dtcc.ppd.daily": _raw_dtcc_df()})
        adapter = DTCCAdapter(source=source, cache_conn=conn)

        manifest = adapter.prefetch(
            "dtcc.ppd",
            asof_range=(pd.Timestamp("2025-01-01").date(), pd.Timestamp("2025-01-31").date()),
        )

        assert len(source.queries) == 1
        assert source.queries[0].table == "dtcc.ppd.daily"
        assert manifest.dataset == "dtcc.ppd"
        assert manifest.source == "dtcc"
        assert manifest.row_count > 0


class TestDTCCAdapterBase:
    def test_base_supports_subclass_dataset_contract(self, tmp_path):
        from alphaforge.data.sources.dtcc import DTCCPPDAdapterBase
        from alphaforge.data.transforms.dtcc_pit import dtcc_daily_to_pit_observations

        class DemoDTCCAdapter(DTCCPPDAdapterBase):
            source_name = "dtcc_demo"
            datasets = frozenset({"dtcc.demo"})

            def _to_pit(self, dataset: str, raw_df: pd.DataFrame) -> pd.DataFrame:
                assert dataset == "dtcc.demo"
                return dtcc_daily_to_pit_observations(raw_df)

        conn = duckdb.connect(str(tmp_path / "cache.duckdb"))
        source = _StubDTCCSource({"dtcc.ppd.daily": _raw_dtcc_df()})
        adapter = DemoDTCCAdapter(source=source, cache_conn=conn)

        result = adapter.fetch(
            Query(
                table="dtcc.demo",
                columns=["value"],
                entities=["dtcc.ppd.daily.dtccppd.fx.eur.trade_count"],
                asof="2025-01-20",
            )
        )

        assert result.source == "dtcc_demo"
        assert result.dataset == "dtcc.demo"
        assert not result.data.empty


class TestDTCCProductFamilyAdapters:
    def test_fx_adapter_filters_to_fx_entities(self, tmp_path):
        from alphaforge.data.sources.dtcc import DTCCFXAdapter

        conn = duckdb.connect(str(tmp_path / "cache.duckdb"))
        source = _StubDTCCSource({"dtcc.ppd.daily": _raw_dtcc_family_df()})
        adapter = DTCCFXAdapter(source=source, cache_conn=conn)

        result = adapter.fetch(
            Query(
                table="dtcc.fx",
                columns=["value"],
                entities=["dtcc.fx.dtccppd.fx.fx_forward.usd.1m.trade_count"],
                asof="2025-01-20",
            )
        )

        assert result.source == "dtcc_fx"
        assert not result.data.empty
        assert result.data["series_key"].str.startswith("dtcc.fx.").all()
        assert result.data["series_key"].str.contains(".fx.", regex=False).all()
        assert result.data["source"].eq("dtcc_ppd_fx").all()

    def test_irs_adapter_filters_to_interest_rate_swaps(self, tmp_path):
        from alphaforge.data.sources.dtcc import DTCCIRSAdapter

        conn = duckdb.connect(str(tmp_path / "cache.duckdb"))
        source = _StubDTCCSource({"dtcc.ppd.daily": _raw_dtcc_family_df()})
        adapter = DTCCIRSAdapter(source=source, cache_conn=conn)

        result = adapter.fetch(
            Query(
                table="dtcc.irs",
                columns=["value"],
                entities=["dtcc.irs.dtccppd.rates.interest_rate_swap.usd.5y.trade_count"],
                asof="2025-01-20",
            )
        )

        assert result.source == "dtcc_irs"
        assert not result.data.empty
        assert result.data["series_key"].str.startswith("dtcc.irs.").all()
        assert result.data["series_key"].str.contains("interest_rate_swap").all()
        assert not result.data["series_key"].str.contains("cross_currency_swap").any()
        assert result.data["source"].eq("dtcc_ppd_irs").all()

    def test_family_adapters_route_through_data_context(self, tmp_path):
        from alphaforge.data.context import DataContext
        from alphaforge.data.sources.dtcc import DTCCFXAdapter, DTCCIRSAdapter

        source = _StubDTCCSource({"dtcc.ppd.daily": _raw_dtcc_family_df()})
        fx = DTCCFXAdapter(source=source, cache_conn=duckdb.connect(str(tmp_path / "fx.duckdb")))
        irs = DTCCIRSAdapter(
            source=source,
            cache_conn=duckdb.connect(str(tmp_path / "irs.duckdb")),
        )

        ctx = DataContext.from_adapters(fx, irs, calendars={}, store=None)

        fx_result = ctx.load(
            "dtcc.fx",
            columns=["value"],
            entities=["dtcc.fx.dtccppd.fx.fx_forward.usd.1m.trade_count"],
            asof="2025-01-20",
        )
        irs_result = ctx.load(
            "dtcc.irs",
            columns=["value"],
            entities=["dtcc.irs.dtccppd.rates.interest_rate_swap.usd.5y.trade_count"],
            asof="2025-01-20",
        )

        assert ctx.default_sources == {"dtcc.fx": "dtcc_fx", "dtcc.irs": "dtcc_irs"}
        assert fx_result.source == "dtcc_fx"
        assert irs_result.source == "dtcc_irs"


# ---------------------------------------------------------------------------
# Adapter fetch (bulk behavior)
# ---------------------------------------------------------------------------


class TestCFTCAdapterFetch:
    def test_fetch_returns_pit_data(self, cftc_adapter):
        q = Query(
            table="cot.tff",
            columns=["value"],
            entities=["cftc.cot.tff.futures.eur.lev_money.cftc.net_positions"],
            asof="2025-01-20",
        )
        result = cftc_adapter.fetch(q)

        from alphaforge.data.types import FetchResult

        assert isinstance(result, FetchResult)
        assert result.source == "cftc"
        assert result.is_pit is True
        # Should filter to the requested series key
        assert len(result.data) > 0

    def test_fetch_filters_to_requested_entity(self, cftc_adapter):
        q = Query(
            table="cot.tff",
            columns=["value"],
            entities=["cftc.cot.tff.futures.eur.lev_money.cftc.net_positions"],
            asof="2025-01-20",
        )
        result = cftc_adapter.fetch(q)
        assert (
            result.data["series_key"] == "cftc.cot.tff.futures.eur.lev_money.cftc.net_positions"
        ).all()


class TestCFTCAdapterBulkCache:
    def test_bulk_fetch_caches_all_series(self, cftc_adapter):
        """First fetch triggers bulk load; second fetch for different series uses cache."""
        fetch_count = [0]
        original_fetcher = cftc_adapter._raw_fetcher

        def counting_fetcher(start, end):
            fetch_count[0] += 1
            return original_fetcher(start, end)

        cftc_adapter._raw_fetcher = counting_fetcher

        q1 = Query(
            table="cot.tff", columns=["value"],
            entities=["cftc.cot.tff.futures.eur.lev_money.cftc.net_positions"],
            asof="2025-01-20",
        )
        cftc_adapter.fetch(q1)
        assert fetch_count[0] == 1

        # Different series, same dataset — should use cache
        q2 = Query(
            table="cot.tff", columns=["value"],
            entities=["cftc.cot.tff.futures.gbp.lev_money.cftc.net_positions"],
            asof="2025-01-20",
        )
        result2 = cftc_adapter.fetch(q2)
        assert fetch_count[0] == 1  # No additional fetch
        assert result2.cached_at is not None
        assert len(result2.data) > 0

    def test_cache_hit_preserves_source_column(self, cftc_adapter):
        """Cache-hit path must return the lineage 'source' column identical to cache-miss."""
        q = Query(
            table="cot.tff", columns=["value"],
            entities=["cftc.cot.tff.futures.eur.lev_money.cftc.net_positions"],
            asof="2025-01-20",
        )
        # First fetch (cache miss) — source column comes from cot_to_pit_observations
        miss_result = cftc_adapter.fetch(q)
        assert "source" in miss_result.data.columns, "cache-miss result must have 'source' column"
        assert miss_result.data["source"].eq("cftc_cot").all(), (
            "cache-miss source must be 'cftc_cot' for cot.tff dataset"
        )

        # Second fetch for same entity — cache hit
        hit_result = cftc_adapter.fetch(q)
        assert hit_result.cached_at is not None, "second fetch should be a cache hit"
        assert "source" in hit_result.data.columns, "cache-hit result must have 'source' column"
        assert hit_result.data["source"].eq("cftc_cot").all(), (
            "cache-hit source must equal cache-miss source for cot.tff dataset"
        )

    def test_cache_hit_preserves_disagg_source_column(self, cftc_multi_adapter):
        """Cache-hit path for disagg dataset must carry the correct 'cftc_cot_disagg' source."""
        q = Query(
            table="cot.disagg", columns=["value"],
            entities=["cftc.cot.disagg.futures.wheat_srw.m_money.cftc.net_positions"],
            asof="2025-01-20",
        )
        # First fetch (cache miss)
        miss_result = cftc_multi_adapter.fetch(q)
        assert "source" in miss_result.data.columns
        assert miss_result.data["source"].eq("cftc_cot_disagg").all()

        # Second fetch (cache hit)
        hit_result = cftc_multi_adapter.fetch(q)
        assert hit_result.cached_at is not None
        assert "source" in hit_result.data.columns
        assert hit_result.data["source"].eq("cftc_cot_disagg").all()


class TestCFTCAdapterPrefetch:
    def test_prefetch_returns_manifest(self, cftc_adapter):
        manifest = cftc_adapter.prefetch("cot.tff")
        assert manifest.source == "cftc"
        assert manifest.dataset == "cot.tff"
        assert manifest.row_count > 0
        assert len(manifest.entity_keys) > 0


class TestCFTCAdapterMultipleDatasets:
    def test_declares_disagg_dataset(self, cftc_multi_adapter):
        assert "cot.tff" in cftc_multi_adapter.datasets
        assert "cot.disagg" in cftc_multi_adapter.datasets

    def test_fetch_disagg_dataset_uses_dataset_prefix(self, cftc_multi_adapter):
        q = Query(
            table="cot.disagg",
            columns=["value"],
            entities=["cftc.cot.disagg.futures.wheat_srw.m_money.cftc.net_positions"],
            asof="2025-01-20",
        )
        result = cftc_multi_adapter.fetch(q)
        assert not result.data.empty
        assert (
            result.data["series_key"]
            == "cftc.cot.disagg.futures.wheat_srw.m_money.cftc.net_positions"
        ).all()
        assert result.data["source"].eq("cftc_cot_disagg").all()

    def test_prefetch_disagg_returns_manifest(self, cftc_multi_adapter):
        manifest = cftc_multi_adapter.prefetch("cot.disagg")
        assert manifest.source == "cftc"
        assert manifest.dataset == "cot.disagg"
        assert manifest.row_count > 0
