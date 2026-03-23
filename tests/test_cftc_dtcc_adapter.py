"""Phase 6 – CFTCAdapter & DTCCAdapter: bulk SourceAdapters for positioning data.

TDD: tests written BEFORE implementation.
Uses synthetic data to test transform logic and cache integration.
"""

from __future__ import annotations

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
        raw_fetcher=lambda start, end: _raw_dtcc_df(),
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


class TestCFTCAdapterPrefetch:
    def test_prefetch_returns_manifest(self, cftc_adapter):
        manifest = cftc_adapter.prefetch("cot.tff")
        assert manifest.source == "cftc"
        assert manifest.dataset == "cot.tff"
        assert manifest.row_count > 0
        assert len(manifest.entity_keys) > 0
