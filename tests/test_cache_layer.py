"""Phase 2 – CacheLayer for unified data caching.

TDD: tests written BEFORE implementation.

The CacheLayer wraps a DuckDB connection and provides store/lookup for both
PIT data (pit_observations table) and non-PIT data (market_observations table),
plus a cache_manifest table for tracking freshness.
"""

from __future__ import annotations

import warnings
from datetime import timedelta

import duckdb
import pandas as pd
import pytest


@pytest.fixture
def cache(tmp_path):
    """CacheLayer backed by a temporary DuckDB database."""
    from alphaforge.data.cache_layer import CacheLayer

    conn = duckdb.connect(str(tmp_path / "test.duckdb"))
    return CacheLayer(conn)


def _pit_df():
    """PIT DataFrame: two obs_dates, each with two revisions, from 'cftc'."""
    return pd.DataFrame(
        {
            "series_key": ["cot.tff.eur.net"] * 4,
            "obs_date": pd.to_datetime(
                ["2025-01-31", "2025-01-31", "2025-02-28", "2025-02-28"]
            ),
            "asof_utc": pd.to_datetime(
                ["2025-02-07", "2025-02-14", "2025-03-07", "2025-03-14"]
            ),
            "value": [1.0, 1.1, 2.0, 2.1],
        }
    )


def _market_df():
    """Non-PIT market DataFrame: daily prices for one ticker."""
    return pd.DataFrame(
        {
            "series_key": ["SPY"] * 3,
            "obs_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]),
            "value": [480.0, 482.5, 479.0],
        }
    )


# ---------------------------------------------------------------------------
# Store / Lookup basics
# ---------------------------------------------------------------------------


class TestCacheStorePIT:
    def test_store_and_lookup(self, cache):
        df = _pit_df()
        cache.store(
            df, dataset="cot.tff", source="cftc", is_pit=True,
        )
        result = cache.lookup(
            series_key="cot.tff.eur.net",
            dataset="cot.tff",
            source="cftc",
            is_pit=True,
        )
        assert result is not None
        data, cached_at = result
        assert len(data) == 4
        assert cached_at is not None

    def test_lookup_filters_series_key(self, cache):
        df = _pit_df()
        # Add a second series
        df2 = df.copy()
        df2["series_key"] = "cot.tff.gbp.net"
        df2["value"] = [10.0, 10.1, 20.0, 20.1]
        combined = pd.concat([df, df2], ignore_index=True)
        cache.store(combined, dataset="cot.tff", source="cftc", is_pit=True)

        result = cache.lookup(
            series_key="cot.tff.eur.net",
            dataset="cot.tff",
            source="cftc",
            is_pit=True,
        )
        assert result is not None
        data, _ = result
        assert (data["series_key"] == "cot.tff.eur.net").all()
        assert len(data) == 4


class TestCacheStoreMarket:
    def test_store_and_lookup(self, cache):
        df = _market_df()
        cache.store(df, dataset="market.ohlcv", source="tiingo", is_pit=False)
        result = cache.lookup(
            series_key="SPY",
            dataset="market.ohlcv",
            source="tiingo",
            is_pit=False,
        )
        assert result is not None
        data, cached_at = result
        assert len(data) == 3
        assert cached_at is not None


class TestCacheLookupMiss:
    def test_returns_none_on_miss(self, cache):
        result = cache.lookup(
            series_key="NONEXISTENT",
            dataset="gdp",
            source="fred",
            is_pit=True,
        )
        assert result is None


class TestCacheSourceIsolation:
    def test_cftc_cached_not_returned_for_bloomberg(self, cache):
        df = _pit_df()
        cache.store(df, dataset="cot.tff", source="cftc", is_pit=True)

        # Same series_key, different source → miss
        result = cache.lookup(
            series_key="cot.tff.eur.net",
            dataset="cot.tff",
            source="bloomberg",
            is_pit=True,
        )
        assert result is None


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


class TestCacheStaleness:
    def test_within_threshold_no_warning(self, cache):
        df = _market_df()
        cache.store(df, dataset="market.ohlcv", source="tiingo", is_pit=False)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = cache.lookup(
                series_key="SPY",
                dataset="market.ohlcv",
                source="tiingo",
                is_pit=False,
                max_staleness=timedelta(hours=24),
            )
            stale_warnings = [
                x for x in w if issubclass(x.category, UserWarning)
                and "stale" in str(x.message).lower()
            ]
            assert len(stale_warnings) == 0
        assert result is not None

    def test_exceeds_threshold_emits_warning(self, cache):
        from alphaforge.data.types import StaleDataWarning

        df = _market_df()
        cache.store(df, dataset="market.ohlcv", source="tiingo", is_pit=False)

        # Force the cached_at to be old by manipulating the manifest
        cache._force_manifest_age(
            dataset="market.ohlcv",
            source="tiingo",
            age=timedelta(hours=25),
        )

        with pytest.warns(StaleDataWarning):
            result = cache.lookup(
                series_key="SPY",
                dataset="market.ohlcv",
                source="tiingo",
                is_pit=False,
                max_staleness=timedelta(hours=24),
            )
        # Still returns data (warn, don't raise)
        assert result is not None

    def test_zero_staleness_forces_miss(self, cache):
        df = _market_df()
        cache.store(df, dataset="market.ohlcv", source="tiingo", is_pit=False)

        result = cache.lookup(
            series_key="SPY",
            dataset="market.ohlcv",
            source="tiingo",
            is_pit=False,
            max_staleness=timedelta(0),
        )
        assert result is None


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestCacheManifest:
    def test_manifest_written_on_store(self, cache):
        df = _pit_df()
        cache.store(df, dataset="cot.tff", source="cftc", is_pit=True)

        manifest = cache.get_manifest(dataset="cot.tff", source="cftc")
        assert manifest is not None
        assert manifest.dataset == "cot.tff"
        assert manifest.source == "cftc"
        assert manifest.row_count == 4
        assert "cot.tff.eur.net" in manifest.entity_keys

    def test_manifest_query_miss(self, cache):
        manifest = cache.get_manifest(dataset="nonexistent", source="fred")
        assert manifest is None

    def test_manifest_updated_on_re_store(self, cache):
        df = _market_df()
        cache.store(df, dataset="market.ohlcv", source="tiingo", is_pit=False)
        m1 = cache.get_manifest(dataset="market.ohlcv", source="tiingo")

        # Store again with more data
        df2 = pd.DataFrame(
            {
                "series_key": ["AAPL"] * 2,
                "obs_date": pd.to_datetime(["2025-01-02", "2025-01-03"]),
                "value": [190.0, 191.0],
            }
        )
        cache.store(df2, dataset="market.ohlcv", source="tiingo", is_pit=False)
        m2 = cache.get_manifest(dataset="market.ohlcv", source="tiingo")

        assert m2 is not None
        assert m1 is not None
        # Row count should reflect accumulated data
        assert m2.row_count >= m1.row_count
        assert m2.populated_at >= m1.populated_at


# ---------------------------------------------------------------------------
# PIT vs non-PIT dispatch
# ---------------------------------------------------------------------------


class TestCacheTableRouting:
    def test_pit_goes_to_pit_table(self, cache):
        df = _pit_df()
        cache.store(df, dataset="cot.tff", source="cftc", is_pit=True)

        # Verify data is in the pit-specific table
        count = cache._conn.execute(
            "SELECT COUNT(*) FROM cache_pit_observations "
            "WHERE source = 'cftc'"
        ).fetchone()[0]
        assert count == 4

    def test_market_goes_to_market_table(self, cache):
        df = _market_df()
        cache.store(df, dataset="market.ohlcv", source="tiingo", is_pit=False)

        count = cache._conn.execute(
            "SELECT COUNT(*) FROM cache_market_observations "
            "WHERE source = 'tiingo'"
        ).fetchone()[0]
        assert count == 3
