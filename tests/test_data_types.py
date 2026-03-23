"""Phase 0 – FetchResult, CacheManifest, StaleDataWarning value objects.

TDD: these tests are written BEFORE the implementation.
"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# FetchResult
# ---------------------------------------------------------------------------


class TestFetchResultFrozen:
    def test_immutable(self):
        from alphaforge.data.types import FetchResult

        result = FetchResult(
            data=pd.DataFrame({"obs_date": [date(2025, 1, 31)], "value": [1.0]}),
            source="fred",
            dataset="gdp",
            is_pit=True,
            cached_at=None,
        )
        with pytest.raises(AttributeError):
            result.source = "bloomberg"  # type: ignore[misc]

    def test_fields_stored(self):
        from alphaforge.data.types import FetchResult

        now = datetime.now(timezone.utc)
        df = pd.DataFrame({"obs_date": [date(2025, 1, 31)], "value": [1.0]})
        result = FetchResult(
            data=df,
            source="cftc",
            dataset="cot.tff",
            is_pit=True,
            cached_at=now,
        )
        assert result.source == "cftc"
        assert result.dataset == "cot.tff"
        assert result.is_pit is True
        assert result.cached_at == now
        pd.testing.assert_frame_equal(result.data, df)


class TestFetchResultCacheAge:
    def test_cache_age_when_cached(self):
        from alphaforge.data.types import FetchResult

        past = datetime.now(timezone.utc) - timedelta(hours=2)
        result = FetchResult(
            data=pd.DataFrame(),
            source="fred",
            dataset="gdp",
            is_pit=True,
            cached_at=past,
        )
        age = result.cache_age
        assert age is not None
        # Should be approximately 2 hours (allow generous tolerance for test runtime)
        assert timedelta(hours=1, minutes=59) <= age <= timedelta(hours=2, minutes=1)

    def test_cache_age_when_fresh(self):
        from alphaforge.data.types import FetchResult

        result = FetchResult(
            data=pd.DataFrame(),
            source="fred",
            dataset="gdp",
            is_pit=True,
            cached_at=None,
        )
        assert result.cache_age is None


class TestFetchResultAsTimeseries:
    def _make_pit_df(self):
        """PIT data: two obs_dates, each with two revisions."""
        return pd.DataFrame(
            {
                "obs_date": pd.to_datetime(
                    [
                        "2025-01-31",
                        "2025-01-31",
                        "2025-02-28",
                        "2025-02-28",
                    ]
                ),
                "asof_utc": pd.to_datetime(
                    [
                        "2025-02-15",  # 1st release of Jan
                        "2025-03-01",  # revision of Jan
                        "2025-03-15",  # 1st release of Feb
                        "2025-04-01",  # revision of Feb
                    ]
                ),
                "value": [1.0, 1.1, 2.0, 2.1],
            }
        )

    def test_pit_as_timeseries_filters_asof(self):
        from alphaforge.data.types import FetchResult

        result = FetchResult(
            data=self._make_pit_df(),
            source="fred",
            dataset="gdp",
            is_pit=True,
            cached_at=None,
        )
        # As of March 20 — should see revision of Jan (1.1) and 1st release of Feb (2.0)
        ts = result.as_timeseries(asof=date(2025, 3, 20))
        assert len(ts) == 2
        assert ts.iloc[0] == pytest.approx(1.1)  # latest Jan revision as-of Mar 20
        assert ts.iloc[1] == pytest.approx(2.0)  # only Feb release as-of Mar 20

    def test_pit_as_timeseries_no_asof_returns_latest(self):
        from alphaforge.data.types import FetchResult

        result = FetchResult(
            data=self._make_pit_df(),
            source="fred",
            dataset="gdp",
            is_pit=True,
            cached_at=None,
        )
        ts = result.as_timeseries()
        assert len(ts) == 2
        assert ts.iloc[0] == pytest.approx(1.1)  # latest Jan
        assert ts.iloc[1] == pytest.approx(2.1)  # latest Feb

    def test_non_pit_as_timeseries_ignores_asof(self):
        from alphaforge.data.types import FetchResult

        df = pd.DataFrame(
            {
                "obs_date": pd.to_datetime(["2025-01-31", "2025-02-28"]),
                "value": [100.0, 101.0],
            }
        )
        result = FetchResult(
            data=df,
            source="tiingo",
            dataset="market.ohlcv",
            is_pit=False,
            cached_at=None,
        )
        ts = result.as_timeseries(asof=date(2025, 2, 15))
        # Non-PIT: asof is ignored, all rows returned
        assert len(ts) == 2
        assert ts.iloc[0] == pytest.approx(100.0)
        assert ts.iloc[1] == pytest.approx(101.0)

    def test_non_pit_as_timeseries_obs_date_indexed(self):
        from alphaforge.data.types import FetchResult

        df = pd.DataFrame(
            {
                "obs_date": pd.to_datetime(["2025-01-31", "2025-02-28"]),
                "value": [100.0, 101.0],
            }
        )
        result = FetchResult(
            data=df,
            source="tiingo",
            dataset="market.ohlcv",
            is_pit=False,
            cached_at=None,
        )
        ts = result.as_timeseries()
        assert ts.index.name == "obs_date"


# ---------------------------------------------------------------------------
# CacheManifest
# ---------------------------------------------------------------------------


class TestCacheManifest:
    def test_frozen(self):
        from alphaforge.data.types import CacheManifest

        now = datetime.now(timezone.utc)
        m = CacheManifest(
            dataset="cot.tff",
            source="cftc",
            entity_keys=["futures.eur.net_long"],
            asof_range=(date(2025, 1, 1), date(2025, 3, 31)),
            populated_at=now,
            row_count=100,
        )
        with pytest.raises(AttributeError):
            m.source = "bloomberg"  # type: ignore[misc]

    def test_fields_stored(self):
        from alphaforge.data.types import CacheManifest

        now = datetime.now(timezone.utc)
        m = CacheManifest(
            dataset="cot.tff",
            source="cftc",
            entity_keys=["a", "b"],
            asof_range=(date(2025, 1, 1), date(2025, 3, 31)),
            populated_at=now,
            row_count=42,
        )
        assert m.dataset == "cot.tff"
        assert m.source == "cftc"
        assert m.entity_keys == ["a", "b"]
        assert m.asof_range == (date(2025, 1, 1), date(2025, 3, 31))
        assert m.populated_at == now
        assert m.row_count == 42

    def test_empty_manifest(self):
        from alphaforge.data.types import CacheManifest

        m = CacheManifest(
            dataset="gdp",
            source="fred",
            entity_keys=[],
            asof_range=(date(2025, 1, 1), date(2025, 1, 1)),
            populated_at=datetime.now(timezone.utc),
            row_count=0,
        )
        assert m.row_count == 0
        assert m.entity_keys == []


# ---------------------------------------------------------------------------
# StaleDataWarning
# ---------------------------------------------------------------------------


class TestStaleDataWarning:
    def test_is_user_warning(self):
        from alphaforge.data.types import StaleDataWarning

        assert issubclass(StaleDataWarning, UserWarning)

    def test_can_be_raised(self):
        from alphaforge.data.types import StaleDataWarning

        with pytest.warns(StaleDataWarning, match="stale"):
            import warnings

            warnings.warn("data is stale", StaleDataWarning)

    def test_message_preserved(self):
        from alphaforge.data.types import StaleDataWarning

        w = StaleDataWarning("cache age: 3h, threshold: 1h")
        assert "3h" in str(w)
        assert "threshold" in str(w)
