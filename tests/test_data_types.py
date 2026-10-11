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


@pytest.mark.parametrize("timezone_name", [None, "UTC", "America/New_York"])
def test_timeseries_normalizes_release_timezone(timezone_name):
    from alphaforge.data.types import FetchResult

    releases = pd.to_datetime(["2025-01-02T15:00Z", "2025-01-03T15:00Z"])
    releases = releases.tz_convert(timezone_name) if timezone_name else releases.tz_localize(None)
    df = pd.DataFrame({
        "obs_date": pd.to_datetime(["2025-01-01"] * 2),
        "asof_utc": releases,
        "value": [1.0, 2.0],
    })
    result = FetchResult(df, "test", "test", True, None)
    assert result.as_timeseries(asof=date(2025, 1, 3)).iloc[0] == 1.0
    assert result.as_timeseries(asof=pd.Timestamp("2025-01-03T10:00-05:00")).iloc[0] == 2.0


def test_timeseries_preserves_latest_null_revision():
    from alphaforge.data.types import FetchResult

    df = pd.DataFrame({
        "obs_date": pd.to_datetime(["2025-01-01"] * 2),
        "asof_utc": pd.to_datetime(["2025-01-02", "2025-01-03"]),
        "value": [1.0, float("nan")],
    })
    result = FetchResult(df, "test", "test", True, None)
    assert result.as_timeseries(asof=date(2025, 1, 2)).iloc[0] == 1.0
    assert pd.isna(result.as_timeseries().iloc[0])


def test_timeseries_requires_entity_for_multi_series_results():
    from alphaforge.data.types import FetchResult

    df = pd.DataFrame({
        "series_key": ["A", "B"], "obs_date": pd.to_datetime(["2025-01-01"] * 2),
        "asof_utc": pd.to_datetime(["2025-01-02"] * 2, utc=True), "value": [1.0, 20.0],
    })
    result = FetchResult(df, "test", "test", True, None)
    with pytest.raises(ValueError, match="entity"):
        result.as_timeseries()
    assert result.as_timeseries(entity="A").tolist() == [1.0]
    assert result.as_timeseries(entity="B").tolist() == [20.0]


def test_timeseries_supports_an_explicit_market_value_column():
    from alphaforge.data.types import FetchResult

    df = pd.DataFrame({"series_key": ["SPY"], "obs_date": [date(2025, 1, 1)],
                       "close": [100.0], "volume": [200.0]})
    result = FetchResult(df, "market", "ohlcv", False, None)
    assert result.as_timeseries(value_column="close").tolist() == [100.0]
    with pytest.raises(ValueError, match="value"):
        result.as_timeseries()


def test_snapshot_cannot_reconstruct_an_earlier_vintage():
    from alphaforge.data.types import FetchResult

    df = pd.DataFrame({"obs_date": [date(2025, 1, 1)],
                       "asof_utc": [pd.Timestamp("2025-02-01", tz="UTC")], "value": [2.0]})
    result = FetchResult(df, "fred", "macro.fred", True, None, result_kind="snapshot",
                         availability_precision="date", availability_basis="snapshot",
                         snapshot_asof=pd.Timestamp("2025-02-01", tz="UTC"))
    with pytest.raises(ValueError, match="snapshot"):
        result.as_timeseries(asof=date(2025, 1, 15))
    assert result.as_timeseries(asof=date(2025, 2, 1)).tolist() == [2.0]


def test_partial_and_mixed_cache_results_expose_each_acquisition():
    from alphaforge.data.types import AcquisitionMetadata, FetchFailure, FetchResult

    oldest = datetime(2025, 1, 1, tzinfo=timezone.utc)
    newest = datetime(2025, 1, 2, tzinfo=timezone.utc)
    acquisitions = (AcquisitionMetadata(entity="A", cached_at=oldest),
                    AcquisitionMetadata(entity="B", cached_at=newest),
                    AcquisitionMetadata(entity="C"))
    result = FetchResult(pd.DataFrame(), "test", "test", True, newest,
                         requested_entities=("A", "B", "C", "D"), acquisitions=acquisitions,
                         failures=(FetchFailure(entity="D", message="unavailable", error_type="Timeout"),))
    assert result.complete is False
    assert result.cached_at == oldest
    assert result.acquisitions[2].cached_at is None
    assert result.failures[0].entity == "D"


def test_legacy_pit_flag_does_not_claim_a_revision_history():
    from alphaforge.data.types import FetchResult

    result = FetchResult(pd.DataFrame(), "test", "test", True, None)
    assert result.result_kind is None
    assert result.availability_basis == "none"
    assert result.complete is True


def test_explicit_observations_do_not_claim_historical_availability():
    from alphaforge.data.types import FetchResult

    result = FetchResult(pd.DataFrame({"obs_date": [date(2025, 1, 1)], "value": [1.0]}),
                         "test", "test", False, None, result_kind="observations")
    with pytest.raises(ValueError, match="asof"):
        result.as_timeseries(asof=date(2025, 1, 2))


def test_selected_snapshot_rejects_later_cutoff_but_accepts_same_day():
    from alphaforge.data.types import FetchResult

    result = FetchResult(
        pd.DataFrame({"obs_date": pd.to_datetime(["2025-01-01"]),
                      "asof_utc": pd.to_datetime(["2025-02-01T18:00Z"], utc=True), "value": [2.0]}),
        "fred", "macro.fred", True, None, result_kind="snapshot",
        availability_precision="date", snapshot_asof=pd.Timestamp("2025-02-01", tz="UTC"),
    )
    with pytest.raises(ValueError, match="snapshot"):
        result.as_timeseries(asof=date(2025, 3, 1))
    assert result.as_timeseries(asof=pd.Timestamp("2025-02-01T15:00Z")).tolist() == [2.0]
