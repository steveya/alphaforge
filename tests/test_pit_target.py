"""Tests for alphaforge.pit.target — TargetPolicy and release resolution."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from alphaforge.pit.target import (
    TargetPolicy,
    get_quarterly_release_observation_stream,
    list_quarterly_target_releases_asof,
    list_quarterly_target_releases_asof_multi,
    quarter_end_date,
    quarter_obs_date,
    quarter_start_date,
    resolve_quarterly_final_target,
    resolve_target_from_releases,
    resolve_target_obs_date_anchor,
)

# ---------------------------------------------------------------------------
# TargetPolicy dataclass
# ---------------------------------------------------------------------------


class TestTargetPolicy:
    def test_latest_available(self):
        p = TargetPolicy(mode="latest_available")
        assert p.mode == "latest_available"
        assert p.nth is None
        assert p.max_release_rank is None

    def test_nth_release(self):
        p = TargetPolicy(mode="nth_release", nth=3, max_release_rank=3)
        assert p.nth == 3
        assert p.max_release_rank == 3

    def test_next_release(self):
        p = TargetPolicy(mode="next_release")
        assert p.mode == "next_release"


# ---------------------------------------------------------------------------
# Quarter calendar helpers
# ---------------------------------------------------------------------------


class TestQuarterCalendar:
    @pytest.mark.parametrize(
        ("ref", "expected"),
        [
            ("2025Q1", date(2025, 3, 31)),
            ("2025Q2", date(2025, 6, 30)),
            ("2025Q3", date(2025, 9, 30)),
            ("2025Q4", date(2025, 12, 31)),
            ("2024q2", date(2024, 6, 30)),
        ],
    )
    def test_quarter_end_date(self, ref, expected):
        assert quarter_end_date(ref) == expected

    def test_quarter_end_date_pd_period(self):
        period = pd.Period("2025Q1", freq="Q")
        assert quarter_end_date(period) == date(2025, 3, 31)

    def test_quarter_end_date_invalid(self):
        with pytest.raises(ValueError, match="Expected quarterly"):
            quarter_end_date("2025-01")

    @pytest.mark.parametrize(
        ("ref", "expected"),
        [
            ("2025Q1", date(2025, 1, 1)),
            ("2025Q2", date(2025, 4, 1)),
            ("2025Q3", date(2025, 7, 1)),
            ("2025Q4", date(2025, 10, 1)),
        ],
    )
    def test_quarter_start_date(self, ref, expected):
        assert quarter_start_date(ref) == expected

    def test_quarter_obs_date_end(self):
        assert quarter_obs_date("2025Q1", obs_date_anchor="end") == date(2025, 3, 31)

    def test_quarter_obs_date_start(self):
        assert quarter_obs_date("2025Q1", obs_date_anchor="start") == date(2025, 1, 1)

    def test_quarter_obs_date_invalid_anchor(self):
        with pytest.raises(ValueError, match="obs_date_anchor"):
            quarter_obs_date("2025Q1", obs_date_anchor="middle")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# resolve_target_obs_date_anchor
# ---------------------------------------------------------------------------


class TestResolveTargetObsDateAnchor:
    def test_explicit_end(self):
        assert resolve_target_obs_date_anchor("end", target_series_key="X", catalog=None) == "end"

    def test_explicit_start(self):
        assert (
            resolve_target_obs_date_anchor("start", target_series_key="X", catalog=None) == "start"
        )

    def test_auto_fallback_no_catalog(self):
        assert resolve_target_obs_date_anchor("auto", target_series_key="X", catalog=None) == "end"

    def test_auto_from_catalog(self):
        catalog = MagicMock()
        meta = MagicMock()
        meta.obs_date_anchor = "start"
        catalog.get.return_value = meta
        result = resolve_target_obs_date_anchor("auto", target_series_key="GDP", catalog=catalog)
        assert result == "start"

    def test_auto_catalog_missing_key(self):
        catalog = MagicMock()
        catalog.get.return_value = None
        result = resolve_target_obs_date_anchor("auto", target_series_key="GDP", catalog=catalog)
        assert result == "end"

    def test_invalid_anchor(self):
        with pytest.raises(ValueError, match="target_obs_date_anchor"):
            resolve_target_obs_date_anchor("foo", target_series_key="X", catalog=None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# resolve_target_from_releases
# ---------------------------------------------------------------------------


def _make_releases(values, asof_dates=None):
    """Helper to build a releases DataFrame."""
    n = len(values)
    if asof_dates is None:
        base = pd.Timestamp("2024-04-15", tz="UTC")
        asof_dates = [base + pd.Timedelta(days=30 * i) for i in range(n)]
    return pd.DataFrame(
        {
            "obs_date": [pd.Timestamp("2024-03-31", tz="UTC")] * n,
            "asof_utc": asof_dates,
            "value": values,
        }
    )


class TestResolveTargetFromReleases:
    def test_latest_available_single(self):
        releases = _make_releases([1.5])
        val, meta = resolve_target_from_releases(releases, TargetPolicy(mode="latest_available"))
        assert val == 1.5
        assert meta["selected_release_rank"] == 1
        assert meta["n_releases_available"] == 1

    def test_latest_available_multiple(self):
        releases = _make_releases([1.5, 2.0, 2.5])
        val, meta = resolve_target_from_releases(releases, TargetPolicy(mode="latest_available"))
        assert val == 2.5
        assert meta["selected_release_rank"] == 3

    def test_latest_available_all_null(self):
        releases = _make_releases([None, None])
        val, meta = resolve_target_from_releases(releases, TargetPolicy(mode="latest_available"))
        assert val is None
        assert meta["selected_release_rank"] == 1

    def test_nth_release_hit(self):
        releases = _make_releases([1.5, 2.0, 2.5])
        val, meta = resolve_target_from_releases(releases, TargetPolicy(mode="nth_release", nth=2))
        assert val == 2.0
        assert meta["selected_release_rank"] == 2

    def test_nth_release_miss(self):
        releases = _make_releases([1.5])
        val, meta = resolve_target_from_releases(releases, TargetPolicy(mode="nth_release", nth=3))
        assert val is None

    def test_nth_release_no_nth(self):
        releases = _make_releases([1.5])
        with pytest.raises(ValueError, match="nth must be set"):
            resolve_target_from_releases(releases, TargetPolicy(mode="nth_release"))

    def test_nth_release_invalid_nth(self):
        releases = _make_releases([1.5])
        with pytest.raises(ValueError, match="nth must be >= 1"):
            resolve_target_from_releases(releases, TargetPolicy(mode="nth_release", nth=0))

    def test_next_release(self):
        releases = _make_releases([1.5, 2.0])
        val, meta = resolve_target_from_releases(releases, TargetPolicy(mode="next_release"))
        assert val is None
        assert meta["target_release_rank"] == 3

    def test_next_release_with_cap(self):
        releases = _make_releases([1.5, 2.0])
        val, meta = resolve_target_from_releases(
            releases, TargetPolicy(mode="next_release", max_release_rank=2)
        )
        assert meta["target_release_rank"] == 2

    def test_max_release_rank(self):
        releases = _make_releases([1.5, 2.0, 2.5, 3.0])
        val, meta = resolve_target_from_releases(
            releases, TargetPolicy(mode="latest_available", max_release_rank=2)
        )
        assert val == 2.0
        assert meta["n_releases_available"] == 2

    def test_max_release_rank_invalid(self):
        releases = _make_releases([1.5])
        with pytest.raises(ValueError, match="max_release_rank must be >= 1"):
            resolve_target_from_releases(
                releases, TargetPolicy(mode="latest_available", max_release_rank=0)
            )

    def test_empty_releases(self):
        releases = pd.DataFrame(
            {
                "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "value": pd.Series(dtype="float64"),
            }
        )
        val, meta = resolve_target_from_releases(releases, TargetPolicy(mode="latest_available"))
        assert val is None
        assert meta["n_raw_releases_available"] == 0

    def test_unknown_mode(self):
        releases = _make_releases([1.5])
        with pytest.raises(ValueError, match="Unknown target policy mode"):
            resolve_target_from_releases(releases, TargetPolicy(mode="bad"))  # type: ignore[arg-type]

    def test_nan_skipped(self):
        releases = _make_releases([float("nan"), 2.0, float("nan")])
        val, meta = resolve_target_from_releases(releases, TargetPolicy(mode="latest_available"))
        assert val == 2.0
        assert meta["n_nonnull_releases_available"] == 1


# ---------------------------------------------------------------------------
# Adapter-dependent functions (mock PITAdapter)
# ---------------------------------------------------------------------------


def _mock_adapter(releases_df):
    adapter = MagicMock(spec=["list_pit_observations_asof"])
    adapter.list_pit_observations_asof.return_value = releases_df
    return adapter


class TestListQuarterlyTargetReleasesAsof:
    def test_basic(self):
        releases = _make_releases([1.5, 2.0])
        adapter = _mock_adapter(releases)
        result = list_quarterly_target_releases_asof(
            adapter, series_key="GDP", ref_quarter="2024Q1", asof_date=date(2024, 12, 31)
        )
        assert len(result) == 2
        assert list(result.columns) == ["obs_date", "asof_utc", "value"]
        adapter.list_pit_observations_asof.assert_called_once()

    def test_sorted_by_asof(self):
        releases = pd.DataFrame(
            {
                "obs_date": [pd.Timestamp("2024-03-31", tz="UTC")] * 2,
                "asof_utc": [
                    pd.Timestamp("2024-06-01", tz="UTC"),
                    pd.Timestamp("2024-05-01", tz="UTC"),
                ],
                "value": [2.0, 1.5],
            }
        )
        adapter = _mock_adapter(releases)
        result = list_quarterly_target_releases_asof(
            adapter, series_key="GDP", ref_quarter="2024Q1", asof_date=date(2024, 12, 31)
        )
        assert result.iloc[0]["value"] == 1.5
        assert result.iloc[1]["value"] == 2.0


class TestListQuarterlyMulti:
    def test_fallback_to_single(self):
        releases = _make_releases([1.5])
        adapter = MagicMock(spec=["list_pit_observations_asof"])
        adapter.list_pit_observations_asof.return_value = releases
        result = list_quarterly_target_releases_asof_multi(
            adapter,
            series_key="GDP",
            ref_quarters=["2024Q1", "2024Q2"],
            asof_date=date(2024, 12, 31),
        )
        assert "2024Q1" in result
        assert "2024Q2" in result
        assert adapter.list_pit_observations_asof.call_count == 2

    def test_empty_refs(self):
        adapter = MagicMock()
        result = list_quarterly_target_releases_asof_multi(
            adapter, series_key="GDP", ref_quarters=[], asof_date=date(2024, 12, 31)
        )
        assert result == {}


class TestResolveQuarterlyFinalTarget:
    def test_basic(self):
        releases = _make_releases([1.5, 2.0, 2.5])
        adapter = _mock_adapter(releases)
        val, meta = resolve_quarterly_final_target(
            adapter, series_key="GDP", ref_quarter="2024Q1", evaluation_asof_date=date(2024, 12, 31)
        )
        assert val == 2.5
        assert meta["selected_release_rank"] == 3


class TestGetQuarterlyReleaseObservationStream:
    def test_basic(self):
        releases = _make_releases([1.5, 2.0])
        adapter = _mock_adapter(releases)
        stream = get_quarterly_release_observation_stream(
            adapter, series_key="GDP", ref_quarter="2024Q1", asof_date=date(2024, 12, 31)
        )
        assert list(stream.columns) == [
            "series_key",
            "ref_quarter",
            "obs_date",
            "release_rank",
            "asof_utc",
            "value",
        ]
        assert len(stream) == 2
        assert list(stream["release_rank"]) == [1, 2]
        assert stream.iloc[0]["series_key"] == "GDP"
        assert stream.iloc[0]["ref_quarter"] == "2024Q1"

    def test_empty(self):
        empty_df = pd.DataFrame(
            {
                "obs_date": pd.Series(dtype="datetime64[ns, UTC]"),
                "asof_utc": pd.Series(dtype="datetime64[ns, UTC]"),
                "value": pd.Series(dtype="float64"),
            }
        )
        adapter = _mock_adapter(empty_df)
        stream = get_quarterly_release_observation_stream(
            adapter, series_key="GDP", ref_quarter="2024Q1", asof_date=date(2024, 12, 31)
        )
        assert stream.empty
        assert "release_rank" in stream.columns

    def test_max_release_rank(self):
        releases = _make_releases([1.5, 2.0, 2.5])
        adapter = _mock_adapter(releases)
        stream = get_quarterly_release_observation_stream(
            adapter,
            series_key="GDP",
            ref_quarter="2024Q1",
            asof_date=date(2024, 12, 31),
            max_release_rank=2,
        )
        assert len(stream) == 2

    def test_max_release_rank_invalid(self):
        releases = _make_releases([1.5])
        adapter = _mock_adapter(releases)
        with pytest.raises(ValueError, match="max_release_rank must be >= 1"):
            get_quarterly_release_observation_stream(
                adapter,
                series_key="GDP",
                ref_quarter="2024Q1",
                asof_date=date(2024, 12, 31),
                max_release_rank=0,
            )
