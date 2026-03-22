"""Tests for vintage view resolvers."""

from __future__ import annotations

from datetime import date

import pytest

from alphaforge.pit.resolvers import (
    FrozenResolver,
    LatestResolver,
    RealtimeResolver,
    VintageResolver,
)
from alphaforge.pit.views import VintageView

# ---------------------------------------------------------------------------
# VintageView value object
# ---------------------------------------------------------------------------


class TestVintageView:
    def test_realtime_factory(self) -> None:
        v = VintageView.realtime()
        assert v.mode == "realtime"
        assert v.n_releases == 3  # default, unused

    def test_latest_factory(self) -> None:
        v = VintageView.latest()
        assert v.mode == "latest"

    def test_frozen_factory_default(self) -> None:
        v = VintageView.frozen()
        assert v.mode == "frozen"
        assert v.n_releases == 3

    def test_frozen_factory_custom(self) -> None:
        v = VintageView.frozen(n=5)
        assert v.mode == "frozen"
        assert v.n_releases == 5

    def test_frozen_immutable(self) -> None:
        v = VintageView.frozen()
        with pytest.raises(AttributeError):
            v.mode = "latest"  # type: ignore[misc]

    def test_equality(self) -> None:
        assert VintageView.realtime() == VintageView(mode="realtime")
        assert VintageView.frozen(3) == VintageView.frozen(3)
        assert VintageView.frozen(3) != VintageView.frozen(5)


# ---------------------------------------------------------------------------
# RealtimeResolver
# ---------------------------------------------------------------------------


class TestRealtimeResolver:
    def test_returns_requested_asof_unchanged(self) -> None:
        resolver = RealtimeResolver()
        asof = date(2024, 6, 15)
        result = resolver.resolve(
            series_key="GDP",
            obs_date=date(2024, 3, 31),
            requested_asof=asof,
            has_pit=True,
        )
        assert result == asof

    def test_returns_requested_asof_for_non_pit(self) -> None:
        resolver = RealtimeResolver()
        asof = date(2024, 6, 15)
        result = resolver.resolve(
            series_key="SP500",
            obs_date=date(2024, 3, 31),
            requested_asof=asof,
            has_pit=False,
        )
        assert result == asof

    def test_view_property(self) -> None:
        resolver = RealtimeResolver()
        assert resolver.view == VintageView.realtime()

    def test_satisfies_protocol(self) -> None:
        assert isinstance(RealtimeResolver(), VintageResolver)


# ---------------------------------------------------------------------------
# LatestResolver
# ---------------------------------------------------------------------------


class TestLatestResolver:
    def test_returns_sentinel_for_pit_series(self) -> None:
        resolver = LatestResolver()
        result = resolver.resolve(
            series_key="GDP",
            obs_date=date(2024, 3, 31),
            requested_asof=date(2024, 6, 15),
            has_pit=True,
        )
        assert result == date(2099, 12, 31)

    def test_returns_requested_asof_for_non_pit(self) -> None:
        resolver = LatestResolver()
        asof = date(2024, 6, 15)
        result = resolver.resolve(
            series_key="SP500",
            obs_date=date(2024, 3, 31),
            requested_asof=asof,
            has_pit=False,
        )
        assert result == asof

    def test_view_property(self) -> None:
        resolver = LatestResolver()
        assert resolver.view == VintageView.latest()

    def test_satisfies_protocol(self) -> None:
        assert isinstance(LatestResolver(), VintageResolver)


# ---------------------------------------------------------------------------
# FrozenResolver
# ---------------------------------------------------------------------------


@pytest.fixture()
def revision_map() -> dict[tuple[str, date], list[date]]:
    """Sample revision map: GDP Q1 2024 was released on three vintages."""
    return {
        ("GDP", date(2024, 3, 31)): [
            date(2024, 4, 25),  # advance
            date(2024, 5, 30),  # second
            date(2024, 6, 27),  # third
        ],
        ("GDP", date(2023, 12, 31)): [
            date(2024, 1, 25),  # advance
            date(2024, 2, 28),  # second
        ],
        ("PAYEMS", date(2024, 3, 31)): [
            date(2024, 4, 5),   # initial
            date(2024, 5, 3),   # revised
            date(2024, 6, 7),   # revised again
            date(2024, 7, 5),   # revised again
        ],
    }


class TestFrozenResolver:
    def test_returns_nth_vintage(self, revision_map: dict) -> None:
        resolver = FrozenResolver(revision_map=revision_map, n_releases=3)
        result = resolver.resolve(
            series_key="GDP",
            obs_date=date(2024, 3, 31),
            requested_asof=date(2024, 12, 1),
            has_pit=True,
        )
        assert result == date(2024, 6, 27)  # 3rd release

    def test_returns_2nd_release(self, revision_map: dict) -> None:
        resolver = FrozenResolver(revision_map=revision_map, n_releases=2)
        result = resolver.resolve(
            series_key="GDP",
            obs_date=date(2024, 3, 31),
            requested_asof=date(2024, 12, 1),
            has_pit=True,
        )
        assert result == date(2024, 5, 30)  # 2nd release

    def test_returns_1st_release(self, revision_map: dict) -> None:
        resolver = FrozenResolver(revision_map=revision_map, n_releases=1)
        result = resolver.resolve(
            series_key="GDP",
            obs_date=date(2024, 3, 31),
            requested_asof=date(2024, 12, 1),
            has_pit=True,
        )
        assert result == date(2024, 4, 25)  # advance

    def test_falls_back_to_latest_when_fewer_releases(self, revision_map: dict) -> None:
        resolver = FrozenResolver(revision_map=revision_map, n_releases=3)
        result = resolver.resolve(
            series_key="GDP",
            obs_date=date(2023, 12, 31),
            requested_asof=date(2024, 12, 1),
            has_pit=True,
        )
        # Only 2 releases exist, so returns the 2nd (latest available)
        assert result == date(2024, 2, 28)

    def test_returns_requested_asof_for_non_pit(self, revision_map: dict) -> None:
        resolver = FrozenResolver(revision_map=revision_map, n_releases=3)
        asof = date(2024, 6, 15)
        result = resolver.resolve(
            series_key="SP500",
            obs_date=date(2024, 3, 31),
            requested_asof=asof,
            has_pit=False,
        )
        assert result == asof

    def test_returns_requested_asof_for_unknown_series(self, revision_map: dict) -> None:
        resolver = FrozenResolver(revision_map=revision_map, n_releases=3)
        asof = date(2024, 6, 15)
        result = resolver.resolve(
            series_key="UNKNOWN",
            obs_date=date(2024, 3, 31),
            requested_asof=asof,
            has_pit=True,
        )
        # Not in revision_map → fall through to requested_asof
        assert result == asof

    def test_view_property(self, revision_map: dict) -> None:
        resolver = FrozenResolver(revision_map=revision_map, n_releases=5)
        assert resolver.view == VintageView.frozen(n=5)

    def test_satisfies_protocol(self, revision_map: dict) -> None:
        assert isinstance(
            FrozenResolver(revision_map=revision_map, n_releases=3),
            VintageResolver,
        )

    def test_different_series_different_vintages(self, revision_map: dict) -> None:
        resolver = FrozenResolver(revision_map=revision_map, n_releases=3)
        gdp_result = resolver.resolve(
            series_key="GDP",
            obs_date=date(2024, 3, 31),
            requested_asof=date(2024, 12, 1),
            has_pit=True,
        )
        payems_result = resolver.resolve(
            series_key="PAYEMS",
            obs_date=date(2024, 3, 31),
            requested_asof=date(2024, 12, 1),
            has_pit=True,
        )
        assert gdp_result == date(2024, 6, 27)
        assert payems_result == date(2024, 6, 7)
        assert gdp_result != payems_result

    def test_lookahead_guard_filters_future_vintages(self, revision_map: dict) -> None:
        """FrozenResolver must not return vintages after requested_asof."""
        resolver = FrozenResolver(revision_map=revision_map, n_releases=3)
        # requested_asof is before the 3rd release (2024-06-27),
        # so only 2 releases are valid → returns 2nd release
        result = resolver.resolve(
            series_key="GDP",
            obs_date=date(2024, 3, 31),
            requested_asof=date(2024, 6, 1),
            has_pit=True,
        )
        assert result == date(2024, 5, 30)  # 2nd release (latest valid)

    def test_lookahead_guard_no_valid_vintages(self, revision_map: dict) -> None:
        """If requested_asof is before all vintages, fall back to requested_asof."""
        resolver = FrozenResolver(revision_map=revision_map, n_releases=3)
        asof = date(2024, 1, 1)
        result = resolver.resolve(
            series_key="GDP",
            obs_date=date(2024, 3, 31),
            requested_asof=asof,
            has_pit=True,
        )
        assert result == asof  # no valid vintages → fall through

    def test_empty_revision_map(self) -> None:
        resolver = FrozenResolver(revision_map={}, n_releases=3)
        asof = date(2024, 6, 15)
        result = resolver.resolve(
            series_key="GDP",
            obs_date=date(2024, 3, 31),
            requested_asof=asof,
            has_pit=True,
        )
        assert result == asof
