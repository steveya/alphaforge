"""Tests for missingness classification."""
from __future__ import annotations

from datetime import date

from alphaforge.pit.missingness import MissingnessReason, classify_missingness
from alphaforge.pit.release_rules import FixedLagMonths


class TestMissingnessReason:
    def test_structural_quarterly_in_monthly(self) -> None:
        result = classify_missingness(
            obs_date=date(2025, 2, 28),
            asof_date=date(2025, 4, 1),
            series_frequency="Q",
            panel_frequency="M",
        )
        assert result == MissingnessReason.STRUCTURAL

    def test_future_obs(self) -> None:
        result = classify_missingness(
            obs_date=date(2025, 6, 30),
            asof_date=date(2025, 5, 15),
            series_frequency="M",
        )
        assert result == MissingnessReason.FUTURE

    def test_ragged_edge_with_release_rule(self) -> None:
        # obs_date=Mar 31, release expected May 1 (2 month lag), asof=Apr 15 (before release)
        rule = FixedLagMonths(months=2)
        result = classify_missingness(
            obs_date=date(2025, 3, 31),
            asof_date=date(2025, 4, 15),
            series_frequency="M",
            release_rule=rule,
        )
        assert result == MissingnessReason.RAGGED_EDGE
