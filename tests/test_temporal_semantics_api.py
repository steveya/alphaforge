"""Tests for the canonical temporal semantics public surface."""

from __future__ import annotations

from datetime import date

from alphaforge import MissingnessReason, ReleaseRule, classify_missingness
from alphaforge.pit.missingness import MissingnessReason as PitMissingnessReason
from alphaforge.pit.release_rules import ReleaseRule as PitReleaseRule
from alphaforge.time import FixedLagMonths


def test_top_level_temporal_semantics_exports_are_canonical() -> None:
    assert ReleaseRule is PitReleaseRule
    assert MissingnessReason is PitMissingnessReason


def test_time_release_rules_drive_core_missingness_classifier() -> None:
    result = classify_missingness(
        obs_date=date(2025, 3, 31),
        asof_date=date(2025, 4, 15),
        series_frequency="M",
        release_rule=FixedLagMonths(months=2),
    )

    assert result == MissingnessReason.RAGGED_EDGE
