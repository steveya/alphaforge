"""Missingness taxonomy and classification for temporal semantics."""

from __future__ import annotations

from datetime import date
from enum import Enum

from .release_rules import ReleaseRule


class MissingnessReason(str, Enum):
    """Why a point-in-time panel cell is missing."""

    STRUCTURAL = "structural"
    FUTURE = "future"
    RAGGED_EDGE = "ragged_edge"
    TRUE_MISSING = "true_missing"


def classify_missingness(
    *,
    obs_date: date,
    asof_date: date,
    series_frequency: str,
    panel_frequency: str = "M",
    release_rule: ReleaseRule | None = None,
    publication_lag_months: int | None = None,
    realized_release_date: date | None = None,
) -> MissingnessReason:
    """Classify why a cell is missing at a given as-of date."""
    if _is_structural(obs_date, series_frequency, panel_frequency):
        return MissingnessReason.STRUCTURAL

    if obs_date > asof_date:
        return MissingnessReason.FUTURE

    expected_release = _resolve_expected_date(
        obs_date,
        release_rule=release_rule,
        publication_lag_months=publication_lag_months,
        realized_release_date=realized_release_date,
    )
    if expected_release is None:
        return MissingnessReason.TRUE_MISSING
    if asof_date < expected_release:
        return MissingnessReason.RAGGED_EDGE
    return MissingnessReason.TRUE_MISSING


_QUARTER_END_MONTHS = {3, 6, 9, 12}


def _is_structural(
    obs_date: date,
    series_frequency: str,
    panel_frequency: str,
) -> bool:
    series_freq = series_frequency.upper()
    panel_freq = panel_frequency.upper()
    if series_freq == "Q" and panel_freq == "M":
        return obs_date.month not in _QUARTER_END_MONTHS
    return False


def _resolve_expected_date(
    obs_date: date,
    *,
    release_rule: ReleaseRule | None,
    publication_lag_months: int | None,
    realized_release_date: date | None,
) -> date | None:
    if realized_release_date is not None:
        return realized_release_date
    if release_rule is not None:
        return release_rule.expected_release_date(obs_date)
    if publication_lag_months is not None:
        month = obs_date.month + publication_lag_months
        year = obs_date.year + (month - 1) // 12
        month = (month - 1) % 12 + 1
        return date(year, month, 1)
    return None


__all__ = ["MissingnessReason", "classify_missingness"]
