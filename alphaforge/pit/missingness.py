"""Missingness taxonomy and classification for nowcasting panels.

Provides a shared classifier so Phase 3 (views) and Phase 4 (imputation)
use identical logic to determine *why* a cell is NaN.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphaforge.pit.release_rules import ReleaseRule


class MissingnessReason(str, Enum):
    """Why a panel cell is NaN."""

    STRUCTURAL = "structural"
    """Frequency mismatch — e.g. quarterly obs_date in a monthly panel."""

    FUTURE = "future"
    """The observation period has not yet ended (obs_date > asof_date)."""

    RAGGED_EDGE = "ragged_edge"
    """Not yet published — the expected release date is after asof_date."""

    TRUE_MISSING = "true_missing"
    """Should have been available by now but is absent — data issue."""


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
    """Classify why a panel cell is NaN.

    Evaluation follows a strict order:

    1. **STRUCTURAL** — frequency mismatch (e.g. quarterly series in a
       monthly panel has NaN for non-quarter-end months).
    2. **FUTURE** — ``obs_date`` is after ``asof_date``.
    3. **RAGGED_EDGE vs TRUE_MISSING** — compare ``asof_date`` against the
       expected (or realized) publication date:

       * If a ``realized_release_date`` is provided (from AlphaForge PIT
         storage), it takes precedence over the expected schedule.
       * Otherwise, ``release_rule.expected_release_date(obs_date)`` is used.
       * As a last fallback, ``publication_lag_months`` gives a coarse
         estimate.
       * If none are available, default to ``TRUE_MISSING``.

    Parameters
    ----------
    obs_date
        The observation / reference-period end date of the NaN cell.
    asof_date
        The "as-of" date of the vintage snapshot.
    series_frequency
        Frequency code of the series (``"Q"``, ``"M"``, ``"W"``, etc.).
    panel_frequency
        Frequency code of the panel grid (default ``"M"`` for monthly).
    release_rule
        Optional structured release schedule from catalog metadata.
    publication_lag_months
        Optional simple lag heuristic (backward compat).
    realized_release_date
        Optional realized publication date from AlphaForge.  When provided
        this always takes precedence over the expected schedule.
    """
    # 1. Structural: quarterly series in a monthly panel at non-quarter-end
    if _is_structural(obs_date, series_frequency, panel_frequency):
        return MissingnessReason.STRUCTURAL

    # 2. Future: observation period hasn't ended
    if obs_date > asof_date:
        return MissingnessReason.FUTURE

    # 3. Ragged edge vs true missing
    expected = _resolve_expected_date(
        obs_date,
        release_rule=release_rule,
        publication_lag_months=publication_lag_months,
        realized_release_date=realized_release_date,
    )

    if expected is None:
        # No release information at all — assume it should be here
        return MissingnessReason.TRUE_MISSING

    if asof_date < expected:
        return MissingnessReason.RAGGED_EDGE

    return MissingnessReason.TRUE_MISSING


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_QUARTER_END_MONTHS = {3, 6, 9, 12}


def _is_structural(
    obs_date: date, series_frequency: str, panel_frequency: str
) -> bool:
    """Return True when the NaN is a frequency-mismatch artifact."""
    sf = series_frequency.upper()
    pf = panel_frequency.upper()

    if sf == "Q" and pf == "M":
        # Quarterly series only has values at quarter-end months
        return obs_date.month not in _QUARTER_END_MONTHS

    return False


def _resolve_expected_date(
    obs_date: date,
    *,
    release_rule: ReleaseRule | None,
    publication_lag_months: int | None,
    realized_release_date: date | None,
) -> date | None:
    """Determine the best-available expected publication date.

    Precedence:
    1. realized_release_date (from AlphaForge PIT storage)
    2. release_rule.expected_release_date(obs_date)
    3. obs_date + publication_lag_months (coarse fallback)
    4. None (no information)
    """
    if realized_release_date is not None:
        return realized_release_date

    if release_rule is not None:
        return release_rule.expected_release_date(obs_date)

    if publication_lag_months is not None:
        m = obs_date.month + publication_lag_months
        y = obs_date.year + (m - 1) // 12
        m = (m - 1) % 12 + 1
        return date(y, m, 1)

    return None
