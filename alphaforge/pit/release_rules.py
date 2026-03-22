"""Release schedule rules for macro series publication timing.

Each rule class models a real-world release schedule pattern and exposes
``expected_release_date(obs_date)`` to compute the day a value is expected
to become publicly available.

These rules are an *expectation layer* — realized PIT timestamps from
AlphaForge always take precedence when available.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

_US_BD = CustomBusinessDay(calendar=USFederalHolidayCalendar())

# ---------------------------------------------------------------------------
# Registry for tagged-union YAML parsing (populated by @_register below)
# ---------------------------------------------------------------------------

RULE_REGISTRY: dict[str, type] = {}


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseRule(ABC):
    """Base class for publication schedule rules."""

    rule_type: str = ""  # overridden by subclass class-var

    @abstractmethod
    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        """Return the expected publication date for a given observation date.

        Parameters
        ----------
        obs_date : date
            The observation / reference-period end date.
        release_number : int | None
            For multi-release series (e.g. GDP), selects which release
            (1 = advance, 2 = preliminary, …).  Ignored by single-release
            rules.
        """

    # Serialization helpers for round-trip YAML
    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict suitable for YAML output."""
        d: dict[str, Any] = {"type": self.rule_type}
        for k, v in self.__dict__.items():
            if k == "rule_type":
                continue
            d[k] = v
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> ReleaseRule:
        """Reconstruct a ReleaseRule from a YAML-style dict."""
        kwargs = dict(d)
        type_key = kwargs.pop("type")
        cls = RULE_REGISTRY.get(type_key)
        if cls is None:
            raise ValueError(
                f"Unknown release rule type '{type_key}'. "
                f"Known types: {sorted(RULE_REGISTRY)}"
            )
        return cls(**kwargs)


def _register(cls: type[ReleaseRule]) -> type[ReleaseRule]:
    """Decorator that adds a ReleaseRule subclass to the registry."""
    RULE_REGISTRY[cls.rule_type] = cls
    return cls


# ---------------------------------------------------------------------------
# Concrete rule classes
# ---------------------------------------------------------------------------


def _month_start(ref: date, offset_months: int) -> date:
    """Return the 1st day of the month that is *offset_months* after *ref*."""
    m = ref.month + offset_months
    y = ref.year + (m - 1) // 12
    m = (m - 1) % 12 + 1
    return date(y, m, 1)


def _nth_business_day(anchor: date, n: int) -> date:
    """Return the *n*-th US business day on or after *anchor*."""
    ts = pd.Timestamp(anchor)
    # Move to the n-th business day (1-indexed)
    result = ts + (n - 1) * _US_BD
    # If anchor itself is not a business day, the offset already handles it
    # but we need to make sure we start counting from the first valid day
    first_bd = ts + 0 * _US_BD  # snap to first business day on or after
    result = first_bd + (n - 1) * _US_BD
    return result.date()


@_register
@dataclass(frozen=True)
class NthBusinessDay(ReleaseRule):
    """Published on the n-th US business day relative to an anchor month.

    Example: Employment Situation — 1st business day of the following month.
    """

    rule_type: str = "nth_business_day"
    n: int = 1
    anchor: str = "following_month"  # "following_month" | "same_month"

    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        if self.anchor == "following_month":
            start = _month_start(obs_date, 1)
        elif self.anchor == "same_month":
            start = _month_start(obs_date, 0)
        else:
            raise ValueError(f"Unknown anchor: {self.anchor}")
        return _nth_business_day(start, self.n)


@_register
@dataclass(frozen=True)
class NthWeekday(ReleaseRule):
    """Published on the n-th occurrence of a weekday in the anchor month.

    Example: CPI — typically released around the 10th–15th business day
    of the following month, often modeled as ~2nd or 3rd week.
    """

    rule_type: str = "nth_weekday"
    n: int = 1
    weekday: str = "Friday"  # Monday–Sunday
    anchor: str = "following_month"

    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        _weekday_map = {
            "Monday": 0,
            "Tuesday": 1,
            "Wednesday": 2,
            "Thursday": 3,
            "Friday": 4,
            "Saturday": 5,
            "Sunday": 6,
        }
        target_wd = _weekday_map[self.weekday]

        if self.anchor == "following_month":
            first = _month_start(obs_date, 1)
        elif self.anchor == "same_month":
            first = _month_start(obs_date, 0)
        else:
            raise ValueError(f"Unknown anchor: {self.anchor}")

        # Find the first occurrence of target weekday in the month
        days_ahead = (target_wd - first.weekday()) % 7
        first_occurrence = first + pd.Timedelta(days=days_ahead)
        # Advance to the n-th occurrence
        result = first_occurrence + pd.Timedelta(weeks=self.n - 1)
        return result.date() if isinstance(result, pd.Timestamp) else result


@_register
@dataclass(frozen=True)
class CalendarDay(ReleaseRule):
    """Published on a specific calendar day of the anchor month.

    Example: "15th of the following month."
    """

    rule_type: str = "calendar_day"
    day: int = 15
    anchor: str = "following_month"

    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        if self.anchor == "following_month":
            start = _month_start(obs_date, 1)
        elif self.anchor == "same_month":
            start = _month_start(obs_date, 0)
        elif self.anchor == "two_months_later":
            start = _month_start(obs_date, 2)
        else:
            raise ValueError(f"Unknown anchor: {self.anchor}")
        return start.replace(day=self.day)


@_register
@dataclass(frozen=True)
class FixedLagMonths(ReleaseRule):
    """Published a fixed number of months after the observation month.

    Fallback rule for series where the exact schedule is not worth encoding.
    """

    rule_type: str = "fixed_lag_months"
    months: int = 1

    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        return _month_start(obs_date, self.months)


@_register
@dataclass(frozen=True)
class QuarterlyRelease(ReleaseRule):
    """GDP-style multi-release schedule.

    Models advance / preliminary / final releases with configurable lag
    (in months after the quarter-end).
    """

    rule_type: str = "quarterly_release"
    advance_lag_months: int = 1
    preliminary_lag_months: int = 2
    final_lag_months: int = 3

    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        rn = release_number or 1
        if rn == 1:
            lag = self.advance_lag_months
        elif rn == 2:
            lag = self.preliminary_lag_months
        else:
            lag = self.final_lag_months
        return _month_start(obs_date, lag)


@_register
@dataclass(frozen=True)
class WeeklyRelease(ReleaseRule):
    """Weekly series released on a fixed weekday with a lag.

    Example: Initial Claims — released Thursday for the prior Saturday week.
    """

    rule_type: str = "weekly"
    release_weekday: str = "Thursday"
    lag_days: int = 5

    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        return obs_date + pd.Timedelta(days=self.lag_days)


@_register
@dataclass(frozen=True)
class CustomRule(ReleaseRule):
    """Free-text description for exotic or not-yet-modeled schedules."""

    rule_type: str = "custom"
    description: str = ""
    approximate_lag_months: int = 1

    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        return _month_start(obs_date, self.approximate_lag_months)
