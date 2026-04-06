"""Release schedule rules for publication and availability semantics.

These rules model expected public release timing for a reference-period
observation. They are an expectation layer: realized PIT timestamps always take
precedence when available from source data.
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

RULE_REGISTRY: dict[str, type["ReleaseRule"]] = {}

_WEEKDAY_MAP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


@dataclass(frozen=True)
class ReleaseRule(ABC):
    """Base class for publication schedule rules."""

    rule_type: str = ""

    @abstractmethod
    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        """Return the expected publication date for an observation date."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize the rule to a YAML-friendly mapping."""
        payload: dict[str, Any] = {"type": self.rule_type}
        for key, value in self.__dict__.items():
            if key != "rule_type":
                payload[key] = value
        return payload

    @staticmethod
    def from_dict(payload: dict[str, Any]) -> "ReleaseRule":
        """Reconstruct a rule from a YAML-style mapping."""
        kwargs = dict(payload)
        type_key = kwargs.pop("type")
        cls = RULE_REGISTRY.get(type_key)
        if cls is None:
            raise ValueError(
                f"Unknown release rule type '{type_key}'. "
                f"Known types: {sorted(RULE_REGISTRY)}"
            )
        return cls(**kwargs)


def _register(cls: type[ReleaseRule]) -> type[ReleaseRule]:
    RULE_REGISTRY[cls.rule_type] = cls
    return cls


def _month_start(ref: date, offset_months: int) -> date:
    """Return the first day of the month *offset_months* after *ref*."""
    month = ref.month + offset_months
    year = ref.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    return date(year, month, 1)


def _nth_business_day(anchor: date, n: int) -> date:
    """Return the n-th US business day on or after *anchor*."""
    first_business_day = pd.Timestamp(anchor) + 0 * _US_BD
    return (first_business_day + (n - 1) * _US_BD).date()


def _resolve_weekday(weekday: str) -> int:
    try:
        return _WEEKDAY_MAP[weekday]
    except KeyError as exc:
        raise ValueError(
            f"Unknown weekday: {weekday!r}. Known values: {sorted(_WEEKDAY_MAP)}"
        ) from exc


@_register
@dataclass(frozen=True)
class NthBusinessDay(ReleaseRule):
    """Published on the n-th US business day of an anchor month."""

    rule_type: str = "nth_business_day"
    n: int = 1
    anchor: str = "following_month"

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
    """Published on the n-th occurrence of a weekday in the anchor month."""

    rule_type: str = "nth_weekday"
    n: int = 1
    weekday: str = "Friday"
    anchor: str = "following_month"

    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        target_weekday = _resolve_weekday(self.weekday)

        if self.anchor == "following_month":
            first = _month_start(obs_date, 1)
        elif self.anchor == "same_month":
            first = _month_start(obs_date, 0)
        else:
            raise ValueError(f"Unknown anchor: {self.anchor}")

        days_ahead = (target_weekday - first.weekday()) % 7
        first_occurrence = first + pd.Timedelta(days=days_ahead)
        result = first_occurrence + pd.Timedelta(weeks=self.n - 1)
        return result.date() if isinstance(result, pd.Timestamp) else result


@_register
@dataclass(frozen=True)
class CalendarDay(ReleaseRule):
    """Published on a specific calendar day of the anchor month."""

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
    """Published a fixed number of months after the observation month."""

    rule_type: str = "fixed_lag_months"
    months: int = 1

    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        return _month_start(obs_date, self.months)


@_register
@dataclass(frozen=True)
class QuarterlyRelease(ReleaseRule):
    """GDP-style multi-release schedule."""

    rule_type: str = "quarterly_release"
    advance_lag_months: int = 1
    preliminary_lag_months: int = 2
    final_lag_months: int = 3

    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        release_rank = release_number or 1
        if release_rank == 1:
            lag_months = self.advance_lag_months
        elif release_rank == 2:
            lag_months = self.preliminary_lag_months
        else:
            lag_months = self.final_lag_months
        return _month_start(obs_date, lag_months)


@_register
@dataclass(frozen=True)
class WeeklyRelease(ReleaseRule):
    """Weekly series released on a fixed weekday after a lag."""

    rule_type: str = "weekly"
    release_weekday: str = "Thursday"
    lag_days: int = 5

    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        anchor = obs_date + pd.Timedelta(days=self.lag_days)
        target_weekday = _resolve_weekday(self.release_weekday)
        days_ahead = (target_weekday - anchor.weekday()) % 7
        result = anchor + pd.Timedelta(days=days_ahead)
        return result.date() if isinstance(result, pd.Timestamp) else result


@_register
@dataclass(frozen=True)
class CustomRule(ReleaseRule):
    """Free-text description for schedules that are not yet modeled."""

    rule_type: str = "custom"
    description: str = ""
    approximate_lag_months: int = 1

    def expected_release_date(
        self, obs_date: date, release_number: int | None = None
    ) -> date:
        return _month_start(obs_date, self.approximate_lag_months)


__all__ = [
    "RULE_REGISTRY",
    "ReleaseRule",
    "NthBusinessDay",
    "NthWeekday",
    "CalendarDay",
    "FixedLagMonths",
    "QuarterlyRelease",
    "WeeklyRelease",
    "CustomRule",
]
