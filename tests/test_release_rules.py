"""Tests for release rule schedule computations."""

from datetime import date

import pytest

from alphaforge.pit.release_rules import (
    RULE_REGISTRY,
    CalendarDay,
    CustomRule,
    FixedLagMonths,
    NthBusinessDay,
    NthWeekday,
    QuarterlyRelease,
    ReleaseRule,
    WeeklyRelease,
)


class TestRuleRegistry:
    def test_all_rules_registered(self):
        expected = {
            "nth_business_day",
            "nth_weekday",
            "calendar_day",
            "fixed_lag_months",
            "quarterly_release",
            "weekly",
            "custom",
        }
        assert set(RULE_REGISTRY.keys()) == expected

    def test_round_trip_from_dict(self):
        rule = NthBusinessDay(n=3, anchor="following_month")
        d = rule.to_dict()
        assert d["type"] == "nth_business_day"
        restored = ReleaseRule.from_dict(d)
        assert isinstance(restored, NthBusinessDay)
        assert restored.n == 3

    def test_from_dict_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown release rule type"):
            ReleaseRule.from_dict({"type": "nonexistent"})


class TestNthBusinessDay:
    def test_first_business_day_following_month(self):
        rule = NthBusinessDay(n=1, anchor="following_month")
        result = rule.expected_release_date(date(2025, 1, 31))
        assert result == date(2025, 2, 3)

    def test_third_business_day(self):
        rule = NthBusinessDay(n=3, anchor="following_month")
        result = rule.expected_release_date(date(2024, 12, 31))
        assert result == date(2025, 1, 6)


class TestQuarterlyRelease:
    def test_advance_release(self):
        rule = QuarterlyRelease(advance_lag_months=1, preliminary_lag_months=2, final_lag_months=3)
        result = rule.expected_release_date(date(2024, 12, 31), release_number=1)
        assert result == date(2025, 1, 1)

    def test_final_release(self):
        rule = QuarterlyRelease(advance_lag_months=1, preliminary_lag_months=2, final_lag_months=3)
        result = rule.expected_release_date(date(2024, 12, 31), release_number=3)
        assert result == date(2025, 3, 1)


class TestWeeklyRelease:
    def test_five_day_lag(self):
        rule = WeeklyRelease(release_weekday="Thursday", lag_days=5)
        result = rule.expected_release_date(date(2025, 1, 4))
        assert result == date(2025, 1, 9)


class TestCustomRule:
    def test_serialization(self):
        rule = CustomRule(description="test", approximate_lag_months=1)
        d = rule.to_dict()
        restored = ReleaseRule.from_dict(d)
        assert isinstance(restored, CustomRule)
