"""Tests for alphaforge.pipeline.health module."""
from __future__ import annotations

import pandas as pd

from alphaforge.pipeline.health import (
    CFTC_COT_HEALTH_POLICY,
    DTCC_PPD_HEALTH_POLICY,
    SourceHealthPolicy,
    assess_source_health,
)


def _ts(days_ago: float) -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days_ago)


_POLICY = SourceHealthPolicy(
    expected_cadence=pd.Timedelta(days=7),
    grace_period=pd.Timedelta(days=2),
    stale_threshold=pd.Timedelta(days=14),
    dead_threshold=pd.Timedelta(days=42),
    weight_decay_half_life=pd.Timedelta(days=7),
)
_NOW = pd.Timestamp.now(tz="UTC")


class TestOkStatus:
    def test_ok_status(self) -> None:
        h = assess_source_health("test", _ts(5), _NOW, _POLICY)
        assert h.status == "ok"
        assert h.weight_factor == 1.0


class TestLateStatus:
    def test_late_status(self) -> None:
        # Past cadence+grace (9d) but within stale (14d)
        h = assess_source_health("test", _ts(12), _NOW, _POLICY)
        assert h.status == "late"
        assert h.weight_factor == 1.0


class TestStaleStatus:
    def test_stale_status(self) -> None:
        h = assess_source_health("test", _ts(21), _NOW, _POLICY)
        assert h.status == "stale"
        assert 0 < h.weight_factor < 1.0


class TestStaleWeightDecay:
    def test_stale_weight_decay(self) -> None:
        # At stale_threshold (14d) + 1 half-life (7d) = 21d → weight ≈ 0.5
        h = assess_source_health("test", _ts(21), _NOW, _POLICY)
        assert abs(h.weight_factor - 0.5) < 0.05

        # At stale_threshold + 2 half-lives = 28d → weight ≈ 0.25
        h2 = assess_source_health("test", _ts(28), _NOW, _POLICY)
        assert abs(h2.weight_factor - 0.25) < 0.05


class TestDeadStatus:
    def test_dead_status(self) -> None:
        h = assess_source_health("test", _ts(50), _NOW, _POLICY)
        assert h.status == "dead"
        assert h.weight_factor == 0.0


class TestEmptyStatus:
    def test_empty_status(self) -> None:
        h = assess_source_health("test", None, _NOW, _POLICY)
        assert h.status == "empty"
        assert h.weight_factor == 0.0


class TestCftcPolicyShutdown:
    def test_cftc_policy_shutdown(self) -> None:
        """Simulate a 35-day gov shutdown (2018 style)."""
        policy = CFTC_COT_HEALTH_POLICY
        # Day 5: ok
        h5 = assess_source_health("cftc", _ts(5), _NOW, policy)
        assert h5.status == "ok"
        # Day 15: late
        h15 = assess_source_health("cftc", _ts(15), _NOW, policy)
        assert h15.status == "late"
        # Day 25: stale
        h25 = assess_source_health("cftc", _ts(25), _NOW, policy)
        assert h25.status == "stale"
        assert h25.weight_factor > 0
        # Day 50: still stale (dead_threshold is 56)
        h50 = assess_source_health("cftc", _ts(50), _NOW, policy)
        assert h50.status == "stale"
        # Day 60: dead
        h60 = assess_source_health("cftc", _ts(60), _NOW, policy)
        assert h60.status == "dead"


class TestDtccPolicyWeekend:
    def test_dtcc_policy_weekend(self) -> None:
        """Saturday/Sunday should be ok due to grace period."""
        policy = DTCC_PPD_HEALTH_POLICY
        # 2 days ago (e.g., Friday data on Sunday)
        h = assess_source_health("dtcc", _ts(2), _NOW, policy)
        assert h.status == "ok"


class TestExpectedNext:
    def test_expected_next(self) -> None:
        latest = _ts(3)
        h = assess_source_health("test", latest, _NOW, _POLICY)
        expected = latest + pd.Timedelta(days=7)
        assert h.expected_next is not None
        assert abs((h.expected_next - expected).total_seconds()) < 1
