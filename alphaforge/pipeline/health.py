"""Source health policy and assessment for data pipelines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from alphaforge.pit.release_rules import ReleaseRule


@dataclass(frozen=True)
class SourceHealthPolicy:
    """Configures how to respond when a data source goes stale."""

    expected_cadence: pd.Timedelta
    release_rule: ReleaseRule | None = None  # precise schedule (optional)
    grace_period: pd.Timedelta = pd.Timedelta(days=2)
    stale_threshold: pd.Timedelta = pd.Timedelta(days=14)
    dead_threshold: pd.Timedelta = pd.Timedelta(days=42)
    weight_decay_start: pd.Timedelta | None = None  # defaults to stale_threshold
    weight_decay_half_life: pd.Timedelta = pd.Timedelta(days=7)


@dataclass(frozen=True)
class SourceHealthStatus:
    """Health assessment for a single data source at a point in time."""

    source_name: str
    latest_obs_date: pd.Timestamp | None
    asof: pd.Timestamp
    age: pd.Timedelta | None
    age_days: float | None
    status: str  # "ok", "late", "stale", "dead", "empty"
    weight_factor: float
    expected_next: pd.Timestamp | None
    message: str


def assess_source_health(
    source_name: str,
    latest_obs_date: pd.Timestamp | None,
    asof: pd.Timestamp,
    policy: SourceHealthPolicy,
) -> SourceHealthStatus:
    """Evaluate the health of a data source."""
    if latest_obs_date is None:
        return SourceHealthStatus(
            source_name=source_name,
            latest_obs_date=None,
            asof=asof,
            age=None,
            age_days=None,
            status="empty",
            weight_factor=0.0,
            expected_next=None,
            message=f"{source_name}: no observations found",
        )

    # Ensure both timestamps are comparable (both tz-aware or both naive)
    _asof = asof
    _latest = latest_obs_date
    if _asof.tzinfo is not None and _latest.tzinfo is None:
        _latest = _latest.tz_localize("UTC")
    elif _asof.tzinfo is None and _latest.tzinfo is not None:
        _asof = _asof.tz_localize("UTC")

    age = _asof - _latest
    age_days = age.total_seconds() / 86400.0
    expected_next = _latest + policy.expected_cadence

    ok_limit = policy.expected_cadence + policy.grace_period
    decay_start = (
        policy.weight_decay_start
        if policy.weight_decay_start is not None
        else policy.stale_threshold
    )

    if age <= ok_limit:
        status = "ok"
        weight = 1.0
        msg = f"{source_name}: OK (age {age_days:.0f}d)"
    elif age <= policy.stale_threshold:
        status = "late"
        weight = 1.0
        msg = f"{source_name}: late (age {age_days:.0f}d, expected every {policy.expected_cadence.days}d)"
    elif age <= policy.dead_threshold:
        status = "stale"
        decay_age = age - decay_start
        hl = policy.weight_decay_half_life
        weight = max(0.0, min(1.0, 2.0 ** (-decay_age / hl)))
        msg = (
            f"{source_name}: stale (age {age_days:.0f}d, weight {weight:.2f})"
        )
    else:
        status = "dead"
        weight = 0.0
        msg = (
            f"{source_name}: dead (age {age_days:.0f}d, "
            f"exceeds dead threshold {policy.dead_threshold.days}d)"
        )

    return SourceHealthStatus(
        source_name=source_name,
        latest_obs_date=latest_obs_date,
        asof=asof,
        age=age,
        age_days=age_days,
        status=status,
        weight_factor=weight,
        expected_next=expected_next,
        message=msg,
    )


# ---------------------------------------------------------------------------
# Default policies for known sources
# ---------------------------------------------------------------------------

CFTC_COT_HEALTH_POLICY = SourceHealthPolicy(
    expected_cadence=pd.Timedelta(days=7),
    grace_period=pd.Timedelta(days=3),
    stale_threshold=pd.Timedelta(days=21),
    dead_threshold=pd.Timedelta(days=56),
    weight_decay_half_life=pd.Timedelta(days=7),
)

DTCC_PPD_HEALTH_POLICY = SourceHealthPolicy(
    expected_cadence=pd.Timedelta(days=1),
    grace_period=pd.Timedelta(days=3),
    stale_threshold=pd.Timedelta(days=7),
    dead_threshold=pd.Timedelta(days=30),
    weight_decay_half_life=pd.Timedelta(days=3),
)


# ---------------------------------------------------------------------------
# Generalized aliases (Source prefix dropped for generic usage)
# ---------------------------------------------------------------------------

HealthPolicy = SourceHealthPolicy
HealthStatus = SourceHealthStatus
assess_health = assess_source_health
