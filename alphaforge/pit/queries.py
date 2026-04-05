"""Typed ref-period PIT query surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Mapping

import pandas as pd

from alphaforge.time.ref_period import (
    ObsDateAnchor,
    RefFreq,
    RefPeriod,
    coerce_ref_period,
    normalize_obs_date_anchor,
    normalize_ref_freq,
)

from .exceptions import PITContractError

RefPeriodLike = RefPeriod | str | pd.Period | pd.Timestamp | date | datetime


def _coerce_series_key(value: object) -> str:
    series_key = str(value).strip()
    if not series_key:
        raise PITContractError("series_key is required.")
    return series_key


def _coerce_timestamp(value: object, *, field_name: str) -> pd.Timestamp:
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise PITContractError(f"{field_name} must be datetime-like.") from exc
    if pd.isna(ts):
        raise PITContractError(f"{field_name} must be datetime-like.")
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _coerce_optional_timestamp(value: object | None, *, field_name: str) -> pd.Timestamp | None:
    if value is None:
        return None
    return _coerce_timestamp(value, field_name=field_name)


def _coerce_ref(
    value: object,
    *,
    field_name: str,
    freq: RefFreq | None,
    obs_date_anchor: ObsDateAnchor,
) -> RefPeriod:
    try:
        return coerce_ref_period(value, freq=freq, obs_date_anchor=obs_date_anchor)
    except ValueError as exc:
        raise PITContractError(f"{field_name}: {exc}") from exc


@dataclass(frozen=True)
class RefSnapshotQuery:
    """Ref-period snapshot request for :meth:`alphaforge.pit.accessor.PITAccessor.snapshot_ref`."""

    series_key: str
    asof: pd.Timestamp
    start_ref: RefPeriodLike | None = None
    end_ref: RefPeriodLike | None = None
    freq: RefFreq | str | None = None
    obs_date_anchor: ObsDateAnchor | str = "end"


@dataclass(frozen=True)
class RefRevisionQuery:
    """Ref-period revision request for :meth:`alphaforge.pit.accessor.PITAccessor.revisions_ref`."""

    series_key: str
    ref: RefPeriodLike
    start_asof: pd.Timestamp | None = None
    end_asof: pd.Timestamp | None = None
    freq: RefFreq | str | None = None
    obs_date_anchor: ObsDateAnchor | str = "end"


def coerce_ref_snapshot_query(
    query: RefSnapshotQuery | Mapping[str, Any],
) -> RefSnapshotQuery:
    """Normalize a ref-period snapshot query into a validated typed object."""

    if isinstance(query, RefSnapshotQuery):
        candidate = query
    elif isinstance(query, Mapping):
        try:
            candidate = RefSnapshotQuery(
                series_key=query["series_key"],
                asof=query["asof"],
                start_ref=query.get("start_ref"),
                end_ref=query.get("end_ref"),
                freq=query.get("freq"),
                obs_date_anchor=query.get("obs_date_anchor", "end"),
            )
        except KeyError as exc:
            raise PITContractError(f"Ref snapshot query missing required field: {exc.args[0]}") from exc
    else:
        raise PITContractError("Ref snapshot query must be RefSnapshotQuery or a mapping.")

    series_key = _coerce_series_key(candidate.series_key)
    asof = _coerce_timestamp(candidate.asof, field_name="asof")
    obs_date_anchor = normalize_obs_date_anchor(candidate.obs_date_anchor)
    freq = normalize_ref_freq(candidate.freq)

    start_ref: RefPeriod | None = None
    if candidate.start_ref is not None:
        start_ref = _coerce_ref(
            candidate.start_ref,
            field_name="start_ref",
            freq=freq,
            obs_date_anchor=obs_date_anchor,
        )
        if freq is None:
            freq = start_ref.freq

    end_ref: RefPeriod | None = None
    if candidate.end_ref is not None:
        end_ref = _coerce_ref(
            candidate.end_ref,
            field_name="end_ref",
            freq=freq,
            obs_date_anchor=obs_date_anchor,
        )
        if freq is None:
            freq = end_ref.freq

    if freq is None:
        raise PITContractError(
            "Ref snapshot query requires freq or at least one bounded reference period."
        )

    if start_ref is not None and end_ref is not None:
        if start_ref.obs_date(anchor=obs_date_anchor) > end_ref.obs_date(anchor=obs_date_anchor):
            raise PITContractError("start_ref must be <= end_ref.")

    return RefSnapshotQuery(
        series_key=series_key,
        asof=asof,
        start_ref=start_ref,
        end_ref=end_ref,
        freq=freq,
        obs_date_anchor=obs_date_anchor,
    )


def coerce_ref_revision_query(
    query: RefRevisionQuery | Mapping[str, Any],
) -> RefRevisionQuery:
    """Normalize a ref-period revision query into a validated typed object."""

    if isinstance(query, RefRevisionQuery):
        candidate = query
    elif isinstance(query, Mapping):
        try:
            candidate = RefRevisionQuery(
                series_key=query["series_key"],
                ref=query["ref"],
                start_asof=query.get("start_asof"),
                end_asof=query.get("end_asof"),
                freq=query.get("freq"),
                obs_date_anchor=query.get("obs_date_anchor", "end"),
            )
        except KeyError as exc:
            raise PITContractError(
                f"Ref revision query missing required field: {exc.args[0]}"
            ) from exc
    else:
        raise PITContractError("Ref revision query must be RefRevisionQuery or a mapping.")

    series_key = _coerce_series_key(candidate.series_key)
    obs_date_anchor = normalize_obs_date_anchor(candidate.obs_date_anchor)
    freq = normalize_ref_freq(candidate.freq)
    ref = _coerce_ref(
        candidate.ref,
        field_name="ref",
        freq=freq,
        obs_date_anchor=obs_date_anchor,
    )
    start_asof = _coerce_optional_timestamp(candidate.start_asof, field_name="start_asof")
    end_asof = _coerce_optional_timestamp(candidate.end_asof, field_name="end_asof")
    if start_asof is not None and end_asof is not None and start_asof > end_asof:
        raise PITContractError("start_asof must be <= end_asof.")

    return RefRevisionQuery(
        series_key=series_key,
        ref=ref,
        start_asof=start_asof,
        end_asof=end_asof,
        freq=ref.freq,
        obs_date_anchor=obs_date_anchor,
    )
