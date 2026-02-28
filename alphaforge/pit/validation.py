from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

_REQUIRED_COLUMNS = ("series_key", "obs_date", "asof_utc", "value")


def _is_naive_timestamp(value: Any) -> bool:
    ts = pd.Timestamp(value)
    return ts.tzinfo is None


def _coerce_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


@dataclass(frozen=True)
class PITValidationReport:
    rows: int
    missing_required_columns: tuple[str, ...] = field(default_factory=tuple)
    null_required_rows: int = 0
    duplicate_key_rows: int = 0
    future_rows: int = 0
    invalid_datetime_rows: int = 0
    naive_asof_rows: int = 0
    naive_release_rows: int = 0

    @property
    def has_errors(self) -> bool:
        return bool(
            self.missing_required_columns
            or self.null_required_rows
            or self.duplicate_key_rows
            or self.future_rows
            or self.invalid_datetime_rows
            or self.naive_asof_rows
            or self.naive_release_rows
        )

    def to_error_message(self) -> str:
        parts: list[str] = []
        if self.missing_required_columns:
            parts.append(f"missing_required_columns={list(self.missing_required_columns)}")
        if self.null_required_rows:
            parts.append(f"null_required_rows={self.null_required_rows}")
        if self.duplicate_key_rows:
            parts.append(f"duplicate_key_rows={self.duplicate_key_rows}")
        if self.future_rows:
            parts.append(f"future_rows={self.future_rows}")
        if self.invalid_datetime_rows:
            parts.append(f"invalid_datetime_rows={self.invalid_datetime_rows}")
        if self.naive_asof_rows:
            parts.append(f"naive_asof_rows={self.naive_asof_rows}")
        if self.naive_release_rows:
            parts.append(f"naive_release_rows={self.naive_release_rows}")
        return "PIT observation validation failed: " + ", ".join(parts)


def validate_pit_observations(df: pd.DataFrame) -> PITValidationReport:
    missing_required_columns = tuple(c for c in _REQUIRED_COLUMNS if c not in df.columns)
    rows = int(len(df))
    if missing_required_columns:
        return PITValidationReport(rows=rows, missing_required_columns=missing_required_columns)

    null_required_rows = int(df[list(_REQUIRED_COLUMNS)].isna().any(axis=1).sum())

    duplicate_key_rows = int(
        df.duplicated(subset=["series_key", "obs_date", "asof_utc"], keep=False).sum()
    )

    invalid_datetime_rows = 0
    naive_asof_rows = 0
    naive_release_rows = 0
    future_rows = 0

    obs_values = df["obs_date"].tolist()
    asof_values = df["asof_utc"].tolist()
    release_values = df["release_time_utc"].tolist() if "release_time_utc" in df.columns else []

    obs_utc: list[pd.Timestamp | None] = []
    asof_utc: list[pd.Timestamp | None] = []

    for obs_val, asof_val in zip(obs_values, asof_values, strict=False):
        if pd.isna(obs_val) or pd.isna(asof_val):
            obs_utc.append(None)
            asof_utc.append(None)
            continue
        try:
            if _is_naive_timestamp(asof_val):
                naive_asof_rows += 1
            obs_utc.append(_coerce_utc(obs_val))
            asof_utc.append(_coerce_utc(asof_val))
        except Exception:
            invalid_datetime_rows += 1
            obs_utc.append(None)
            asof_utc.append(None)

    if "release_time_utc" in df.columns:
        for release_val in release_values:
            if pd.isna(release_val):
                continue
            try:
                if _is_naive_timestamp(release_val):
                    naive_release_rows += 1
                _coerce_utc(release_val)
            except Exception:
                invalid_datetime_rows += 1

    for obs_ts, asof_ts in zip(obs_utc, asof_utc, strict=False):
        if obs_ts is None or asof_ts is None:
            continue
        if obs_ts > asof_ts:
            future_rows += 1

    return PITValidationReport(
        rows=rows,
        missing_required_columns=missing_required_columns,
        null_required_rows=null_required_rows,
        duplicate_key_rows=duplicate_key_rows,
        future_rows=future_rows,
        invalid_datetime_rows=invalid_datetime_rows,
        naive_asof_rows=naive_asof_rows,
        naive_release_rows=naive_release_rows,
    )

