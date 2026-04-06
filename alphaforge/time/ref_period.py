from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Literal

import pandas as pd
from pandas.tseries.offsets import MonthEnd

ObsDateAnchor = Literal["start", "end"]
RefPeriodInput = "RefPeriod | str | pd.Period | pd.Timestamp | date | datetime"


class RefFreq(str, Enum):
    A = "A"
    Q = "Q"
    M = "M"


def _ts_utc_midnight(value: pd.Timestamp | date | datetime | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("D")


def normalize_ref_freq(value: RefFreq | str | None) -> RefFreq | None:
    if value is None:
        return None
    if isinstance(value, RefFreq):
        return value

    text = str(value).strip().upper()
    aliases = {
        "A": RefFreq.A,
        "Y": RefFreq.A,
        "A-DEC": RefFreq.A,
        "Y-DEC": RefFreq.A,
        "YEAR": RefFreq.A,
        "ANNUAL": RefFreq.A,
        "Q": RefFreq.Q,
        "Q-DEC": RefFreq.Q,
        "QUARTER": RefFreq.Q,
        "QUARTERLY": RefFreq.Q,
        "M": RefFreq.M,
        "MONTH": RefFreq.M,
        "MONTHLY": RefFreq.M,
    }
    try:
        return aliases[text]
    except KeyError as exc:
        raise ValueError(f"Unsupported reference frequency: {value!r}") from exc


def normalize_obs_date_anchor(anchor: ObsDateAnchor | str) -> ObsDateAnchor:
    text = str(anchor).strip().lower()
    if text not in {"start", "end"}:
        raise ValueError(f"obs_date_anchor must be 'start' or 'end', got {anchor!r}")
    return text  # type: ignore[return-value]


def _coerce_explicit_datetime_input(
    value: object,
) -> pd.Timestamp | None:
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return _ts_utc_midnight(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("Reference period string is required.")
        if not re.match(r"^\d{4}([-/]\d{2}){1,2}$", text):
            return None
        try:
            return _ts_utc_midnight(text)
        except (TypeError, ValueError):
            return None
    return None


@dataclass(frozen=True)
class RefPeriod:
    freq: RefFreq
    year: int
    period: int

    @staticmethod
    def parse(
        value: object,
        *,
        freq: RefFreq | str | None = None,
        obs_date_anchor: ObsDateAnchor | str = "end",
    ) -> "RefPeriod":
        requested_freq = normalize_ref_freq(freq)
        ref_period: RefPeriod | None

        if isinstance(value, RefPeriod):
            ref_period = value
        elif isinstance(value, pd.Period):
            ref_period = RefPeriod.from_period(value)
        else:
            text = str(value).strip()
            if not text:
                raise ValueError("Reference period string is required.")

            ref_period = _parse_text_ref_period(text)
            if (
                ref_period is not None
                and requested_freq is not None
                and ref_period.freq != requested_freq
            ):
                obs_ts = _coerce_explicit_datetime_input(value)
                if obs_ts is not None:
                    ref_period = RefPeriod.from_obs_date(
                        obs_ts,
                        freq=requested_freq,
                        obs_date_anchor=obs_date_anchor,
                    )
            if ref_period is None:
                obs_ts = _coerce_explicit_datetime_input(value)
                if obs_ts is None or requested_freq is None:
                    raise ValueError(
                        "Invalid reference period format. Expected YYYY, YYYYQq, YYYY-MM, "
                        "YYYY/MM, YYYY-MM-DD, pandas Period, or an explicit observation "
                        "date with freq=... ."
                    )
                ref_period = RefPeriod.from_obs_date(
                    obs_ts,
                    freq=requested_freq,
                    obs_date_anchor=obs_date_anchor,
                )

        if ref_period is None:
            raise ValueError("Reference period could not be resolved.")
        if requested_freq is not None and ref_period.freq != requested_freq:
            raise ValueError(
                "Reference period frequency does not match the requested frequency."
            )
        return ref_period

    @staticmethod
    def from_period(period: pd.Period) -> "RefPeriod":
        freq = normalize_ref_freq(period.freqstr)
        if freq == RefFreq.A:
            return RefPeriod(freq=freq, year=period.year, period=1)
        if freq == RefFreq.Q:
            return RefPeriod(freq=freq, year=period.year, period=period.quarter)
        if freq == RefFreq.M:
            return RefPeriod(freq=freq, year=period.year, period=period.month)
        raise ValueError(f"Unsupported reference frequency: {period.freqstr}")

    @staticmethod
    def from_obs_date(
        value: pd.Timestamp | date | datetime | str,
        *,
        freq: RefFreq | str,
        obs_date_anchor: ObsDateAnchor | str = "end",
    ) -> "RefPeriod":
        resolved_freq = normalize_ref_freq(freq)
        if resolved_freq is None:
            raise ValueError("freq is required when normalizing observation dates.")

        obs_ts = _ts_utc_midnight(value)
        anchor = normalize_obs_date_anchor(obs_date_anchor)

        if resolved_freq == RefFreq.A:
            candidate = RefPeriod(freq=resolved_freq, year=obs_ts.year, period=1)
        elif resolved_freq == RefFreq.Q:
            candidate = RefPeriod(
                freq=resolved_freq,
                year=obs_ts.year,
                period=((obs_ts.month - 1) // 3) + 1,
            )
        elif resolved_freq == RefFreq.M:
            candidate = RefPeriod(freq=resolved_freq, year=obs_ts.year, period=obs_ts.month)
        else:
            raise ValueError(f"Unsupported reference frequency: {resolved_freq}")

        if candidate.obs_date(anchor=anchor) != obs_ts:
            raise ValueError(
                "obs_date does not match the requested reference period under the "
                f"{anchor!r} anchor."
            )
        return candidate

    def to_key(self) -> str:
        if self.freq == RefFreq.A:
            return f"{self.year:04d}"
        if self.freq == RefFreq.Q:
            return f"{self.year:04d}Q{self.period}"
        if self.freq == RefFreq.M:
            return f"{self.year:04d}-{self.period:02d}"
        raise ValueError(f"Unsupported reference frequency: {self.freq}")

    def __str__(self) -> str:
        return self.to_key()

    def start_obs_date(self) -> pd.Timestamp:
        if self.freq == RefFreq.A:
            return pd.Timestamp(self.year, 1, 1, tz="UTC")
        if self.freq == RefFreq.Q:
            month = ((self.period - 1) * 3) + 1
            return pd.Timestamp(self.year, month, 1, tz="UTC")
        if self.freq == RefFreq.M:
            return pd.Timestamp(self.year, self.period, 1, tz="UTC")
        raise ValueError(f"Unsupported reference frequency: {self.freq}")

    def end_obs_date(self) -> pd.Timestamp:
        return self.obs_date(anchor="end")

    def obs_date(self, anchor: ObsDateAnchor | str = "end") -> pd.Timestamp:
        resolved_anchor = normalize_obs_date_anchor(anchor)
        if resolved_anchor == "start":
            return self.start_obs_date().floor("D")
        if self.freq == RefFreq.A:
            end = pd.Timestamp(self.year, 12, 31, tz="UTC")
        elif self.freq == RefFreq.Q:
            month = self.period * 3
            end = pd.Timestamp(self.year, month, 1, tz="UTC") + MonthEnd(0)
        else:
            end = self.start_obs_date() + MonthEnd(0)
        return end.floor("D")

    @staticmethod
    def from_obs_date_end(ts: pd.Timestamp, freq: RefFreq | str) -> "RefPeriod":
        return RefPeriod.from_obs_date(ts, freq=freq, obs_date_anchor="end")


def coerce_ref_period(
    value: object,
    *,
    freq: RefFreq | str | None = None,
    obs_date_anchor: ObsDateAnchor | str = "end",
) -> RefPeriod:
    return RefPeriod.parse(value, freq=freq, obs_date_anchor=obs_date_anchor)


def _parse_text_ref_period(text: str) -> RefPeriod | None:
    match = re.match(r"^(\d{4})[Qq]([1-4])$", text)
    if match:
        return RefPeriod(RefFreq.Q, int(match.group(1)), int(match.group(2)))

    match = re.match(r"^(\d{4})[-/](\d{2})$", text)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        if not 1 <= month <= 12:
            raise ValueError(f"Invalid reference period month: {text}")
        return RefPeriod(RefFreq.M, year, month)

    match = re.match(r"^(\d{4})$", text)
    if match:
        return RefPeriod(RefFreq.A, int(match.group(1)), 1)

    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", text)
    if match:
        ts = pd.Timestamp(text)
        if ts.day != (ts + MonthEnd(0)).day:
            return None
        return RefPeriod(RefFreq.M, ts.year, ts.month)

    return None


__all__ = [
    "ObsDateAnchor",
    "RefPeriodInput",
    "RefFreq",
    "RefPeriod",
    "normalize_ref_freq",
    "normalize_obs_date_anchor",
    "coerce_ref_period",
]
