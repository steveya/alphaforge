from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

import pandas as pd
from pandas.tseries.offsets import MonthEnd


class RefFreq(str, Enum):
    A = "A"
    Q = "Q"
    M = "M"


def _ts_utc_midnight(value: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.floor("D")


@dataclass(frozen=True)
class RefPeriod:
    freq: RefFreq
    year: int
    period: int

    @staticmethod
    def parse(s: str) -> "RefPeriod":
        text = str(s).strip()
        if not text:
            raise ValueError("Reference period string is required.")

        match = re.match(r"^(\d{4})[Qq]([1-4])$", text)
        if match:
            return RefPeriod(RefFreq.Q, int(match.group(1)), int(match.group(2)))

        match = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", text)
        if match:
            try:
                ts = pd.Timestamp(text)
            except ValueError as exc:
                raise ValueError(f"Invalid reference period date: {text}") from exc
            if ts.day != (ts + MonthEnd(0)).day:
                raise ValueError(
                    "Date reference periods must be month-end (YYYY-MM-DD)."
                )
            return RefPeriod(RefFreq.M, ts.year, ts.month)

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

        raise ValueError(
            "Invalid reference period format. Expected YYYY, YYYYQq, YYYY-MM, "
            "YYYY/MM, or YYYY-MM-DD."
        )

    def to_key(self) -> str:
        if self.freq == RefFreq.A:
            return f"{self.year:04d}"
        if self.freq == RefFreq.Q:
            return f"{self.year:04d}Q{self.period}"
        if self.freq == RefFreq.M:
            return f"{self.year:04d}-{self.period:02d}"
        raise ValueError(f"Unsupported reference frequency: {self.freq}")

    def end_obs_date(self) -> pd.Timestamp:
        if self.freq == RefFreq.A:
            ts = pd.Timestamp(self.year, 12, 31, tz="UTC")
        elif self.freq == RefFreq.Q:
            month = self.period * 3
            ts = pd.Timestamp(self.year, month, 1, tz="UTC") + MonthEnd(0)
        elif self.freq == RefFreq.M:
            ts = pd.Timestamp(self.year, self.period, 1, tz="UTC") + MonthEnd(0)
        else:
            raise ValueError(f"Unsupported reference frequency: {self.freq}")
        return ts.floor("D")

    @staticmethod
    def from_obs_date_end(ts: pd.Timestamp, freq: RefFreq) -> "RefPeriod":
        obs_ts = _ts_utc_midnight(ts)
        if freq == RefFreq.A:
            period = 1
        elif freq == RefFreq.Q:
            period = (obs_ts.month - 1) // 3 + 1
        elif freq == RefFreq.M:
            period = obs_ts.month
        else:
            raise ValueError(f"Unsupported reference frequency: {freq}")

        candidate = RefPeriod(freq=freq, year=obs_ts.year, period=period)
        if candidate.end_obs_date() != obs_ts:
            raise ValueError(
                "obs_date_end does not match the end of the requested reference period."
            )
        return candidate
