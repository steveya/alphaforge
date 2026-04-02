# alphaforge/features/event_dates.py
"""
Feature template producing event-proximity features for scheduled market events.

Supported events:
  - FOMC: Federal Open Market Committee meeting dates (hardcoded schedule).
  - OpEx: Monthly equity options expiration (3rd Friday of each month).
  - IMM:  SPX futures quarterly expiration / IMM dates (3rd Wednesday of
          March, June, September, December).

For each event type the template produces:
  - is_{event}       : 1.0 on the event date, 0.0 otherwise
  - days_to_{event}  : business days until next event, capped at ``max_days``
  - is_near_{event}  : 1.0 when days_to <= ``near_window``, 0.0 otherwise
"""
from __future__ import annotations

import calendar as _cal
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from .frame import FeatureFrame
from .ids import group_path, make_feature_id
from .template import ParamSpec, SliceSpec

# ---------------------------------------------------------------------------
# FOMC meeting dates  (source: federalreserve.gov historical schedule)
# ---------------------------------------------------------------------------
_FOMC_DATES_RAW: Sequence[str] = (
    # 2000
    "2000-02-02",
    "2000-03-21",
    "2000-05-16",
    "2000-06-28",
    "2000-08-22",
    "2000-10-03",
    "2000-11-15",
    "2000-12-19",
    # 2001  (includes emergency 2001-01-03, 2001-04-18, 2001-09-17)
    "2001-01-03",
    "2001-01-31",
    "2001-03-20",
    "2001-04-18",
    "2001-05-15",
    "2001-06-27",
    "2001-08-21",
    "2001-09-17",
    "2001-10-02",
    "2001-11-06",
    "2001-12-11",
    # 2002
    "2002-01-30",
    "2002-03-19",
    "2002-05-07",
    "2002-06-26",
    "2002-08-13",
    "2002-09-24",
    "2002-11-06",
    "2002-12-10",
    # 2003
    "2003-01-29",
    "2003-03-18",
    "2003-05-06",
    "2003-06-25",
    "2003-08-12",
    "2003-09-16",
    "2003-10-28",
    "2003-12-09",
    # 2004
    "2004-01-28",
    "2004-03-16",
    "2004-05-04",
    "2004-06-30",
    "2004-08-10",
    "2004-09-21",
    "2004-11-10",
    "2004-12-14",
    # 2005
    "2005-02-02",
    "2005-03-22",
    "2005-05-03",
    "2005-06-30",
    "2005-08-09",
    "2005-09-20",
    "2005-11-01",
    "2005-12-13",
    # 2006
    "2006-01-31",
    "2006-03-28",
    "2006-05-10",
    "2006-06-29",
    "2006-08-08",
    "2006-09-20",
    "2006-10-25",
    "2006-12-12",
    # 2007
    "2007-01-31",
    "2007-03-21",
    "2007-05-09",
    "2007-06-28",
    "2007-08-07",
    "2007-08-17",  # emergency discount-rate
    "2007-09-18",
    "2007-10-31",
    "2007-12-11",
    # 2008  (includes emergency 2008-01-22, 2008-03-11, 2008-10-08)
    "2008-01-22",
    "2008-01-30",
    "2008-03-11",
    "2008-03-18",
    "2008-04-30",
    "2008-06-25",
    "2008-08-05",
    "2008-09-16",
    "2008-10-08",
    "2008-10-29",
    "2008-12-16",
    # 2009
    "2009-01-28",
    "2009-03-18",
    "2009-04-29",
    "2009-06-24",
    "2009-08-12",
    "2009-09-23",
    "2009-11-04",
    "2009-12-16",
    # 2010
    "2010-01-27",
    "2010-03-16",
    "2010-04-28",
    "2010-06-23",
    "2010-08-10",
    "2010-09-21",
    "2010-11-03",
    "2010-12-14",
    # 2011
    "2011-01-26",
    "2011-03-15",
    "2011-04-27",
    "2011-06-22",
    "2011-08-09",
    "2011-09-21",
    "2011-11-02",
    "2011-12-13",
    # 2012
    "2012-01-25",
    "2012-03-13",
    "2012-04-25",
    "2012-06-20",
    "2012-08-01",
    "2012-09-13",
    "2012-10-24",
    "2012-12-12",
    # 2013
    "2013-01-30",
    "2013-03-20",
    "2013-05-01",
    "2013-06-19",
    "2013-07-31",
    "2013-09-18",
    "2013-10-30",
    "2013-12-18",
    # 2014
    "2014-01-29",
    "2014-03-19",
    "2014-04-30",
    "2014-06-18",
    "2014-07-30",
    "2014-09-17",
    "2014-10-29",
    "2014-12-17",
    # 2015
    "2015-01-28",
    "2015-03-18",
    "2015-04-29",
    "2015-06-17",
    "2015-07-29",
    "2015-09-17",
    "2015-10-28",
    "2015-12-16",
    # 2016
    "2016-01-27",
    "2016-03-16",
    "2016-04-27",
    "2016-06-15",
    "2016-07-27",
    "2016-09-21",
    "2016-11-02",
    "2016-12-14",
    # 2017
    "2017-02-01",
    "2017-03-15",
    "2017-05-03",
    "2017-06-14",
    "2017-07-26",
    "2017-09-20",
    "2017-11-01",
    "2017-12-13",
    # 2018
    "2018-01-31",
    "2018-03-21",
    "2018-05-02",
    "2018-06-13",
    "2018-08-01",
    "2018-09-26",
    "2018-11-08",
    "2018-12-19",
    # 2019
    "2019-01-30",
    "2019-03-20",
    "2019-05-01",
    "2019-06-19",
    "2019-07-31",
    "2019-09-18",
    "2019-10-04",  # emergency repo
    "2019-10-30",
    "2019-12-11",
    # 2020  (includes emergency 2020-03-03, 2020-03-15)
    "2020-01-29",
    "2020-03-03",
    "2020-03-15",
    "2020-04-29",
    "2020-06-10",
    "2020-07-29",
    "2020-09-16",
    "2020-11-05",
    "2020-12-16",
    # 2021
    "2021-01-27",
    "2021-03-17",
    "2021-04-28",
    "2021-06-16",
    "2021-07-28",
    "2021-09-22",
    "2021-11-03",
    "2021-12-15",
    # 2022
    "2022-01-26",
    "2022-03-16",
    "2022-05-04",
    "2022-06-15",
    "2022-07-27",
    "2022-09-21",
    "2022-11-02",
    "2022-12-14",
    # 2023
    "2023-02-01",
    "2023-03-22",
    "2023-05-03",
    "2023-06-14",
    "2023-07-26",
    "2023-09-20",
    "2023-11-01",
    "2023-12-13",
    # 2024
    "2024-01-31",
    "2024-03-20",
    "2024-05-01",
    "2024-06-12",
    "2024-07-31",
    "2024-09-18",
    "2024-11-07",
    "2024-12-18",
    # 2025
    "2025-01-29",
    "2025-03-19",
    "2025-05-07",
    "2025-06-18",
    "2025-07-30",
    "2025-09-17",
    "2025-10-29",
    "2025-12-17",
    # 2026
    "2026-01-28",
    "2026-03-18",
    "2026-04-29",
    "2026-06-17",
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-09",
)


def _fomc_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Return FOMC meeting dates within [start, end]."""
    all_dates = pd.to_datetime(list(_FOMC_DATES_RAW))
    mask = (all_dates >= start) & (all_dates <= end)
    return pd.DatetimeIndex(all_dates[mask])


# ---------------------------------------------------------------------------
# Options expiration (3rd Friday of each month)
# ---------------------------------------------------------------------------


def _third_weekday(year: int, month: int, weekday: int) -> pd.Timestamp:
    """Return the 3rd occurrence of *weekday* (0=Mon…6=Sun) in year/month."""
    # first day of month
    first_dow = _cal.weekday(year, month, 1)  # 0=Mon
    # days until first target weekday
    offset = (weekday - first_dow) % 7
    day = 1 + offset + 14  # +14 → 3rd occurrence
    return pd.Timestamp(year=year, month=month, day=day)


def _monthly_opex(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """3rd Friday of every month between start and end."""
    dates: list[pd.Timestamp] = []
    y, m = start.year, start.month
    while True:
        ts = _third_weekday(y, m, 4)  # Friday = 4
        if ts > end:
            break
        if ts >= start:
            dates.append(ts)
        m += 1
        if m > 12:
            m = 1
            y += 1
    return pd.DatetimeIndex(dates)


def _imm_dates(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """
    IMM dates: 3rd Wednesday of March, June, September, December.
    These are SPX futures quarterly expiration dates.
    """
    dates: list[pd.Timestamp] = []
    for y in range(start.year, end.year + 1):
        for m in (3, 6, 9, 12):
            ts = _third_weekday(y, m, 2)  # Wednesday = 2
            if start <= ts <= end:
                dates.append(ts)
    return pd.DatetimeIndex(dates)


# ---------------------------------------------------------------------------
# Proximity helper
# ---------------------------------------------------------------------------


def _days_to_next(
    index: pd.DatetimeIndex,
    event_dates: pd.DatetimeIndex,
    cap: int,
) -> np.ndarray:
    """
    For each date in *index*, compute the number of business days
    until the next event in *event_dates*.  Capped at *cap*.
    """
    # Normalise both to tz-naive for consistent comparison
    idx_naive = index.tz_convert(None) if index.tz is not None else index
    ev_naive = (
        event_dates.tz_convert(None) if event_dates.tz is not None else event_dates
    )

    result = np.full(len(idx_naive), float(cap))

    # Sort events for searchsorted
    events_sorted = np.sort(ev_naive.values)
    if len(events_sorted) == 0:
        return result

    # Vectorised: find position of next event for each date
    positions = np.searchsorted(events_sorted, idx_naive.values, side="left")

    for i in range(len(idx_naive)):
        pos = positions[i]
        if pos < len(events_sorted):
            if idx_naive.values[i] == events_sorted[pos]:
                result[i] = 0.0
            else:
                # Count business days between (exclusive)
                dt = idx_naive[i]
                ne = pd.Timestamp(events_sorted[pos])
                bdays = np.busday_count(dt.date(), ne.date())
                result[i] = min(float(bdays), float(cap))
    return result


# ---------------------------------------------------------------------------
# EventDateTemplate
# ---------------------------------------------------------------------------


@dataclass
class EventDateTemplate:
    """
    Feature template for scheduled market event proximity features.

    Produces binary indicator, business-day countdown, and near-event flag
    for FOMC meetings, monthly equity options expiration (OpEx), and
    quarterly SPX futures expiration (IMM dates).

    Parameters (via ``params`` dict):
        calendar : str          Exchange calendar name (default "XNYS").
        events   : tuple[str]   Which event families to generate.
                                Choices: "fomc", "opex", "imm".
        max_days : int          Cap for days_to_* countdown (default 30).
        near_window : int       Threshold for is_near_* flag (default 3).
    """

    name: str = "event_dates"
    version: str = "1.0"

    param_space: Dict[str, ParamSpec] = field(
        default_factory=lambda: {
            "calendar": ParamSpec("categorical", default="XNYS", choices=None),
            "events": ParamSpec(
                "categorical",
                default=("fomc", "opex", "imm"),
                choices=None,
            ),
            "max_days": ParamSpec("int", default=30, low=1, high=252),
            "near_window": ParamSpec("int", default=3, low=0, high=30),
        }
    )

    # ------------------------------------------------------------------

    def requires(self, params: Dict[str, Any]) -> List[Tuple[str, Any]]:
        return []  # calendar + timestamps only; no data source needed

    def fit(self, ctx, params: Dict[str, Any], fit_slice: SliceSpec):
        return None  # stateless

    def transform(
        self,
        ctx,
        params: Dict[str, Any],
        slice: SliceSpec,
        state,
    ) -> FeatureFrame:
        cal_name = str(params.get("calendar", "XNYS"))
        events_requested = params.get("events", ("fomc", "opex", "imm"))
        max_days = int(params.get("max_days", 30))
        near_window = int(params.get("near_window", 3))

        if slice.entities is None or len(slice.entities) == 0:
            raise ValueError("EventDateTemplate requires slice.entities")

        # Build session dates from calendar
        cal = ctx.calendars[cal_name]
        start = (
            pd.Timestamp(slice.start).tz_convert(None)
            if pd.Timestamp(slice.start).tzinfo
            else pd.Timestamp(slice.start)
        )
        end = (
            pd.Timestamp(slice.end).tz_convert(None)
            if pd.Timestamp(slice.end).tzinfo
            else pd.Timestamp(slice.end)
        )

        sessions = pd.to_datetime(cal.sessions(str(start.date()), str(end.date())))

        idx = pd.MultiIndex.from_product(
            [sessions, list(slice.entities)],
            names=["ts_utc", "entity_id"],
        )

        dates_rep = pd.DatetimeIndex(idx.get_level_values("ts_utc"))
        # Strip tz for consistent comparison with tz-naive event dates
        dates_norm = dates_rep.normalize()
        dates_norm_naive = (
            dates_norm.tz_convert(None) if dates_norm.tz is not None else dates_norm
        )

        X_cols: Dict[str, pd.Series] = {}
        cat: List[Dict[str, Any]] = []
        gp = group_path("events", "event_dates", {"calendar": cal_name})

        def _add_col(col_name: str, values, meta: Dict[str, Any]):
            fid = make_feature_id(f"events.{cal_name}", "*", "events", col_name, {})
            X_cols[fid] = pd.Series(np.asarray(values, dtype=float), index=idx)
            cat.append(
                {
                    "feature_id": fid,
                    "group_path": gp,
                    "family": "events",
                    "transform": col_name,
                    "source_table": f"events.{cal_name}",
                    "source_col": "(derived)",
                    **meta,
                }
            )

        events_requested = (
            tuple(events_requested)
            if isinstance(events_requested, (list, tuple))
            else (events_requested,)
        )

        # --- Event generators ---
        _generators = {
            "fomc": lambda: _fomc_dates(start, end),
            "opex": lambda: _monthly_opex(start, end),
            "imm": lambda: _imm_dates(start, end),
        }

        for ev_name in events_requested:
            if ev_name not in _generators:
                raise ValueError(
                    f"Unknown event '{ev_name}'. Choose from: {list(_generators)}"
                )
            ev_dates = _generators[ev_name]()
            ev_set = set(ev_dates.normalize())

            # is_{event}
            is_event = np.array(
                [1.0 if d in ev_set else 0.0 for d in dates_norm_naive],
                dtype=float,
            )
            _add_col(f"is_{ev_name}", is_event, {"event": ev_name})

            # days_to_{event}
            days_arr = _days_to_next(dates_norm_naive, ev_dates, cap=max_days)
            _add_col(
                f"days_to_{ev_name}", days_arr, {"event": ev_name, "max_days": max_days}
            )

            # is_near_{event}
            is_near = (days_arr <= near_window).astype(float)
            _add_col(
                f"is_near_{ev_name}",
                is_near,
                {"event": ev_name, "near_window": near_window},
            )

        X = pd.DataFrame(X_cols, index=idx).sort_index()
        catalog = pd.DataFrame(cat).set_index("feature_id").sort_index()

        return FeatureFrame(
            X=X,
            catalog=catalog,
            meta={"template": self.name, "version": self.version},
        )
