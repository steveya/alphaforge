# alphaforge/features/calendar_flags.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import pandas as pd

from .frame import FeatureFrame
from .ids import group_path, make_feature_id
from .template import ParamSpec, SliceSpec


@dataclass
class CalendarFlagsTemplate:
    """
    Non-OHLC feature template based on the TradingCalendar and timestamps only.

    Produces:
      - dow: integer 0..6 (or one-hot if one_hot=True)
      - is_month_end / is_quarter_end / is_year_end
      - month: integer 1..12
    """

    name: str = "calendar_flags"
    version: str = "1.0"

    param_space = {
        "calendar": ParamSpec("categorical", default="XNYS", choices=None),
        "flags": ParamSpec(
            "categorical", default=("dow", "is_month_end"), choices=None
        ),
        "one_hot_dow": ParamSpec("bool", default=False),
    }

    def requires(self, params: Dict[str, Any]) -> List[Tuple[str, Any]]:
        # no data source required; uses calendar + time grid only
        return []

    def transform(
        self, ctx, params: Dict[str, Any], slice: SliceSpec, state
    ) -> FeatureFrame:
        cal_name = str(params.get("calendar", "XNYS"))
        flags = params.get("flags", ("dow", "is_month_end"))
        one_hot = bool(params.get("one_hot_dow", False))

        if slice.entities is None or len(slice.entities) == 0:
            raise ValueError("CalendarFlagsTemplate requires slice.entities")

        # sessions are returned tz-naive in your current TradingCalendar
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

        sessions = cal.sessions(str(start.date()), str(end.date()))
        # sessions returned as tz-aware UTC timestamps already
        sessions = pd.to_datetime(sessions)

        idx = pd.MultiIndex.from_product(
            [sessions, list(slice.entities)],
            names=["ts_utc", "entity_id"],
        )

        X_cols: Dict[str, pd.Series] = {}
        cat: List[Dict[str, Any]] = []
        gp = group_path("calendar", "flags", {"calendar": cal_name})

        # Base time index (repeated for entities)
        dates_rep = idx.get_level_values("ts_utc")

        def _add_col(col_name: str, values, meta: Dict[str, Any]):
            fid = make_feature_id(f"calendar.{cal_name}", "*", "calendar", col_name, {})
            vals = values.values if hasattr(values, "values") else values
            X_cols[fid] = pd.Series(vals, index=idx)
            cat.append(
                {
                    "feature_id": fid,
                    "group_path": gp,
                    "family": "calendar",
                    "transform": col_name,
                    "source_table": f"calendar.{cal_name}",
                    "source_col": "(derived)",
                    **meta,
                }
            )

        flags = tuple(flags) if isinstance(flags, (list, tuple)) else (flags,)

        if "dow" in flags:
            dow = dates_rep.dayofweek.astype(int)
            if one_hot:
                for k in range(7):
                    _add_col(
                        f"dow_{k}",
                        (dow == k).astype(float),
                        {"dow": k, "one_hot": True},
                    )
            else:
                _add_col("dow", dow.astype(float), {"one_hot": False})

        if "is_month_end" in flags:
            _add_col("is_month_end", dates_rep.is_month_end.astype(float), {})

        if "is_quarter_end" in flags:
            _add_col("is_quarter_end", dates_rep.is_quarter_end.astype(float), {})

        if "is_year_end" in flags:
            _add_col("is_year_end", dates_rep.is_year_end.astype(float), {})

        if "month" in flags:
            _add_col("month", dates_rep.month.astype(float), {})

        X = pd.DataFrame(X_cols, index=idx).sort_index()
        catalog = pd.DataFrame(cat).set_index("feature_id").sort_index()

        return FeatureFrame(
            X=X, catalog=catalog, meta={"template": self.name, "version": self.version}
        )
