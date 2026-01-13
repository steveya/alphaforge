import pandas as pd
from alphaforge.features.dataset_builder import build_dataset
from alphaforge.features.dataset_spec import (
    DatasetSpec,
    UniverseSpec,
    TimeSpec,
    FeatureRequest,
    TargetRequest,
)
from alphaforge.features.template import SliceSpec
from alphaforge.features.frame import FeatureFrame
from alphaforge.time.calendar import TradingCalendar
from alphaforge.data.context import DataContext


class MultiDayTinyTemplate:
    name = "md_tiny"
    version = "1.0"
    param_space = {}

    def requires(self, params):
        return []

    def transform(self, ctx: DataContext, params, slice: SliceSpec, state):
        cal = ctx.calendars["XNYS"]
        minutes = cal.trading_minutes_utc(slice.start, slice.end, freq="5min")
        # emit the first minute of each session in the range
        sessions = cal.sessions(str(slice.start.date()), str(slice.end.date()))
        chosen = []
        for s in sessions:
            open_utc = cal.session_open_utc(s)
            close_utc = cal.session_close_utc(s)
            # generate per-session local minute range directly to avoid cross-day filtering
            open_local = open_utc.tz_convert(cal.tz)
            close_local = close_utc.tz_convert(cal.tz)
            rng_local = pd.date_range(
                start=open_local, end=close_local, freq="5min", inclusive="right"
            )
            rng_utc = rng_local.tz_convert("UTC")
            if len(rng_utc):
                chosen.append(rng_utc[0])
        if not chosen:
            return FeatureFrame(
                X=pd.DataFrame(columns=["f"]), catalog=pd.DataFrame(), meta={}
            )
        idx = pd.MultiIndex.from_product(
            [pd.DatetimeIndex(chosen), slice.entities], names=["ts_utc", "entity_id"]
        )
        X = pd.DataFrame({"f": [1.0] * len(idx)}, index=idx)
        catalog = pd.DataFrame([{"feature_id": "f", "family": "md_tiny"}]).set_index(
            "feature_id"
        )
        return FeatureFrame(X=X, catalog=catalog, meta={})


def test_multi_day_intraday_dataset_across_dst():
    # choose a range that includes DST start (US 2021-03-14)
    cal = TradingCalendar("XNYS", tz="America/New_York")
    ctx = DataContext(sources={}, calendars={"XNYS": cal}, store=None)

    start = pd.Timestamp("2021-03-12T00:00:00Z")
    end = pd.Timestamp("2021-03-16T23:59:59Z")

    spec = DatasetSpec(
        universe=UniverseSpec(entities=["AAA"]),
        time=TimeSpec(start=start, end=end, calendar="XNYS", grid="5min", asof=None),
        target=TargetRequest(template=MultiDayTinyTemplate()),
        features=[FeatureRequest(template=MultiDayTinyTemplate())],
    )

    art = build_dataset(ctx, spec, persist=False)

    # should produce at least one row per session
    X = art.X
    assert len(X) >= 1
    # timestamps should be tz-aware UTC
    assert X.index.get_level_values("ts_utc").tz is not None
